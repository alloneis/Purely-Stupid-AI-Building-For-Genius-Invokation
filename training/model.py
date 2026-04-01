"""
TCGNeuralEvaluator — Set Transformer model for Genius Invokation TCG.

All default dimensions are sourced from:
  packages/core/src/decoupled/neural/constants.ts

Architecture:

  ┌─────────────────┐ ┌────────────────┐ ┌────────────────┐
  │ global_features  │ │ self_characters │ │ oppo_characters │
  │    [B, 27]       │ │   [B, 3, 32]   │ │   [B, 3, 32]   │
  └────────┬─────────┘ └──┬─────────────┘ └──┬─────────────┘
           │           entity pool          entity pool
           │         [B,3,ent_summary]    [B,3,ent_summary]
           │              concat              concat
           │         [B,3,32+pool_d]     [B,3,32+pool_d]
           │              flatten              flatten
           └────────────────┬──────────────────┘
                            ▼
                     Context MLP → [B, 256]
                            │
      ┌──────────┬──────────┼──────────┬──────────┐
      │          │          │          │          │
      ▼          ▼          │          ▼          ▼
  hand_cards  summons    context   supports  combat_sts
  [B,10,16]   [B,4,16]  [B,256]   [B,4,16]  [B,10,16]
    │ embed     │ embed      │     │ embed    │ embed
    │ ISAB+PMA  │ PMA        │     │ PMA      │ PMA
    ▼           ▼            │     ▼          ▼
  [B,256]    [B,128]         │   [B,128]   [B,128]
      └──────────┴───────────┴─────┴──────────┘
                            ▼
                     Trunk MLP → [B, 256]
                            │
                    ┌───────┼───────┐
                    ▼       ▼       ▼
                Value   Policy   Auxiliary
                [B,1]   [B,128]  ├─ next_hp      [B,1]
                tanh   log_sfmx  ├─ card_play    [B,10]  sigmoid
                                 ├─ oppo_belief  [B,16]  sigmoid
                                 ├─ kill_pred    [B,6]   sigmoid (3 self + 3 oppo)
                                 ├─ reaction     [B,1]   sigmoid
                                 └─ dice_eff     [B,1]

Total parameters: ~1.4 M  (well under the 3 M budget)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelOutput(NamedTuple):
    value: torch.Tensor        # [B, 1]  ∈ [-1, 1]
    log_policy: torch.Tensor   # [B, action_slots]  masked log-probs
    next_hp: torch.Tensor      # [B, 1]  predicted active-char HP in 5 turns (/ 10)
    card_play: torch.Tensor    # [B, max_hand]  sigmoid: playable next turn?
    oppo_belief: torch.Tensor  # [B, CARD_FEATURE_DIM]  opponent hand aggregate
    kill_pred: torch.Tensor    # [B, 2*max_chars]  sigmoid: char death within 3 turns
    reaction_pred: torch.Tensor  # [B, 1]  sigmoid: reaction triggers on next attack
    dice_efficiency: torch.Tensor  # [B, 1]  effective actions from current dice

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Constants — must match packages/core/src/decoupled/neural/constants.ts ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GLOBAL_FEATURE_DIM = 27  # 1+1+1 + 8+8 + 1+1+1+1+1+1+1+1
CHARACTER_FEATURE_DIM = 32  # 6 scalar + 7 aura + 2 binary + 7 elem + 6 weapon + 4 counts
CARD_FEATURE_DIM = 16  # 1 id(raw→embedding) + 6 type + 1 cost + 1 fast + 6 tag bools + 1 effectless
ENTITY_FEATURE_DIM = 16  # 1 id(raw→embedding) + 6 type + 4 vars + 4 tag bools + 1 visible
CARD_STRUCT_DIM = CARD_FEATURE_DIM - 1    # structural features (everything except raw id)
ENTITY_STRUCT_DIM = ENTITY_FEATURE_DIM - 1
MAX_CHARACTERS = 3
MAX_HAND_CARDS = 10
MAX_SUMMONS = 4
MAX_SUPPORTS = 4
MAX_COMBAT_STATUSES = 10
MAX_CHARACTER_ENTITIES = 8
MAX_ACTION_SLOTS = 128


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Model Configuration                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class ModelConfig:
    # --- encoder dimensions (from TS) ---
    global_dim: int = GLOBAL_FEATURE_DIM
    char_dim: int = CHARACTER_FEATURE_DIM
    card_dim: int = CARD_FEATURE_DIM
    entity_dim: int = ENTITY_FEATURE_DIM
    max_chars: int = MAX_CHARACTERS
    max_hand: int = MAX_HAND_CARDS
    max_summons: int = MAX_SUMMONS
    max_supports: int = MAX_SUPPORTS
    max_combat_statuses: int = MAX_COMBAT_STATUSES
    max_char_entities: int = MAX_CHARACTER_ENTITIES
    action_slots: int = MAX_ACTION_SLOTS
    # --- ID embedding (replaces raw definitionId scalar) ---
    num_id_buckets: int = 512
    id_embed_dim: int = 8
    # --- model hyper-parameters ---
    d_context: int = 256
    d_set: int = 128
    d_trunk: int = 256
    n_heads: int = 4
    hand_seeds: int = 2
    summon_seeds: int = 1
    support_seeds: int = 1
    combat_status_seeds: int = 1
    char_entity_pool_dim: int = 32
    hand_inducing: int = 4
    ff_mult: int = 2
    use_isab_hand: bool = True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Attention Primitives  (manual, ONNX-safe — no nn.MultiheadAttention)  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class CrossAttention(nn.Module):
    """Scaled dot-product cross-attention with optional padding mask."""

    def __init__(self, d_q: int, d_kv: int, n_heads: int) -> None:
        super().__init__()
        assert d_q % n_heads == 0, f"d_q={d_q} not divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = d_q // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_q, d_q)
        self.k_proj = nn.Linear(d_kv, d_q)
        self.v_proj = nn.Linear(d_kv, d_q)
        self.out_proj = nn.Linear(d_q, d_q)

    def forward(
        self,
        q: torch.Tensor,  # [B, Sq, d_q]
        kv: torch.Tensor,  # [B, Skv, d_kv]
        pad_mask: torch.Tensor | None = None,  # [B, Skv]  True = IGNORE
    ) -> torch.Tensor:
        B, Sq, _ = q.shape
        Skv = kv.size(1)
        H, D = self.n_heads, self.head_dim

        q_h = self.q_proj(q).reshape(B, Sq, H, D).transpose(1, 2)  # [B,H,Sq,D]
        k_h = self.k_proj(kv).reshape(B, Skv, H, D).transpose(1, 2)
        v_h = self.v_proj(kv).reshape(B, Skv, H, D).transpose(1, 2)

        scores = (q_h @ k_h.transpose(-2, -1)) * self.scale  # [B,H,Sq,Skv]
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], -1e9)
        attn = scores.softmax(dim=-1)

        out = (attn @ v_h).transpose(1, 2).reshape(B, Sq, H * D)
        return self.out_proj(out)


class MAB(nn.Module):
    """Multihead Attention Block: LN(X + Attn(X,Y)) → LN(· + FFN(·))"""

    def __init__(self, d: int, d_kv: int, n_heads: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.attn = CrossAttention(d, d_kv, n_heads)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * ff_mult),
            nn.GELU(),
            nn.Linear(d * ff_mult, d),
        )
        self.ln2 = nn.LayerNorm(d)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(q + self.attn(q, kv, pad_mask))
        return self.ln2(h + self.ff(h))


class ISAB(nn.Module):
    """Induced Set Attention Block — compresses set through *m* inducing pts."""

    def __init__(self, d: int, n_heads: int, m: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, m, d) * 0.02)
        self.mab_down = MAB(d, d, n_heads, ff_mult)  # I ← attend(I, X)
        self.mab_up = MAB(d, d, n_heads, ff_mult)  # X ← attend(X, H)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        I = self.inducing.expand(x.size(0), -1, -1)
        H = self.mab_down(I, x, pad_mask)  # [B, m, d]
        return self.mab_up(x, H)  # [B, S, d]  (H has no padding)


class PMA(nn.Module):
    """Pooling by Multihead Attention with *k* learned seed vectors."""

    def __init__(self, d: int, n_heads: int, k: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, k, d) * 0.02)
        self.mab = MAB(d, d, n_heads, ff_mult)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        S = self.seeds.expand(x.size(0), -1, -1)
        return self.mab(S, x, pad_mask)  # [B, k, d]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SetEncoder  =  project → [ISAB] → PMA → flatten                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class SetEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d: int,
        n_heads: int,
        n_seeds: int,
        n_inducing: int = 4,
        use_isab: bool = True,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.GELU())
        self.isab = ISAB(d, n_heads, n_inducing, ff_mult) if use_isab else None
        self.pma = PMA(d, n_heads, n_seeds, ff_mult)
        self.output_dim = n_seeds * d

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [B, S, in_dim]  raw features
            valid_mask: [B, S]          1.0 = real item, 0.0 = padding
        Returns:
            [B, n_seeds * d]  flattened pool
        """
        h = self.proj(x)
        pad_mask = ~valid_mask.bool()  # True = ignore
        if self.isab is not None:
            h = self.isab(h, pad_mask)
        pooled = self.pma(h, pad_mask)  # [B, seeds, d]
        return pooled.flatten(1)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TCGNeuralEvaluator  —  the full dual-headed model                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TCGNeuralEvaluator(nn.Module):
    """
    Dual-headed AlphaZero-style evaluator for Genius Invokation TCG
    with auxiliary prediction heads for sample-efficient training.

    Inputs:  dict of Float32 tensors matching the TS NeuralStateEncoder.
    Outputs: ModelOutput(value, log_policy, next_hp, card_play, oppo_belief,
                         kill_pred, reaction_pred, dice_efficiency)
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        c = cfg or ModelConfig()
        self.cfg = c

        # ── 0. Learned ID embeddings (replaces raw definitionId) ──────
        self.card_id_embed = nn.Embedding(c.num_id_buckets, c.id_embed_dim)
        self.entity_id_embed = nn.Embedding(c.num_id_buckets, c.id_embed_dim)
        card_in = CARD_STRUCT_DIM + c.id_embed_dim     # 15 + 8 = 23
        entity_in = ENTITY_STRUCT_DIM + c.id_embed_dim  # 15 + 8 = 23

        # ── 1. Character entity pooling (masked mean → summary per char) ─
        self.char_entity_proj = nn.Sequential(
            nn.Linear(entity_in, c.char_entity_pool_dim),
            nn.GELU(),
        )
        augmented_char_dim = c.char_dim + c.char_entity_pool_dim

        # ── 2. Fixed-feature encoder ──────────────────────────────────
        fixed_dim = c.global_dim + c.max_chars * augmented_char_dim * 2
        self.context_mlp = nn.Sequential(
            nn.Linear(fixed_dim, c.d_context),
            nn.GELU(),
            nn.LayerNorm(c.d_context),
            nn.Linear(c.d_context, c.d_context),
            nn.GELU(),
            nn.LayerNorm(c.d_context),
        )

        # ── 2. Set encoders for variable-length inputs ────────────────
        self.hand_enc = SetEncoder(
            in_dim=card_in,
            d=c.d_set,
            n_heads=c.n_heads,
            n_seeds=c.hand_seeds,
            n_inducing=c.hand_inducing,
            use_isab=c.use_isab_hand,
            ff_mult=c.ff_mult,
        )
        self.summon_enc = SetEncoder(
            in_dim=entity_in,
            d=c.d_set,
            n_heads=c.n_heads,
            n_seeds=c.summon_seeds,
            n_inducing=2,
            use_isab=False,
            ff_mult=c.ff_mult,
        )

        self.support_enc = SetEncoder(
            in_dim=entity_in,
            d=c.d_set,
            n_heads=c.n_heads,
            n_seeds=c.support_seeds,
            n_inducing=2,
            use_isab=False,
            ff_mult=c.ff_mult,
        )
        self.combat_status_enc = SetEncoder(
            in_dim=entity_in,
            d=c.d_set,
            n_heads=c.n_heads,
            n_seeds=c.combat_status_seeds,
            n_inducing=2,
            use_isab=False,
            ff_mult=c.ff_mult,
        )

        # ── 4. Trunk ──────────────────────────────────────────────────
        trunk_in = (
            c.d_context
            + self.hand_enc.output_dim
            + self.summon_enc.output_dim
            + self.support_enc.output_dim * 2    # self + oppo
            + self.combat_status_enc.output_dim * 2  # self + oppo
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, c.d_trunk),
            nn.GELU(),
            nn.LayerNorm(c.d_trunk),
            nn.Linear(c.d_trunk, c.d_trunk),
            nn.GELU(),
            nn.LayerNorm(c.d_trunk),
        )

        # ── 5. Value head → scalar ∈ [-1, 1] ─────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(c.d_trunk, c.d_trunk // 2),
            nn.GELU(),
            nn.Linear(c.d_trunk // 2, 1),
            nn.Tanh(),
        )

        # ── 6. Policy head → logits of size action_slots ─────────────
        self.policy_head = nn.Sequential(
            nn.Linear(c.d_trunk, c.d_trunk // 2),
            nn.GELU(),
            nn.Linear(c.d_trunk // 2, c.action_slots),
        )

        # ── 7. Auxiliary: active-character HP in 5 turns ───────────────
        self.next_hp_head = nn.Sequential(
            nn.Linear(c.d_trunk, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # ── 8. Auxiliary: per-hand-card playability next turn ──────────
        self.card_playability_head = nn.Sequential(
            nn.Linear(c.d_trunk, 64),
            nn.GELU(),
            nn.Linear(64, c.max_hand),
        )

        # ── 9. Auxiliary: opponent hidden-hand belief state ───────────
        self.oppo_belief_head = nn.Sequential(
            nn.Linear(c.d_trunk, 64),
            nn.GELU(),
            nn.Linear(64, c.card_dim),
        )

        # ── 10. Auxiliary: kill prediction (char death within 3 turns) ─
        self.kill_pred_head = nn.Sequential(
            nn.Linear(c.d_trunk, 64),
            nn.GELU(),
            nn.Linear(64, c.max_chars * 2),  # 3 self + 3 oppo
        )

        # ── 11. Auxiliary: element reaction on next attack ────────────
        self.reaction_pred_head = nn.Sequential(
            nn.Linear(c.d_trunk, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # ── 12. Auxiliary: dice efficiency (effective actions count) ──
        self.dice_efficiency_head = nn.Sequential(
            nn.Linear(c.d_trunk, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _embed_ids(
        self,
        raw: torch.Tensor,          # [B, S, feat_dim]  position 0 = raw id
        embed: nn.Embedding,
    ) -> torch.Tensor:
        """Extract raw ID at position 0, hash→embed, concat with rest."""
        ids = (raw[:, :, 0] * 10000).long().abs() % self.cfg.num_id_buckets
        id_vecs = embed(ids)                          # [B, S, embed_dim]
        struct = raw[:, :, 1:]                        # [B, S, feat_dim-1]
        return torch.cat([id_vecs, struct], dim=-1)   # [B, S, embed_dim + feat_dim-1]

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.policy_head[-1].weight.normal_(std=0.01)
            self.policy_head[-1].bias.zero_()
            self.value_head[-2].weight.normal_(std=0.01)
            self.value_head[-2].bias.zero_()
            self.next_hp_head[-1].weight.normal_(std=0.01)
            self.next_hp_head[-1].bias.zero_()
            self.card_playability_head[-1].weight.normal_(std=0.01)
            self.card_playability_head[-1].bias.zero_()
            self.oppo_belief_head[-1].weight.normal_(std=0.01)
            self.oppo_belief_head[-1].bias.zero_()
            self.kill_pred_head[-1].weight.normal_(std=0.01)
            self.kill_pred_head[-1].bias.zero_()
            self.reaction_pred_head[-1].weight.normal_(std=0.01)
            self.reaction_pred_head[-1].bias.zero_()
            self.dice_efficiency_head[-1].weight.normal_(std=0.01)
            self.dice_efficiency_head[-1].bias.zero_()

    def _pool_char_entities(
        self,
        entities: torch.Tensor,    # [B, max_chars * max_ent, feat_dim]
        mask: torch.Tensor,        # [B, max_chars * max_ent]
    ) -> torch.Tensor:
        """Masked mean pool of character entities → [B, max_chars, pool_dim]."""
        c = self.cfg
        B = entities.size(0)
        ent = entities.view(B, c.max_chars, c.max_char_entities, -1)
        m = mask.view(B, c.max_chars, c.max_char_entities)
        embedded = self._embed_ids(
            ent.reshape(B * c.max_chars, c.max_char_entities, -1),
            self.entity_id_embed,
        )
        projected = self.char_entity_proj(embedded)  # [B*3, E, pool_d]
        projected = projected.view(B, c.max_chars, c.max_char_entities, -1)
        m_expanded = m.unsqueeze(-1)  # [B, 3, E, 1]
        summed = (projected * m_expanded).sum(dim=2)
        count = m_expanded.sum(dim=2).clamp(min=1.0)
        return summed / count  # [B, 3, pool_d]

    # -------------------------------------------------------------- forward
    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> ModelOutput:
        """
        Args:
            batch: dict with float32 tensors —
                global_features          [B, 27]
                self_characters          [B, 3, 32]
                oppo_characters          [B, 3, 32]
                hand_cards               [B, 10, 16]
                hand_mask                [B, 10]       1 = real, 0 = pad
                summons                  [B, 4, 16]
                summons_mask             [B, 4]        1 = real, 0 = pad
                self_supports            [B, 4, 16]
                self_supports_mask       [B, 4]
                oppo_supports            [B, 4, 16]
                oppo_supports_mask       [B, 4]
                self_combat_statuses     [B, 10, 16]
                self_combat_statuses_mask[B, 10]
                oppo_combat_statuses     [B, 10, 16]
                oppo_combat_statuses_mask[B, 10]
                self_char_entities       [B, 24, 16]   (3 chars × 8 ent)
                self_char_entities_mask  [B, 24]
                oppo_char_entities       [B, 24, 16]
                oppo_char_entities_mask  [B, 24]
                action_mask              [B, 128]      1 = legal, 0 = illegal
        Returns:
            ModelOutput(value, log_policy, next_hp, card_play, oppo_belief,
                        kill_pred, reaction_pred, dice_efficiency)
        """
        # --- character entity pooling → augmented character features ---
        sc_ent = self._pool_char_entities(
            batch["self_char_entities"], batch["self_char_entities_mask"],
        )  # [B, 3, pool_d]
        oc_ent = self._pool_char_entities(
            batch["oppo_char_entities"], batch["oppo_char_entities_mask"],
        )  # [B, 3, pool_d]

        sc_aug = torch.cat([batch["self_characters"], sc_ent], dim=-1).flatten(1)
        oc_aug = torch.cat([batch["oppo_characters"], oc_ent], dim=-1).flatten(1)

        # --- fixed context ---
        g = batch["global_features"]
        ctx = self.context_mlp(torch.cat([g, sc_aug, oc_aug], dim=-1))  # [B, d_context]

        # --- embed IDs and replace raw scalar with dense vector ---
        hand_feats = self._embed_ids(batch["hand_cards"], self.card_id_embed)
        summ_feats = self._embed_ids(batch["summons"], self.entity_id_embed)
        self_sup_feats = self._embed_ids(batch["self_supports"], self.entity_id_embed)
        oppo_sup_feats = self._embed_ids(batch["oppo_supports"], self.entity_id_embed)
        self_cs_feats = self._embed_ids(batch["self_combat_statuses"], self.entity_id_embed)
        oppo_cs_feats = self._embed_ids(batch["oppo_combat_statuses"], self.entity_id_embed)

        # --- variable sets ---
        h_pool = self.hand_enc(hand_feats, batch["hand_mask"])
        s_pool = self.summon_enc(summ_feats, batch["summons_mask"])
        self_sup_pool = self.support_enc(self_sup_feats, batch["self_supports_mask"])
        oppo_sup_pool = self.support_enc(oppo_sup_feats, batch["oppo_supports_mask"])
        self_cs_pool = self.combat_status_enc(self_cs_feats, batch["self_combat_statuses_mask"])
        oppo_cs_pool = self.combat_status_enc(oppo_cs_feats, batch["oppo_combat_statuses_mask"])

        # --- trunk ---
        trunk = self.trunk(torch.cat([
            ctx, h_pool, s_pool,
            self_sup_pool, oppo_sup_pool,
            self_cs_pool, oppo_cs_pool,
        ], dim=-1))

        # --- primary heads ---
        value = self.value_head(trunk)
        logits = self.policy_head(trunk)
        logits = logits.masked_fill(~batch["action_mask"].bool(), -1e9)
        log_policy = F.log_softmax(logits, dim=-1)

        # --- auxiliary heads ---
        next_hp = self.next_hp_head(trunk)
        card_play = torch.sigmoid(self.card_playability_head(trunk))
        oppo_belief = torch.sigmoid(self.oppo_belief_head(trunk))
        kill_pred = torch.sigmoid(self.kill_pred_head(trunk))
        reaction_pred = torch.sigmoid(self.reaction_pred_head(trunk))
        dice_efficiency = self.dice_efficiency_head(trunk)

        return ModelOutput(
            value, log_policy, next_hp, card_play, oppo_belief,
            kill_pred, reaction_pred, dice_efficiency,
        )

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (value, policy_probs) — convenience for inference.
        Auxiliary heads are not returned here; use forward() for training."""
        self.eval()
        out = self.forward(batch)
        return out.value, out.log_policy.exp()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Training Loss                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class LossOutput(NamedTuple):
    total: torch.Tensor
    value_loss: torch.Tensor
    policy_loss: torch.Tensor
    hp_loss: torch.Tensor
    card_play_loss: torch.Tensor
    belief_loss: torch.Tensor
    kill_loss: torch.Tensor
    reaction_loss: torch.Tensor
    dice_eff_loss: torch.Tensor


def compute_loss(
    output: ModelOutput,
    target_value: torch.Tensor,    # [B]     TD(λ) return or game outcome ∈ [-1, +1]
    target_policy: torch.Tensor,   # [B, 64] MCTS visit-count distribution
    target_next_hp: torch.Tensor,  # [B]     actual HP of active char in 5 turns (/ 10)
    target_card_played: torch.Tensor,  # [B, max_hand]  binary: card playable next turn?
    hand_mask: torch.Tensor,       # [B, max_hand]  1 = real card, 0 = pad
    target_oppo_hand_features: torch.Tensor,  # [B, CARD_FEATURE_DIM] aggregate opponent hand
    target_kill: torch.Tensor,     # [B, 6]  binary: char dies within 3 turns (3 self + 3 oppo)
    target_reaction: torch.Tensor,  # [B]    binary: reaction triggers on next attack
    target_dice_eff: torch.Tensor,  # [B]    effective action count from current dice (/ 10)
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
    aux_weight: float = 0.1,
    per_sample: bool = False,
) -> LossOutput:
    """
    Combined loss:
        Value_MSE + Policy_CE
        + aux * (HP_MSE + Card_BCE + Belief_MSE + Kill_BCE + Reaction_BCE + DiceEff_MSE)

    If per_sample=True, returns per-sample [B] total (for IS-weighted PER).
    Component losses are always returned as scalars for logging.
    """
    value_sq = (output.value.squeeze(-1) - target_value) ** 2  # [B]
    policy_per = -(target_policy * output.log_policy).sum(dim=-1)  # [B]
    hp_sq = (output.next_hp.squeeze(-1) - target_next_hp) ** 2  # [B]

    card_bce = F.binary_cross_entropy(
        output.card_play, target_card_played, reduction="none",
    )  # [B, H]
    card_per = (card_bce * hand_mask).sum(dim=-1) / hand_mask.sum(dim=-1).clamp(min=1)  # [B]

    belief_sq = ((output.oppo_belief - target_oppo_hand_features) ** 2).mean(dim=-1)  # [B]

    kill_bce = F.binary_cross_entropy(
        output.kill_pred, target_kill, reduction="none",
    ).mean(dim=-1)  # [B]
    reaction_bce = F.binary_cross_entropy(
        output.reaction_pred.squeeze(-1), target_reaction, reduction="none",
    )  # [B]
    dice_sq = (output.dice_efficiency.squeeze(-1) - target_dice_eff) ** 2  # [B]

    total_per_sample = (
        value_weight * value_sq
        + policy_weight * policy_per
        + aux_weight * (
            hp_sq + card_per + belief_sq
            + kill_bce + reaction_bce + dice_sq
        )
    )  # [B]

    total = total_per_sample if per_sample else total_per_sample.mean()
    return LossOutput(
        total,
        value_sq.mean(),
        policy_per.mean(),
        hp_sq.mean(),
        card_per.mean(),
        belief_sq.mean(),
        kill_bce.mean(),
        reaction_bce.mean(),
        dice_sq.mean(),
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Utilities                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _rand_entity_block(
    batch_size: int, slots: int, feat_dim: int, device: str,
) -> torch.Tensor:
    t = torch.randn(batch_size, slots, feat_dim, device=device)
    t[:, :, 0] = torch.randint(0, 5000, (batch_size, slots), device=device).float() / 10000
    return t


def make_dummy_batch(
    batch_size: int = 4, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Create a random batch for smoke-testing / shape verification."""
    char_ent_slots = MAX_CHARACTERS * MAX_CHARACTER_ENTITIES  # 24
    return {
        "global_features": torch.randn(batch_size, GLOBAL_FEATURE_DIM, device=device),
        "self_characters": torch.randn(
            batch_size, MAX_CHARACTERS, CHARACTER_FEATURE_DIM, device=device
        ),
        "oppo_characters": torch.randn(
            batch_size, MAX_CHARACTERS, CHARACTER_FEATURE_DIM, device=device
        ),
        "hand_cards": _rand_entity_block(batch_size, MAX_HAND_CARDS, CARD_FEATURE_DIM, device),
        "hand_mask": torch.ones(batch_size, MAX_HAND_CARDS, device=device),
        "summons": _rand_entity_block(batch_size, MAX_SUMMONS, ENTITY_FEATURE_DIM, device),
        "summons_mask": torch.ones(batch_size, MAX_SUMMONS, device=device),
        "self_supports": _rand_entity_block(batch_size, MAX_SUPPORTS, ENTITY_FEATURE_DIM, device),
        "self_supports_mask": torch.ones(batch_size, MAX_SUPPORTS, device=device),
        "oppo_supports": _rand_entity_block(batch_size, MAX_SUPPORTS, ENTITY_FEATURE_DIM, device),
        "oppo_supports_mask": torch.ones(batch_size, MAX_SUPPORTS, device=device),
        "self_combat_statuses": _rand_entity_block(batch_size, MAX_COMBAT_STATUSES, ENTITY_FEATURE_DIM, device),
        "self_combat_statuses_mask": torch.ones(batch_size, MAX_COMBAT_STATUSES, device=device),
        "oppo_combat_statuses": _rand_entity_block(batch_size, MAX_COMBAT_STATUSES, ENTITY_FEATURE_DIM, device),
        "oppo_combat_statuses_mask": torch.ones(batch_size, MAX_COMBAT_STATUSES, device=device),
        "self_char_entities": _rand_entity_block(batch_size, char_ent_slots, ENTITY_FEATURE_DIM, device),
        "self_char_entities_mask": torch.ones(batch_size, char_ent_slots, device=device),
        "oppo_char_entities": _rand_entity_block(batch_size, char_ent_slots, ENTITY_FEATURE_DIM, device),
        "oppo_char_entities_mask": torch.ones(batch_size, char_ent_slots, device=device),
        "action_mask": (
            torch.rand(batch_size, MAX_ACTION_SLOTS, device=device) > 0.5
        ).float(),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ONNX Export  (for onnxruntime-web in TS worker)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class _OnnxWrapper(nn.Module):
    """Unwrap dict → positional args for torch.onnx.export."""

    def __init__(self, inner: TCGNeuralEvaluator) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self,
        global_features: torch.Tensor,
        self_characters: torch.Tensor,
        oppo_characters: torch.Tensor,
        hand_cards: torch.Tensor,
        hand_mask: torch.Tensor,
        summons: torch.Tensor,
        summons_mask: torch.Tensor,
        self_supports: torch.Tensor,
        self_supports_mask: torch.Tensor,
        oppo_supports: torch.Tensor,
        oppo_supports_mask: torch.Tensor,
        self_combat_statuses: torch.Tensor,
        self_combat_statuses_mask: torch.Tensor,
        oppo_combat_statuses: torch.Tensor,
        oppo_combat_statuses_mask: torch.Tensor,
        self_char_entities: torch.Tensor,
        self_char_entities_mask: torch.Tensor,
        oppo_char_entities: torch.Tensor,
        oppo_char_entities_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch = {
            "global_features": global_features,
            "self_characters": self_characters,
            "oppo_characters": oppo_characters,
            "hand_cards": hand_cards,
            "hand_mask": hand_mask,
            "summons": summons,
            "summons_mask": summons_mask,
            "self_supports": self_supports,
            "self_supports_mask": self_supports_mask,
            "oppo_supports": oppo_supports,
            "oppo_supports_mask": oppo_supports_mask,
            "self_combat_statuses": self_combat_statuses,
            "self_combat_statuses_mask": self_combat_statuses_mask,
            "oppo_combat_statuses": oppo_combat_statuses,
            "oppo_combat_statuses_mask": oppo_combat_statuses_mask,
            "self_char_entities": self_char_entities,
            "self_char_entities_mask": self_char_entities_mask,
            "oppo_char_entities": oppo_char_entities,
            "oppo_char_entities_mask": oppo_char_entities_mask,
            "action_mask": action_mask,
        }
        out = self.inner(batch)
        return (
            out.value, out.log_policy, out.next_hp, out.card_play,
            out.oppo_belief, out.kill_pred, out.reaction_pred, out.dice_efficiency,
        )


_INPUT_NAMES = [
    "global_features",
    "self_characters",
    "oppo_characters",
    "hand_cards",
    "hand_mask",
    "summons",
    "summons_mask",
    "self_supports",
    "self_supports_mask",
    "oppo_supports",
    "oppo_supports_mask",
    "self_combat_statuses",
    "self_combat_statuses_mask",
    "oppo_combat_statuses",
    "oppo_combat_statuses_mask",
    "self_char_entities",
    "self_char_entities_mask",
    "oppo_char_entities",
    "oppo_char_entities_mask",
    "action_mask",
]

_OUTPUT_NAMES = [
    "value", "log_policy", "next_hp", "card_play",
    "oppo_belief", "kill_pred", "reaction_pred", "dice_efficiency",
]


def export_onnx(
    model: TCGNeuralEvaluator,
    path: str = "tcg_evaluator.onnx",
    opset: int = 17,
) -> None:
    """Export to ONNX with dynamic batch axis (uses legacy TorchScript tracer)."""
    model.eval()
    wrapper = _OnnxWrapper(model)
    dummy = make_dummy_batch(batch_size=1)
    args = tuple(dummy[k] for k in _INPUT_NAMES)

    torch.onnx.export(
        wrapper,
        args,
        path,
        input_names=_INPUT_NAMES,
        output_names=_OUTPUT_NAMES,
        dynamic_axes={
            name: {0: "batch"} for name in _INPUT_NAMES + _OUTPUT_NAMES
        },
        opset_version=opset,
        dynamo=False,
    )
    print(f"Exported to {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Smoke Test                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    torch.manual_seed(42)

    model = TCGNeuralEvaluator()
    n = count_parameters(model)
    print(f"Parameters: {n:,}  ({n / 1e6:.2f} M)")
    assert n < 3_000_000, f"Over budget! {n:,} params"

    B = 8
    batch = make_dummy_batch(batch_size=B)
    out = model(batch)

    print(f"Value shape:      {out.value.shape}")          # [8, 1]
    print(f"Log-policy shape: {out.log_policy.shape}")      # [8, 128]
    print(f"Next-HP shape:    {out.next_hp.shape}")         # [8, 1]
    print(f"Card-play shape:  {out.card_play.shape}")       # [8, 10]
    print(f"Oppo-belief shape:{out.oppo_belief.shape}")     # [8, 16]
    print(f"Kill-pred shape:  {out.kill_pred.shape}")       # [8, 6]
    print(f"Reaction shape:   {out.reaction_pred.shape}")   # [8, 1]
    print(f"Dice-eff shape:   {out.dice_efficiency.shape}") # [8, 1]
    print(f"Value range:      [{out.value.min().item():.4f}, {out.value.max().item():.4f}]")
    prob_sum = out.log_policy.exp().sum(dim=-1)
    print(f"Policy prob sum:  [{prob_sum.min().item():.6f}, {prob_sum.max().item():.6f}]")
    print(f"Card-play range:  [{out.card_play.min().item():.4f}, {out.card_play.max().item():.4f}]")
    print(f"Kill-pred range:  [{out.kill_pred.min().item():.4f}, {out.kill_pred.max().item():.4f}]")
    print(f"Reaction range:   [{out.reaction_pred.min().item():.4f}, {out.reaction_pred.max().item():.4f}]")

    # Training round-trip with all losses
    target_v = torch.sign(torch.randn(B))
    target_p = F.softmax(torch.randn(B, MAX_ACTION_SLOTS) * 0.1, dim=-1)
    target_p = target_p * batch["action_mask"]
    target_p = target_p / target_p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target_hp = torch.rand(B) * 0.8
    target_card = (torch.rand(B, MAX_HAND_CARDS) > 0.7).float()
    target_oppo = torch.rand(B, CARD_FEATURE_DIM).clamp(0, 1)
    target_kill = (torch.rand(B, MAX_CHARACTERS * 2) > 0.8).float()
    target_react = (torch.rand(B) > 0.6).float()
    target_dice = torch.rand(B) * 0.5

    loss = compute_loss(
        out, target_v, target_p, target_hp, target_card,
        batch["hand_mask"], target_oppo,
        target_kill, target_react, target_dice,
    )
    loss.total.backward()
    print(
        f"Loss:  total={loss.total.item():.4f}  "
        f"value={loss.value_loss.item():.4f}  "
        f"policy={loss.policy_loss.item():.4f}  "
        f"hp={loss.hp_loss.item():.4f}  "
        f"card={loss.card_play_loss.item():.4f}  "
        f"belief={loss.belief_loss.item():.4f}  "
        f"kill={loss.kill_loss.item():.4f}  "
        f"react={loss.reaction_loss.item():.4f}  "
        f"dice={loss.dice_eff_loss.item():.4f}"
    )

    grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    print(f"Grad norm: {grad_norm:.4f}")

    # Verify per_sample mode for PER IS-weighting
    model.zero_grad()
    out2 = model(batch)
    loss_ps = compute_loss(
        out2, target_v, target_p, target_hp, target_card,
        batch["hand_mask"], target_oppo,
        target_kill, target_react, target_dice,
        per_sample=True,
    )
    assert loss_ps.total.shape == (B,), f"per_sample total should be [B], got {loss_ps.total.shape}"
    is_weights = torch.rand(B)
    is_weights /= is_weights.max()
    (loss_ps.total * is_weights).mean().backward()
    print(f"Per-sample IS-weighted backward OK  (shape: {loss_ps.total.shape})")
    print("All checks passed")
