import type { AssetsManager } from "@gi-tcg/assets-manager";
import type { Deck } from "@gi-tcg/typings";

const LEGEND = "GCG_TAG_LEGEND";

const takeRandomOne = <T>(source: T[]) => {
  const i = Math.floor(Math.random() * source.length);
  return source.splice(i, 1)[0];
};

export const generateRandomDeck = async (assets: AssetsManager) => {
  const deckData = await assets.getDeckData();
  const characters = deckData.characters.values().toArray();
  const characterTags: string[] = [];

  const decks: Deck = {
    characters: [],
    cards: [],
  };
  for (let i = 0; i < 3; i++) {
    const ch = takeRandomOne(characters);
    decks.characters.push(ch.id);
    characterTags.push(...ch.tags);
  }

  const actionCards = deckData.actionCards
    .values()
    .flatMap((c) => {
      let included = true;
      if (
        typeof c.relatedCharacterId === "number" &&
        !decks.characters.includes(c.relatedCharacterId)
      ) {
        included &&= false;
      }
      if (
        typeof c.relatedCharacterTag === "number" &&
        characterTags.filter((t) => t === c.relatedCharacterTag).length < 2
      ) {
        included &&= false;
      }
      if (included) {
        return c.tags.includes(LEGEND) ? [c] : [c, c];
      } else {
        return [];
      }
    })
    .toArray();

  for (let i = 0; i < 30; i++) {
    const ac = takeRandomOne(actionCards);
    decks.cards.push(ac.id);
  }

  return decks;
};
