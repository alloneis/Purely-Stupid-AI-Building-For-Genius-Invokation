import { DiceType, type PbDiceRequirement } from "@gi-tcg/typings";
import { type ComponentProps, createMemo, For, splitProps } from "solid-js";

import { Dice, type DiceColor } from "./Dice";
import { isDeepEqual } from "remeda";
import { Key } from "@solid-primitives/keyed";

interface DiceCostProps extends ComponentProps<"div"> {
  cost: readonly PbDiceRequirement[];
  size: number;
  realCost?: readonly PbDiceRequirement[];
}

export function DiceCost(props: DiceCostProps) {
  const [local, restProps] = splitProps(props, ["cost", "size", "realCost"]);
  const diceMap = createMemo(
    () => {
      const costMap = new Map(
        local.cost.map(({ type, count }) => [type as DiceType, count]),
      );
      const realCostMap = new Map(
        local.realCost?.map(({ type, count }) => [type as DiceType, count]),
      );
      type DiceTuple = readonly [
        type: DiceType,
        count: number,
        color: DiceColor,
      ];
      let result: DiceTuple[] = [];
      if (local.realCost) {
        for (const [type, originalCount] of costMap) {
          const realCount = realCostMap.get(type) ?? 0;
          const color =
            realCount > originalCount
              ? "increased"
              : realCount < originalCount
                ? "decreased"
                : "normal";
          result.push([type, realCount, color]);
          realCostMap.delete(type);
        }
        result.push(
          ...realCostMap
            .entries()
            .filter(([, count]) => count > 0)
            .map(([type, count]) => [type, count, "increased"] as const),
        );
      } else {
        result = costMap
          .entries()
          .map(([type, count]) => [type, count, "normal"] as const)
          .toArray();
      }
      return result;
    },
    [],
    { equals: isDeepEqual },
  );
  return (
    <div {...restProps}>
      <Key each={diceMap()} by={0} /* by-type */>
        {(item) => (
          <Dice
            type={item()[0]}
            text={item()[0] === DiceType.Legend ? "" : `${item()[1]}`}
            size={local.size}
            color={item()[2]}
          />
        )}
      </Key>
    </div>
  );
}
