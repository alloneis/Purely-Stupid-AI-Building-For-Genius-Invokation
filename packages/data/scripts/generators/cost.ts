import type { PlayCost } from "@gi-tcg/assets-manager";

function costMap(s: string) {
  if (s === "GCG_COST_ENERGY") return "Energy";
  return s[14] + s.substring(15).toLowerCase();
}

export function inlineCostDescription(cost: PlayCost[]): string {
  return cost.map((c) => `${c.count}*${costMap(c.type)}`).join(", ");
}

export function isLegend(playcost: PlayCost[]) {
  return playcost.find((c) => c.type === "GCG_COST_LEGEND");
}

export function getCostCode(playCost: PlayCost[]): string {
  let resultArr = playCost
    .filter((c) => c.type !== "GCG_COST_LEGEND")
    .map((c) => `.cost${costMap(c.type)}(${c.count})`);
  if (isLegend(playCost)) {
    resultArr.push(".legend()");
  }
  if (resultArr.length > 0) {
    return `\n  ${resultArr.join("\n  ")}`;
  } else {
    return "";
  }
}
