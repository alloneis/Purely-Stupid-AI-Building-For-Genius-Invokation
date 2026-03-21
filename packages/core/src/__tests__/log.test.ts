import { test, expect } from "bun:test";
import { DetailLogType, DetailLogger } from "../log";

test("detail logger", () => {
  const logger = new DetailLogger();
  logger.log(DetailLogType.Other, "top level");
  {
    using _ = logger.subLog(DetailLogType.Skill, "skill");
    logger.log(DetailLogType.Other, "in skill");
  }
  logger.log(DetailLogType.Other, "top level again");

  const logs = logger.getLogs();
  expect(logs).toEqual([
    { type: DetailLogType.Other, value: "top level" },
    {
      type: DetailLogType.Skill,
      value: "skill",
      children: [{ type: DetailLogType.Other, value: "in skill" }],
    },
    { type: DetailLogType.Other, value: "top level again" },
  ]);
});
