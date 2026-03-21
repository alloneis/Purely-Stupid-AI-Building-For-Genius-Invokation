import { resolveOfficialVersion, type Version } from "@gi-tcg/core";
import { registry, scope } from "./begin";

scope.end();
registry.freeze();

export { registry };

export default (version?: Version) => {
  return registry.resolve((x) => resolveOfficialVersion(x, version))
};
