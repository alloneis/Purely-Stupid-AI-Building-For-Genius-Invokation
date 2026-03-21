import {
  defineConfig,
  presetWind3,
  transformerDirectives,
  Variant,
} from "unocss";

export default defineConfig({
  presets: [presetWind3()],
  variants: [
    ...Object.entries({
      DP: "deck-page-control",
      FM: "filter-menu-control",
    }).map<Variant>(([prefix, cls]) => {
      return (matcher) => {
        if (!matcher.startsWith(`${prefix}:`)) {
          return matcher;
        }
        return {
          matcher: matcher.slice(prefix.length + 1),
          layer: "reactive",
          selector: (s) =>
            `${s}:is(.gi-tcg-deck-builder:has(.${cls}:checked) *)`,
        };
      };
    }),
  ],
  // https://github.com/unocss/unocss/discussions/3444
  postprocess: (obj) => {
    const scope = ".gi-tcg-deck-builder";
    obj.selector += `:where(${scope},${scope} *)`;
  },
  transformers: [transformerDirectives()],
});
