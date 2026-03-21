import { defineConfig, presetWind3, transformerDirectives } from "unocss";

export default defineConfig({
  presets: [presetWind3()],
  transformers: [transformerDirectives()],
  postprocess: (obj) => {
    const scope = `.gi-tcg-chessboard-new`;
    obj.selector += `:where(${scope},${scope} *)`;
  },
});
