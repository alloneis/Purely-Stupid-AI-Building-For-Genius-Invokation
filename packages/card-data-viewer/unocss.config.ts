import { defineConfig, presetUno, transformerDirectives } from "unocss";

export default defineConfig({
  presets: [presetUno()],
  // https://github.com/unocss/unocss/discussions/3444
  postprocess: (obj) => {
    obj.selector = ".gi-tcg-card-data-viewer " + obj.selector;
  },
  transformers: [transformerDirectives()],
});
