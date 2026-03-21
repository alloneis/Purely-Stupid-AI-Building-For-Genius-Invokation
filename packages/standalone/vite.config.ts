import { defaultClientConditions, defineConfig } from "vite";
import solid from "vite-plugin-solid";
import babel from "@rollup/plugin-babel";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  esbuild: {
    target: "ES2020",
  },
  plugins: [
    solid(),
    babel({
      babelHelpers: "bundled",
    }),
    viteStaticCopy({
      watch: null,
      silent: true,
      targets: [
        {
          src: "../data-code-analyzer/src/result.json",
          rename: "data-code-analyze-result.json",
          dest: "."
        }
      ]
    })
  ],
});
