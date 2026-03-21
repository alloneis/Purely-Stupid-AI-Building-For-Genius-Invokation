import { resolve } from "node:path";
import { defaultClientConditions, defineConfig, Plugin } from "vite";
import unoCss from "unocss/vite";
import solid from "vite-plugin-solid";
import nodeExternals from "rollup-plugin-node-externals";
import dts from "vite-plugin-dts";

export default defineConfig({
  resolve: {
    conditions: ["bun", ...defaultClientConditions],
  },
  plugins: [
    {
      ...nodeExternals(),
      enforce: "pre",
    },
    unoCss(),
    solid(),
    !process.env.NO_TYPING && dts({ rollupTypes: true }),
  ],
  build: {
    sourcemap: true,
    lib: {
      entry: resolve(__dirname, "src/index.ts"),
      formats: ["es"],
      fileName: "index",
      cssFileName: "style",
    },
  },
});
