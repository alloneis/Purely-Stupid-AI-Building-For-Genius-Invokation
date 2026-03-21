import { resolve } from "node:path";
import { defineConfig } from "vite";
import nodeExternals from "rollup-plugin-node-externals";
import solid from "vite-plugin-solid";
import dts from "vite-plugin-dts";

export default defineConfig({
  plugins: [
    {
      ...nodeExternals(),
      enforce: "pre",
    },
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
