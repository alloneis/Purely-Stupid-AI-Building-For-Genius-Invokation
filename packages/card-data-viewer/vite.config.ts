import { resolve } from "node:path";
import { defineConfig } from "vite";
import unoCss from "unocss/vite";
// import devtools from "solid-devtools/vite";
import solid from "vite-plugin-solid";
import nodeExternals from "rollup-plugin-node-externals";
import dts from "vite-plugin-dts";

export default defineConfig({
  plugins: [
    {
      ...nodeExternals({ exclude: /\.css$/ }),
      enforce: "pre",
    },
    unoCss(),
    solid(),
    !process.env.NO_TYPING && dts({ rollupTypes: true }),
  ],
  build: {
    sourcemap: true,
    lib: {
      entry: resolve(__dirname, "src/index.tsx"),
      formats: ["es"],
      fileName: "index",
      cssFileName: "style",
    },
  },
});
