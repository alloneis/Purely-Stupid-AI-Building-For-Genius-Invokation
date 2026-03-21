import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "./src/index.ts"
  },
  format: "esm",
  clean: true,
  sourcemap: true,
  dts: !process.env.NO_TYPING,
  minify: true,
});
