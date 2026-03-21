import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    extension: "./src/extension.ts",
  },
  external: ["vscode"],
  format: "cjs",
  clean: true,
  sourcemap: true,
  minify: true,
});
