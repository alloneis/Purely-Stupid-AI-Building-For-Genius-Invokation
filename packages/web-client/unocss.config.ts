import {
  defineConfig,
  presetUno,
  transformerDirectives,
  transformerVariantGroup,
} from "unocss";
import presetUna from "@una-ui/preset";
import presetIcons from "@unocss/preset-icons";

export default defineConfig<object>({
  presets: [
    presetUno(),
    presetIcons({
      collections: {
        mdi: () => import("@iconify-json/mdi").then((i) => i.icons),
      },
    }),
    presetUna({
      primary: "yellow" as any,
    }),
  ],
  transformers: [transformerDirectives(), transformerVariantGroup()],
});
