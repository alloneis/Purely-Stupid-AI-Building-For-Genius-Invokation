/* @refresh reload */
import { render } from "solid-js/web";
import { inject } from "@vercel/analytics";

import "./style.css";

(async () => {
  if (import.meta.env.PROD) {
    await import("core-js");
  }
  const { App } = await import("./App");
  const root = document.getElementById("root")!;
  root.innerHTML = "";
  render(() => <App />, root);
})();

inject();
