import { render } from "solid-js/web";

import "./style.css";
import "virtual:uno.css";
import "@una-ui/preset/una.css";
import "@unocss/reset/tailwind-compat.css";

import axios from "axios";
import { BACKEND_BASE_URL } from "./config";

async function prepareServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }
  await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`, {
    scope: import.meta.env.BASE_URL,
  });
  navigator.serviceWorker.ready.then((sw) => {
    sw.active?.postMessage({
      type: "config",
      payload: {
        backendBaseUrl: BACKEND_BASE_URL,
      }
    });
  });
}

async function main() {
  await prepareServiceWorker();
  if (import.meta.env.PROD) {
    await import("core-js");
  }
  axios.defaults.baseURL = BACKEND_BASE_URL;
  axios.interceptors.request.use((config) => {
    if (config.url?.includes("https://")) {
      // non-backend request
      return config;
    }
    const accessToken = localStorage.getItem("accessToken");
    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`;
    }
    return config;
  });

  const app = document.getElementById("app")!;
  const { default: App } = await import("./App");
  render(() => <App />, app);
}

main();
