
import { Route, Router } from "@solidjs/router";
import {
  createContext,
  createResource,
  createSignal,
  onMount,
  onCleanup,
  Resource,
  useContext,
  type Accessor,
  lazy,
} from "solid-js";
import axios from "axios";
import { useAuth } from "./auth";

const Home = lazy(() => import("./pages/Home"));
const User = lazy(() => import("./pages/User"));
const Decks = lazy(() => import("./pages/Decks"));
const EditDeck = lazy(() => import("./pages/EditDeck"));
const Room = lazy(() => import("./pages/Room"));
// Temporarily not lazy loading AIBattle for debugging
import AIBattle from "./pages/AIBattle";
const NotFound = lazy(() => import("./pages/NotFound"));

export interface VersionContextValue {
  versionInfo: Resource<any>;
}

const VersionContext = createContext<VersionContextValue>({
  versionInfo: createResource(() => Promise.resolve({}))[0],
});
export const useVersionContext = () => useContext(VersionContext)!;

const MobileContext = createContext<Accessor<boolean>>();
export const useMobile = () => useContext(MobileContext)!;

function App() {
  const [versionInfo] = createResource(() =>
    axios.get("version").then((res) => res.data),
  );
  const versionContextValue: VersionContextValue = {
    versionInfo,
  };

  const mobileMediaQuery = window.matchMedia("(max-width: 768px)");
  const [mobile, setMobile] = createSignal(mobileMediaQuery.matches);
  const handleMobileChange = (e: MediaQueryListEvent) => {
    setMobile(e.matches);
  };

  const { refresh } = useAuth();
  const onReceiveToken = async (e: MessageEvent) => {
    if (e.data && e.data.type === "login" && e.data.token) {
      localStorage.setItem("accessToken", e.data.token);
      window.githubOAuthPopup?.close();
      await refresh();
    }
  };

  onMount(() => {
    mobileMediaQuery.addEventListener("change", handleMobileChange);
    window.addEventListener("message", onReceiveToken);
  });
  onCleanup(() => {
    mobileMediaQuery.removeEventListener("change", handleMobileChange);
    window.removeEventListener("message", onReceiveToken);
  });

  return (
    <VersionContext.Provider value={versionContextValue}>
      <MobileContext.Provider value={mobile}>
        <Router base={import.meta.env.BASE_URL.replace(/(.+)\/$/, "$1")}>
          <Route path="/" component={Home} />
          <Route path="/user/:id" component={User} />
          <Route path="/decks/:id" component={EditDeck} />
          <Route path="/decks" component={Decks} />
          <Route path="/rooms/:code" component={Room} />
          <Route path="/ai-battle" component={AIBattle} />
          <Route path="*" component={NotFound} />
        </Router>
      </MobileContext.Provider>
    </VersionContext.Provider>
  );
}

export default App;
