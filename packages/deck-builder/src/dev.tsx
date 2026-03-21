import { render } from "solid-js/web";
import { DeckBuilder } from ".";
import type { Deck } from "@gi-tcg/typings";
import { createEffect, createSignal } from "solid-js";
import { AssetsManager } from "@gi-tcg/assets-manager";

const EMPTY_DECK: Deck = {
  characters: [],
  cards: [],
};

function App() {
  const [deck, setDeck] = createSignal<Deck>(EMPTY_DECK);
  createEffect(() => {
    console.log(deck());
  });
  const assetsManager = new AssetsManager({
    apiEndpoint: `https://static-data.piovium.org/api/v4`
  })
  return (
    <DeckBuilder
      assetsManager={assetsManager}
      class="mobile"
      deck={deck()}
      onChangeDeck={setDeck}
      // version="v3.3.0"
    />
  );
}

render(() => <App />, document.getElementById("root")!);
