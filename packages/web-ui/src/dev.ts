import { createClient } from "./index";

const io0 = createClient(document.querySelector("#player0")!, 0);
const io1 = createClient(document.querySelector("#player1")!, 0);

Reflect.set(window, "io0", io0);
Reflect.set(window, "io1", io1);
