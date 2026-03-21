import { createSignal } from "solid-js";

export interface MessageBoxController {
  confirm: (question: string) => Promise<boolean>;
}

export function createMessageBox() {
  let dialogEl: HTMLDialogElement | undefined;
  const [question, setQuestion] = createSignal<string>("");
  let resolver: PromiseWithResolvers<boolean> | null = null;

  return [
    {
      confirm: async (q) => {
        setQuestion(q), (resolver = Promise.withResolvers<boolean>());
        dialogEl?.showModal();
        return await resolver.promise.finally(() => {
          dialogEl?.close();
        });
      },
    } as MessageBoxController,
    () => (
      <MessageBox
        ref={dialogEl!}
        question={question()}
        onConfirm={() => resolver?.resolve(true)}
        onCancel={() => resolver?.resolve(false)}
      />
    ),
  ] as const;
}

interface MessageBoxProps {
  ref: HTMLDialogElement;
  question: string;
  onConfirm: () => void;
  onCancel: () => void;
}

function MessageBox(props: MessageBoxProps) {
  return (
    <dialog
      ref={props.ref}
      class="bg-#ebdab7 p-4 rounded-3 shadow-lg w-96 h-48 border-#735a3f border-2"
    >
      <p class="h-24 font-size-6 font-bold mt-4 text-center">
        {props.question}
      </p>
      <div class="flex justify-center gap-2">
        <button
          class="px-3 py-1 w-36 font-bold font-size-5 color-black bg-#e9e2d3 rounded-full border-#735a3f b-2 hover:bg-#e9e2d3 hover:shadow-[inset_0_0_16px_rgba(255,255,255,1)] hover:border-white"
          onClick={props.onCancel}
        >
          取消
        </button>
        <button
          class="px-3 py-1 w-36 font-bold font-size-5 color-black bg-#e9e2d3 rounded-full border-#735a3f b-2 hover:bg-#e9e2d3 hover:shadow-[inset_0_0_16px_rgba(255,255,255,1)] hover:border-white"
          onClick={props.onConfirm}
        >
          确定
        </button>
      </div>
    </dialog>
  );
}
