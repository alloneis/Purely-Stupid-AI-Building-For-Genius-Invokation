import { children, type JSX } from "solid-js";
import ButtonNormal from "../svg/ButtonNormal.svg?fb";
import ButtonHover from "../svg/ButtonHover.svg?fb";
import ButtonActive from "../svg/ButtonActive.svg?fb";

export interface ButtonProps {
  class?: string;
  children: JSX.Element;
  onClick: (e: MouseEvent) => void;
}

export function Button(props: ButtonProps) {
  const ch = children(() => props.children);
  return (
    <button
      class={`grid h-10.8 w-45 group/confirm_btn bg-transparent ${
        props.class ?? ""
      }`}
      onClick={(e) => props.onClick(e)}
    >
      <ButtonActive
        class="grid-area-[1/1] w-45 h-10.8 hidden group-active/confirm_btn:block" 
      />
      <ButtonHover
        class="grid-area-[1/1] w-45 h-10.8 hidden group-[:hover:not(:active)]/confirm_btn:block" 
      />
      <ButtonNormal
        class="grid-area-[1/1] w-45 h-10.8 block group-[:is(:hover,:active)]/confirm_btn:hidden" 
      />
      <div class="grid-area-[1/1] h-full w-full flex items-center justify-center text-lg font-bold text-black/70 transition-colors line-height-none">
      {ch()}        
      </div>
    </button>
  );
}
