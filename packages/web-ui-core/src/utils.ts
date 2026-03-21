export const checkPointerEvent = (e: PointerEvent) => {
  return (
    (e.pointerType === "mouse" && e.buttons === 1) ||
    e.pointerType === "touch" ||
    e.pointerType === "pen"
  );
};
