import { useEffect, useRef } from "hono/jsx";

export function useAutoScroll(deps: unknown[]) {
  const ref = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);
  const isScrollingRef = useRef(false);

  useEffect(() => {
    if (!autoScrollRef.current || !ref.current) return;

    isScrollingRef.current = true;
    ref.current.scrollTop = ref.current.scrollHeight;

    requestAnimationFrame(() => {
      isScrollingRef.current = false;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- deps are passed dynamically by the caller
  }, deps);

  const handleScroll = () => {
    if (isScrollingRef.current) return;

    const el = ref.current;

    if (!el) return;

    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    autoScrollRef.current = atBottom;
  };

  const scrollToBottom = () => {
    autoScrollRef.current = true;
  };

  return { ref, handleScroll, scrollToBottom };
}
