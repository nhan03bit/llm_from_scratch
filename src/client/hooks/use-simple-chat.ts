/** Hook for SSE for pattern matching word, stream word once at a time */
import { useSSEChat } from "./use-sse-chat";

export function useSimpleChat() {
  return useSSEChat<{ words: string[] }, { word?: string}>({
    endpoint: "/simple-chat",
    title: "Simple Chat Bot",
    tagline:"a simple pattern matching chat bot",
    initState: () => ({ words: []}),
    onEvent: (parsed, state) => {
      if (parsed.word) {
        state.words.push(parsed.word);
        return state.words.join(" ");
      }
    }
  });
}