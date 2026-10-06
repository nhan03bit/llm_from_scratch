import type { Child } from "hono/jsx";

export type Message = {
  content: Child;
  id: string;
  role: "assistant" | "user";
};