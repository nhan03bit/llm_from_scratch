/**
 * Simple chat with scripted responses. SSE - Server Sent Events
 * @see https://dl.acm.org/doi/10.1145/365153.365168 - ELIZA: Uses string manipulation to simulate conversation
 *
 * * SSE event flow:
 * 1. "start" — signals the UI to show a loading state
 * 2. "word" × N — one event per word, streamed with 200ms delays
 * 3. "done" — signals completion
 */
import { zValidator } from "@hono/zod-validator";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { createEmitter } from "../server/lib/sse";
import ChatRequest from "../schemas/chat-request";



const greetings = [
  "hello",
  "hi",
  "hey",
  "morning",
  "afternoon",
  "evening",
];

/** Return random fallback response when no pattern matches */
function getRandomContinuation(): string {
   const continuations = [
    "Please go on.",
    "Can you tell me more about that?",
    "Tell me more.",
    "How does that make you feel?",
  ];

  return continuations[Math.floor(Math.random() * continuations.length)];
}

/**
 * Pattern-matches the user's message and returns a canned response.
 *
 * This is the entire "AI" — a chain of string checks. It handles:
 * - Greetings ("hello", "hi", "hey")
 * - "I feel X" → "Why do you feel X?" (ELIZA's signature move)
 * - "my X" → "Tell me more about your X"
 * - "worried" → a follow-up question
 * - Everything else → random continuation
 *
 * @example
 * getSimpleChatResponse("I feel anxious")  // => "Why do you feel anxious?"
 * getSimpleChatResponse("hello there")     // => "Hello! How can I help you today?"
 * getSimpleChatResponse("my dog is sick")  // => "Tell me more about your dog."
 */
function getSimpleChatResponse(message: string): string {
  const messageLower = message.toLowerCase();
  const words = messageLower
    .split(" ")
    .map(word => word.replaceAll(/[!?.;,:"']/g, ""))
    .filter(word => word.trim() !== "");

  if (greetings.some(greet => words.includes(greet))) {
    return "Hello! How can I help you today?";
  }

  const iFeel = "i feel";
  if (messageLower.startsWith(iFeel)) {
        const feeling = messageLower.slice(iFeel.length + 1).replace(" i ", " you ");
    return `Why do you feel ${feeling}?`;
  }

    const subjectIndex = words.indexOf("my") + 1;
  if (subjectIndex > 0 && subjectIndex < words.length) {
    const subject = words[subjectIndex];
    return `Tell me more about your ${subject}.`;
  }

  if (messageLower.includes("worried")) {
    return "How long have you been worried about this?";
  }

  return getRandomContinuation();
}

export default new Hono()
  .post(
    "/simple-chat",
    zValidator("json", ChatRequest),
    (c) => {
      const { message } = c.req.valid("json");
      const response = getSimpleChatResponse(message);
      const words = response.split(" ");

      return streamSSE(c, async (stream) => {
        const { emit } = createEmitter(stream);
        await emit({}, "start", 10000);

        for (const word of words) {
          await emit({ word }, "word", 200);
        }

        await emit({}, "done");
      })
    }
  );