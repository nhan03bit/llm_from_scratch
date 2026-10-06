# Stage 0 — Setup & ELIZA ✅ (mostly done)

## Goal
Get a working "chat" UI streaming responses over SSE, using a bot with **zero learning**. Every later stage replaces the "brain" and keeps the same plumbing.

## Concepts
- **Pattern matching as "AI"**: ELIZA (Weizenbaum 1966, see [RESEARCH.md §1](RESEARCH.md#1-eliza-1966--stage-0)).
- **Server-Sent Events (SSE)**: a one-way HTTP stream of `event:` / `data:` frames. It lets the UI render tokens and training epochs as they happen.

## What you already built
| File | Role |
|---|---|
| `src/index.tsx` | Hono app, JSX shell, mounts routes |
| `src/server/lib/sse.ts` | `createEmitter(stream)` → `emit(data, event, delay?)`, `emitError(err)` |
| `src/routes/simple-chat.ts` | ELIZA rules + `start` / `word` / `done` events |
| `src/schemas/chat-request.ts` | `{ message: string }` zod schema |
| `src/client/hooks/use-sse-chat.ts` | Generic streaming-chat hook every stage reuses |
| `src/client/hooks/use-simple-chat.ts` | The ELIZA page's hook |

## Housekeeping TODOs
- [ ] **0.5a** `zod` is imported but only installed transitively. Add it as a direct dependency. *Hint: which package manager lockfile do you have?*
- [ ] **0.5b** Re-read the error message in `src/schemas/chat-request.ts`. Does it say what you mean?
- [ ] **0.6** `src/client/routes.tsx` uses hooks that don't exist yet, so `tsc` fails. Comment out each unfinished route line for now and restore it when you finish that stage.
- [ ] **0.7** Understand the event flow. Trace one message from `<ChatInput>` → `useSSEChat.sendMessage` → `POST /simple-chat` → `emit(...)` → `readSSE` → `onEvent` → re-render. Write the 6 hops down in your own words.

## Checkpoint
```bash
bunx tsc -b --noEmit        # no errors
bun run dev
curl -N -X POST localhost:5173/simple-chat -H 'content-type: application/json' -d '{"message":"I feel tired"}'
```
Expected: one `event: start`, then `event: word` frames spelling *"Why do you feel tired?"*, then `event: done`.

## Questions to answer before moving on
1. Why does the server `emit` a `start` event with a long delay *before* the words? (Look at what the UI does on `start`.)
2. ELIZA can't handle "I don't feel good". What would you need to add? Why doesn't that approach scale?
3. What is the fundamental difference between how ELIZA gets its rules and how every later stage gets them?

## Stretch goals
- Add ELIZA's pronoun reflection table (`my→your`, `am→are`, `me→you`) instead of one `replace`.
- Add a keyword-priority system like the original DOCTOR script.
