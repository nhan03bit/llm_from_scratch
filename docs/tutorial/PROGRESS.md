# Progress

Tick a box only when its checkpoint passes. The tutor cross-checks this against your code.

## Stage 0 — Setup & ELIZA  ([guide](00-setup-and-eliza.md))

- [X] 0.1 Hono server + JSX renderer (`src/index.tsx`)
- [X] 0.2 SSE emitter (`src/server/lib/sse.ts`)
- [X] 0.3 ELIZA pattern-matching route (`src/routes/simple-chat.ts`)
- [X] 0.4 Client hook `use-simple-chat.ts` + `useSSEChat`
- [X] 0.5 Housekeeping: add `zod` to `package.json`, fix schema error message
- [X] 0.6 `bunx tsc -b --noEmit` is clean (comment out unfinished routes in `routes.tsx` until each stage lands)

## Stage 1 — BPE Tokenizer  ([guide](01-bpe-tokenizer.md))

- [ ] 1.0 Read the UI contract, set up `scratch/bpe.test.ts`, hand-train the hug/pug/hugs table
- [ ] 1.1 `PRE_TOKEN_RE` + `countWords`
- [ ] 1.2 `mergeTokens`
- [ ] 1.3 `trainBpe` (pair counting weighted by word frequency)
- [ ] 1.4 `applyMerges` + `trainBpeOnText`
- [ ] 1.5 Route `src/routes/bpe-tokenize.ts` (events: `init`, `merge`, `result`)
- [ ] 1.6 Hook `use-bpe-tokenize-chat.tsx`, wired in `routes.tsx` + mounted in `index.tsx`

## Stage 2 — XOR Neural Net  ([guide](02-xor-neural-net.md))

- [X] 2.1 Schema `neural-net-request.ts`
- [X] 2.2 `sigmoid`, `sigmoidDeriv`, `randWeight`
- [X] 2.3 `trainSingleLayer` (should FAIL — and you can explain why)
- [X] 2.4 `trainMultiLayer` 2→4→1 with backprop (should SUCCEED)
- [X] 2.5 `serialize.ts` → weights saved in `.data/`
- [X] 2.6 Route (events: `epoch`, `done`) + hook `use-neural-net-chat.tsx`

## Stage 3 — Word Embeddings  ([guide](03-word-embeddings.md))

- [ ] 3.1 `math.ts` → `cosineSimilarity`
- [ ] 3.2 `corpus.ts` → corpus, BPE merges, `tokenize`, `buildVocab`
- [ ] 3.3 Training pairs from a sliding window
- [ ] 3.4 Negative-sampling table (unigram^0.75)
- [ ] 3.5 Skip-gram SGD update (positive + K negatives)
- [ ] 3.6 Nearest neighbours, similarities, analogies
- [ ] 3.7 Schema + route (events: `init`, `epoch`, `done`) + hook `use-train-embed-chat.tsx`

## Stage 4 — Transformer  ([guide](04-transformer.md))

- [ ] 4a `matrix.ts` (seeded RNG, matmul ×3, xavierInit, helpers)
- [ ] 4b Config/weights/grad types, `initWeights`, `zeroGrads`, `countParams`
- [ ] 4c LayerNorm forward + backward (grad-checked)
- [ ] 4d Causal multi-head self-attention forward
- [ ] 4e Transformer block forward (pre-LN, residuals, FFN)
- [ ] 4f Full `forward` + cross-entropy loss (init loss ≈ ln(vocabSize))
- [ ] 4g Backward pass (every piece grad-checked)
- [ ] 4h Adam optimizer
- [ ] 4i Training loop — model overfits a single sentence
- [ ] 4j `generateText` with temperature + top-p
- [ ] 4k (optional) Worker threads + SharedArrayBuffer data parallelism
- [ ] 4l Schema + route (events: `init`, `epoch`, `done`) + hook `use-train-transformer-chat.tsx`

## Finished 🎉

- [ ] Compared each stage against the reference repo and wrote down 3 differences
- [ ] Read all 11 papers' "what to read" sections in [RESEARCH.md](RESEARCH.md)
