# Stage 1 — BPE Tokenizer

## Goal
Turn raw text into **subword tokens** by learning merge rules from data, the same algorithm behind GPT's tiktoken. Stream every merge step to the UI so you can watch the vocabulary grow.

**You're done when** you can open `/bpe-token`, paste any paragraph, watch the merges animate, and get a lossless tokenization with a compression ratio > 1. You also need to be able to explain every number on screen.

## Concepts
- **Why not characters?** Sequences get too long. **Why not words?** The vocabulary explodes and unseen words become `<unk>`. **Subwords** sit in between.
- **BPE training**: start with characters, count adjacent pairs, merge the most frequent pair, repeat.
- **Pre-tokenization**: split text into words/spaces/punctuation *first*, so merges never cross word boundaries.
- **Word-frequency trick**: count each *unique* word once, weighted by how often it appears. That is what lets BPE scale.
- **Merge order matters at inference**: replay merges in the order they were learned.
- 📚 Sennrich, Haddow & Birch (2016), see [RESEARCH.md §7](RESEARCH.md#7-byte-pair-encoding-for-nlp-2016--stage-1). Read Algorithm 1 **after** you finish 1.3.

## Files you'll create
```
src/server/lib/bpe.ts                     ← the algorithm (reused in Stages 3 & 4!)
src/routes/bpe-tokenize.ts                ← already exists, empty
src/client/hooks/use-bpe-tokenize-chat.tsx
scratch/bpe.test.ts                       ← your checkpoint runner (optional but recommended)
```
and edits to `src/index.tsx` (mount) and `src/client/routes.tsx` (un-comment).

## Signatures you'll implement (bodies are yours)
```ts
export const PRE_TOKEN_RE: RegExp;
export type Merge = { pair: [string, string]; merged: string; frequency: number };

export function countWords(text: string, regex?: RegExp): Map<string, number>;
export function mergeTokens(tokens: string[], pair: [string, string], merged: string): string[];
export function trainBpe(wordFreqs: Map<string, number>, maxMerges?: number): { merges: Merge[]; /* + whatever else you need */ };
export function applyMerges(text: string, merges: Merge[], regex?: RegExp): string[];
export function trainBpeOnText(text: string, maxMerges?: number, regex?: RegExp): { merges: Merge[] /* … */ };
```

---

## Step 1.0 — Before you start
- [ ] **Read the contract.** Open `src/client/components/bpe-tokenize-result/index.tsx` and read `BpeInit`, `MergeStep` and `BpeResult`. For every field, write down in one line *where the value comes from*. You'll check your route against this list in 1.5.
- [ ] **Set up a checkpoint runner.** Create `scratch/bpe.test.ts` that imports from `src/server/lib/bpe.ts` and prints (or `console.assert`s) the checkpoint cases below. Run it with `bun scratch/bpe.test.ts` after every step. Bun also has a built-in `bun test` with `expect` if you prefer.
- [ ] **Train BPE by hand once.** Fill in this table on paper for the corpus `{"hug": 10, "pug": 5, "hugs": 5}` (the classic Hugging Face course example). Doing this by hand makes 1.3 much easier.

  | round | pair counts (top 3) | winner | words after merge |
  |---|---|---|---|
  | 0 | — | — | `h u g` ×10, `p u g` ×5, `h u g s` ×5 |
  | 1 | `u g`: ?, `h u`: ?, `g s`: ? | ? | ? |
  | 2 | ? | ? | ? |

  <details><summary>Check your table</summary>

  Round 1: `u+g` = 20 wins (`h+u` = 15). Round 2: `h+ug` = 15 wins. Round 3: `p+ug` = 5 and `hug+s` = 5 tie, so your tie-break rule decides.
  </details>

**Tie-break rule used for every checkpoint in this file:** highest frequency wins; on a tie, the pair **first seen** while scanning words in insertion order (then left to right) wins. With a `Map` and a strict `>` comparison you get this for free. Use a different rule and your merge order on ties may differ, which is fine as long as it's deterministic.

---

## Step 1.1 — `PRE_TOKEN_RE` and `countWords`
**Tier 1.** You want a regex that yields "word characters", "a single whitespace char", or "a single punctuation char". Then count how many times each piece appears. What JS structure maps string → count?

**Tier 2.**
1. Define a global regex with three alternatives: runs of `\w`, one `\s`, one "not word and not space".
2. `text.match(regex)` (or `matchAll`) → array of pre-tokens. What does `match` return on an empty string?
3. For each piece: `freq[piece] = (freq[piece] ?? 0) + 1`.

**Checkpoints.**
- `countWords("low low lower")` → `Map { "low" → 2, " " → 2, "lower" → 1 }`
- `countWords("")` → an empty `Map` (not a crash)
- `countWords("hi, bob!")` has 5 keys: `hi`, `,`, ` `, `bob`, `!`

---

## Step 1.2 — `mergeTokens`
**Tier 1.** Given `['l','o','w']` and the pair `('l','o')`, produce `['lo','w']`. What happens with overlapping pairs like `['a','a','a']` merging `('a','a')`? Decide left-to-right, non-overlapping.

**Tier 2.**
1. `out = []`, `i = 0`.
2. While `i < tokens.length`: if `tokens[i]` and `tokens[i+1]` equal the pair, push `merged` and `i += 2`; otherwise push `tokens[i]` and `i += 1`.

**Checkpoints.**
- `mergeTokens(['c','a','t'], ['c','a'], 'ca')` → `['ca','t']`
- `mergeTokens(['a','a','a'], ['a','a'], 'aa')` → `['aa','a']`
- `mergeTokens(['x'], ['x','y'], 'xy')` → `['x']` (no out-of-bounds read)
- The input array is **not mutated** (log it after the call).

---

## Step 1.3 — `trainBpe` ⭐ the core
**Tier 1.** Represent each unique word as an array of symbols (initially its characters). In each round, count every adjacent pair **across all words**, where each pair contributes the *word's frequency*, not 1. Merge the winner inside every word. Record the merge. Stop at `maxMerges` or when no pair is left.

**Tier 2.**
```
words = [ (chars(w), freq) for each (w, freq) in wordFreqs ]
merges = []
repeat up to maxMerges:
    pairCounts = {}
    for (symbols, freq) in words:
        for each adjacent (a, b) in symbols:
            pairCounts[a+SEP+b] += freq        ← choose a SEP that can't appear in text
    if pairCounts is empty: stop
    (best, bestFreq) = argmax(pairCounts)
    merged = a + b
    for each word: symbols = mergeTokens(symbols, [a, b], merged)
    merges.push({ pair: [a, b], merged, frequency: bestFreq })
return merges (+ anything the route needs, e.g. per-step token counts)
```

**Things to decide yourself.**
- How do you key a pair in a `Map`? (A string with a separator, or nested maps: what are the trade-offs? What goes wrong with the separator `""`: is `"ab"+"c"` the same key as `"a"+"bc"`?)
- The UI's `MergeStep` wants `vocabSize` and `tokenCount` after each merge. How do you compute those **without re-scanning**?
  - A merge adds exactly **1** to the vocabulary. *Why never 0?*
  - A merge reduces the total token count by exactly **`frequency`**. *Why?*
  - So you only need the starting values: vocab = number of unique characters; tokenCount = Σ `word.length × freq`.
- Should `trainBpe` return everything at the end, or be a generator that yields each step? (The route streams steps. Either works; think about which is simpler to test.)

**Checkpoint A — the paper's example.** `trainBpe(new Map([["low",5],["lower",2],["newest",6],["widest",3]]), 6)`:

| step | pair → merged | frequency | vocabSize | tokenCount |
|---|---|---|---|---|
| 1 | `e+s → es` | 9 | 11 | 70 |
| 2 | `es+t → est` | 9 | 12 | 61 |
| 3 | `l+o → lo` | 7 | 13 | 54 |
| 4 | `lo+w → low` | 7 | 14 | 47 |
| 5 | `n+e → ne` | 6 | 15 | 41 |
| 6 | `ne+w → new` | 6 | 16 | 35 |

(Starting values: 10 unique characters, 79 tokens.)

**Checkpoint B — it stops on its own.** `trainBpe(countWords("the cat sat on the mat. the cat ate the rat."), 1000)` produces exactly **9** merges, not 1000. After that, every word is a single token and no pairs remain. The first four are `a+t → at` (6), `t+h → th` (4), `th+e → the` (4), `c+at → cat` (2).

---

## Step 1.4 — `applyMerges` and `trainBpeOnText`
**Tier 1.** At inference you don't re-count anything: pre-tokenize the new text, split each piece into characters, then replay **every learned merge in order**. Why would applying `('es','t')` before `('e','s')` break things?

**Tier 2.**
1. `pieces = text.match(regex)`
2. For each piece: `symbols = [...piece]`; for each merge in order: `symbols = mergeTokens(symbols, merge.pair, merge.merged)`.
3. Concatenate all pieces' symbols.
4. `trainBpeOnText` = `countWords` → `trainBpe`. It's a convenience wrapper for Stages 3 and 4.

**Checkpoints.**
- **Unseen word:** train 5 merges on `"low lower lowest newer"`. The merges are `lo, low, lowe, lower, lowes`. Then `applyMerges("lowest", merges)` → `["lowes", "t"]`. Why not `["lowest"]`? (Count how many merges you'd need.)
- **Lossless:** for any text, `applyMerges(text, merges).join("") === text`. Try text with punctuation, double spaces and newlines.
- **Words outside the training data** still tokenize (as characters and known pieces), with no crash and no `<unk>`.

---

## Step 1.5 — Route `src/routes/bpe-tokenize.ts`
**Tier 1.** This is `simple-chat.ts` with a different brain. The user's message *is* the training corpus, so reuse `ChatRequest`. Instead of streaming words, you stream the training process itself: one snapshot at the start, one event per merge, one summary at the end.

**SSE event flow (must match the contract)**
1. `init` → `BpeInit`: `corpus`, `characters` (cap at ~200 for the UI), `charCount`, `wordCount` = number of **unique** pre-tokens
2. `merge` × N → `MergeStep`: `step`, `pair`, `frequency`, `newToken`, `vocabSize`, `tokenCount`
3. `result` → `BpeResult`: `inputTokens`, `tokenCount`, `originalCharCount`, `compressionRatio` (a **string**)

**Tier 2.**
```
POST /bpe-tokenize  (validate { message })
  wordFreqs  = countWords(message)
  characters = [...message]                       ← why spread, not split("")? (try an emoji)
  emit init   { corpus, characters(capped), charCount, wordCount }
  train, then for each step: emit merge {…}, small delay (~50–150 ms)
  tokens = applyMerges(message, merges)
  emit result { inputTokens: tokens, tokenCount, originalCharCount, compressionRatio: (chars/tokens).toFixed(2) }
  on any throw: emitError(err)
```
Think about the delay: 1000 merges × 150 ms is 2.5 minutes. How would you scale the delay with the number of merges?

**Checkpoint.**
```bash
curl -N -X POST localhost:5173/bpe-tokenize -H 'content-type: application/json' \
  -d '{"message":"the cat sat on the mat. the cat ate the rat."}'
```
Expected:
- `init`: `charCount: 44`, `wordCount: 9`
- 9 × `merge`. The first is `{"step":1,"pair":["a","t"],"frequency":6,"newToken":"at","vocabSize":13,"tokenCount":38}`.
- `result`: `tokenCount: 23`, `compressionRatio: "1.91"`, `inputTokens` = `["the"," ","cat"," ","sat", …, "rat","."]`

(The route isn't reachable until it's mounted. That's the first TODO of 1.6, so do it now if curl returns 404.)

---

## Step 1.6 — Client hook + wiring
**Tier 1.** `use-simple-chat.ts` accumulates words into a string. Here you accumulate *three different event shapes* into one state object, and render the result component every time.

**Tier 2.**
1. `src/index.tsx`: import the route and add another `.route("/", bpeTokenize)`, chained next to `simpleChat` so the exported `App` type includes it.
2. `src/client/hooks/use-bpe-tokenize-chat.tsx` (`.tsx` because it returns JSX):
   - `endpoint: "/bpe-tokenize"`, plus a title and tagline.
   - `initState: () => ({ mergeSteps: [] })`, typed as `{ init?: BpeInit; mergeSteps: MergeStep[]; result?: BpeResult }`.
   - `onEvent(parsed, state)`: work out which event this is **from its fields**. For example, only `init` has `corpus`, only `merge` has `newToken`, and only `result` has `inputTokens`. Update the state and return `<BpeTokenizeResult init={…} mergeSteps={…} result={…} />`.
   - Type `TEvent` as a union of the three shapes. Can you narrow it with the `in` operator?
3. `src/client/routes.tsx`: un-comment the `useBpeTokenizeChat` import and keep the `/bpe-token` route.

**Checkpoints.**
- `bunx tsc -b --noEmit` has no errors from your new files.
- Open `http://localhost:5173/bpe-token`, paste a paragraph, and watch the merges animate. Final tokens show `␣` for spaces.
- Send a second message. The display should start fresh, not append to the previous run. (Where does `initState` get called?)

---

## Common pitfalls
- Regex without the `g` flag: `match` returns only the first hit.
- `text.match(re)` returns `null` (not `[]`) when nothing matches.
- A separator that can appear in the text (or `""`), so `"ab"+"c"` and `"a"+"bc"` collide.
- Mutating a `Map` while iterating it.
- Counting a pair +1 instead of +word frequency, which gives the wrong merges on real text.
- Non-deterministic tie-breaking, so merges differ between runs (and Stage 3/4 vocabularies drift).
- Forgetting the loop can run out of pairs before `maxMerges`.
- `split("")` breaks emoji and other surrogate pairs; `[...str]` doesn't.
- Route not mounted in `src/index.tsx`, so every request is a 404 and the page shows nothing.

## Questions to answer before moving on
1. Why does counting unique words × frequency give *exactly* the same merges as running BPE over the whole raw text split into words, only faster?
2. In Checkpoint B, why did training stop at 9 merges? What does that say about how `maxMerges` relates to vocabulary size on a real corpus?
3. Your tokenizer never merges across a space. What would happen to the vocabulary if it did?
4. `"The"` and `"the"` become different tokens. Stages 3 and 4 lower-case the corpus. What does that gain, and what does it cost?
5. Look at the GPT-4 tokenizer (e.g. https://tiktokenizer.vercel.app). Find one thing it does that yours doesn't.

## Stretch goals
- Byte-level BPE (start from UTF-8 bytes instead of chars), as in GPT-2. Try it with emoji.
- GPT-style pre-tokenization that attaches the leading space to the word (`/\s\w+|[^\w\s]/g`). You'll use exactly this in Stage 4. Why is it better?
- Speed-up: only recount pairs in words that changed.
- Add an `encode` (tokens → IDs) / `decode` (IDs → text) pair with a vocabulary map. You'll need it in Stage 3 anyway.

## After this stage
1. Tick 1.1–1.6 in [PROGRESS.md](PROGRESS.md).
2. Read Sennrich et al. **Algorithm 1** and compare it line by line with your `trainBpe`. What does their `</w>` marker do that your pre-tokenization handles differently?
3. Compare with the reference [`src/server/lib/bpe.ts`](https://github.com/w3cj/how-llms-work/blob/main/src/server/lib/bpe.ts) and note 3 differences.
4. Next: [Stage 2 — XOR Neural Net](02-xor-neural-net.md).
