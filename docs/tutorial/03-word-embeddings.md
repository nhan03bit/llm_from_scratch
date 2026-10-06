# Stage 3 — Word Embeddings (Word2Vec Skip-gram + Negative Sampling)

## Goal
Learn a vector for every token so that **meaning becomes geometry**: `cat` near `dog`, and `king − man + woman ≈ queen`. You train on a small curated corpus tokenized with **your Stage 1 BPE**.

## Concepts
- **Distributional hypothesis**: "you shall know a word by the company it keeps" (Firth, 1957).
- **Skip-gram**: given a centre token, predict the tokens inside a ±`windowSize` window.
- **Two matrices**: `W_in` (the embeddings you keep) and `W_out` (context vectors), each `vocabSize × dim`.
- **Negative sampling**: instead of a softmax over the whole vocabulary, do a binary "real neighbour or random noise?" classification with K negatives.
- **Cosine similarity**: compares directions, ignoring vector length.
- 📚 Mikolov et al. 2013a & 2013b, [RESEARCH.md §4–5](RESEARCH.md#4-word2vec-2013a--stage-3).

## Files you'll create
```
src/server/lib/math.ts                      ← cosineSimilarity (reused in Stage 4 demos)
src/routes/train-embed/corpus.ts            ← CORPUS, STORIES, BPE merges, tokenize, buildVocab
src/routes/train-embed/train.ts             ← trainSkipGram generator
src/routes/train-embed/index.ts             ← route
src/schemas/train-embed-request.ts
src/client/hooks/use-train-embed-chat.tsx
```

## The contract
`src/client/components/train-embed-result/index.tsx`: `InitData { vocabSize, sentenceCount, embeddingDim, windowSize, totalPairs }`, `EpochData`, `WordEmbedding`, `Neighbor`, `SimilarityPair`, `Analogy`, plus optional `warnings: string[]`.

## Signatures
```ts
export function cosineSimilarity(a: number[], b: number[]): number;

export const CORPUS: string[];
export const STORIES: string[];                     // Stage 4 trains on these
export const BPE_MERGES: Merge[];
export function tokenize(text: string): string[];
export function buildVocab(corpus: string[]): { wordToIndex: Map<string, number>; indexToWord: string[]; /* counts? */ };

// schema: words: string[1..10]; epochs 10..10000 (default 10000); dimensions 4..64 (32);
//         windowSize 1..5 (2); negativeSamples 1..10 (5)
export async function* trainSkipGram(opts: TrainOpts): AsyncGenerator<InitResult | EpochResult | TrainEmbedResult>;
```

---

## Step 3.1 — `cosineSimilarity`
`cos(a, b) = (a·b) / (‖a‖ ‖b‖)`. Guard against a zero-length vector.
**Checkpoints.** `cos([1,0],[1,0]) = 1`, `cos([1,0],[0,1]) = 0`, `cos([1,2],[-1,-2]) = -1`.

## Step 3.2 — Corpus & vocabulary
**Tier 1.** Write roughly 100 **short, parallel sentences** so related words share contexts:
"the king is a man", "the queen is a woman", "the cat chased the mouse", "the dog chased the ball"…
Word2Vec can only learn relationships the corpus *shows*. Then write a few dozen tiny stories (`STORIES`) for Stage 4.

**Tier 2.**
1. Train BPE on the lower-cased corpus text (the reference uses ~500 merges) and export the merges.
2. `tokenize(text)` = `applyMerges(text.toLowerCase(), BPE_MERGES)`, then **drop pure-whitespace tokens** for Skip-gram.
3. `buildVocab`: unique tokens → index. Keep counts too, since you'll need them for 3.4.

**Checkpoint.** `tokenize("the king is a man")` returns whole-word tokens for common words. If `king` splits into pieces, add more merges or more sentences.

## Step 3.3 — Training pairs
**Tier 2.**
```
pairs = []
for sentence in corpus:
    ids = tokenize(sentence).map(toIndex)
    for i, target in ids:
        for j in [i−window, i+window], j ≠ i, in bounds:
            pairs.push([target, ids[j]])
```
**Checkpoint.** `"a b c"` with window 1 gives 4 pairs: (a,b), (b,a), (b,c), (c,b).

## Step 3.4 — Negative-sampling table
**Tier 1.** Negatives shouldn't be uniform (too many rare words) or raw frequency (too many "the"). Mikolov uses `count^0.75`. How do you sample from a weighted distribution quickly?

**Tier 2.** Build a big array (e.g. 1e5–1e6 slots). Fill it with each word index in proportion to `count^0.75 / Σ count^0.75`. Sampling is then just `table[randInt]`.
Use a **seeded RNG** (mulberry32) so runs are reproducible. You'll write the same RNG again in Stage 4.

## Step 3.5 — Skip-gram update ⭐
**Tier 1.** For a real pair (t, c), you want `σ(W_in[t] · W_out[c])` → 1. For each negative n, you want `σ(W_in[t] · W_out[n])` → 0. Loss = `−log σ(pos) − Σ log σ(−neg)`.

**Tier 2.** For each pair (after shuffling the pairs every epoch):
```
v = W_in[t]                        grad_v = zeros(dim)
s = σ(v · W_out[c]);   g = lr · (1 − s)       ← positive: pull together
    grad_v += g · W_out[c];   W_out[c] += g · v
repeat K times: n = sampleNegative() (skip if n == c or n == t)
    s = σ(v · W_out[n]);   g = lr · s         ← negative: push apart
    grad_v −= g · W_out[n];   W_out[n] −= g · v
W_in[t] += grad_v                  ← apply once, at the end
loss += −log(σ(pos)) − Σ log(σ(−neg))
```
Learning rate: **linear decay** from `0.025` → `0.001` over the epochs (as in the original word2vec).
Initialise `W_in` with small random values and `W_out` with zeros (as the original C code does), or both small random.

**Checkpoints.**
- The loss goes down across epochs (it will be noisy).
- Clamp the sigmoid input (e.g. to ±10) or you'll get `log(0) = -Infinity`.

## Step 3.6 — Results
For each query word the user sends:
- **embeddings**: `W_in[idx]` (the vector).
- **neighbours**: top-5 by cosine similarity, excluding the word itself.
- **similarities**: every pair of query words.
- **analogies**: `a − b + c` for triples like `["king","man","woman"]` → nearest word excluding a, b, c.
- **warnings**: query words not in the vocabulary (or split into several BPE pieces).

**Checkpoint.** After full training on a well-designed corpus, `king` and `queen` land in each other's top neighbours, and `king − man + woman` returns `queen` (or at least ranks it in the top 3).

## Step 3.7 — Schema, route, hook
**SSE event flow**
1. `init` → `InitData`
2. `epoch` × ~50 → `{ epoch, loss }`
3. `done` → `{ embeddings, neighbors, similarities, analogies, warnings }`

`buildBody` in the hook: split the user's input on commas or spaces → `{ words }`. Un-comment `/train-embed` and mount the route.

---

## Common pitfalls
- Updating `W_in[t]` inside the negative loop, which changes `v` mid-step.
- Not shuffling pairs, so the model learns sentence order artefacts.
- Whitespace tokens in the vocabulary: every word ends up "near" the space token.
- Blocking the event loop: 10 000 epochs in a tight loop freezes SSE. `await` a tick (e.g. `setImmediate`) when you yield.

## Stretch goals
- Subsampling frequent words (paper 2013b §2.3).
- A 2-D PCA projection of the embeddings shown in the UI.
- CBOW (context → centre) and a comparison with Skip-gram.
