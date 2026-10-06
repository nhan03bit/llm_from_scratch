# Stage 4 — GPT From Scratch (Decoder-only Transformer)

## Goal
Implement a tiny GPT with **no ML libraries**: every matmul, softmax, layer norm, attention head, gradient and Adam step is written by you on flat `Float32Array`s. Train it on `STORIES` (from Stage 3) tokenized with your Stage 1 BPE, and stream the loss plus generated text as it learns.

This is the longest stage. It's split into 12 sub-steps, and **each one has its own checkpoint. Don't skip them.** The backward pass is only manageable if every piece has been gradient-checked on its own.

## Concepts
- **Next-token prediction**: input `[t0..t15]`, target `[t1..t16]`, cross-entropy at every position. 📚 Radford 2018 ([§10](RESEARCH.md#10-gpt-1-2018--stage-4-overall))
- **Self-attention**: `softmax(QKᵀ/√d_k + mask) · V`, one per head, then concatenate and project. 📚 Vaswani 2017 ([§9](RESEARCH.md#9-attention-is-all-you-need-2017--stage-4d4e))
- **Causal mask**: position i may attend only to j ≤ i. This is what makes the model *generative*.
- **Residual connections + pre-LayerNorm**: `x = x + Attn(LN(x))`, `x = x + FFN(LN(x))`. 📚 Ba 2016 ([§8](RESEARCH.md#8-layer-normalization-2016--stage-4c))
- **Xavier init** 📚 Glorot 2010 ([§3](RESEARCH.md#3-xavierglorot-initialisation-2010--stages-2--4)), **Adam** 📚 Kingma 2014 ([§6](RESEARCH.md#6-adam-2014--stage-4h)), **top-p sampling** 📚 Holtzman 2019 ([§11](RESEARCH.md#11-nucleus-top-p-sampling-2019--stage-4j)).

## Architecture you're building
```
tokenIds (S)
  → tokEmb[id] + posEmb[pos]                      (S × D)
  → N × Block:
        a = LN1(x);  x = x + Wo · MultiHeadCausalAttn(a·Wq, a·Wk, a·Wv)
        f = LN2(x);  x = x + ff2( ReLU( ff1(f) ) )  (D → F → D)
  → LNf(x) → headW·x + headB → logits (S × V) → softmax → probs
```
Defaults (match the schema/UI): `contextLen 32`, `embDim D=32`, `numHeads 2` (d_k = 16), `ffDim F=128`, `numLayers 2`, training `seqLen 16`, `lr 0.001`, epochs 300, `temperature 0.8`, `topP 0.9`.

## Files you'll create
```
src/routes/train-transformer/matrix.ts
src/routes/train-transformer/transformer.ts      ← the big one
src/routes/train-transformer/train.ts
src/routes/train-transformer/index.ts
src/schemas/train-transformer-request.ts
src/client/hooks/use-train-transformer-chat.tsx
(optional 4k) weight-layout.ts, train-worker.ts
```

## The contract
`src/client/components/train-transformer-result/index.tsx`:
- `InitData { vocabSize, contextLen, embeddingDim, numHeads, ffDim, numLayers, totalParams, temperature, topP, corpusSentences, trainingSequences }`
- `EpochData { epoch, loss, sample? }`
- `Sample { epoch, text }`
- `TransformerSummary { architecture, finalLoss }`

## Signatures & types (bodies are yours)
```ts
// matrix.ts — row-major: element (i, j) of an M×N matrix lives at i*N + j
export let rand: () => number;                    // seeded mulberry32
export function resetRand(seed?: number): void;
export function matmul(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array;       // (M×K)(K×N)
export function matmulTransB(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array; // (M×K)(N×K)ᵀ
export function matmulTransA(a: Float32Array, b: Float32Array, K: number, M: number, N: number): Float32Array; // (K×M)ᵀ(K×N)
export function addBias(out: Float32Array, bias: Float32Array, rows: number, cols: number): void;
export function addInPlace(target: Float32Array, source: Float32Array): void;
export function zeros(len: number): Float32Array;
export function ones(len: number): Float32Array;
export function xavierInit(len: number, fanIn: number, fanOut: number): Float32Array;
export function sumCols(mat: Float32Array, rows: number, cols: number): Float32Array; // bias gradients

// transformer.ts
export type TransformerConfig = { vocabSize: number; contextLen: number; embDim: number; numHeads: number; ffDim: number; numLayers: number };
export type BlockWeights = {
  ln1Gamma; ln1Beta; wQ; bQ; wK; bK; wV; bV; wO; bO;   // all Float32Array
  ln2Gamma; ln2Beta; ff1W; ff1B; ff2W; ff2B;
};
export type TransformerWeights = { tokEmb; posEmb; blocks: BlockWeights[]; lnFGamma; lnFBeta; headW; headB };
export type TransformerGrads = /* same shape as weights */;
export type ForwardCache = { /* every intermediate you'll need in backward */ };

export function initWeights(cfg: TransformerConfig): TransformerWeights;
export function zeroGrads(cfg: TransformerConfig): TransformerGrads;
export function countParams(w: TransformerWeights): number;
export function forward(tokenIds: number[], w: TransformerWeights, cfg: TransformerConfig): ForwardCache;
export function crossEntropyLoss(probs: Float32Array, targets: number[], seqLen: number, V: number): number;
export function backward(cache: ForwardCache, targets: number[], w: TransformerWeights, cfg: TransformerConfig, grads: TransformerGrads): void; // ACCUMULATES into grads
export function initAdam(w: TransformerWeights): AdamBuf;
export function adamUpdate(w: TransformerWeights, grads: TransformerGrads, buf: AdamBuf, lr: number, t: number, beta1?: number, beta2?: number, eps?: number): void;
export function generateText(w: TransformerWeights, cfg: TransformerConfig, seedIds: number[], maxLen: number, idxToToken: string[], temperature?: number, topP?: number, seqLen?: number): string;
```

---

## Before you start: build a gradient checker 🧪
You'll use it in every step from 4c to 4g. Write a small **test script** (e.g. `scratch/gradcheck.ts`, run with `bun`):
```
for a few random indices i of some parameter p:
    save = p[i]
    p[i] = save + ε ; L+ = loss(...)
    p[i] = save − ε ; L− = loss(...)
    p[i] = save
    numeric = (L+ − L−) / 2ε
    relErr  = |numeric − analytic[i]| / max(1e-8, |numeric| + |analytic[i]|)
```
Use `ε ≈ 1e-3` (Float32 is imprecise), and use a *scalar* loss such as `sum(output * randomFixedWeights)`. Pass: `relErr < 1e-2` in Float32 (`< 1e-4` if you temporarily use Float64).

---

## 4a — `matrix.ts`
**Tier 1.** Everything is a 1-D array plus dimensions you track yourself. Write `matmul` first; the two transposed variants are what the backward pass needs (`dA = dC·Bᵀ`, `dB = Aᵀ·dC`), so you never have to materialise a transpose.

**Tier 2.**
- mulberry32: a 32-bit integer PRNG (look up the ~6-line algorithm and implement it from the description).
- `matmul`: triple loop `i, k, j`. The **i-k-j** loop order is much faster than i-j-k. Why? (Think about the memory layout.)
- `xavierInit`: `limit = sqrt(6/(fanIn+fanOut))`, values `(rand()*2−1)*limit`.

**Checkpoints.**
- `matmul([1,2,3,4],[5,6,7,8],2,2,2)` → `[19,22,43,50]`
- `matmulTransB(A, B, …)` equals `matmul(A, transpose(B))` on a random 3×4 / 5×4 case.
- `resetRand(42)` twice gives the same sequence.

## 4b — Types, `initWeights`, `zeroGrads`, `countParams`
**Tier 1.** Write out the shape of every tensor (e.g. `wQ: D×D`, `ff1W: D×F`, `headW: D×V`, `tokEmb: V×D`, `posEmb: contextLen×D`). γ starts at 1, β and the biases at 0.

**Checkpoint.** With V=100, D=32, F=128, L=2, ctx=32, derive `countParams` **by hand on paper first**, then check the code agrees.
Per block = 2D (ln1) + 4(D·D + D) (Q,K,V,O) + 2D (ln2) + (D·F + F) + (F·D + D). Add the embeddings and the head.

## 4c — LayerNorm forward & backward
**Tier 2 (forward, per row of length D).**
`μ = mean(x)`, `σ² = mean((x−μ)²)`, `x̂ = (x−μ)/sqrt(σ²+ε)`, `y = γ·x̂ + β`, with `ε = 1e-5`. Cache `x̂` and `1/sqrt(σ²+ε)` per row.

**Tier 2 (backward).** `dγ += Σ_rows dy·x̂`, `dβ += Σ_rows dy`. For the input:
```
dx̂ = dy · γ
dx = invStd · ( dx̂ − mean(dx̂) − x̂ · mean(dx̂ · x̂) )
```
Derive this yourself on paper. It is the chain rule through μ and σ, both of which depend on every x.

**Checkpoint.** The forward output has per-row mean ≈ 0 and variance ≈ 1 (with γ=1, β=0). All three gradients pass the gradient checker.

## 4d — Causal multi-head self-attention (forward)
**Tier 1.** Q, K, V are each `S×D`. Head h owns columns `[h·dk, (h+1)·dk)`. For each head, compute scores `S×S`, mask the future, softmax each row, multiply by V's slice, and write the result into the head's columns of the output. Then apply `wO`.

**Tier 2.**
```
for head h:
    for i in 0..S−1:
        for j in 0..S−1:
            score[i][j] = j > i ? −∞ : (Q[i,h-slice] · K[j,h-slice]) / sqrt(dk)
        attn[i] = softmax(score[i])            ← subtract row max first!
        out[i, h-slice] = Σ_j attn[i][j] · V[j, h-slice]
proj = out · wO + bO
```
Cache `Q, K, V, attn` (per head) and the pre-projection `out`. You'll need them all in 4g.

**Checkpoints.**
- Every attention row sums to 1, and entries with `j > i` are exactly 0.
- **Causality test**: change the *last* token of the input. The outputs at positions `0..S−2` must be bit-for-bit unchanged.

## 4e — Block forward
**Tier 2.**
```
a    = LN1(x)
x1   = x + Attn(a)
f    = LN2(x1)
h    = ReLU(f · ff1W + ff1B)       (S×F)
x2   = x1 + (h · ff2W + ff2B)
```
**Checkpoint.** The output shape is `S×D`. With all block weights zeroed (γ=1), the block is the identity. Why? (Residuals.)

## 4f — Full forward + cross-entropy
**Tier 2.**
```
x = tokEmb[id_i] + posEmb[i]     for each position i
for each block: x = blockForward(x)
z = LNf(x); logits = z · headW + headB; probs = softmaxRows(logits)
loss = −(1/S) Σ_i log(probs[i, target_i] + 1e-10)
```
**Checkpoint ⭐.** At initialisation the loss should be **≈ ln(vocabSize)** (e.g. V=300 → ≈ 5.7), because an untrained model is close to uniform. If it's much larger, your init scale is off.

## 4g — Backward pass ⭐⭐ (the hard one)
Work **backwards through 4f → 4e → 4d → 4c**, one piece at a time, grad-checking each piece before moving on. `backward` must **add** into `grads`, so that gradients from many sequences accumulate.

**Tier 2 — the pieces, in order.**
1. **Softmax + cross-entropy**: `dlogits = (probs − onehot(target)) / S`. This is the nicest identity in deep learning. Derive it.
2. **Linear layer** `Y = X·W + b`: `dW += Xᵀ·dY` (`matmulTransA`), `db += sumCols(dY)`, `dX = dY·Wᵀ` (`matmulTransB`).
3. **LNf** → your 4c backward.
4. **Per block, reversed**:
   - residual: `dx1 = dx2` (+ whatever flows back through the FFN branch)
   - FFN: linear ff2 → ReLU (`dh = dout · (pre > 0)`) → linear ff1 → LN2 backward → add to `dx1`
   - attention output: linear wO backward gives `dOut` (S×D)
   - per head: `dV[j] += Σ_i attn[i][j]·dOut[i]`, `dAttn[i][j] = dOut[i]·V[j]`
   - softmax backward per row: `dScore[i][j] = attn[i][j] · (dAttn[i][j] − Σ_k attn[i][k]·dAttn[i][k])`
   - `dQ[i] += Σ_j dScore[i][j]·K[j] / √dk`, `dK[j] += Σ_i dScore[i][j]·Q[i] / √dk`
   - linear backward for wQ, wK, wV (each from input `a`), sum the three `da`s → LN1 backward → add to `dx`
5. **Embeddings**: `dTokEmb[id_i] += dx[i]`, `dPosEmb[i] += dx[i]`. Why `+=`? (The same token can appear twice.)

**Checkpoints.** Grad-check `headW`, `lnFGamma`, `ff2W`, `ff1B`, `wO`, `wV`, `wQ`, `wK`, `ln1Beta`, `tokEmb`, `posEmb` individually on a tiny config (V=10, D=8, heads=2, F=16, L=1, S=4). **Don't move on until they all pass.**

## 4h — Adam
**Tier 2** (per parameter element, t = step count starting at 1):
```
m = β1·m + (1−β1)·g
v = β2·v + (1−β2)·g²
m̂ = m / (1 − β1^t) ;  v̂ = v / (1 − β2^t)
p −= lr · m̂ / (sqrt(v̂) + ε)          β1=0.9 β2=0.999 ε=1e-8 lr=0.001
```
`initAdam`: one zero `m` and one zero `v` buffer per weight array. Walk the weights and grads in the **same order** (a keys list like `BLOCK_KEYS` helps).

**Checkpoint.** On the very first step, every parameter with a non-zero gradient moves by ≈ `lr` (≈ 0.001). Can you see why from the formulas?

## 4i — Training loop (`train.ts`)
**Tier 2.**
1. Corpus: `STORIES`, lower-cased, each prefixed with a space, BPE with the GPT-style regex `/\s\w+|[^\w\s]/g` (~1000 merges).
2. Vocab sorted by frequency → `wordToIndex`, `indexToWord`.
3. Sliding windows: `input = ids[i..i+seqLen)`, `target = ids[i+1..i+seqLen+1)`.
4. Each epoch: `zeroGrads` → for each sequence, `forward` + `backward` (accumulate) → average the grads over the sequences → `adamUpdate(…, t = epoch+1)`.
5. Every ~epochs/50, yield `{ epoch, loss }`. Every so often also include `sample: generateText(...)`.
6. Cache the trained weights in `.data/` under a filename that encodes the config, and skip training on a cache hit.

**Checkpoints.**
- **Overfit test first**: train on ONE short sentence. The loss should fall toward ~0, and generation should reproduce the sentence. If it can't overfit one sentence, you have a bug (usually in 4g).
- On the full corpus, the loss falls steadily from ≈ ln V, and samples go from word salad to story-like phrases.

## 4j — `generateText` (temperature + top-p)
**Tier 2.**
```
ids = seedIds
repeat maxLen times:
    window = last `seqLen` ids
    probs  = forward(window).probs at the LAST position
    logits = log(probs) / temperature → softmax again
    sort indices by prob desc; keep the smallest prefix with cumulative ≥ topP
    renormalise; sample with rand()
    ids.push(sampled)
return ids.map(i => idxToToken[i]).join("")
```
**Checkpoints.**
- `temperature → 0.1` gives near-greedy, repetitive output. `2.0` gives chaos.
- `topP = 1.0` is plain sampling. `topP → 0.1` is almost greedy.
- Write down what Holtzman et al. would predict for each setting, then check it.

## 4k — (Optional) Data parallelism with worker threads
**Tier 1.** Forward+backward for each sequence is independent, so split the sequences across `cpus().length − 2` workers. Weights and gradients live in **one `SharedArrayBuffer`**: a layout table maps each tensor to an offset, and every thread builds `Float32Array` views onto it. Workers write gradients into their own grad regions. The main thread sums them, runs Adam, and signals the next epoch.

**Signatures**
```ts
export type LayoutEntry = { key: string; block?: number; offset: number; length: number };
export function computeLayout(cfg: TransformerConfig): { layout: LayoutEntry[]; totalFloats: number };
export function packWeightsToSab(w: TransformerWeights, layout: LayoutEntry[], totalFloats: number): SharedArrayBuffer;
export function createWeightViews(sab: SharedArrayBuffer, layout: LayoutEntry[], numLayers: number): TransformerWeights;
export function createGradViews(sab: SharedArrayBuffer, layout: LayoutEntry[], numLayers: number): TransformerGrads;
```
**Checkpoint.** With a fixed seed, loss after 10 epochs matches the single-threaded run (up to float summation order), and wall time drops.

## 4l — Schema, route, hook
Schema defaults: `epochs 300 (50..2000)`, `temperature 0.8 (0.1..2)`, `topP 0.9 (0.1..1)`, `numLayers 2 (1..6)`, `maxTokens 40 (3..500)`.

**SSE event flow**
1. `init` → `InitData`
2. `epoch` × ~50 → `{ epoch, loss, sample? }`
3. `done` → `{ architecture, finalLoss, samples }`

The hook keeps `{ init?, epochs: EpochData[], samples: Sample[], summary? }`. Un-comment `/train-transformer` and mount the route.

---

## Common pitfalls
- **Softmax overflow**: always subtract the row max.
- Masking with `-1e9` and then letting a fully masked row produce NaN. That can't happen with a causal mask; if you see it, your indices are wrong.
- Forgetting the `1/√dk` in the *backward* pass for dQ/dK.
- Overwriting `grads` instead of accumulating (`=` vs `+=`).
- Mixing up `matmulTransA` and `matmulTransB` argument orders. Grad-check every linear layer.
- Using the post-update weights in backward (the same bug as Stage 2).
- The event loop is blocked for minutes: yield/`await` between epochs, or use 4k.

## Stretch goals
- Swap ReLU for GELU and learned positions for sinusoidal ones (Vaswani §3.5). Compare the loss curves.
- Weight tying: share `tokEmb` and `headW` (as GPT-2 does). How many params do you save?
- A KV-cache in `generateText` so generation is O(S) per token instead of O(S²).
- Visualise one head's attention matrix in the UI.
- Learning-rate warmup + cosine decay.
