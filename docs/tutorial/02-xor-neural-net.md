# Stage 2 — XOR Neural Net (Perceptron vs. Backprop)

## Goal
Train two tiny networks on XOR and stream the loss live:
- a **single-layer perceptron**, which *must fail*;
- a **2 → 4 → 1 multi-layer network** trained with backpropagation, which *succeeds*.

This is the smallest possible version of "how every neural network, including GPT, learns".

## Concepts
- **Neuron** = weighted sum + bias → non-linearity (sigmoid).
- **Loss** = how wrong we are (mean squared error here).
- **Gradient descent**: nudge each weight opposite to ∂loss/∂weight.
- **Linear separability**: one neuron draws one straight line. Can you separate XOR's `[0,1],[1,0]` from `[0,0],[1,1]` with one line?
- **Backpropagation**: the chain rule, applied from the output backwards, gives every hidden weight its gradient.
- 📚 Rumelhart, Hinton & Williams (1986), [RESEARCH.md §2](RESEARCH.md#2-backpropagation-1986--stage-2); Glorot & Bengio (2010), [§3](RESEARCH.md#3-xavierglorot-initialisation-2010--stages-2--4).

## Files you'll create
```
src/schemas/neural-net-request.ts
src/routes/neural-net/train.ts
src/routes/neural-net/serialize.ts
src/routes/neural-net/index.ts
src/client/hooks/use-neural-net-chat.tsx
.data/                                  ← weights get saved here (add to .gitignore?)
```

## The contract
`src/client/components/neural-net-result/index.tsx` → `EpochData { epoch, loss }`, `Prediction { input, expected, actual }`, `NeuralNetSummary { architecture, predictions, verdict }`.

## Signatures
```ts
// schema: { mode: "single-layer" | "multi-layer", epochs: int 100..100000, default 5000 }
function sigmoid(x: number): number;
function sigmoidDeriv(s: number): number;           // takes the sigmoid OUTPUT
function randWeight(): number;
export type EpochResult = { epoch: number; loss: number };
export type TrainResult = { architecture: string; predictions: Prediction[]; verdict: string; /* + weights */ };
export async function* trainSingleLayer(epochs: number): AsyncGenerator<EpochResult | TrainResult>;
export async function* trainMultiLayer(epochs: number): AsyncGenerator<EpochResult | TrainResult>;
export async function saveNetwork(network: SavedNetwork, path: string): Promise<void>;
```

---

## Step 2.1 — Schema
zod object with `mode` (enum) and `epochs` (int, min/max/default). The client's `buildBody` option in `useSSEChat` turns the typed input into this JSON. How will the user choose the mode from a chat box?

## Step 2.2 — Activation helpers
**Tier 2.**
- `σ(x) = 1 / (1 + e^(−x))`
- `σ'(x) = σ(x) · (1 − σ(x))`, so if you already have `s = σ(x)`, the derivative is `s(1 − s)`.
- Random init: small values around 0 (e.g. uniform in [−1, 1]). *Why not all zeros?* (Think: symmetry, and what backprop does to identical hidden units.)

**Checkpoint.** `sigmoid(0) = 0.5`, `sigmoidDeriv(0.5) = 0.25`.

## Step 2.3 — `trainSingleLayer` (expected to FAIL)
**Tier 1.** One output neuron: `out = σ(w1·x1 + w2·x2 + b)`. For each of the 4 XOR examples, compute the error and push the weights to reduce it.

**Tier 2.**
```
for epoch in 1..epochs:
    totalLoss = 0
    for (x, target) in XOR:
        out   = σ(w·x + b)
        err   = out − target
        totalLoss += err²
        delta = err · σ'(out)
        w_i  -= lr · delta · x_i ;  b -= lr · delta
    every ~epochs/50: yield { epoch, loss: totalLoss / 4 }
yield final { architecture, predictions, verdict }
```
The reference uses `lr = 1.0`.

**Checkpoint.** The loss plateaus around **0.25** and the predictions all hover near 0.5. Now **write down why**, in one sentence, using the phrase "linearly separable".

## Step 2.4 — `trainMultiLayer` ⭐ backprop
**Tier 1.** Add a hidden layer of 4 sigmoid neurons. The output neuron's error is easy. The hidden neurons' error has to be *inferred*: each gets the output error, weighted by how much it contributed to the output (its outgoing weight).

**Tier 2.** For each example:
```
FORWARD
  h_j   = σ( Σ_k W1[k][j]·x_k + b1_j )        j = 0..3
  out   = σ( Σ_j W2[j]·h_j + b2 )
BACKWARD
  δ_out = (out − target) · σ'(out)
  δ_h_j = δ_out · W2[j] · σ'(h_j)              ← use W2 BEFORE you update it!
UPDATE
  W2[j]    -= lr · δ_out · h_j ;   b2   -= lr · δ_out
  W1[k][j] -= lr · δ_h_j · x_k ;   b1_j -= lr · δ_h_j
```

**Checkpoints.**
- The loss drops below **0.01** within the default 5000 epochs (most seeds).
- `predictions.map(p => Math.round(p.actual))` → `[0, 1, 1, 0]`.
- If it gets stuck at ~0.25 on some runs: that's a local minimum or saturation. Try another init range or more hidden units, and explain what you saw.

**Gradient check (strongly recommended — you'll reuse this skill in Stage 4).**
Pick one weight, compute `(L(w+ε) − L(w−ε)) / 2ε` with `ε = 1e-4`, and compare it with your analytic gradient. The relative error should be below `1e-4`.

## Step 2.5 — `serialize.ts`
Save the trained weights as JSON to `.data/<mode>.json`, using Node `fs/promises`. Define `SingleLayerWeights` and `MultiLayerWeights` types, plus a union `SavedNetwork` with a discriminant field. *Why save weights at all?* Because the weights **are** the model. Everything learned lives in those numbers.

## Step 2.6 — Route + hook
**SSE event flow**
1. `epoch` × ~50 → `{ epoch, loss }`
2. `done` → `NeuralNetSummary`

In the route, `for await (const r of generator)`, and use a type guard to tell `EpochResult` from `TrainResult`. After `done`, call `saveNetwork`.
Hook: `use-neural-net-chat.tsx`, with state `{ epochs: EpochData[]; summary?: NeuralNetSummary }`. Un-comment `/neural-net-xor` in `routes.tsx` and mount the route.

---

## Common pitfalls
- Updating `W2` before computing `δ_h`. The hidden deltas then use the *new* weights.
- Passing the pre-activation to `sigmoidDeriv` when it expects the output.
- Yielding every epoch floods SSE. Throttle to ~50 updates.
- `lr` too small (nothing happens) or too large (oscillation).

## Stretch goals
- Swap sigmoid for `tanh` or ReLU in the hidden layer. What changes?
- Replace MSE with binary cross-entropy. The `σ'` term cancels. Derive why.
- Visualise the decision boundary on a 20×20 grid in the UI.
