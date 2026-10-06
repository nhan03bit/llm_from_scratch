# Research Notes

The 11 papers cited in [w3cj/how-llms-work](https://github.com/w3cj/how-llms-work), in the order you'll meet them. Each one has: **key idea**, **where it shows up in your code**, and **what to read** (you don't need the whole paper).

> Tip: read the paper *after* finishing the step it belongs to. The code gives you the intuition; the paper gives you the vocabulary and the "why".

---

## 1. ELIZA (1966) — Stage 0
**Weizenbaum, J.** "ELIZA — A Computer Program for the Study of Natural Language Communication Between Man and Machine." *Communications of the ACM* 9(1).
🔗 https://dl.acm.org/doi/10.1145/365153.365168

**Key idea.** A convincing conversation can come from **keyword rules plus decomposition/reassembly templates**, with no understanding at all. "I feel X" → "Why do you feel X?" works because the *user* supplies the meaning. Weizenbaum was alarmed at how readily people attributed intelligence to it (the "ELIZA effect"). That is still worth keeping in mind with modern LLMs.

**In your code.** `src/routes/simple-chat.ts` → `getSimpleChatResponse`: greetings, "i feel", "my X", fallback continuations. Pronoun swapping (`" i " → " you "`) is ELIZA's reassembly step.

**What to read.** The sections on *keywords*, *decomposition rules* and *reassembly rules*, and the DOCTOR script example. Also read the concluding discussion on how people perceived the program.

**Contrast with an LLM.** ELIZA's rules are **hand-written**. Everything from Stage 2 onwards **learns** its rules from data.

---

## 2. Backpropagation (1986) — Stage 2
**Rumelhart, D. E., Hinton, G. E., & Williams, R. J.** "Learning Representations by Back-Propagating Errors." *Nature* 323.
🔗 https://www.nature.com/articles/323533a0

**Key idea.** To train a network with **hidden layers**, apply the chain rule backwards from the loss: each unit's error signal δ is a weighted sum of the δs of the units it feeds, times its own activation derivative. Hidden units then learn **internal representations** that no one designed. This was the answer to Minsky & Papert's (1969) critique that single-layer perceptrons can't solve XOR.

**In your code.** `src/routes/neural-net/train.ts` → `trainMultiLayer`. You compute δ_output, propagate it through the hidden→output weights, multiply by `sigmoidDeriv`, and update every weight with `w -= lr * δ * input`. The same idea is behind every `...Backward` function in Stage 4.

**What to read.** The whole paper is 4 pages. Focus on the equations for ∂E/∂w, the generalised delta rule, and the symmetry example (Fig. 1).

---

## 3. Xavier/Glorot initialisation (2010) — Stages 2 & 4
**Glorot, X., & Bengio, Y.** "Understanding the Difficulty of Training Deep Feedforward Neural Networks." *AISTATS*.
🔗 https://proceedings.mlr.press/v9/glorot10a.html

**Key idea.** If initial weights are too big, activations saturate. If they're too small, signals shrink to nothing layer by layer. To keep the **variance of activations and gradients roughly constant** across layers, draw weights from `U(−a, a)` with `a = sqrt(6 / (fanIn + fanOut))`.

**In your code.** `matrix.ts` → `xavierInit(len, fanIn, fanOut)`, used by `initWeights` for every weight matrix in the transformer.

**What to read.** Section 4.2 ("Gradients at initialization") and equation (16), the normalised initialisation. Figures 3–4 show activations saturating.

---

## 4. Word2Vec (2013a) — Stage 3
**Mikolov, T., Chen, K., Corrado, G., & Dean, J.** "Efficient Estimation of Word Representations in Vector Space."
🔗 https://arxiv.org/abs/1301.3781

**Key idea.** A **shallow** network trained to predict nearby words (Skip-gram: centre → context) learns dense vectors where **geometry encodes meaning**. Similar words end up close together, and relationships become vector offsets: `vec(king) − vec(man) + vec(woman) ≈ vec(queen)`.

**In your code.** `src/routes/train-embed/train.ts` → `trainSkipGram`: the `W_in` rows become the embeddings. Your analogy output uses `cosineSimilarity` from `math.ts`.

**What to read.** Section 3.2 (Continuous Skip-gram, Fig. 1 right) and Section 4.1 (the analogy test set, Table 1).

---

## 5. Negative sampling (2013b) — Stage 3
**Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J.** "Distributed Representations of Words and Phrases and their Compositionality." *NeurIPS*.
🔗 https://arxiv.org/abs/1310.4546

**Key idea.** A full softmax over the vocabulary is expensive. Instead, turn each (target, context) pair into **binary classification**: push `σ(v_ctx · v_target)` → 1 for the real pair, and `σ(v_neg · v_target)` → 0 for **K random "negative" words** drawn from the unigram distribution **raised to the ¾ power**.

**In your code.** `sampleNegative()`, the negative-sampling table, and the per-pair update in `trainSkipGram`. The `negativeSamples` hyperparameter (default 5) is K.

**What to read.** Section 2.2 (Negative Sampling, equation 4) and the paragraph on the U(w)^{3/4} noise distribution. Section 2.3 (subsampling frequent words) is an optional stretch goal.

---

## 6. Adam (2014) — Stage 4h
**Kingma, D. P., & Ba, J.** "Adam: A Method for Stochastic Optimization." *ICLR 2015*.
🔗 https://arxiv.org/abs/1412.6980

**Key idea.** Keep an exponential moving average of the gradient (`m`, the "momentum") and of the squared gradient (`v`, a per-parameter scale). Correct both for their zero-initialisation bias, then step by `lr · m̂ / (sqrt(v̂) + ε)`. Each parameter effectively gets its own adaptive learning rate.

**In your code.** `transformer.ts` → `initAdam` (the `m`, `v` buffers per weight) and the Adam step in the training loop.

**What to read.** **Algorithm 1**, which is the whole algorithm in 10 lines, and Section 3 on bias correction. Defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e-8.

---

## 7. Byte Pair Encoding for NLP (2016) — Stage 1
**Sennrich, R., Haddow, B., & Birch, A.** "Neural Machine Translation of Rare Words with Subword Units." *ACL*.
🔗 https://arxiv.org/abs/1508.07909

**Key idea.** Start from characters, then **repeatedly merge the most frequent adjacent pair** into a new symbol. Frequent words become single tokens, and rare words break into meaningful subwords (`lowest → low + est`). There is no "unknown word" problem, and the vocabulary size is a single knob (the number of merges). Counting pairs **per unique word × its frequency**, instead of over the raw text, is what makes it fast.

**In your code.** `src/server/lib/bpe.ts` → `countWords`, `mergeTokens`, `trainBpe`, `applyMerges`. It is reused by Stage 3 (`corpus.ts`) and Stage 4 (`train.ts`). The same algorithm underlies GPT-2/GPT-4 (tiktoken) and SentencePiece.

**What to read.** Section 3.2 and **Algorithm 1** (a ~15-line Python listing). Check your `trainBpe` against it *after* you've written yours. Also look at Table 1 and the `low / lower / newest / widest` example.

---

## 8. Layer Normalization (2016) — Stage 4c
**Ba, J. L., Kiros, J. R., & Hinton, G. E.** "Layer Normalization."
🔗 https://arxiv.org/abs/1607.06450

**Key idea.** Normalise each position's activation vector to **zero mean and unit variance across its features**, then rescale with learned `γ` and shift with learned `β`. Unlike batch norm, it doesn't depend on batch statistics, so it works for sequences and at batch size 1.

**In your code.** `layerNormForward` / `layerNormBackward` in `transformer.ts`. They are used twice per block (pre-attention and pre-FFN) and once before the output head (`lnFGamma`, `lnFBeta`).

**What to read.** Section 3, equations (3)–(4). For the backward pass, derive it yourself. The trick is that the mean and variance depend on *every* input, so the gradient has three terms.

---

## 9. Attention Is All You Need (2017) — Stage 4d/4e
**Vaswani, A., et al.** "Attention Is All You Need." *NeurIPS*.
🔗 https://arxiv.org/abs/1706.03762

**Key idea.** Replace recurrence with **self-attention**. Each position projects to a query `Q`, key `K` and value `V`, and computes `softmax(QKᵀ / √d_k) V`. **Multi-head** runs this in parallel over `h` sub-spaces. Stacking blocks of (attention + feed-forward), each with a residual connection and normalisation, plus **positional information**, gives the Transformer.

**In your code.** `blockForward` in `transformer.ts`: Q/K/V projections (`wQ`, `wK`, `wV`), per-head attention, **causal mask** (position i sees only ≤ i, which makes it decoder-only), output projection, residual, FFN, residual. You use **learned** positional embeddings (`posEmb`) as GPT does, not the paper's sinusoids.

**What to read.** Section 3.2.1 (Scaled Dot-Product Attention: why divide by √d_k?), 3.2.2 (Multi-Head, Fig. 2), 3.3 (FFN), and 3.5 (Positional Encoding). Note that the paper puts the norm *after* each sub-layer (post-LN). You will use **pre-LN**, which trains more stably. Can you work out why?

---

## 10. GPT-1 (2018) — Stage 4 overall
**Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I.** "Improving Language Understanding by Generative Pre-Training." OpenAI.
🔗 https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

**Key idea.** Take **only the decoder** of the Transformer (masked self-attention), and pre-train it on raw text with one objective: **predict the next token** (maximise Σ log P(uᵢ | uᵢ₋ₖ … uᵢ₋₁)). The resulting model transfers to many tasks. Every modern chat LLM is a scaled-up descendant.

**In your code.** `train-transformer/train.ts` → `trainTransformer`: BPE the stories, build (input, target) pairs where target = input shifted by one, and minimise cross-entropy.

**What to read.** Section 3.1 (Unsupervised pre-training, equation 1–2) and Section 4.1 (Model specifications: 12 layers, 768 dims, 12 heads). Compare them with your 2 layers, 32 dims, 2 heads.

---

## 11. Nucleus (top-p) sampling (2019) — Stage 4j
**Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y.** "The Curious Case of Neural Text Degeneration." *ICLR 2020*.
🔗 https://arxiv.org/abs/1904.09751

**Key idea.** Greedy and beam search produce bland, repetitive loops, while pure sampling wanders into the unreliable long tail. **Nucleus sampling**: sort tokens by probability, keep the smallest set whose cumulative probability ≥ p, renormalise, and sample from that. Combine it with **temperature** (divide logits by T before softmax: T < 1 sharpens, T > 1 flattens).

**In your code.** `generateText(..., temperature = 0.8, topP = 0.9)` in `transformer.ts`, exposed as request params in `train-transformer-request.ts`.

**What to read.** Section 3.1 (Fig. 2, the repetition problem), Section 3.2 (the long tail), and **Section 4 / equation 3** (nucleus sampling definition).

---

## Paper → stage map

| Stage | Papers |
|---|---|
| 0 ELIZA | 1 |
| 1 BPE | 7 |
| 2 XOR | 2, 3 |
| 3 Embeddings | 4, 5, 7 |
| 4 Transformer | 3, 6, 7, 8, 9, 10, 11 |

## Optional further reading (not in the reference repo)
- Minsky & Papert (1969), *Perceptrons*: the XOR critique that Stage 2 answers.
- Radford et al. (2019), "Language Models are Unsupervised Multitask Learners" (GPT-2): the byte-level BPE regex and pre-LN.
- Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture": why pre-LN trains better. https://arxiv.org/abs/2002.04745
- Karpathy, "Let's build GPT" (video) and `micrograd`: great companions to Stages 2 and 4.
