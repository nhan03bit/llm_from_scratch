# Build an LLM From Scratch — Tutorial

You are rebuilding [w3cj/how-llms-work](https://github.com/w3cj/how-llms-work) by hand: an app that walks every stage of the LLM pipeline, from pattern matching to training a GPT-style transformer, with **no ML libraries**.

This tutorial gives you **TODOs, hints and checkpoints — not solutions.** You write every line.

## The pipeline you're building

```
 "the cat sat"                                   Stage 0: ELIZA — the "before" picture:
      │                                          no learning, just if-statements
      ▼
 ┌──────────────┐  Stage 1: BPE tokenizer        text → subword tokens → integer IDs
 │  Tokenizer   │  (Sennrich 2016)
 └──────┬───────┘
        ▼
 ┌──────────────┐  Stage 2: XOR neural net       how ANY network learns:
 │  Learning    │  (Rumelhart 1986, Glorot 2010) forward → loss → backprop → update
 └──────┬───────┘
        ▼
 ┌──────────────┐  Stage 3: Word2Vec Skip-gram   token IDs → vectors that capture meaning
 │  Embeddings  │  (Mikolov 2013a/b)
 └──────┬───────┘
        ▼
 ┌──────────────┐  Stage 4: GPT transformer      attention + layer norm + FFN, trained
 │ Transformer  │  (Vaswani 2017, Ba 2016,       with Adam to predict the next token
 │              │   Kingma 2014, Radford 2018)
 └──────┬───────┘
        ▼
 ┌──────────────┐  Stage 4j: sampling            logits → temperature → top-p → next token
 │  Sampling    │  (Holtzman 2019)
 └──────────────┘
```

## Files in this folder

| File | What it's for |
|---|---|
| [PROGRESS.md](PROGRESS.md) | Your checklist. Tick boxes as you go; the tutor reads it. |
| [RESEARCH.md](RESEARCH.md) | The 11 papers behind this project: key idea, where it appears in the code, what to read. |
| [00-setup-and-eliza.md](00-setup-and-eliza.md) | Recap of what you've built + housekeeping. |
| [01-bpe-tokenizer.md](01-bpe-tokenizer.md) | Byte Pair Encoding from scratch. |
| [02-xor-neural-net.md](02-xor-neural-net.md) | Perceptron vs. multi-layer net + backprop. |
| [03-word-embeddings.md](03-word-embeddings.md) | Word2Vec Skip-gram with negative sampling. |
| [04-transformer.md](04-transformer.md) | Decoder-only transformer, backward pass, Adam, sampling. |

## Using the tutor agent

The agent lives in `.claude/agents/llm-tutor.md`. The quickest way to use it is the `/tutor` skill (`.claude/skills/tutor/SKILL.md`), which runs the agent with your PROGRESS.md and source tree already loaded:

```
/tutor
/tutor more hint
/tutor stuck
/tutor check src/server/lib/bpe.ts
/tutor explain why we divide by sqrt(d_k)
```

You can also mention the agent directly:

```
@llm-tutor what's next?
@llm-tutor I'm stuck on trainBpe — more hint
@llm-tutor check my code in src/server/lib/bpe.ts
@llm-tutor explain why we divide by sqrt(d_k)
```

It reads `PROGRESS.md` and your `src/`, then answers with **📍 Where you are → ✅ Next TODOs → 💡 Hint → 📚 Research → 🧪 How to verify**.

### The hint ladder

| Tier | You get | Ask with |
|---|---|---|
| 1 | Concept + a guiding question | (default) |
| 2 | Pseudocode / equations | "more hint", "tier 2" |
| 3 | Signature + skeleton with `// TODO` bodies | "stuck", "tier 3" |
| after 3 | It walks through *your* code with a concrete input until you find the bug | — |

It will **never** write a full function for you. That's the point.

## Ground rules for yourself

1. **The UI types are the spec.** Every `src/client/components/*-result/index.tsx` exports the exact shapes your server must emit (`BpeInit`, `MergeStep`, `BpeResult`, …). Read them first.
2. **Run a checkpoint before moving on.** Each step has one.
3. **Read the paper section after you finish the step**, not before. It will make much more sense.
4. **Compare with the reference only after a stage is done.**

## Running

```bash
bun install
bun run dev          # http://localhost:5173
bunx tsc -b --noEmit # type-check
```

Watch raw SSE from a route:

```bash
curl -N -X POST localhost:5173/bpe-tokenize -H 'content-type: application/json' -d '{"message":"low lower lowest"}'
```
