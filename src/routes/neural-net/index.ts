/**
 * Neural network route — trains a tiny network on XOR and streams the results.
 *
 * Takes a mode ("single-layer" or "multi-layer") and epoch count. Delegates to
 * the training generators in `train.ts`, streaming each epoch's loss as it goes.
 * On completion, persists the trained weights to disk and emits the final result
 * with predictions and a pass/fail verdict.
 *
 * SSE event flow:
 * 1. "epoch" × ~50 — loss values during training (throttled to ~50 updates)
 * 2. "done" — architecture, predictions for all 4 XOR inputs, and verdict
 */

import { Hono } from "hono";
import NerualNetRequest from "../../schemas/neural-net-request.js";

import type { TrainResult } from "./train";
import { zValidator } from "@hono/zod-validator";

import { streamSSE } from "hono/streaming";
import { createEmitter } from "../../server/lib/sse.js";
import { saveNetwork } from "./serialize.js";
import { trainSingleLayer, trainMultiLayer } from "./train.js";

/** Type guard to distinguish the final TrainResult from intermediate EpochResults. */
function isTrainResult(value: unknown): value is TrainResult {
  return typeof value === "object" && value !== null && "predictions" in value;
}

export default new Hono().post(
  "/neural-net",
  zValidator("json", NerualNetRequest),
  (c) => {
    const { mode, epochs } = c.req.valid("json");
    const trainer = mode === "single-layer"
      ? trainSingleLayer(epochs)
      : trainMultiLayer(epochs);

    return streamSSE(c, async (stream) => {
      const { emit } = createEmitter(stream);

      for await (const result of trainer) {
        if (isTrainResult(result)) {
          await saveNetwork(result.weights, `.data/${mode}-weights.json`);
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { weights: _, ...payload } = result;
          await emit(payload, "done");
        } else {
          await emit(result, "epoch", 20);
        }
      }
    });
  }
)