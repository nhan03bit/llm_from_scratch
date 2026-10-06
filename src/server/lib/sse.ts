/**
 * Server-Sent Events (SSE) streaming JSON to browser
 *
 * SSE is the protocal streaming that data piece by piece to browser.
 * instead of waiting for entire response
 */

import type { streamSSE } from "hono/streaming";

type SSEStream = Parameters<Parameters<typeof streamSSE>[1]>[0];

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Create an emitter send JSON events to SSE.
 */
export function createEmitter(stream: SSEStream) {
  return {
    /**
     *  Serirliazed data as JSON then write to SSE event.
     *  @param data - The data to send as JSON.
     *  @param event - The event name to send.
     *  @delay optional delay in milliseconds before sending the event.
     */
    async emit<T>(data: T, event: string, delay?: number) {
      await stream.writeSSE({ data: JSON.stringify(data), event });
      if (delay) await sleep(delay);
    },

    /** Write error envent. Client readSSE() as surface result.error */
    async emitError(err: unknown) {
      await stream.writeSSE({
        data: JSON.stringify({ error: err instanceof Error ? err.message : String(err) }),
        event: "error"
      });
    }
  }
}