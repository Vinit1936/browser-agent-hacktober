// DeepSeek chat-completions client — Browy's direct (no-Copilot) backend.
//
// When the user picks the DeepSeek provider in Settings and pastes an API
// key, the Agent drives its browser tools through this module instead of the
// GitHub Copilot SDK. Both endpoints this module can hit are OpenAI-
// compatible chat completions streamed as SSE:
//
//   - DeepSeek's own API (https://api.deepseek.com) for plain model ids
//     (deepseek-chat, deepseek-reasoner), with a DeepSeek key.
//   - OpenRouter (https://openrouter.ai/api/v1) for vendor-prefixed ids such
//     as deepseek/deepseek-v4.1-flash-1017, with an OpenRouter key.
//
// Kept free of Agent state so the SSE reduction is unit-testable with a
// mocked fetch (tests/deepseek-sse.mjs).

import type { ModelOption } from '../types.js';

export const DEEPSEEK_BASE_URL = 'https://api.deepseek.com';
export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

/** OpenRouter ids are "vendor/model" (e.g. deepseek/deepseek-v4.1-flash-1017);
 *  DeepSeek's own ids never contain a slash. The prefix decides the endpoint,
 *  and with it which key the pasted API key must be. */
export function isOpenRouterModel(model: string): boolean {
  return model.includes('/');
}

/** The chat-completions endpoint that serves `model`. */
export function chatCompletionsUrl(model: string): string {
  return isOpenRouterModel(model)
    ? `${OPENROUTER_BASE_URL}/chat/completions`
    : `${DEEPSEEK_BASE_URL}/chat/completions`;
}

/** The models Browy offers when the DeepSeek provider is active. */
export function deepseekModelOptions(): ModelOption[] {
  return [
    { id: 'deepseek-chat', name: 'DeepSeek Chat', vendor: 'deepseek' },
    { id: 'deepseek-reasoner', name: 'DeepSeek Reasoner', vendor: 'deepseek' },
    { id: 'deepseek/deepseek-v4.1-flash', name: 'DeepSeek V4.1 Flash', vendor: 'openrouter' },
    { id: 'deepseek/deepseek-v4.1-flash-1017', name: 'DeepSeek V4.1 Flash (1017)', vendor: 'openrouter' },
  ];
}

/** One tool call the model asked for, arguments still as raw JSON text. */
export interface DeepSeekToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** Everything one completions request produced, after the stream ended. */
export interface DeepSeekTurnResult {
  content: string;
  reasoning: string;
  toolCalls: DeepSeekToolCall[];
  finishReason?: string;
}

/** Mutable accumulator for one streamed response. Exported for tests. */
export interface DeepSeekAccumulator {
  content: string;
  reasoning: string;
  finishReason?: string;
  toolCalls: Map<number, DeepSeekToolCall>;
}

export function newDeepSeekAccumulator(): DeepSeekAccumulator {
  return { content: '', reasoning: '', toolCalls: new Map() };
}

/** Reduce one chat-completions delta into `acc`. Returns the pieces that
 *  were new, so the caller can stream them on without re-sending. Handles
 *  both reasoning spellings: DeepSeek sends `reasoning_content`, OpenRouter
 *  sends `reasoning`. */
export function applyDeepSeekDelta(
  acc: DeepSeekAccumulator,
  delta: {
    content?: string | null;
    reasoning_content?: string | null;
    reasoning?: string | null;
    tool_calls?: Array<{
      index?: number;
      id?: string;
      function?: { name?: string; arguments?: string };
    }>;
  },
): { text?: string; reasoning?: string } {
  const out: { text?: string; reasoning?: string } = {};
  if (delta.content) {
    acc.content += delta.content;
    out.text = delta.content;
  }
  const reasoning = delta.reasoning_content || delta.reasoning;
  if (reasoning) {
    acc.reasoning += reasoning;
    out.reasoning = reasoning;
  }
  for (const [i, tc] of (delta.tool_calls || []).entries()) {
    const index = typeof tc.index === 'number' ? tc.index : i;
    let slot = acc.toolCalls.get(index);
    if (!slot) {
      slot = { id: tc.id || `call_${index}`, name: '', arguments: '' };
      acc.toolCalls.set(index, slot);
    }
    if (tc.id) slot.id = tc.id;
    if (tc.function?.name) slot.name += tc.function.name;
    if (tc.function?.arguments) slot.arguments += tc.function.arguments;
  }
  return out;
}

/** Tool calls in model order, ready to be executed and echoed back as
 *  `role: "tool"` messages. */
export function finalizeDeepSeekToolCalls(acc: DeepSeekAccumulator): DeepSeekToolCall[] {
  return [...acc.toolCalls.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, tc]) => tc);
}

export interface DeepSeekStreamOptions {
  apiKey: string;
  model: string;
  messages: unknown[];
  tools?: unknown[];
  maxTokens?: number;
  /** Aborted when the user presses Stop. */
  signal?: AbortSignal;
  /** Idle timeout: the request dies when no SSE event arrives for this long.
   *  Default 5 minutes, matching the Copilot path's watchdog. */
  idleMs?: number;
  /** Test seam. */
  fetchImpl?: typeof fetch;
  onText?: (chunk: string) => void;
  onReasoning?: (chunk: string) => void;
}

/** One streamed chat-completions request. Resolves after the stream ends;
 *  text and reasoning chunks stream out via the callbacks as they arrive. */
export async function streamDeepSeekChat(opts: DeepSeekStreamOptions): Promise<DeepSeekTurnResult> {
  const fetchImpl = opts.fetchImpl || fetch;
  const idleMs = opts.idleMs ?? 5 * 60 * 1000;

  // One controller covers both Stop (outer signal) and the idle watchdog.
  const controller = new AbortController();
  const onOuterAbort = () => controller.abort();
  if (opts.signal) {
    if (opts.signal.aborted) controller.abort();
    else opts.signal.addEventListener('abort', onOuterAbort, { once: true });
  }
  let idleTimer: NodeJS.Timeout | undefined;
  const armIdle = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      controller.abort(new Error(`DeepSeek sent nothing for ${Math.round(idleMs / 1000)} s, so Browy stopped waiting.`));
    }, idleMs);
  };

  const body: Record<string, unknown> = {
    model: opts.model,
    messages: opts.messages,
    stream: true,
    max_tokens: opts.maxTokens,
  };
  if (opts.tools && opts.tools.length > 0) {
    body.tools = opts.tools;
    body.tool_choice = 'auto';
  }

  try {
    armIdle();
    // Vendor-prefixed ids route to OpenRouter (with an OpenRouter key);
    // plain ids go to DeepSeek's own API. X-Title is OpenRouter's optional
    // app-attribution header.
    const openrouter = isOpenRouterModel(opts.model);
    const res = await fetchImpl(chatCompletionsUrl(opts.model), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${opts.apiKey}`,
        ...(openrouter ? { 'X-Title': 'Browy' } : {}),
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => res.statusText);
      const via = openrouter ? 'OpenRouter' : 'DeepSeek';
      throw new Error(`${via} ${res.status}: ${String(errText).slice(0, 300)}`);
    }
    if (!res.body) throw new Error('DeepSeek returned an empty response body');

    const acc = newDeepSeekAccumulator();
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let done = false;
    while (!done) {
      const chunk = await reader.read();
      done = chunk.done;
      if (chunk.value) buffer += decoder.decode(chunk.value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') { done = true; break; }
        armIdle();
        let evt: any;
        try { evt = JSON.parse(payload); } catch { continue; }
        const choice = evt?.choices?.[0];
        if (choice?.finish_reason) acc.finishReason = choice.finish_reason;
        const delta = choice?.delta;
        if (!delta) continue;
        const parts = applyDeepSeekDelta(acc, delta);
        if (parts.text && opts.onText) opts.onText(parts.text);
        if (parts.reasoning && opts.onReasoning) opts.onReasoning(parts.reasoning);
      }
    }

    return {
      content: acc.content,
      reasoning: acc.reasoning,
      toolCalls: finalizeDeepSeekToolCalls(acc),
      finishReason: acc.finishReason,
    };
  } catch (e) {
    // An abort we didn't cause carries its reason (the idle watchdog's).
    if (controller.signal.aborted && !opts.signal?.aborted) {
      const reason = controller.signal.reason;
      if (reason instanceof Error) throw reason;
    }
    throw e;
  } finally {
    clearTimeout(idleTimer);
    if (opts.signal) opts.signal.removeEventListener('abort', onOuterAbort);
  }
}
