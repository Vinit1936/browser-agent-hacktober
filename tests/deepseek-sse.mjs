// DeepSeek SSE reduction (agent/deepseek.ts).
//
// The DeepSeek backend is Browy's direct (no-Copilot) path: one chat-
// completions request streams SSE deltas, tool-call arguments arrive split
// across chunks, and the reduction must reassemble them in model order.
// These tests pin the reducer and the fetch loop with a mocked fetch.
//
// Run: node tests/deepseek-sse.mjs   (Node 23.6+ strips the .ts types itself)

import {
  applyDeepSeekDelta,
  newDeepSeekAccumulator,
  finalizeDeepSeekToolCalls,
  streamDeepSeekChat,
} from '../src/agent/deepseek.ts';

const failures = [];
const check = (name, cond, detail = '') => { if (!cond) failures.push(`${name}${detail ? `\n    ${detail}` : ''}`); };

// ── Reducer ────────────────────────────────────────────────────────────────
{
  const acc = newDeepSeekAccumulator();
  const p1 = applyDeepSeekDelta(acc, { content: 'Hel' });
  const p2 = applyDeepSeekDelta(acc, { content: 'lo' });
  check('reducer returns only the new text chunks', p1.text === 'Hel' && p2.text === 'lo');
  applyDeepSeekDelta(acc, { reasoning_content: 'thinking…' });
  check('reasoning accumulates separately', acc.reasoning === 'thinking…');
  check('content accumulates', acc.content === 'Hello');
}

{
  // Tool calls streamed as chunks: id+name first, arguments split after,
  // two calls at different indices that must not interleave.
  const acc = newDeepSeekAccumulator();
  applyDeepSeekDelta(acc, { tool_calls: [{ index: 0, id: 'call_9', function: { name: 'click_index', arguments: '{"ind' } }] });
  applyDeepSeekDelta(acc, { tool_calls: [{ index: 0, function: { arguments: 'ex": 3}' } }] });
  applyDeepSeekDelta(acc, { tool_calls: [{ index: 1, id: 'call_2', function: { name: 'scroll', arguments: '{"direction":"down"}' } }] });
  const calls = finalizeDeepSeekToolCalls(acc);
  check('two tool calls reassembled', calls.length === 2, JSON.stringify(calls));
  check('model order (by index)', calls[0].name === 'click_index' && calls[1].name === 'scroll');
  check('arguments joined across chunks', calls[0].arguments === '{"index": 3}', calls[0].arguments);
  check('ids kept', calls[0].id === 'call_9' && calls[1].id === 'call_2');
}

// ── Streamed request over a mocked fetch ───────────────────────────────────
{
  const enc = new TextEncoder();
  const sse = [
    'data: {"choices":[{"delta":{"content":"Hi"}}]}',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"get_page_info","arguments":"{}"}}]}}]}',
    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
    'data: [DONE]',
  ].map((l) => l + '\n\n').join('');
  const body = new ReadableStream({ start(c) { c.enqueue(enc.encode(sse)); c.close(); } });
  let seenUrl = '';
  let seenAuth = '';
  const res = await streamDeepSeekChat({
    apiKey: 'sk-test',
    model: 'deepseek-chat',
    messages: [{ role: 'user', content: 'hello' }],
    onText: () => {}, // full assertions below
    fetchImpl: (async (url, init) => {
      seenUrl = String(url);
      seenAuth = String(init?.headers?.Authorization || '');
      return { ok: true, status: 200, body };
    }),
  });
  check('endpoint is the chat-completions API', seenUrl === 'https://api.deepseek.com/chat/completions', seenUrl);
  check('key travels as a bearer token', seenAuth === 'Bearer sk-test', seenAuth);
  check('streamed tool call extracted', res.toolCalls.length === 1 && res.toolCalls[0].name === 'get_page_info' && res.toolCalls[0].id === 'c1', JSON.stringify(res.toolCalls));
  check('finish reason captured', res.finishReason === 'tool_calls', String(res.finishReason));
}

// ── Non-OK responses surface the status line ───────────────────────────────
{
  const err = await streamDeepSeekChat({
    apiKey: 'sk-bad',
    model: 'deepseek-chat',
    messages: [],
    fetchImpl: (async () => ({ ok: false, status: 401, text: async () => 'Authentication Fails, Your api key is invalid' })),
  }).then(() => null, (e) => e);
  check('401 becomes a readable error', err instanceof Error && /DeepSeek 401/.test(err.message) && /invalid/i.test(err.message), String(err));
}

// ── Endpoint routing: prefixed ids go to OpenRouter ────────────────────────
{
  const enc = new TextEncoder();
  // OpenRouter streams reasoning as delta.reasoning (DeepSeek's own API uses
  // reasoning_content) — both must surface as reasoning deltas.
  const sse = [
    'data: {"choices":[{"delta":{"reasoning":"thinking…"}}]}',
    'data: {"choices":[{"delta":{"content":"ok"}}]}',
    'data: [DONE]',
  ].map((l) => l + '\n\n').join('');
  let seenUrl = '';
  let seenTitle = '';
  const body = new ReadableStream({ start(c) { c.enqueue(enc.encode(sse)); c.close(); } });
  const res = await streamDeepSeekChat({
    apiKey: 'sk-or-test',
    model: 'deepseek/deepseek-v4.1-flash-1017',
    messages: [],
    fetchImpl: (async (url, init) => {
      seenUrl = String(url);
      seenTitle = String(init?.headers?.['X-Title'] || '');
      return { ok: true, status: 200, body };
    }),
  });
  check('prefixed id routes to openrouter', seenUrl === 'https://openrouter.ai/api/v1/chat/completions', seenUrl);
  check('openrouter attribution header set', seenTitle === 'Browy', seenTitle);
  check('openrouter reasoning field streams', res.reasoning === 'thinking…', JSON.stringify(res.reasoning));
  check('content still assembled', res.content === 'ok');
}
{
  let seenUrl = '';
  await streamDeepSeekChat({
    apiKey: 'sk-test',
    model: 'deepseek-chat',
    messages: [],
    fetchImpl: (async (url) => {
      seenUrl = String(url);
      return { ok: true, status: 200, body: new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode('data: [DONE]\n\n')); c.close(); } }) };
    }),
  });
  check('plain id stays on deepseek api', seenUrl === 'https://api.deepseek.com/chat/completions', seenUrl);
}
{
  const err = await streamDeepSeekChat({
    apiKey: 'sk-bad',
    model: 'deepseek/deepseek-v4.1-flash-1017',
    messages: [],
    fetchImpl: (async () => ({ ok: false, status: 401, text: async () => 'No auth credentials found' })),
  }).then(() => null, (e) => e);
  check('openrouter errors name the service', err instanceof Error && /OpenRouter 401/.test(err.message), String(err));
}

if (failures.length) {
  console.error(`\n  ✗ deepseek-sse: ${failures.length} failure${failures.length > 1 ? 's' : ''}`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
} else {
  console.log('  ✅ deepseek-sse');
}
