const HTML = __HTML__;
const SYSTEM_PROMPT = __PROMPT__;
const KNOWLEDGE = __KNOWLEDGE__;
const CASES_RAW = __CASES__;

const PERSONAS = { teacher: "教师端", student: "学生端", parent: "家长端" };
const RULES = {
  teacher: "你的服务对象是高中化学教师。给建议，不给指令，最终结论由教师决定。每条错因必须有学生作答中的证据；证据不足时要追问。禁止推断学生的智力、能力、态度或心理状态；禁止给出分数或等级。",
  student: "你的服务对象是学生。只引导，不给答案：不得直接给出正确答案、完整解题步骤或最终结果。用苏格拉底式提问引导学生自己发现差异，每次最多问 1—2 个问题。不得展示教师的分析草稿、班级统计或他人信息。",
  parent: "你的服务对象是家长。显示进步，不显示排名：可以说明孩子本次相对自身的变化、值得关注的知识点；不得展示班级/年级排名、其他学生的信息，也尽量少用分数。语言需通俗，避免术语堆砌。",
};
const TOOL_DESC = {
  kb_search: "检索化学学科知识与常见错因的知识库。args: {\"query\": \"关键词\"}",
  student_profile: "读取示例学生的历史错因档案。args: {\"student_id\": \"可选\"}",
  class_overview: "读取教师专属班级概览。args: {}",
  request_more_info: "声明当前证据不足，需要补充信息。args: {\"fields\": [\"...\"]}",
};
const TOOL_PERSONAS = {
  kb_search: ["teacher", "student", "parent"],
  student_profile: ["teacher", "parent"],
  class_overview: ["teacher"],
  request_more_info: ["teacher", "student", "parent"],
};
const HISTORY_MAX_MESSAGES = 8;
const HISTORY_MAX_CHARS = 12000;
const sessions = new Map();
const ipHits = new Map();
const quota = { date: "", count: 0 };
const cases = CASES_RAW.split(/\r?\n/).filter(Boolean).map((line) => {
  const item = JSON.parse(line);
  return {
    id: item.id, topic: item.topic, question: item.question,
    reference_answer: item.reference_answer, student_answer: item.student_answer,
    ambiguity: item.ambiguity || "none",
  };
});
const kb = KNOWLEDGE.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}
function getClientIp(request) {
  return request.headers.get("CF-Connecting-IP") || request.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() || "unknown";
}
function quotaSnapshot(ip) {
  const now = Date.now();
  const hits = (ipHits.get(ip) || []).filter((t) => now - t < 3600000);
  ipHits.set(ip, hits);
  return { ip_used_last_hour: hits.length, ip_limit_per_hour: 20, global_used_today: quota.count, global_limit_per_day: 250 };
}
function reserve(ip) {
  const day = new Date().toISOString().slice(0, 10);
  if (quota.date !== day) { quota.date = day; quota.count = 0; }
  const current = quotaSnapshot(ip);
  if (quota.count >= 250) return "演示额度已用完（当前服务实例今日共享额度上限），请稍后再来。";
  if (current.ip_used_last_hour >= 20) return "同一访客每小时最多 20 次，请稍后再试。";
  ipHits.set(ip, [...(ipHits.get(ip) || []), Date.now()]);
  quota.count += 1;
  return "";
}
function getSession(id, persona) {
  let bucket = sessions.get(id);
  if (!bucket) { bucket = {}; sessions.set(id, bucket); }
  if (!bucket[persona]) bucket[persona] = { persona, history: [], retrieved: [], corrections: [], denials: [] };
  return bucket[persona];
}
function resetSession(id, persona) {
  if (!persona) { sessions.delete(id); return; }
  const bucket = sessions.get(id);
  if (bucket) { delete bucket[persona]; if (!Object.keys(bucket).length) sessions.delete(id); }
}
function recentHistory(history) {
  const items = history.slice(-HISTORY_MAX_MESSAGES).map(({ role, content }) => ({ role, content: String(content || "") }));
  let left = HISTORY_MAX_CHARS;
  const selected = [];
  for (let i = items.length - 1; i >= 0; i -= 1) {
    let content = items[i].content;
    if (content.length > left) {
      if (!selected.length) selected.push({ role: items[i].role, content: "[较早内容已截断]\n" + content.slice(-(left - 12)) });
      break;
    }
    selected.push(items[i]); left -= content.length;
    if (left <= 0) break;
  }
  return selected.reverse();
}
function buildSystem(persona) {
  const tools = Object.entries(TOOL_DESC)
    .filter(([name]) => TOOL_PERSONAS[name].includes(persona))
    .map(([name, desc]) => `- ${name}: ${desc}`).join("\n") || "（无可用工具）";
  return `${SYSTEM_PROMPT.trim()}\n\n---\n\n## 端别与边界\n${RULES[persona]}\n\n---\n\n## 工具使用（Agent 运行规范）\n需要调用工具时单独输出一行合法格式：<tool_call>{"name":"kb_search","args":{"query":"气体摩尔体积 22.4"}}</tool_call>\n可用工具：\n${tools}\n一次只调用一个工具，最多 3 次；收到工具结果后继续推理，最终回答不得包含工具标记。`;
}
function parseToolCall(text) {
  const match = String(text || "").match(/<tool_call>\s*(\{.*?\})\s*<\/tool_call>/s);
  if (!match) return null;
  try { const item = JSON.parse(match[1]); return item && item.name ? item : null; } catch { return null; }
}
function execTool(name, args, session) {
  const allowed = TOOL_PERSONAS[name];
  if (!allowed) return JSON.stringify({ error: `未知工具 ${name}` });
  if (!allowed.includes(session.persona)) {
    session.denials.push({ tool: name, persona: session.persona, at: new Date().toISOString() });
    return JSON.stringify({ error: `拒绝：${session.persona} 端无权调用 ${name}` });
  }
  if (name === "kb_search") {
    const query = String(args.query || "").toLowerCase();
    const words = query.split(/[\s，。；、,.;:!?！？]+/).filter((w) => w.length > 1);
    const hits = kb.map((item) => ({ item, score: [...(item.keywords || []), item.topic || "", item.content || ""]
      .reduce((n, text) => n + (words.some((word) => String(text).toLowerCase().includes(word)) ? 1 : 0), 0) }))
      .filter((x) => x.score > 0).sort((a, b) => b.score - a.score).slice(0, 3).map(({ item }) => item);
    hits.forEach((item) => session.retrieved.push(item.id));
    return JSON.stringify({ results: hits.map(({ id, topic, content }) => ({ id, topic, content })), note: hits.length ? undefined : "知识库无匹配，请基于题目本身推理" });
  }
  if (name === "student_profile") return JSON.stringify({ student_id: args.student_id || "示例学生", note: "示例档案（非真实学生数据）", recent_topics: ["气体摩尔体积", "氧化还原概念"], open_issues: ["单位与适用条件类错误反复出现"] });
  if (name === "class_overview") return JSON.stringify({ note: "示例数据（非真实班级）", sample: "示例班级 40 人，本次该题错误率约 35%" });
  return JSON.stringify({ recorded: true, fields: args.fields || [], note: "已记录信息缺口，请在最终回答的「需要补充的信息」中列出。" });
}
function guardrail(persona, answer) {
  const hard = [];
  const soft = [];
  if (["student", "parent"].includes(persona)) {
    const match = String(answer).match(/班级第\s*\d+|年级第\s*\d+|成绩排名/);
    if (match) hard.push(match[0]);
  }
  for (const word of ["逻辑能力", "智商", "不适合学", "学习态度", "心理问题", "天赋", "太笨", "学习潜力"]) {
    if (String(answer).includes(word)) soft.push(word);
  }
  if (persona === "student" && /答案是/.test(answer) && !/[吗？?]/.test(answer)) soft.push("学生端疑似直接给出答案");
  const notes = [];
  if (hard.length) notes.push(`风险命中（未阻断）：${hard.join("、")}`);
  if (soft.length) notes.push(`规则层待复核：${soft.join("、")}`);
  return { hard_hits: hard, soft_hits: soft, banned_hits: [...hard, ...soft], notes, clean: !hard.length && !soft.length };
}
async function callModel(messages, env) {
  const base = (env.LLM_BASE_URL || "https://api.deepseek.com/v1").replace(/\/$/, "");
  const model = env.LLM_MODEL || "deepseek-flash";
  const body = { model, messages, temperature: 0.2 };
  if (model.startsWith("deepseek")) body.thinking = { type: "disabled" };
  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${env.LLM_API_KEY}` },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(45000),
  });
  if (!response.ok) throw new Error(`model_http_${response.status}`);
  const data = await response.json();
  return { content: data.choices?.[0]?.message?.content || "", usage: data.usage || {} };
}
function estimateCost(model) {
  const entries = { "deepseek-flash": [2, 8], "deepseek-chat": [2, 8], "qwen-plus": [0.8, 2], "qwen-turbo": [0.3, 0.6], "glm-4-flash": [0, 0] };
  const found = Object.entries(entries).find(([key]) => model.toLowerCase().includes(key));
  return found ? { input: found[1][0], output: found[1][1], currency: "CNY" } : null;
}
async function chat(body, request, env) {
  if (!env.LLM_API_KEY) return json({ error: "服务端未配置 API Key，暂时无法对话。" }, 503);
  const persona = body.persona || "teacher";
  const message = typeof body.message === "string" ? body.message.trim() : "";
  let sessionId = body.session_id;
  if (sessionId != null && typeof sessionId !== "string") return json({ error: "session_id 必须是字符串" }, 400);
  sessionId = sessionId || crypto.randomUUID();
  if (sessionId.length > 128) return json({ error: "session_id 过长（上限 128 字符）" }, 400);
  if (!RULES[persona]) return json({ error: `未知端别 ${persona}` }, 400);
  if (!message) return json({ error: "message 不能为空" }, 400);
  if (message.length > 3000) return json({ error: "输入过长（上限 3000 字）。" }, 413);
  const ip = getClientIp(request);
  const denied = reserve(ip);
  if (denied) return json({ error: denied, quota: quotaSnapshot(ip) }, 429);
  const session = getSession(sessionId, persona);
  const messages = [{ role: "system", content: buildSystem(persona) }, ...recentHistory(session.history), { role: "user", content: message }];
  const steps = [];
  const toolCalls = [];
  let answer = "";
  let totalPrompt = 0;
  let totalCompletion = 0;
  let start = Date.now();
  try {
    for (let i = 0; i <= 3; i += 1) {
      const result = await callModel(messages, env);
      totalPrompt += Number(result.usage.prompt_tokens || 0);
      totalCompletion += Number(result.usage.completion_tokens || 0);
      const call = parseToolCall(result.content);
      if (!call) { answer = result.content; break; }
      const toolResult = execTool(call.name, call.args || {}, session);
      toolCalls.push({ name: call.name, args: call.args || {} });
      steps.push({ tool: call.name, args: call.args || {}, result: toolResult.slice(0, 500) });
      messages.push({ role: "assistant", content: result.content });
      messages.push({ role: "user", content: `[工具结果] ${toolResult}` });
      if (i === 3) {
        messages.push({ role: "user", content: "已达到工具调用轮次上限，请基于已有信息直接给出最终回答，不要再调用工具。" });
        const final = await callModel(messages, env);
        totalPrompt += Number(final.usage.prompt_tokens || 0);
        totalCompletion += Number(final.usage.completion_tokens || 0);
        answer = final.content.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim();
      }
    }
  } catch (error) {
    return json({ error: error.message?.startsWith("model_http_") ? `模型服务暂不可用（${error.message.slice(11)}），请稍后再试。` : "模型服务暂时连接失败，请稍后重试。" }, 502);
  }
  answer = String(answer).replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim();
  session.history.push({ role: "user", content: message }, { role: "assistant", content: answer });
  session.history = session.history.slice(-HISTORY_MAX_MESSAGES);
  while (session.history.reduce((n, item) => n + item.content.length, 0) > HISTORY_MAX_CHARS && session.history.length > 2) session.history.shift();
  const model = env.LLM_MODEL || "deepseek-flash";
  const prices = estimateCost(model);
  const cost = prices ? (totalPrompt * prices.input + totalCompletion * prices.output) / 1e6 : null;
  return json({
    persona, persona_label: PERSONAS[persona], session_id: sessionId, answer,
    tool_calls: toolCalls.map((item) => item.name), steps,
    guardrail: guardrail(persona, answer), denials: session.denials,
    metrics: { calls: toolCalls.length + 1, avg_latency_s: Number(((Date.now() - start) / 1000 / (toolCalls.length + 1)).toFixed(3)), cost, cost_currency: prices?.currency || null },
    quota: quotaSnapshot(ip),
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: { "access-control-allow-methods": "GET, POST, OPTIONS", "access-control-allow-headers": "Content-Type" } });
    if (request.method === "GET" && ["/", "/index.html"].includes(url.pathname)) return new Response(HTML, { headers: { "content-type": "text/html; charset=utf-8" } });
    if (request.method === "GET" && url.pathname === "/api/health") return json({ ok: true, has_key: Boolean(env.LLM_API_KEY), personas: PERSONAS, cases: cases.length, quota: quotaSnapshot(getClientIp(request)) });
    if (request.method === "GET" && url.pathname === "/api/cases") return json({ cases });
    if (request.method === "POST" && url.pathname === "/api/chat") {
      let body; try { body = await request.json(); } catch { return json({ error: "请求格式无效" }, 400); }
      return chat(body || {}, request, env);
    }
    if (request.method === "POST" && url.pathname === "/api/reset") {
      let body; try { body = await request.json(); } catch { return json({ error: "请求格式无效" }, 400); }
      resetSession(body.session_id || "default", body.persona);
      return json({ ok: true });
    }
    return json({ error: "not found", path: url.pathname }, 404);
  },
};
