#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChemAI 三端 Agent —— 可运行实现（教师端 / 学生端 / 家长端）

这不是一份方案，是一段能跑起来的代码。它实现了四件事：

  1. Persona 路由：按端别装配不同的角色提示、可用工具与输出规范。
     - 教师端：给建议，不给指令（不替老师下结论）
     - 学生端：只引导，不给答案（苏格拉底式追问）
     - 家长端：显示进步，不显示排名
  2. 工具调用：模型通过 <tool_call>{"name":...,"args":{...}}</tool_call> 协议
     请求 kb_search（知识库检索）/ student_profile（学生档案）/ class_overview（班级概览，
     教师专属）/ request_more_info（声明信息不足，需补充证据）。
  3. 有限对话上下文：同一会话与角色的最近消息会回填给模型；历史保存在内存，
     最多保留 8 条 user/assistant 消息和 12000 字符上下文。
  4. 护栏：persona 隔离（学生端/家长端调用教师专属工具会被拒绝并留痕）+
     越界输出过滤（能力/态度/心理归因、给分数等级）。

零第三方依赖（仅标准库），接口为任意 OpenAI 兼容 /chat/completions，
支持 DeepSeek / 通义 / Kimi / 智谱 / 火山方舟 / OpenAI 等。

被 eval/run_eval.py 以 --agent 模式调用；也可单独命令行运行：

    export LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=...
    python eval/agent/chemai_agent.py --persona teacher \
        --case eval/cases.jsonl --id C04
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_KB = AGENT_DIR / "knowledge.jsonl"

# ---------------------------------------------------------------------------
# 价格表：用于按 token 数估算成本（单位：元 / 每百万 token）
# 说明：以下为各厂商公开定价的近似值（未独立核实），仅用于实验内相对比较；
#       真实账单以厂商控制台为准。
# ---------------------------------------------------------------------------
PRICE_TABLE = {
    "deepseek-flash": {"in": 2.0, "out": 8.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "deepseek-v4-pro": {"in": 9.0, "out": 27.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    # 以下保留历史配置键；不据此推断当前可用性或底层模型版本。
    # 原价格表仅用于演示估算（请求行为需以供应商实际文档确认），
    "deepseek-chat": {"in": 2.0, "out": 8.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "deepseek-reasoner": {"in": 4.0, "out": 16.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "qwen-plus": {"in": 0.8, "out": 2.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "qwen-turbo": {"in": 0.3, "out": 0.6, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "qwen-max": {"in": 2.4, "out": 9.6, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "glm-4-flash": {"in": 0.0, "out": 0.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "glm-4-plus": {"in": 50.0, "out": 50.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "moonshot-v1-8k": {"in": 12.0, "out": 12.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "gpt-4o-mini": {"in": 1.0, "out": 4.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
    "gpt-4o": {"in": 18.0, "out": 72.0, "cur": "CNY", "src": "示例价格假设，未核实当前价目，非账单"},
}

# 硬违规输出（结构化、可直接判定）：非教师端出现排名类信息
HARD_OUTPUT = ["成绩排名", "班级第", "年级第"]

# 软违规输出（能力/态度/心理归因类，语境敏感）：只记「待复核」，
# 因为关键词匹配无法区分「断言」与「引用后否定 / A不A 疑问式」，存在已知误报。
SOFT_OUTPUT = [
    "逻辑能力", "逻辑思维有", "智商", "不适合学", "不适合学化学", "学习态度",
    "心理问题", "能力有问题", "天赋", "太笨", "态度不端正", "学习潜力",
]

# 向后兼容（旧代码可能引用）
BANNED_OUTPUT = HARD_OUTPUT + SOFT_OUTPUT

# Bounded context window shared with the runnable app prototype.
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 12000

# 语境豁免（修复护栏误报）：模型「引用并否定」越界说法时会写出这些词，
# 例如「不足以支持"逻辑能力有问题"这类结论」——这是正确行为，不应判违规。
_NEG_BEFORE = (
    "不是", "不属于", "不能", "不足以", "不足", "无法", "不应", "不宜", "拒绝", "不建议",
    "而非", "并非", "并不是", "也算不上", "算不上", "谈不上", "不支持", "没有证据",
    "不成立", "不意味着", "还不足以", "不宜下", "不轻易", "不应当", "不要", "别",
)
_NEG_AFTER = (
    "这类结论", "此类结论", "这类判断", "这类标签", "这类说法", "的说法", "的标签", "这种说法",
)


def _asserted(text: str, word: str) -> bool:
    """word 是否出现在「肯定/断言」语境（排除引用后否定、排除 A不A 疑问式）。"""
    idx = text.find(word)
    while idx >= 0:
        before = text[max(0, idx - 30):idx]
        after = text[idx + len(word): idx + len(word) + 12]
        # A不A 疑问式：如「适不适合学」含子串「不适合学」，但这是提问不是断言
        is_question = (idx >= 1 and text[idx] == "不" and idx + 1 < len(text)
                       and text[idx - 1] == text[idx + 1])
        if (not is_question
                and not any(m in before for m in _NEG_BEFORE)
                and not any(m in after for m in _NEG_AFTER)):
            return True
        idx = text.find(word, idx + len(word))
    return False


def _student_leaks_answer(text: str) -> bool:
    """学生端是否「直接给出了答案」。

    朴素子串匹配会误报：模型复述学生提问（如「这题答案是 0.5 mol 吗」）或
    声明「不能直接给答案」时也包含「答案是」。这里加引号 / 疑问语境豁免。
    """
    for m in re.finditer(r"答案是", text or ""):
        i = m.start()
        before = text[max(0, i - 12):i]
        after = text[m.end(): m.end() + 12]
        if any(q in before for q in ("「", "“", "\"", "问", "是不是", "是否",
                                     "无法", "不能", "不给", "不做")):
            continue
        if ("吗" in after) or ("？" in after) or ("?" in after):
            continue
        return True
    return False

# ---------------------------------------------------------------------------
# 工具注册表，以及"哪些 persona 可以用"
# ---------------------------------------------------------------------------
TOOLS = {
    "kb_search": {
        "desc": "检索化学学科知识与常见错因的知识库。args: {\"query\": \"关键词\"}",
        "personas": {"teacher", "student", "parent"},
    },
    "student_profile": {
        "desc": "读取当前学生的历史错因档案（会话状态/记忆）。args: {\"student_id\": \"可选\"}",
        "personas": {"teacher", "parent"},
    },
    "class_overview": {
        "desc": "读取班级整体情况（教师专属）。args: {}",
        "personas": {"teacher"},
    },
    "request_more_info": {
        "desc": "声明当前证据不足，需要补充信息。args: {\"fields\": [\"...\"]}",
        "personas": {"teacher", "student", "parent"},
    },
}

PERSONA_RULES = {
    "teacher": (
        "你的服务对象是高中化学教师。给建议，不给指令：你做错因假设与讲评参考，"
        "最终结论由教师决定。每条错因必须有学生作答中的证据；证据不足时要追问。"
        "禁止推断学生的智力、能力、态度或心理状态；禁止给出分数或等级。"
    ),
    "student": (
        "你的服务对象是学生。只引导，不给答案：不得直接给出正确答案、完整解题步骤或最终结果。"
        "用苏格拉底式提问引导学生自己发现差异与原因，每次最多问 1—2 个问题。"
        "不得展示教师的分析草稿、班级统计或他人信息。"
    ),
    "parent": (
        "你的服务对象是家长。显示进步，不显示排名：可以说明孩子本次相对自身的变化、"
        "值得关注的知识点；不得展示班级/年级排名、其他学生的信息、也尽量少用分数。"
        "语言需通俗，避免术语堆砌。"
    ),
}

TOOL_PROTOCOL = """
## 工具使用（Agent 运行规范）

你可以在需要时调用工具。**调用工具时必须单独输出一行**，格式严格如下（合法 JSON）：

<tool_call>{{"name": "kb_search", "args": {{"query": "气体摩尔体积 22.4"}}}}</tool_call>

可用工具：
{tools}

规则：
- 一次只调用一个工具；收到「工具结果」后继续推理，直到可以给出最终回答。
- 给出最终回答时不要包含任何 <tool_call> 标记。
- 最多允许 {max_steps} 轮工具调用，请只在真正需要外部信息时使用。
"""


# ---------------------------------------------------------------------------
# LLM 客户端（记录每次调用的延迟与 token 用量）
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.2, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.calls: list[dict] = []  # 每次调用的指标

    def chat(self, messages: list, temperature: float | None = None) -> str:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        # 以下请求附加参数为现有适配逻辑，实际兼容性需按服务验证：
        # 会额外消耗推理 token、令 temperature 失效，并改变输出形态。
        # 本项目需要可复现的确定性输出，故显式关闭（仅对 DeepSeek 系模型传该参数）。
        if self.model.startswith("deepseek"):
            payload["thinking"] = {"type": "disabled"}
        data = json.dumps(payload).encode("utf-8")
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                dt = time.perf_counter() - t0
                content = body["choices"][0]["message"]["content"]
                usage = body.get("usage") or {}
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                if pt is None:
                    # 极少数接口不回 usage，用字符数粗略折算，并标记为估算
                    pt = sum(len(str(m.get("content", ""))) for m in messages) // 3
                    ct = len(content) // 3
                    est = True
                else:
                    est = False
                self.calls.append({
                    "latency_s": round(dt, 3),
                    "prompt_tokens": int(pt or 0),
                    "completion_tokens": int(ct or 0),
                    "estimated": est,
                })
                return content
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:400]}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
            if attempt < self.max_retries:
                time.sleep(2 * attempt)
        raise RuntimeError(f"调用模型失败（重试 {self.max_retries} 次）：{last_err}")

    # ---------- 指标汇总 ----------
    def reset(self) -> None:
        self.calls = []

    def summary(self, model: str) -> dict:
        lat = sorted(c["latency_s"] for c in self.calls)
        pt = sum(c["prompt_tokens"] for c in self.calls)
        ct = sum(c["completion_tokens"] for c in self.calls)
        est = any(c["estimated"] for c in self.calls)

        def pct(p):
            if not lat:
                return 0.0
            i = min(len(lat) - 1, int(round((p / 100) * (len(lat) - 1))))
            return lat[i]

        price = match_price(model)
        cost = None
        if price:
            cost = pt / 1_000_000 * price["in"] + ct / 1_000_000 * price["out"]
        return {
            "calls": len(self.calls),
            "total_latency_s": round(sum(lat), 3),
            "avg_latency_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
            "p50_latency_s": round(pct(50), 3),
            "p95_latency_s": round(pct(95), 3),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "tokens_estimated": est,
            "cost": round(cost, 6) if cost is not None else None,
            "cost_currency": price["cur"] if price else None,
            "cost_source": price["src"] if price else "未定价（请在 PRICE_TABLE 中补充）",
        }


def match_price(model: str) -> dict | None:
    m = (model or "").lower()
    for key, val in PRICE_TABLE.items():
        if key in m:
            return val
    return None


# ---------------------------------------------------------------------------
# 会话状态（记忆）
# ---------------------------------------------------------------------------
@dataclass
class Session:
    persona: str = "teacher"
    history: list = field(default_factory=list)      # 对话历史
    retrieved: list = field(default_factory=list)    # 检索到的知识
    corrections: list = field(default_factory=list)  # 教师修正记录
    denials: list = field(default_factory=list)      # persona 隔离拒绝记录

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Agent 本体
# ---------------------------------------------------------------------------
class ChemAIAgent:
    def __init__(self, base_url: str, api_key: str, model: str,
                 system_prompt: str = "", kb_path: Path | str = DEFAULT_KB,
                 temperature: float = 0.2, max_steps: int = 3):
        self.client = LLMClient(base_url, api_key, model, temperature)
        self.model = model
        self.system_prompt = system_prompt
        self.kb = _load_kb(Path(kb_path)) if Path(kb_path).exists() else []
        self.max_steps = max_steps

    # ---------- 系统提示装配 ----------
    def _build_system(self, persona: str) -> str:
        rules = PERSONA_RULES.get(persona, "")
        allowed = {n: t["desc"] for n, t in TOOLS.items() if persona in t["personas"]}
        tools_txt = "\n".join(f"- {n}: {d}" for n, d in allowed.items()) or "（无可用工具）"
        parts = [self.system_prompt.strip()] if self.system_prompt.strip() else []
        parts.append(f"## 端别与边界\n{rules}")
        parts.append(TOOL_PROTOCOL.format(tools=tools_txt, max_steps=self.max_steps))
        return "\n\n---\n\n".join(parts)

    # ---------- 工具执行 ----------
    def _exec_tool(self, name: str, args: dict, session: Session) -> str:
        spec = TOOLS.get(name)
        if spec is None:
            return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
        if session.persona not in spec["personas"]:
            session.denials.append({"tool": name, "persona": session.persona,
                                    "at": datetime.now().isoformat(timespec="seconds")})
            return json.dumps(
                {"error": f"拒绝：{session.persona} 端无权调用 {name}",
                 "hint": "当前端别能力边界不允许该操作，请用本端可用信息回答或说明无法提供。"},
                ensure_ascii=False)

        if name == "kb_search":
            q = str(args.get("query", ""))
            hits = _kb_search(self.kb, q, top_k=3)
            for h in hits:
                session.retrieved.append(h["id"])
            if not hits:
                return json.dumps({"results": [], "note": "知识库无匹配，请基于题目本身推理"},
                                  ensure_ascii=False)
            return json.dumps({"results": [{"id": h["id"], "topic": h["topic"], "content": h["content"]}
                                           for h in hits]}, ensure_ascii=False)

        if name == "student_profile":
            return json.dumps({
                "student_id": args.get("student_id", "示例学生"),
                "note": "示例档案（非真实学生数据）",
                "recent_topics": ["气体摩尔体积", "氧化还原概念"],
                "open_issues": ["单位与适用条件类错误反复出现"],
            }, ensure_ascii=False)

        if name == "class_overview":
            return json.dumps({
                "note": "示例数据（非真实班级）",
                "sample": "示例班级 40 人，本次该题错误率约 35%",
            }, ensure_ascii=False)

        if name == "request_more_info":
            return json.dumps({
                "recorded": True,
                "fields": args.get("fields", []),
                "note": "已记录信息缺口，请在最终回答的「需要补充的信息」中列出。",
            }, ensure_ascii=False)

        return json.dumps({"error": "未实现"}, ensure_ascii=False)

    # ---------- 主循环 ----------
    def run(self, user_input: str, persona: str = "teacher",
            session: Session | None = None) -> dict:
        session = session or Session(persona=persona)
        session.persona = persona
        messages = [
            {"role": "system", "content": self._build_system(persona)},
        ]
        messages.extend(_recent_history(session.history))
        messages.append({"role": "user", "content": user_input})

        steps, tool_calls = [], []
        answer = ""
        for _ in range(self.max_steps + 1):
            content = self.client.chat(messages)
            call = _parse_tool_call(content)
            if call is None:
                answer = content
                break
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            result = self._exec_tool(name, args, session)
            tool_calls.append({"name": name, "args": args})
            steps.append({"tool": name, "args": args, "result": result[:500]})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"[工具结果] {result}"})
        else:
            # 用尽轮次仍在调工具：强制收口
            messages.append({
                "role": "user",
                "content": "已达到工具调用轮次上限，请基于已有信息直接给出最终回答，不要再调用工具。",
            })
            answer = self.client.chat(messages)

        answer = _strip_tool_blocks(answer)
        session.history.append({"role": "user", "content": user_input})
        guard = self._guardrail(persona, answer, session)
        session.history.append({"role": "assistant", "content": answer})
        session.history[:] = session.history[-MAX_HISTORY_MESSAGES:]

        return {
            "persona": persona,
            "answer": answer,
            "tool_calls": tool_calls,
            "steps": steps,
            "guardrail": guard,
            "session": session.to_dict(),
        }

    # ---------- 护栏 ----------
    def _guardrail(self, persona: str, answer: str, session: Session) -> dict:
        """硬/软分级护栏。

        - hard：非教师端出现排名类信息（结构化、可直接判定）→ 真越界；
        - soft：能力/态度/心理归因类词 → 仅「待复核」。关键词无法判断语境，
          已知会误报「引用后否定」「A不A 疑问式」等，故不作否决，交人工/裁判复核。
        """
        soft = [w for w in SOFT_OUTPUT if _asserted(answer, w)]
        hard = []
        if persona in ("student", "parent"):
            for m in re.finditer(r"班级第\s*\d+|年级第\s*\d+|成绩排名", answer):
                i = m.start()
                before = answer[max(0, i - 30):i]
                after = answer[m.end(): m.end() + 12]
                if any(x in before for x in _NEG_BEFORE) or any(x in after for x in _NEG_AFTER):
                    continue
                hard.append(m.group(0))
                break
        if persona == "student" and _student_leaks_answer(answer):
            soft.append("学生端疑似直接给出答案")

        notes = []
        if hard:
            notes.append("风险命中（未阻断）：" + "、".join(hard))
        if soft:
            notes.append("规则层待复核（可能为引用式否定导致的误报）：" + "、".join(soft))
        return {"hard_hits": hard, "soft_hits": soft, "banned_hits": hard + soft,
                "notes": notes, "clean": not (hard or soft)}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parse_tool_call(text: str) -> dict | None:
    m = _TOOL_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict) and "name" in obj:
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _strip_tool_blocks(text: str) -> str:
    return _TOOL_RE.sub("", text or "").strip()


def _recent_history(history: list[dict]) -> list[dict]:
    """Return a bounded, recent user/assistant window for the next model call."""
    candidates = [
        {"role": item["role"], "content": str(item.get("content", ""))}
        for item in history
        if item.get("role") in ("user", "assistant")
    ][-MAX_HISTORY_MESSAGES:]

    selected = []
    remaining = MAX_HISTORY_CHARS
    for item in reversed(candidates):
        content = item["content"]
        if len(content) > remaining:
            if not selected:
                marker = "[较早内容已截断]\n"
                content = marker + content[-max(0, remaining - len(marker)):]
                selected.append({"role": item["role"], "content": content})
            break
        selected.append(item)
        remaining -= len(content)
        if remaining <= 0:
            break
    selected.reverse()
    return selected


def _load_kb(path: Path) -> list:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _kb_search(kb: list, query: str, top_k: int = 3) -> list:
    """极简关键词打分检索（演示用，非向量检索）。"""
    q = (query or "").lower()
    toks = [t for t in re.split(r"[\s,，、/]+", q) if t]
    scored = []
    for item in kb:
        text = (item.get("topic", "") + " " + " ".join(item.get("keywords", [])) + " "
                + item.get("content", "")).lower()
        score = sum(2 if t and t in item.get("keywords", []) else (1 if t and t in text else 0)
                    for t in toks)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored[:top_k]]


def build_case_input(case: dict) -> str:
    """把一条评测样例装成教师端输入（与 run_eval 保持一致）。"""
    return (
        f"【题目】\n{case['question']}\n\n"
        f"【参考答案】\n{case['reference_answer']}\n\n"
        f"【学生作答】\n{case['student_answer']}\n\n"
        f"【样例标注的歧义情况】{case.get('ambiguity', 'none')}\n"
    )


# ---------------------------------------------------------------------------
# 命令行入口（便于手动演示 Agent 行为）
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="ChemAI 三端 Agent（可运行）")
    ap.add_argument("--persona", default="teacher", choices=["teacher", "student", "parent"])
    ap.add_argument("--case", default="", help="cases.jsonl 路径（配合 --id 使用）")
    ap.add_argument("--id", default="", help="要运行的样例编号，如 C04")
    ap.add_argument("--input", default="", help="直接给一段用户输入")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--prompt", default="", help="教师端诊断 system prompt 文件")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    api_key = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
               or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ARK_API_KEY") or "")
    if not api_key:
        raise SystemExit("[错误] 未设置 LLM_API_KEY。")
    if not args.model:
        raise SystemExit("[错误] 未指定模型（--model 或 LLM_MODEL）。")

    sys_prompt = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else ""
    agent = ChemAIAgent(args.base_url, api_key, args.model, sys_prompt, temperature=args.temperature)

    if args.case and args.id:
        case = None
        for line in Path(args.case).read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["id"] == args.id:
                case = json.loads(line)
                break
        if case is None:
            raise SystemExit(f"[错误] 未找到样例 {args.id}")
        user_input = build_case_input(case)
    else:
        user_input = args.input or "请分析这份学生作答的错因。"

    out = agent.run(user_input, persona=args.persona)
    print("=" * 60)
    print(f"persona = {out['persona']}｜工具调用 = {[c['name'] for c in out['tool_calls']]}")
    print("=" * 60)
    print(out["answer"])
    print("=" * 60)
    print("指标：", json.dumps(agent.client.summary(args.model), ensure_ascii=False))


if __name__ == "__main__":
    main()
