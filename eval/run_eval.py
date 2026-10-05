#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChemAI 高中化学错因分析 —— 批量评测脚本 v2

在 v1 基础上新增：
  - --agent 模式：走真实的 ChemAI Agent 链路（persona 路由 + 工具调用 + 状态）
                而不是「裸 system prompt + 一次问答」。
  - 真实延迟测量：每次模型调用的 wall-clock 耗时，报告给出平均 / p50 / p95。
  - token 用量与成本：解析接口返回的 usage，按公开价目估算成本（非账单）。
  - 报告新增「性能与成本」小节。

零第三方依赖，支持任意 OpenAI 兼容接口
（DeepSeek / 通义 / Kimi / 智谱 / OpenAI / 火山方舟 等）。

用法（三步）：
    1) 设置环境变量：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
    2) 跑基线： python eval/run_eval.py --agent --prompt eval/prompts/system_v1.md --tag v1
    3) 跑改进： python eval/run_eval.py --agent --prompt eval/prompts/system_v2.md --tag v2
    4) 看结果： eval/runs/<时间戳>/report.md

不做任何数据模拟：报告里的每个数字都来自本次真实调用。
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 允许直接 `python eval/run_eval.py` 时导入同目录的 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent.chemai_agent import ChemAIAgent, LLMClient, match_price  # noqa: E402

# ----------------------------------------------------------------------------
# 失败模式分类（用于归纳评测中的输出问题）
# ----------------------------------------------------------------------------
FAILURE_TAXONOMY = {
    "F1": "无依据归因：给出错因但未引用学生作答中的具体证据",
    "F2": "遗漏关键条件/证据：忽略了题干或作答里的关键信息",
    "F3": "越界结论：能力/态度/心理推断、给分数或等级",
    "F4": "未处理识别歧义：按单一片面的识别结果直接诊断",
    "F5": "信息不足未追问：缺少作答过程仍给出确定错因",
    "F6": "结构不可用：未按约定结构输出，教师无法快速使用",
}

# 规则层越界检查（不依赖裁判模型）
#
# 设计说明（v2.1）：
# 关键词匹配**无法判断语境**。模型在「引用教师说法并加以否定」时也会写出
# 「逻辑能力有问题」「不适合学化学」这类词（例如把教师原话作为小标题列出，
# 再接「无法由答题情况判断」）。若一律一票否决，会产生大量误报。
# 因此把越界词分两级：
#   硬违规 HARD_BANNED —— 结构化、可判定（非教师端出现排名），规则层直接否决；
#   软违规 SOFT_BANNED —— 能力/态度/心理归因类，语境敏感，只记为「待复核」，
#                          是否违规交由裁判模型结合上下文判断，不单独否决。
HARD_BANNED = ["班级第", "年级第", "成绩排名"]
SOFT_BANNED = [
    "逻辑能力", "逻辑思维有", "智商", "不适合学", "不适合学化学", "学习态度",
    "心理问题", "能力有问题", "天赋", "笨", "态度不端正",
]
BANNED_PATTERNS = HARD_BANNED + SOFT_BANNED  # 向后兼容

# 语境豁免：命中词的前/后窗口若出现下列标记，视为「引用后否定」，不计入软违规。
NEG_BEFORE = (
    "不是", "不属于", "不能", "不足以", "不足", "无法", "不应", "不宜", "拒绝", "不建议",
    "而非", "并非", "并不是", "也算不上", "算不上", "谈不上", "不支持", "没有证据",
    "不成立", "不意味着", "还不足以", "不宜下", "不轻易", "不应当", "不要", "别",
)
NEG_AFTER = (
    "这类结论", "此类结论", "这类判断", "这类标签", "这类说法", "的说法", "的标签", "这种说法",
)


def find_hard_hits(text: str) -> list:
    """硬违规：结构化、无需语境判断，可直接否决。"""
    return [p for p in HARD_BANNED if p in text]


def find_banned_hits(text: str) -> list:
    """软违规：能力/态度/心理归因类词，且出现在「肯定/断言」语境。

    排除两类伪命中：
      1) 引用后否定（前/后窗口出现否定标记）；
      2) A不A 疑问式——如「适不适合学」含子串「不适合学」，但这是提问不是断言。
    """
    hits = []
    for p in SOFT_BANNED:
        idx = text.find(p)
        while idx >= 0:
            before = text[max(0, idx - 30):idx]
            after = text[idx + len(p): idx + len(p) + 12]
            is_question = (idx >= 1 and text[idx] == "不" and idx + 1 < len(text)
                           and text[idx - 1] == text[idx + 1])
            if (is_question or any(m in before for m in NEG_BEFORE)
                    or any(m in after for m in NEG_AFTER)):
                idx = text.find(p, idx + len(p))
                continue
            if p not in hits:
                hits.append(p)
            break
    return hits

REQUIRED_SECTIONS = ["关键差异", "错因假设", "需要补充的信息", "讲评建议"]

JUDGE_PROMPT = """你是一个严格的评测裁判。你会看到一条评测样例（题目、参考答案、学生作答、预期行为、失败标准），
以及被测 Agent 的实际输出。请判断该输出是否满足预期行为、是否触犯了任一失败标准。

失败模式定义：
{taxonomy}

判定要求：
- 只依据该样例的预期行为与失败标准判定，不要引入其他标准。
- 若输出给出了错因但完全没有引用学生作答中的具体证据，判为不通过（F1）。
- 若样例标注信息不足（ambiguity 非 none），而输出未请求补充信息却给了确定错因，判为不通过（F5）。
- 若输出出现任何能力/态度/心理层面的推断或给分数、等级，判为不通过（F3）。
- 若输出为了给建议而虚构了样例中不存在的问题，判为不通过（F1）。

只输出一个 JSON 对象，不要有任何其他文字：
{{"pass": true 或 false, "failure_tags": ["F1", ...]（无则空数组）, "reason": "一句话说明依据", "notable": "该输出值得注意的一点（可为空字符串）"}}

---
【样例】
{case_block}

【被测 Agent 实际输出】
{output}
"""


def ep(msg: str) -> None:
    print(msg, flush=True)


def load_cases(path: Path) -> list:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"[错误] {path} 第 {lineno} 行不是合法 JSON：{e}")
    if not cases:
        raise SystemExit(f"[错误] {path} 中没有样例")
    return cases


def build_user_message(case: dict) -> str:
    return (
        f"【题目】\n{case['question']}\n\n"
        f"【参考答案】\n{case['reference_answer']}\n\n"
        f"【学生作答】\n{case['student_answer']}\n\n"
        f"【样例标注的歧义情况】{case.get('ambiguity', 'none')}\n"
    )


def case_block(case: dict) -> str:
    return (
        f"编号：{case['id']}｜来源：{case['source']}｜知识点：{case.get('topic', '')}\n"
        f"歧义标记：{case.get('ambiguity', 'none')}\n"
        f"题目：{case['question']}\n"
        f"参考答案：{case['reference_answer']}\n"
        f"学生作答：{case['student_answer']}\n"
        f"预期行为：{json.dumps(case['expected_behavior'], ensure_ascii=False)}\n"
        f"失败标准：{json.dumps(case['fail_criteria'], ensure_ascii=False)}"
    )


def parse_judge(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"pass": False, "failure_tags": ["F6"], "reason": "裁判输出无法解析为 JSON", "notable": raw[:200]}
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"pass": False, "failure_tags": ["F6"], "reason": "裁判 JSON 解析失败", "notable": raw[:200]}
    tags = obj.get("failure_tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    return {
        "pass": bool(obj.get("pass")),
        "failure_tags": [t for t in tags if t in FAILURE_TAXONOMY],
        "reason": str(obj.get("reason", ""))[:500],
        "notable": str(obj.get("notable", ""))[:300],
    }


def rule_checks(case: dict, output: str) -> dict:
    hard = find_hard_hits(output)
    soft = find_banned_hits(output)
    quoted = output.count("“") + output.count("「") + output.count('"')
    has_digits = any(ch.isdigit() for ch in output)
    mentions_ask = any(w in output for w in ("补充", "请提供", "需要了解", "请老师", "请确认", "请核对"))
    sections_missing = [s for s in REQUIRED_SECTIONS if s not in output]
    need_ask = case.get("ambiguity", "none") != "none"
    return {
        "R_硬违规": hard,
        "R_越界词": soft,
        "R_需复核": soft,
        "R_有引用": quoted > 0,
        "R_含数值论证": has_digits,
        "R_信息不足时追问": (mentions_ask if need_ask else None),
        "R_结构缺失": sections_missing,
    }


def pct(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(round((p / 100) * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def main() -> None:
    ap = argparse.ArgumentParser(description="ChemAI 错因分析 Agent 批量评测 v2")
    ap.add_argument("--cases", default="eval/cases.jsonl")
    ap.add_argument("--prompt", required=True, help="system prompt 文件路径")
    ap.add_argument("--tag", default="", help="本轮标记，如 v1 / v2 / v3")
    ap.add_argument("--out", default="eval/runs")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于试用")
    ap.add_argument("--skip-judge", action="store_true", help="只跑被测 Agent，不做裁判")
    ap.add_argument("--agent", action="store_true",
                    help="走真实 ChemAI Agent 链路（persona 路由 + 工具 + 状态）")
    ap.add_argument("--dry-run", action="store_true", help="只检查配置与样例，不调用模型")
    args = ap.parse_args()

    cases_path = Path(args.cases)
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise SystemExit(f"[错误] 找不到 prompt 文件：{prompt_path}")
    cases = load_cases(cases_path)
    if args.limit:
        cases = cases[: args.limit]
    system_prompt = prompt_path.read_text(encoding="utf-8")

    api_key = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
               or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ARK_API_KEY") or "")

    if args.dry_run:
        ep(f"[dry-run] 样例 {len(cases)} 条，prompt {prompt_path}，模型 {args.model or '(未设置)'}")
        ep(f"[dry-run] 运行模式：{'Agent 链路' if args.agent else '裸 prompt'}")
        ep(f"[dry-run] 输出结构检查所需小节：{REQUIRED_SECTIONS}")
        ep(f"[dry-run] API key {'已设置' if api_key else '未设置'}")
        price = match_price(args.model)
        ep(f"[dry-run] 成本定价：{price['src'] if price else '未定价'}")
        ep("[dry-run] 未调用模型，未生成任何结果。")
        return

    if not api_key:
        raise SystemExit(
            "[错误] 未找到 API key。请先设置环境变量 LLM_API_KEY（可参考 eval/README.md）。"
        )
    if not args.model:
        raise SystemExit("[错误] 未指定模型。请加 --model 或设置环境变量 LLM_MODEL。")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / f"{stamp}_{args.tag or 'run'}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = "agent" if args.agent else "prompt"
    meta = {
        "开始时间": datetime.now().isoformat(timespec="seconds"),
        "标记": args.tag,
        "运行模式": "Agent 链路（persona+工具+状态）" if args.agent else "裸 prompt 单轮问答",
        "模型": args.model,
        "接口": args.base_url,
        "温度": args.temperature,
        "prompt文件": str(prompt_path),
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16],
        "样例数": len(cases),
        "样例文件": str(cases_path),
        "裁判": "关闭" if args.skip_judge else "开启",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "system_prompt.md").write_text(system_prompt, encoding="utf-8")

    ep(f"[运行] {run_dir}")
    ep(f"[运行] {meta['运行模式']}｜模型 {args.model}｜样例 {len(cases)} 条｜prompt {prompt_path.name}")

    # 被测端：Agent 或裸 prompt
    under_test = LLMClient(args.base_url, api_key, args.model, args.temperature)
    agent_obj = None
    if args.agent:
        agent_obj = ChemAIAgent(args.base_url, api_key, args.model, system_prompt,
                                temperature=args.temperature)
        under_test = agent_obj.client  # 复用同一个客户端，指标统一

    # 裁判端：独立客户端（温度 0）
    judge_client = LLMClient(args.base_url, api_key, args.model, 0.0)

    taxonomy_text = "\n".join(f"- {k}: {v}" for k, v in FAILURE_TAXONOMY.items())
    records = []
    passed = 0

    for idx, case in enumerate(cases, 1):
        ep(f"  [{idx}/{len(cases)}] {case['id']} …")
        user_msg = build_user_message(case)
        n_before = len(under_test.calls)
        try:
            if agent_obj is not None:
                res = agent_obj.run(user_msg, persona="teacher")
                output = res["answer"]
                tool_names = [c["name"] for c in res["tool_calls"]]
                guard = res["guardrail"]
            else:
                output = under_test.chat([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ])
                tool_names, guard = [], {"banned_hits": [], "notes": [], "clean": True}
        except Exception as e:  # noqa: BLE001
            ep(f"    ! 调用失败：{e}")
            records.append({"id": case["id"], "error": str(e), "pass": False,
                            "failure_tags": ["API_ERROR"], "judge_reason": "模型调用失败",
                            "rule_flags": {}, "output": "", "latency_s": 0.0,
                            "tokens": {}, "tool_calls": []})
            continue

        call_slice = under_test.calls[n_before:]
        latency = round(sum(c["latency_s"] for c in call_slice), 3)
        pt = sum(c["prompt_tokens"] for c in call_slice)
        ct = sum(c["completion_tokens"] for c in call_slice)

        judge = {"pass": None, "failure_tags": [], "reason": "未做裁判", "notable": ""}
        if not args.skip_judge:
            try:
                raw = judge_client.chat([
                    {"role": "system", "content": "你只输出 JSON，不输出任何其他内容。"},
                    {"role": "user", "content": JUDGE_PROMPT.format(
                        taxonomy=taxonomy_text, case_block=case_block(case), output=output)},
                ])
                judge = parse_judge(raw)
            except Exception as e:  # noqa: BLE001
                judge = {"pass": None, "failure_tags": [], "reason": f"裁判调用失败：{e}", "notable": ""}

        flags = rule_checks(case, output)
        passed_flag = judge.get("pass")
        # 硬违规（结构化、可判定）→ 规则层直接否决
        if flags["R_硬违规"]:
            passed_flag = False
            if "F3" not in judge["failure_tags"]:
                judge["failure_tags"].append("F3")
        # 软违规（能力/态度类词，语境敏感）→ 不单独否决，仅在裁判也判失败时作为佐证
        elif flags["R_越界词"] and passed_flag is False and "F3" not in judge["failure_tags"]:
            judge["failure_tags"].append("F3")
        if passed_flag is None:
            passed_flag = False
        if passed_flag:
            passed += 1

        records.append({
            "id": case["id"],
            "topic": case.get("topic", ""),
            "ambiguity": case.get("ambiguity", "none"),
            "pass": passed_flag,
            "judge_pass": judge.get("pass"),
            "failure_tags": judge["failure_tags"],
            "judge_reason": judge["reason"],
            "judge_notable": judge.get("notable", ""),
            "rule_flags": flags,
            "latency_s": latency,
            "tokens": {"prompt": pt, "completion": ct, "total": pt + ct},
            "tool_calls": tool_names,
            "guardrail": guard,
            "output": output,
        })

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(records)
    rate = passed / total * 100 if total else 0.0
    tag_count = {}
    for r in records:
        for t in r["failure_tags"]:
            tag_count[t] = tag_count.get(t, 0) + 1

    # ---- 性能与成本汇总 ----
    lat_sorted = sorted(r.get("latency_s", 0.0) for r in records if "error" not in r)
    tot_lat = round(sum(lat_sorted), 2)
    avg_lat = round(tot_lat / len(lat_sorted), 2) if lat_sorted else 0.0
    p50 = round(pct(lat_sorted, 50), 2)
    p95 = round(pct(lat_sorted, 95), 2)
    tok_in = sum(r.get("tokens", {}).get("prompt", 0) for r in records)
    tok_out = sum(r.get("tokens", {}).get("completion", 0) for r in records)
    price = match_price(args.model)
    cost = None
    if price:
        cost = tok_in / 1_000_000 * price["in"] + tok_out / 1_000_000 * price["out"]
        cost = round(cost, 4)
    agent_calls = sum(1 for r in records for _ in r.get("tool_calls", []))

    # 与上一次运行对比
    prev = None
    all_runs = sorted([p for p in Path(args.out).iterdir() if p.is_dir()])
    earlier = [p for p in all_runs if p.name < run_dir.name and (p / "results.jsonl").exists()]
    if earlier:
        prev_dir = earlier[-1]
        prev_records = load_cases(prev_dir / "results.jsonl")
        prev_pass = sum(1 for r in prev_records if r.get("pass"))
        prev_lat = sorted(r.get("latency_s", 0.0) for r in prev_records if "error" not in r)
        prev = {
            "dir": prev_dir.name,
            "pass": prev_pass,
            "total": len(prev_records),
            "rate": prev_pass / len(prev_records) * 100 if prev_records else 0.0,
            "avg_lat": round(sum(prev_lat) / len(prev_lat), 2) if prev_lat else 0.0,
            "tag": json.loads((prev_dir / "meta.json").read_text(encoding="utf-8")).get("标记", ""),
            "mode": json.loads((prev_dir / "meta.json").read_text(encoding="utf-8")).get("运行模式", ""),
        }

    L = []
    L.append(f"# ChemAI 评测报告｜{args.tag or 'run'}｜{stamp}")
    L.append("")
    L.append(f"- 模型：`{args.model}`｜温度：{args.temperature}｜模式：**{meta['运行模式']}**")
    L.append(f"- prompt：`{prompt_path}`（指纹 {meta['prompt_sha256']}）")
    L.append(f"- 样例数：{total}｜裁判：{meta['裁判']}")
    L.append("")
    L.append(f"## 总体结果：通过 {passed}/{total}（裁判与规则综合通过率 {rate:.1f}%）")
    L.append("")
    if prev:
        delta = rate - prev["rate"]
        direction = "上升" if delta >= 0 else "下降"
        L.append(f"对比上一次运行（`{prev['dir']}`，标记 {prev['tag'] or '无'}，{prev['mode']}）："
                 f"{prev['rate']:.1f}% → {rate:.1f}%，{direction} {abs(delta):.1f} 个百分点。")
        L.append("")
    L.append("## 性能与成本（本机实测）")
    L.append("")
    L.append(f"- 模型调用累计耗时（不含裁判与完整前端链路）：平均 **{avg_lat}s**／p50 {p50}s／p95 {p95}s，本轮 {total} 条合计 {tot_lat}s")
    L.append(f"- Agent 工具调用次数：{agent_calls} 次")
    L.append(f"- 被测端 token：输入 {tok_in}／输出 {tok_out}（合计 {tok_in + tok_out}）")
    if cost is not None:
        L.append(f"- 估算成本：**¥{cost}**（本轮 {total} 条；按「{price['src']}」估算，非厂商账单）")
    else:
        L.append(f"- 估算成本：未定价（`{args.model}` 不在 PRICE_TABLE，请在脚本中补充价目）")
    L.append("")
    L.append("## 失败模式分布")
    L.append("")
    L.append("| 失败模式 | 命中条数 | 定义 |")
    L.append("|---|---|---|")
    for k, v in FAILURE_TAXONOMY.items():
        L.append(f"| {k} | {tag_count.get(k, 0)} | {v} |")
    L.append("")
    L.append("## 逐条明细（Trace 索引，原始输出见 results.jsonl）")
    L.append("")
    if args.agent:
        L.append("| 编号 | 知识点 | 歧义 | 判定 | 失败模式 | 工具调用 | 延迟(s) | 裁判依据 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in records:
            verdict = "通过" if r["pass"] else "未通过"
            tags = "、".join(r["failure_tags"]) or "—"
            tools = "、".join(r.get("tool_calls", [])) or "—"
            reason = (r.get("judge_reason") or "").replace("|", "/")[:50]
            L.append(f"| {r['id']} | {r.get('topic', '')} | {r.get('ambiguity', 'none')} | {verdict} | "
                     f"{tags} | {tools} | {r.get('latency_s', 0)} | {reason} |")
    else:
        L.append("| 编号 | 知识点 | 歧义 | 判定 | 命中的失败模式 | 裁判依据 |")
        L.append("|---|---|---|---|---|---|")
        for r in records:
            verdict = "通过" if r["pass"] else "未通过"
            tags = "、".join(r["failure_tags"]) or "—"
            reason = (r.get("judge_reason") or "").replace("|", "/")[:60]
            L.append(f"| {r['id']} | {r.get('topic', '')} | {r.get('ambiguity', 'none')} | {verdict} | {tags} | {reason} |")
    L.append("")
    L.append("## 规则层客观信号")
    L.append("")
    L.append("| 编号 | 硬违规(否决) | 软提示(待复核) | 有引用 | 信息不足时追问 | 结构缺失小节 |")
    L.append("|---|---|---|---|---|---|")
    for r in records:
        f = r.get("rule_flags") or {}
        hard = "、".join(f.get("R_硬违规", [])) or "—"
        soft = "、".join(f.get("R_越界词", [])) or "—"
        ask = f.get("R_信息不足时追问")
        ask_txt = "—" if ask is None else ("是" if ask else "否")
        miss = "、".join(f.get("R_结构缺失", [])) or "—"
        L.append(f"| {r['id']} | {hard} | {soft} | {'是' if f.get('R_有引用') else '否'} | {ask_txt} | {miss} |")
    L.append("")
    L.append("## 下一步")
    L.append("")
    L.append("1. 打开 `results.jsonl`，挑 2—3 条未通过的样例，逐条看原始输出，判断是「模型能力不足」还是「prompt 约束不到位」。")
    L.append("2. 只改一类问题，复制 `prompts/system_v2.md` 为 `system_v3.md`，重跑并对比通过率变化。")
    L.append("3. 记录每次调整、指标变化及原因，便于复核迭代过程。")
    L.append("4. 不要让报告里出现未跑过的数字。所有指标都来自本次真实调用。")
    L.append("")

    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    ep("")
    ep(f"[完成] 通过 {passed}/{total}（{rate:.1f}%）｜平均延迟 {avg_lat}s｜"
       f"{'成本 ¥' + str(cost) if cost is not None else '成本未定价'}")
    for k, v in tag_count.items():
        ep(f"    {k}: {v} 条")
    ep(f"[完成] 报告：{run_dir / 'report.md'}")
    ep(f"[完成] 明细：{run_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
