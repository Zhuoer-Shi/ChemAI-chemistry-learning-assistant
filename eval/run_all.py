#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键评测：跑「基线 v1 → 改进 v2」两轮，并打印对比结论。

用法：
    export LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=...
    python eval/run_all.py

可选参数会原样透传给 run_eval.py，例如：
    python eval/run_all.py --limit 3            # 先花几分钱试水
    python eval/run_all.py --cases eval/cases.jsonl
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(script_args: list) -> int:
    cmd = [PY, str(HERE / "run_eval.py")] + script_args
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(HERE.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default="eval/cases.jsonl")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--no-agent", action="store_true", help="用裸 prompt 模式（默认走 Agent）")
    ap.add_argument("--out", default="eval/runs")
    ap.add_argument("--prompt-v1", default="eval/prompts/system_v1.md")
    ap.add_argument("--prompt-v2", default="eval/prompts/system_v2.md")
    args = ap.parse_args()

    if not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ARK_API_KEY")):
        raise SystemExit("[错误] 未设置 LLM_API_KEY，无法真实跑批。请先 export LLM_API_KEY=...")

    common = ["--cases", args.cases, "--out", args.out]
    if args.limit:
        common += ["--limit", str(args.limit)]
    if args.skip_judge:
        common += ["--skip-judge"]
    if not args.no_agent:
        common += ["--agent"]

    print("=" * 66)
    print("ChemAI 评测：第一轮（基线 v1）")
    print("=" * 66)
    if run(["--prompt", args.prompt_v1, "--tag", "v1"] + common) != 0:
        raise SystemExit("[中断] 第一轮失败。")

    print("\n" + "=" * 66)
    print("ChemAI 评测：第二轮（改进 v2）")
    print("=" * 66)
    if run(["--prompt", args.prompt_v2, "--tag", "v2"] + common) != 0:
        raise SystemExit("[中断] 第二轮失败。")

    print("\n" + "=" * 66)
    print("两轮跑完。最新一份报告已包含 v1 → v2 的对比行与性能/成本数据：")
    runs = sorted([p for p in Path(args.out).iterdir() if p.is_dir()])
    if runs:
        print("   " + str(runs[-1] / "report.md"))
    print("=" * 66)


if __name__ == "__main__":
    main()
