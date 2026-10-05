#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChemAI 三端产品 —— 可运行原型（服务端 · 自包含部署包）

把 Agent 包成一个真实可用的产品：
  - 网页界面：教师端 / 学生端 / 家长端 三端切换
  - REST 接口：POST /api/chat（省略 session_id 时会新建会话，并在响应中返回 ID）

只依赖 Python 标准库（http.server），无需安装任何第三方包。
目录自包含：Agent、知识库、样例、prompt 都在本目录内，可直接整包发布。

配置（优先读环境变量，其次读同目录 .env）：
  LLM_API_KEY     必填
  LLM_BASE_URL    默认 https://api.deepseek.com/v1
  LLM_MODEL       默认 deepseek-flash（配置值，需确认服务支持）
  PORT            默认 8787（发布环境由平台注入）
  RATE_PER_HOUR   单 IP 每小时最大请求数，默认 20
  DAILY_CAP       全局每日最大请求数，默认 250

本地启动：
  python app/server.py
然后浏览器打开 http://127.0.0.1:8787
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# 路径：本目录即部署根，所有资源都在其下（自包含）
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))          # 让 `import agent.*` 生效

AGENT_MODULE_DIR = APP_DIR / "agent"
KB_PATH = AGENT_MODULE_DIR / "knowledge.jsonl"
PROMPT_PATH = APP_DIR / "prompts" / "system_v2.md"
CASES_PATH = APP_DIR / "data" / "cases.jsonl"


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取（KEY=VALUE，不覆盖已有环境变量）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv(APP_DIR / ".env")

CONFIG = {
    "api_key": os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
    "base_url": os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    "model": os.environ.get("LLM_MODEL", "deepseek-flash"),
    "port": int(os.environ.get("PORT", "8787")),
    "host": os.environ.get("HOST", "0.0.0.0"),
    "prompt_file": os.environ.get("CHEMAI_PROMPT", str(PROMPT_PATH)),
    "rate_per_hour": int(os.environ.get("RATE_PER_HOUR", "20")),
    "daily_cap": int(os.environ.get("DAILY_CAP", "250")),
    "max_chars": int(os.environ.get("MAX_CHARS", "3000")),
}

PERSONAS = {"teacher": "教师端", "student": "学生端", "parent": "家长端"}

# ---------------------------------------------------------------------------
# 载入 Agent
# ---------------------------------------------------------------------------
from agent.chemai_agent import ChemAIAgent, Session  # noqa: E402

_system_prompt = ""
if Path(CONFIG["prompt_file"]).exists():
    _system_prompt = Path(CONFIG["prompt_file"]).read_text(encoding="utf-8")

AGENT = ChemAIAgent(
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
    model=CONFIG["model"],
    system_prompt=_system_prompt,
    kb_path=KB_PATH,
    temperature=0.2,
)

# session_id -> { persona -> Session }
_SESSIONS: dict[str, dict[str, Session]] = {}
_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# 额度控制：单 IP 每小时限流 + 全局每日总量上限（防止密钥额度被刷）
# ---------------------------------------------------------------------------
_ip_hits: dict[str, list[float]] = {}
_day = {"date": "", "count": 0}


def _client_ip(handler: "Handler") -> str:
    xff = handler.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return handler.client_address[0]


def _reserve_quota(ip: str) -> tuple[bool, str]:
    """检查并预占一个调用额度。返回 (是否允许, 拒绝原因)。"""
    now = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    with _LOCK:
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
        if _day["count"] >= CONFIG["daily_cap"]:
            return False, "演示额度已用完（今日共享额度上限），请明天再来。"
        wins = [t for t in _ip_hits.get(ip, []) if now - t < 3600]
        if len(wins) >= CONFIG["rate_per_hour"]:
            _ip_hits[ip] = wins
            return False, f"同一访客每小时最多 {CONFIG['rate_per_hour']} 次，请稍后再试。"
        wins.append(now)
        _ip_hits[ip] = wins
        _day["count"] += 1
    return True, ""


def _quota_left(ip: str) -> dict:
    now = time.time()
    with _LOCK:
        wins = len([t for t in _ip_hits.get(ip, []) if now - t < 3600])
        return {
            "ip_used_last_hour": wins,
            "ip_limit_per_hour": CONFIG["rate_per_hour"],
            "global_used_today": _day["count"],
            "global_limit_per_day": CONFIG["daily_cap"],
        }


def get_session(session_id: str, persona: str) -> Session:
    with _LOCK:
        bucket = _SESSIONS.setdefault(session_id, {})
        if persona not in bucket:
            bucket[persona] = Session(persona=persona)
        return bucket[persona]


def reset_session(session_id: str, persona: str | None = None) -> None:
    with _LOCK:
        if persona is None:
            _SESSIONS.pop(session_id, None)
        else:
            _SESSIONS.get(session_id, {}).pop(persona, None)


def _load_cases() -> list:
    out = []
    if CASES_PATH.exists():
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    c = json.loads(line)
                    out.append({"id": c.get("id"), "topic": c.get("topic"),
                                "question": c.get("question"),
                                "reference_answer": c.get("reference_answer"),
                                "student_answer": c.get("student_answer"),
                                "ambiguity": c.get("ambiguity", "none")})
                except json.JSONDecodeError:
                    pass
    return out


CASES = _load_cases()


class Handler(BaseHTTPRequestHandler):
    server_version = "ChemAI/1.1"

    # ---------- 工具 ----------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def log_message(self, fmt, *args):  # 精简日志
        sys.stderr.write("[ChemAI] %s\n" % (fmt % args))

    # ---------- 路由 ----------
    def do_OPTIONS(self):  # noqa: N802
        self._send(204, b"", "text/plain")

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            f = APP_DIR / "index.html"
            if not f.exists():
                self._json({"error": "index.html 缺失"}, 500)
                return
            self._send(200, f.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/health":
            # 注意：不返回模型名与 base_url，避免对外暴露具体模型/供应商。
            self._json({
                "ok": True,
                "has_key": bool(CONFIG["api_key"]),
                "personas": PERSONAS,
                "cases": len(CASES),
                "quota": _quota_left(_client_ip(self)),
            })
            return
        if path == "/api/cases":
            self._json({"cases": CASES})
            return
        self._json({"error": "not found", "path": path}, 404)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/chat":
            self._handle_chat()
            return
        if path == "/api/reset":
            body = self._read_body()
            reset_session(body.get("session_id", "default"), body.get("persona"))
            self._json({"ok": True})
            return
        self._json({"error": "not found", "path": path}, 404)

    # ---------- 核心：对话 ----------
    def _handle_chat(self) -> None:
        if not CONFIG["api_key"]:
            self._json({"error": "服务端未配置 LLM_API_KEY，无法调用模型。"}, 500)
            return
        body = self._read_body()
        persona = body.get("persona") or "teacher"
        message = (body.get("message") or "").strip()
        session_id = body.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            self._json({"error": "session_id 必须是字符串"}, 400)
            return
        session_id = session_id or secrets.token_urlsafe(24)
        if len(session_id) > 128:
            self._json({"error": "session_id 过长（上限 128 字符）"}, 400)
            return
        if persona not in PERSONAS:
            self._json({"error": f"未知端别 {persona}"}, 400)
            return
        if not message:
            self._json({"error": "message 不能为空"}, 400)
            return
        if len(message) > CONFIG["max_chars"]:
            self._json({"error": f"输入过长（上限 {CONFIG['max_chars']} 字）。"}, 413)
            return

        allowed, reason = _reserve_quota(_client_ip(self))
        if not allowed:
            self._json({"error": reason, "quota": _quota_left(_client_ip(self))}, 429)
            return

        session = get_session(session_id, persona)
        try:
            out = AGENT.run(message, persona=persona, session=session)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"Agent 运行失败：{type(e).__name__}: {e}"}, 500)
            return

        summary = AGENT.client.summary(CONFIG["model"])
        self._json({
            "persona": persona,
            "persona_label": PERSONAS[persona],
            "session_id": session_id,
            "answer": out["answer"],
            "tool_calls": [c["name"] for c in out["tool_calls"]],
            "steps": out["steps"],
            "guardrail": out["guardrail"],
            "denials": out["session"].get("denials", []),
            "metrics": {
                "calls": summary["calls"],
                "avg_latency_s": summary["avg_latency_s"],
                "cost": summary["cost"],
                "cost_currency": summary["cost_currency"],
            },
            "quota": _quota_left(_client_ip(self)),
        })


def main() -> None:
    print("=" * 62)
    print("ChemAI 三端产品原型")
    print(f"  模型      : {CONFIG['model']}")
    print(f"  base_url  : {CONFIG['base_url']}")
    print(f"  API Key   : {'已配置' if CONFIG['api_key'] else '【缺失】请设置 LLM_API_KEY'}")
    print(f"  system    : {Path(CONFIG['prompt_file']).name}")
    print(f"  样例数    : {len(CASES)}")
    print(f"  额度      : 单 IP {CONFIG['rate_per_hour']}/小时 · 全局 {CONFIG['daily_cap']}/天")
    print("=" * 62)
    if not CONFIG["api_key"]:
        print("[警告] 未设置 LLM_API_KEY，接口会报错。")
    addr = (CONFIG["host"], CONFIG["port"])
    print(f"  监听： http://{addr[0]}:{addr[1]}")
    print("  接口： POST /api/chat   GET /api/health   GET /api/cases")
    print("=" * 62)
    ThreadingHTTPServer(addr, Handler).serve_forever()


if __name__ == "__main__":
    main()
