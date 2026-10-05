# ChemAI 三端原型

Python 标准库实现的教师、学生、家长三种对话模式。需要有效模型 API 配置才能生成回答。

## 运行

使用 Python 3.10 或更新版本，在仓库根目录执行：

```powershell
if (-not (Test-Path app/.env)) { Copy-Item app/.env.example app/.env }
# 编辑 app/.env，填写自己的 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
python app/server.py
```

浏览器访问 <http://127.0.0.1:8787>。无需安装第三方包。模型标识以所用服务实际支持的值为准，配置模板的默认值不代表已核实可用。环境变量优先于 `.env`。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 服务状态、是否有密钥、角色、样例数、额度；不返回密钥 |
| GET | `/api/cases` | 15 条内置样例 |
| POST | `/api/chat` | `{persona, message, session_id?}`；不传会生成会话 ID，响应会返回 `session_id` |
| POST | `/api/reset` | 按会话与可选角色重置记录 |

## 实现与局限

- Agent 按角色组装提示和可用工具；教师专属班级工具对学生/家长角色拒绝调用。
- 关键词知识库检索；学生档案与班级概览均为示例数据。
- 输出检测产生风险标记，**不会阻断或替换原回答**。提示词约束不能保证输出始终合规。
- 每次模型请求会带入同一会话、同一角色最近最多 8 条用户/助手消息（约 4 轮），上下文总长最多 12,000 字符；较早内容会被舍弃或截断。
- 每个会话只保留最近 8 条消息于内存，服务重启即丢；不同角色的对话上下文隔离。
- 单 IP 每小时和全局每日额度由 `RATE_PER_HOUR`、`DAILY_CAP` 配置。
- 角色参数和会话 ID 都不等于登录鉴权；API 使用者应保留响应中的 ID 继续对话。未接入真实学校数据、OCR、确定性方程式校验或完整教学闭环。

文件入口：[服务端](server.py)、[网页](index.html)、[Agent](agent/chemai_agent.py)、[提示词](prompts/system_v2.md)。
