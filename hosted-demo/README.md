# ChemAI 在线演示部署

在线地址：[ChemAI 化学学习助手](https://chemai-study-demo.sleepearly26.chatgpt.site)

此目录将现有三端网页、提示词、样例和知识库打包为一个 Cloudflare Workers 兼容的 Worker，供 Sites 托管。主要 Python 原型仍在项目根目录的 `app/`。

## 能力与边界

- 保留教师、学生、家长三端页面，内置案例、角色提示词、工具调用和最近对话上下文。
- API Key 仅通过托管平台的服务端环境变量 `LLM_API_KEY` 配置；不要写入源码或提交到仓库。
- 会话、每 IP 限流和每日演示额度使用 Worker 实例内存；实例重启或扩缩容时状态可能重置。因此额度是演示保护，不是可靠的计费上限。
- 该演示不连接真实学生或班级数据，请勿输入个人敏感信息。

## 源码构建

`scripts/build-worker.mjs` 从 `app/` 读取页面、提示词和 JSONL 数据，生成 `worker/index.js` 及 `dist/` 部署工件。站点的模型配置由服务端环境变量提供。
