import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (name) => readFile(path.join(root, name), "utf8");
const [template, html, prompt, knowledge, cases, manifest] = await Promise.all([
  read("worker/runtime.template.js"),
  read("app/index.html"),
  read("app/prompts/system_v2.md"),
  read("app/agent/knowledge.jsonl"),
  read("app/data/cases.jsonl"),
  read(".openai/hosting.json"),
]);
const values = {
  __HTML__: JSON.stringify(html),
  __PROMPT__: JSON.stringify(prompt),
  __KNOWLEDGE__: JSON.stringify(knowledge.trim()),
  __CASES__: JSON.stringify(cases.trim()),
};
let output = template;
for (const [marker, value] of Object.entries(values)) output = output.replaceAll(marker, value);
if (/__[A-Z_]+__/.test(output)) throw new Error("Worker template has unresolved placeholders.");
JSON.parse(manifest);
await mkdir(path.join(root, "worker"), { recursive: true });
await writeFile(path.join(root, "worker/index.js"), output, "utf8");
await mkdir(path.join(root, "dist/server"), { recursive: true });
await mkdir(path.join(root, "dist/.openai"), { recursive: true });
await cp(path.join(root, "worker/index.js"), path.join(root, "dist/server/index.js"));
await cp(path.join(root, ".openai/hosting.json"), path.join(root, "dist/.openai/hosting.json"));
console.log("Built the embedded ChemAI Worker artifact.");
