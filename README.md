# 图寻 Agent

图像地理定位实验应用，包含 FastAPI 后端和 Vue 3 前端。系统结合视觉分析、OCR、GeoCLIP、CLIP/FAISS、地理知识库和地图服务，输出带证据的候选位置。

## 仓库结构

```text
.
├── .github/                  # CI、Issue 和 PR 模板
├── backend/                  # FastAPI、LangGraph Agent、工具和测试
├── frontend/                 # Vue 3 + Vite 应用和前端单元测试
├── docs/                     # 架构说明
├── docker-compose.yml        # 容器编排入口
├── setup-windows.*           # Windows 依赖安装
├── start-windows.*           # Windows 启动
└── stop-windows.*            # Windows 停止
```

详细索引见 [docs/README.md](docs/README.md)。

## Windows 原生启动

不需要 WSL、Linux shell 或 Docker。先安装 Python 3.11 x64 和 Node.js 20+ LTS，
然后在 PowerShell 或双击命令文件执行：

```powershell
.\setup-windows.cmd
# 编辑 backend\.env，至少填写 QWEN_API_KEY
.\start-windows.cmd
```

浏览器会打开 `http://127.0.0.1:5173`。服务会在根目录 `.run` 写入日志，停止时执行：

```powershell
.\stop-windows.cmd
```

`setup-windows.ps1 -CoreOnly` 可只安装 API 和前端依赖，缺少的 GeoCLIP、OCR 或
FAISS 会以可见的降级状态运行；默认安装器会逐项尝试这些可选 Windows wheel，
某一个失败不会阻断其它组件。首次启动且 `MODEL_OFFLINE=false` 时，模型会从配置的
模型源下载到 Hugging Face 缓存，下载完成后可将其改为 `true`。

## 本地环境

- Python 3.11+
- Node.js 20+
- 后端依赖：`python -m pip install -r backend/requirements.txt`
- 前端依赖：`cd frontend; npm ci`

模型和 FAISS 数据不会随 Docker 构建上下文复制。启动前请准备模型缓存和
`backend/data/geo_image_db_v2`，或在配置中关闭对应能力；缺少模型/索引时
readiness 会明确报告原因。

## 启动

```powershell
cd backend
uvicorn app.main:app --reload --port 8000

cd frontend
npm run dev
```

## API Key 配置

安装脚本会从 `backend/.env.example` 创建本机配置文件 `backend/.env`。手动配置时可执行：

```powershell
Copy-Item backend\.env.example backend\.env
```

只在 `backend/.env` 中填写真实 Key。该文件已被 `.gitignore` 排除，不要把真实 Key
写入 README、源码、前端环境变量或提交到 Git。项目使用的 Key 如下：

| 配置项 | 是否必需 | 用途与来源 |
| --- | --- | --- |
| `QWEN_API_KEY` | 是 | 阿里云百炼 / DashScope 的 OpenAI 兼容 API Key，用于 Qwen 文本和视觉模型。 |
| `AMAP_SERVER_KEY` | 按需 | 高德 Web 服务 Key，供后端地理编码和 POI 查询使用；启用 `MAP_SERVICE=amap` 或 `MAP_SERVICE_FALLBACK=amap` 时填写。 |
| `AMAP_WEB_KEY` | 按需 | 高德 JS API Key，供浏览器地图使用。它会发送到浏览器，必须在高德控制台限制可用域名和配额，且不能与服务端 Key 共用。 |
| `TENCENT_MAP_KEY` | 按需 | 腾讯位置服务 Key，供后端查询街景元数据使用。 |
| `SERPAPI_API_KEY` | 按需 | SerpAPI Key；仅在 `SEARCH_SERVICE=serpapi` 时填写。 |
| `BING_SEARCH_API_KEY` | 按需 | Bing Web Search Key；仅在 `SEARCH_SERVICE=bing` 时填写。 |

默认 `MAP_SERVICE=nominatim` 不需要地图 Key。旧变量 `AMAP_API_KEY` 仅用于兼容已有
本地配置，新配置应使用 `AMAP_SERVER_KEY`。`GOOGLE_MAPS_API_KEY` 和
`BING_VISUAL_API_KEY` 是预留项，当前版本不读取，保持为空即可。

提交前可用 `git check-ignore backend/.env` 确认本机配置仍被忽略。若 Key 曾经进入
提交历史，仅从当前文件删除是不够的，应立即在服务商控制台撤销并重新生成 Key，
然后清理 Git 历史。

## 评测与诊断

评测集必须与 FAISS 检索索引按图片哈希和来源 ID 隔离，并为每张图片标注
`landmark`、`street`、`nature`、`ocr_strong`、`ocr_weak`、`night`、
`low_resolution` 或 `non_china` 切片。用 JSONL 运行 baseline：

```powershell
cd backend
..\.venv\Scripts\python.exe scripts\evaluate_predictions.py dataset.jsonl --output report.json
```

校准文件使用 `method=isotonic`、`version=1`、`thresholds` 和 `probabilities`
字段；未配置校准文件时，界面只显示“候选得分”，不会伪装成概率。

生产诊断端点为 `/health/live`、`/health/ready` 和 `/metrics`。默认界面不展示
原始步骤 JSON；仅在开发环境设置 `VITE_DIAGNOSTICS=true` 时开启诊断详情。

## 检查

```powershell
cd frontend
npm run typecheck
npm run lint
npm run test:unit -- --run
npm run build
```

后端检查：

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

## 文档与贡献

- [系统架构](docs/architecture.md)
- [贡献指南](CONTRIBUTING.md)
- [MIT License](LICENSE)

提交代码前请运行与 `.github/workflows/ci.yml` 对应的后端和前端检查。Issue 和 Pull Request 请使用仓库提供的模板。
