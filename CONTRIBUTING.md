# 贡献指南

## 开始之前

1. 使用 Python 3.11+ 和 Node.js 20+。
2. 复制 `backend/.env.example` 为 `backend/.env`，填写必要的服务密钥。
3. 按根目录 `README.md` 启动项目。

## 开发约定

- 后端逻辑、API 契约和测试放在 `backend/`。
- 前端组件、状态和测试放在 `frontend/src/`。
- 架构和长期有效的行为说明放在 `docs/`。
- 临时浏览器截图、运行日志、模型和上传文件不应提交到 Git。
- 不要提交 API Key、个人图片或本地环境文件。

## 提交前检查

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider

cd ..\frontend
npm run test:unit -- --run
npm run typecheck
npm run lint
npm run build
```

## Pull Request

请说明变更目的、测试命令和任何配置/迁移影响。涉及用户界面时，附上桌面和移动端截图；涉及行为修复时，补充回归测试。
