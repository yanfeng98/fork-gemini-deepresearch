# 前端

## 1. 安装依赖

```bash
node -v
npm -v
```

## 创建 React + Vite 项目

```bash
# 选 React + JavaScript
npm create vite@latest vite-frontend
```

进入项目并安装依赖

```bash
cd my-react-app
npm install
```

启动开发服务器

```bash
npm run dev
```

## 调用 LangGraph API

### 1️⃣ 配置 Vite 代理

`vite.config.js`

```js
export default {
  server: {
    proxy: {
      "/api": "http://localhost:3001",
    },
  },
};
``

## 构建生产环境版本

```bash
npm run build
```

生成的文件在：

```bash
dist/
```

本地预览构建结果：

```bash
npm run preview
```

# 后端

## 1️⃣ 创建 Python 虚拟环境

```bash
mkdir langgraph-backend
cd langgraph-backend
python -m venv .venv
source .venv/bin/activate
```

## 2️⃣ 安装依赖

```bash
pip install fastapi uvicorn langgraph langchain langchain-openai
```

## 3️⃣ 启动后端

```bash
uvicorn main:app --reload --port 3001
```

## 4️⃣ 测试后端

```bash
curl -X POST http://localhost:3001/api/chat-stream \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，LangGraph"}'
```

# 🎨 前端 UI 升级（接近 ChatGPT）

## 1️⃣ 安装依赖

```bash
cd vite-frontend
npm install react-markdown remark-gfm rehype-highlight highlight.js
```