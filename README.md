# Minimal CLIP Web Demo

一个最简单的网页 demo：输入一段文本或上传一张图，后端调用 CLIP 模型做编码，并把 feature vector 返回给前端展示。

## 1) 安装依赖

```bash
uv sync
```

首次运行会下载模型（`openai/clip-vit-base-patch32`）。

## 2) 启动

```bash
uv run python app.py
```

浏览器打开：

`http://127.0.0.1:52189`

## 3) Docker 打包（最简单分享）

```bash
docker build -t clip-demo:latest .
docker run --rm -p 52189:52189 clip-demo:latest
```

打开：

`http://127.0.0.1:52189`

## 4) 发布到 GitHub（给别人链接）

```bash
git init
git add .
git commit -m "Initial CLIP demo"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

仓库链接就是：

`https://github.com/<your-username>/<your-repo>`

## 5) 说明

- 模型：`openai/clip-vit-base-patch32`
- 设备优先级：`mps`（Apple Silicon）> `cpu`
- 返回结果包含：
  - `source`: `text`、`image` 或 `both`
  - `text`: 文本向量（有文本输入时）
  - `image`: 图片向量（有图片输入时）
  - `cosine_similarity`: 同时输入文本和图片时返回
