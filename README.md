---
title: CLIP Playground
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 52189
---

# Minimal CLIP Web Demo

A minimal web demo where users can:
- Enter text
- Upload an image
- Get CLIP feature vectors for text/image
- See cosine similarity when both are provided

The backend uses `openai/clip-vit-base-patch32` and returns normalized vectors.

## 1) Install Dependencies

```bash
uv sync
```

On first run, model files are downloaded from Hugging Face.

## 2) Run Locally

```bash
uv run python app.py
```

Open:

`http://127.0.0.1:52189`

## 3) Run with Docker

```bash
docker build -t clip-demo:latest .
docker run --rm -p 52189:52189 clip-demo:latest
```

Open:

`http://127.0.0.1:52189`

## 4) Share via GitHub

```bash
git init
git add .
git commit -m "Initial CLIP demo"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Repository link format:

`https://github.com/<your-username>/<your-repo>`

## 5) API Response Shape

- `source`: `text`, `image`, or `both`
- `text`: text vector payload when text input is provided
  - `dimension`: vector length (typically `512`)
  - `vector`: normalized CLIP text embedding
- `image`: image vector payload when image input is provided
  - `dimension`: vector length (typically `512`)
  - `vector`: normalized CLIP image embedding
- `cosine_similarity`: returned only when both text and image are provided

## 6) Runtime Notes

- Device priority: `mps` (Apple Silicon) > `cpu`
- Default port: `52189`
- Override port with environment variable:

```bash
PORT=8000 uv run python app.py
```
