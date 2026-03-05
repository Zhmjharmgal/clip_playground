from __future__ import annotations

import os
from io import BytesIO
from typing import Dict, List

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

app = Flask(__name__)

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    tensor_a = torch.tensor(vec_a, dtype=torch.float32)
    tensor_b = torch.tensor(vec_b, dtype=torch.float32)
    return torch.nn.functional.cosine_similarity(tensor_a, tensor_b, dim=0).item()


def _to_feature_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    raise TypeError(f"Unsupported feature output type: {type(output)!r}")


@torch.no_grad()
def encode_text(text: str) -> List[float]:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    features = _to_feature_tensor(model.get_text_features(**inputs))
    features = _normalize(features)
    return features[0].detach().cpu().tolist()


@torch.no_grad()
def encode_image(image_bytes: bytes) -> List[float]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    features = _to_feature_tensor(model.get_image_features(**inputs))
    features = _normalize(features)
    return features[0].detach().cpu().tolist()


@app.get("/")
def index():
    return render_template("index.html", model_name=MODEL_NAME, device=DEVICE)


@app.post("/embed")
def embed():
    text = (request.form.get("text") or "").strip()
    image_file = request.files.get("image")

    if not text and not image_file:
        return jsonify({"error": "Please provide text or an image."}), 400

    text_vector: List[float] | None = None
    image_vector: List[float] | None = None

    try:
        if text:
            text_vector = encode_text(text)
        if image_file:
            image_vector = encode_image(image_file.read())
    except Exception as exc:
        return jsonify({"error": f"Failed to encode input: {exc}"}), 400

    source = (
        "both" if text_vector and image_vector else "text" if text_vector else "image"
    )
    cosine_similarity = (
        _cosine_similarity(text_vector, image_vector)
        if text_vector and image_vector
        else None
    )

    payload: Dict[str, object] = {
        "source": source,
        "text": (
            {"dimension": len(text_vector), "vector": text_vector}
            if text_vector
            else None
        ),
        "image": (
            {"dimension": len(image_vector), "vector": image_vector}
            if image_vector
            else None
        ),
        "cosine_similarity": cosine_similarity,
    }
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "52189"))
    debug = os.getenv("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
