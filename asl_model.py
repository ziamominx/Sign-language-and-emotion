"""Inference adapter for the published SignBart WLASL-2000 checkpoint.

Architecture and preprocessing follow the SignBart paper and checkpoint layout.
The checkpoint is downloaded separately; no training is required.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import requests
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


REPOSITORY = "tinh2312/SignBart-WLASL-2000"
REVISION = "fb5924360c26a65912dcae40993270eece8406a7"
FILES = {
    "config.json": "0abd65ef4ae4dcd55a9960e54f41b223b75e09656948781e66ca79ac80b45e7b",
    "model.safetensors": "7743a9b842cd299e7fc5e56619fede010e0d6b178aed91c608f00a75a42579c6",
}
MODEL_DIR = Path(__file__).resolve().parent / ".models" / "signbart-wlasl-2000"


def ensure_model_files(directory=MODEL_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = directory / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        temp = path.with_suffix(path.suffix + ".download")
        digest = hashlib.sha256()
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temp.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"Model download checksum failed: {name}")
        temp.replace(path)
    return directory


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.heads = heads
        self.head_width = width // heads

    def forward(self, query, key_value, mask=None):
        batch, target, width = query.shape
        source = key_value.shape[1]
        q = self.q_proj(query).reshape(batch, target, self.heads, self.head_width).transpose(1, 2)
        k = self.k_proj(key_value).reshape(batch, source, self.heads, self.head_width).transpose(1, 2)
        v = self.v_proj(key_value).reshape(batch, source, self.heads, self.head_width).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) * self.head_width**-0.5
        if mask is not None:
            scores = scores + mask
        result = F.softmax(scores, dim=-1) @ v
        return self.out_proj(result.transpose(1, 2).reshape(batch, target, width))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["d_model"]
        self.self_attn = Attention(width, config["encoder_attention_heads"])
        self.self_attn_layer_norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, config["encoder_ffn_dim"])
        self.fc2 = nn.Linear(config["encoder_ffn_dim"], width)
        self.final_layer_norm = nn.LayerNorm(width)

    def forward(self, hidden):
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, hidden))
        return self.final_layer_norm(hidden + self.fc2(F.gelu(self.fc1(hidden))))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["d_model"]
        heads = config["decoder_attention_heads"]
        self.self_attn = Attention(width, heads)
        self.self_attn_layer_norm = nn.LayerNorm(width)
        self.encoder_attn = Attention(width, heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, config["decoder_ffn_dim"])
        self.fc2 = nn.Linear(config["decoder_ffn_dim"], width)
        self.final_layer_norm = nn.LayerNorm(width)

    def forward(self, hidden, encoded, causal_bias):
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, hidden, causal_bias))
        hidden = self.encoder_attn_layer_norm(hidden + self.encoder_attn(hidden, encoded))
        return self.final_layer_norm(hidden + self.fc2(F.gelu(self.fc1(hidden))))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["d_model"]
        self.embed_positions = nn.Embedding(config["max_position_embeddings"] + 2, width)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config["encoder_layers"]))
        self.layernorm_embedding = nn.LayerNorm(width)

    def forward(self, inputs):
        positions = torch.arange(inputs.shape[1], device=inputs.device) + 2
        hidden = self.layernorm_embedding(inputs + self.embed_positions(positions))
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["d_model"]
        self.embed_positions = nn.Embedding(config["max_position_embeddings"] + 2, width)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config["decoder_layers"]))
        self.layernorm_embedding = nn.LayerNorm(width)

    def forward(self, inputs, encoded):
        length = inputs.shape[1]
        positions = torch.arange(length, device=inputs.device) + 2
        hidden = self.layernorm_embedding(inputs + self.embed_positions(positions))
        causal_bias = torch.full((length, length), float("-inf"), device=inputs.device)
        causal_bias = torch.triu(causal_bias, diagonal=1)
        causal_bias = causal_bias + torch.tril(torch.ones_like(causal_bias))
        for layer in self.layers:
            hidden = layer(hidden, encoded, causal_bias)
        return hidden


class SignBart(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.joint_idx = config["joint_idx"]
        width = config["d_model"]
        self.projection = nn.Module()
        self.projection.proj_x1 = nn.Linear(len(self.joint_idx), width)
        self.projection.proj_y1 = nn.Linear(len(self.joint_idx), width)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.classification_head = nn.Module()
        self.classification_head.out_proj = nn.Linear(width, len(config["id2label"]))

    def forward(self, points):
        points = points[:, :, self.joint_idx, :]
        x = self.projection.proj_x1(points[..., 0])
        y = self.projection.proj_y1(points[..., 1])
        encoded = self.encoder(x)
        decoded = self.decoder(y, encoded)
        return self.classification_head.out_proj(decoded[:, -1])


def normalize_keypoints(points):
    """Match SignBart's per-frame body/left-hand/right-hand box normalization."""
    points = np.clip(np.asarray(points, dtype=np.float32), 0, 1).copy()
    if points.ndim != 3 or points.shape[1:] != (75, 2) or not np.isfinite(points).all():
        raise ValueError("Expected a finite sequence of 75 (x, y) landmarks")
    if len(points) > 64:
        points = points[np.linspace(0, len(points) - 1, 64).astype(int)]
    for frame in points:
        for indices in (range(11, 17), range(33, 54), range(54, 75)):
            part = frame[list(indices)]
            minimum = part.min(axis=0)
            maximum = part.max(axis=0)
            width, height = maximum - minimum
            if width > height:
                pad_x = 0.05 * width
                pad_y = pad_x + (width - height) / 2
            else:
                pad_y = 0.05 * height
                pad_x = pad_y + (height - width) / 2
            start = np.maximum(0, minimum - [pad_x, pad_y])
            end = np.minimum(1, maximum + [pad_x, pad_y])
            span = end - start
            for axis in (0, 1):
                if span[axis] > 0:
                    frame[list(indices), axis] = (part[:, axis] - start[axis]) / span[axis]
    return points


class ASLRecognizer:
    def __init__(self, directory=MODEL_DIR):
        directory = ensure_model_files(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.labels = config["id2label"]
        self.model = SignBart(config)
        state = load_file(str(directory / "model.safetensors"), device="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or set(unexpected) != {"encoder.embed_tokens.weight", "decoder.embed_tokens.weight"}:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.model.eval()
        torch.set_num_threads(2)

    def predict(self, points, top_k=3):
        points = normalize_keypoints(points)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(points).unsqueeze(0))
            probabilities = F.softmax(logits, dim=-1)[0]
            scores, indices = torch.topk(probabilities, top_k)
        return [{"label": self.labels[str(index.item())], "score": round(score.item(), 4)}
                for score, index in zip(scores, indices)]
