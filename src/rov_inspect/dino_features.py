"""DINO-style image embeddings used for the optional novelty backend."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
_INSTALL_HINT = (
    "DINO backend requires torch, transformers and Pillow. "
    "Install them or run with --descriptor-backend classical."
)


@dataclass(frozen=True)
class DinoModel:
    """Loaded DINO model, processor, and device."""

    model: object
    processor: object
    device: str
    model_name: str


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return torch


def choose_device(device: str) -> str:
    """Resolve auto/cpu/mps/cuda device selection."""

    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, mps, cuda")

    torch = _require_torch()
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested --device mps, but PyTorch MPS is not available")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available")
    return device


def load_dino_model(model_name: str = DEFAULT_DINO_MODEL, device: str = "auto") -> DinoModel:
    """Load a Hugging Face DINO-style vision model."""

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    resolved_device = choose_device(device)
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load DINO model '{model_name}'. "
            "Check the model name, Hugging Face access, network connectivity, or local cache."
        ) from exc

    model.to(resolved_device)
    model.eval()
    return DinoModel(model=model, processor=processor, device=resolved_device, model_name=model_name)


def compute_dino_embedding(frame: np.ndarray, dino_model: DinoModel) -> np.ndarray:
    """Embed one OpenCV BGR frame as a normalized 1D vector."""

    torch = _require_torch()
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = dino_model.processor(images=Image.fromarray(rgb), return_tensors="pt")
    inputs = {name: value.to(dino_model.device) for name, value in inputs.items()}

    with torch.no_grad():
        outputs = dino_model.model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if norm > 0.0:
        embedding = embedding / norm
    return embedding
