from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..io_utils import ensure_dir
from ..schemas import PartEvidence, SceneEvidence


def _read_mask(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image) > 127


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand(box: tuple[int, int, int, int], w: int, h: int, ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * ratio))
    py = int(round(bh * ratio))
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def _square_pad(image: Image.Image, size: int = 512) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def build_part_crops(scene: SceneEvidence, part: PartEvidence, output_dir: str | Path) -> dict[str, str]:
    if not scene.input_image:
        raise FileNotFoundError("Scene has no input image for crop generation.")
    if not part.mask_path:
        raise FileNotFoundError(f"Part {part.part_id} has no mask.")
    output = ensure_dir(output_dir)
    image = Image.open(scene.input_image).convert("RGB")
    arr = np.asarray(image)
    mask = _read_mask(part.mask_path)
    if mask.shape[:2] != arr.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)) > 127
    h, w = mask.shape
    box = _bbox(mask)
    padded_box = _expand(box, w, h, 0.30)

    white = np.full_like(arr, 255)
    white[mask] = arr[mask]
    part_only = Image.fromarray(white)

    dim = arr.copy()
    dim[~mask] = (dim[~mask].astype(np.float32) * 0.25 + 255.0 * 0.75).astype(np.uint8)
    context = Image.fromarray(dim)

    crops = {
        "tight": part_only.crop(box),
        "padded": part_only.crop(padded_box),
        "context_dim": context.crop(padded_box),
        "isolated_full": part_only,
    }
    paths: dict[str, str] = {}
    for name, crop in crops.items():
        path = output / f"crop_{name}.png"
        _square_pad(crop).save(path)
        paths[name] = str(path)
    return paths

