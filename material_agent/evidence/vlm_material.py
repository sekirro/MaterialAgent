from __future__ import annotations

import base64
import json
import math
import os
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..constants import PHYSGM_MATERIALS, normalize_material
from ..io_utils import ensure_dir, write_json
from ..schemas import PartEvidence, SceneEvidence
from .crops import build_part_crops


class VLMPartMaterialPriorExtractor:
    def __init__(
        self,
        provider: str = "none",
        model: str | None = None,
        api_base: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 180,
    ):
        self.provider = provider
        self.model = model or "gpt-4o-mini"
        self.api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = int(timeout)

    def available(self) -> bool:
        return self.provider == "openai_compatible" and bool(os.environ.get(self.api_key_env))

    def extract_for_scene(self, scene: SceneEvidence, output_dir: str | Path) -> dict[int, list[dict[str, Any]]]:
        output = ensure_dir(output_dir)
        priors: dict[int, list[dict[str, Any]]] = {}
        for part in scene.parts:
            part_output = ensure_dir(output / f"part_{part.part_id:03d}_{part.name}")
            try:
                crops = self._existing_crops(part) or build_part_crops(scene, part, part_output / "crops")
                image_path = crops.get("context_dim") or crops.get("padded") or next(iter(crops.values()))
                prior = self.extract_one(scene, part, image_path)
            except Exception as exc:
                prior = {
                    "part_id": part.part_id,
                    "part_name": part.name,
                    "source": "vlm_material_prior",
                    "ok": False,
                    "warning": str(exc),
                    "material_probs": {},
                }
            priors[part.part_id] = [prior]
            write_json(part_output / "vlm_material_prior.json", prior)
        write_json(output / "vlm_material_priors.json", {str(k): v for k, v in priors.items()})
        return priors

    def extract_one(self, scene: SceneEvidence, part: PartEvidence, image_path: str) -> dict[str, Any]:
        if not self.available():
            raise RuntimeError(f"VLM provider unavailable or env var {self.api_key_env} is not set.")
        prompt = self._prompt(scene, part)
        data = self._call_json(prompt, image_path)
        probs = self._normalize_material_probs(data)
        material = normalize_material(data.get("material") or data.get("top_material") or (max(probs, key=probs.get) if probs else None))
        confidence = float(data.get("confidence", probs.get(material, 0.0) if probs else 0.0) or 0.0)
        E_value = self._positive_float(data.get("E") or data.get("youngs_modulus") or data.get("young_modulus"))
        nu_value = self._bounded_float(data.get("nu") or data.get("poisson_ratio"), 0.0, 0.499)
        return {
            "part_id": part.part_id,
            "part_name": part.name,
            "source": "vlm_material_prior",
            "ok": True,
            "image_path": str(image_path),
            "material": material,
            "material_probs": probs or {material: max(0.01, min(1.0, confidence))},
            "confidence": max(0.0, min(1.0, confidence)),
            "E": E_value,
            "nu": nu_value,
            "reason": str(data.get("reason", "")),
            "raw": data,
        }

    def _prompt(self, scene: SceneEvidence, part: PartEvidence) -> str:
        materials = ", ".join(PHYSGM_MATERIALS)
        expected = ", ".join(part.expected_materials) if part.expected_materials else "none"
        return (
            "You are estimating physical material priors for part-aware 3D Gaussian simulation.\n"
            "Look at the highlighted or isolated object part in the image. Return strict JSON only.\n"
            f"Object: {scene.object_name}\n"
            f"Part name: {part.name}\n"
            f"Part physical role: {part.physical_role or part.physics_group}\n"
            f"Existing weak expected materials: {expected}\n"
            f"Allowed material classes: {materials}\n"
            "Choose material ONLY from the allowed 14 PhysGM classes. Do not invent rare material names; map them to the nearest allowed class.\n"
            "Prefer real-world physical material behavior over color. For cake/food, use Foam or Plasticine for soft edible bodies/cream; "
            "use Ceramic/Glass/Plastic for plates or rigid support objects; use Paper for labels/signs when appropriate.\n"
            "Also provide a rough Young's modulus E in Pascal and Poisson ratio nu if visually inferable; otherwise use null.\n"
            "Required schema: {\n"
            "  \"material\": \"Foam\",\n"
            "  \"confidence\": 0.75,\n"
            "  \"material_probs\": {\"Foam\": 0.55, \"Plasticine\": 0.35, \"Plastic\": 0.10},\n"
            "  \"E\": 100000.0,\n"
            "  \"nu\": 0.35,\n"
            "  \"reason\": \"short visual/physical reason\"\n"
            "}"
        )

    def _image_url(self, image_path: str) -> str:
        data = Path(image_path).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _call_json(self, prompt: str, image_path: str) -> dict[str, Any]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"API key env var {self.api_key_env} is not set.")
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": self._image_url(image_path)}},
                        ],
                    }
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return self._extract_json(data["choices"][0]["message"]["content"])
        except (urllib.error.URLError, TimeoutError, socket.timeout, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"VLM material prior call failed: {exc}") from exc


    def _positive_float(self, value: Any) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        if not math.isfinite(out) or out <= 0:
            return None
        return out

    def _bounded_float(self, value: Any, lo: float, hi: float) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return max(float(lo), min(float(hi), out))

    def _normalize_material_probs(self, data: dict[str, Any]) -> dict[str, float]:
        raw = data.get("material_probs") or data.get("materials") or {}
        scores: dict[str, float] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    scores[normalize_material(key)] = scores.get(normalize_material(key), 0.0) + max(0.0, float(value))
                except Exception:
                    continue
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                material = normalize_material(item.get("material") or item.get("name"))
                try:
                    scores[material] = scores.get(material, 0.0) + max(0.0, float(item.get("probability", item.get("score", 0.0))))
                except Exception:
                    continue
        total = sum(scores.values())
        if total <= 1e-12:
            material = normalize_material(data.get("material") or data.get("top_material"))
            conf = max(0.01, min(1.0, float(data.get("confidence", 0.5) or 0.5)))
            return {material: conf}
        return {material: score / total for material, score in scores.items() if score > 0}

    def _existing_crops(self, part: PartEvidence) -> dict[str, str]:
        if not part.part_dir:
            return {}
        root = Path(part.part_dir)
        mapping = {}
        for variant in ["tight", "padded", "context_dim", "isolated_full"]:
            path = root / f"crop_{variant}.png"
            if path.exists():
                mapping[variant] = str(path)
        return mapping
