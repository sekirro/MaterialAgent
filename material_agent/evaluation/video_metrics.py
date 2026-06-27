from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .critic import is_cohesive_deformable_material
from ..schemas import CandidateSet


BAD_STDERR_TOKENS = ["nan", "inf", "cuda error", "traceback", "runtimeerror"]


class VideoEvaluator:
    def evaluate(self, candidate: CandidateSet, run_result: dict) -> dict:
        score = 0.0
        reasons: list[str] = []
        failed = False
        if run_result.get("returncode", 1) != 0:
            failed = True
            reasons.append("simulation returned non-zero")
        stderr_path = run_result.get("stderr")
        stderr = ""
        if stderr_path and Path(stderr_path).exists():
            stderr = Path(stderr_path).read_text(encoding="utf-8", errors="ignore").lower()
        if any(token in stderr for token in BAD_STDERR_TOKENS):
            failed = True
            reasons.append("stderr contains numerical/runtime error token")
        video_path = run_result.get("video_path")
        if run_result.get("status") != "mock" and (not video_path or not Path(video_path).exists()):
            reasons.append("video missing")
            score -= 0.15
        elif video_path and Path(video_path).exists():
            size = Path(video_path).stat().st_size
            if size < 4096:
                reasons.append("video file is too small")
                score -= 0.2
            else:
                score += 0.2
                reasons.append("video exists")
        frame_metrics = self._frame_metrics(run_result.get("output_path"))
        score += self._score_frame_metrics(frame_metrics, reasons)
        if failed:
            return {
                "candidate_id": candidate.candidate_id,
                "ok": False,
                "score": -1.0,
                "reasons": reasons,
                "frame_metrics": frame_metrics,
            }

        avg_conf = sum(part.confidence for part in candidate.parts) / max(1, len(candidate.parts))
        diversity = len({p.visual_material for p in candidate.parts}) / max(1, len(candidate.parts))
        stiffness_order = self._stiffness_role_score(candidate)
        density_score = self._density_score(candidate)
        solver_response = self._solver_response_score(candidate)
        projection_penalty = self._projection_constraint_penalty(candidate)
        score += 0.27 * avg_conf + 0.10 * diversity + 0.25 * stiffness_order + 0.15 * candidate.score_prior + 0.08 * density_score + 0.15 * solver_response
        score -= projection_penalty
        reasons.append(f"avg material confidence {avg_conf:.3f}")
        reasons.append(f"material diversity {diversity:.3f}")
        reasons.append(f"role stiffness score {stiffness_order:.3f}")
        reasons.append(f"density stability score {density_score:.3f}")
        reasons.append(f"solver response score {solver_response:.3f}")
        if projection_penalty > 0.0:
            reasons.append(f"projection constraint penalty -{projection_penalty:.3f}")
        return {
            "candidate_id": candidate.candidate_id,
            "ok": True,
            "score": float(score),
            "reasons": reasons,
            "frame_metrics": frame_metrics,
        }

    def _stiffness_role_score(self, candidate: CandidateSet) -> float:
        if not candidate.parts:
            return 0.5
        score = 0.0
        for part in candidate.parts:
            name = part.part_name.lower()
            if any(k in name for k in ("head", "blade", "tip", "plate", "support", "stand", "tray", "dish", "shell", "frame")):
                score += 1.0 if part.raw_E >= 1e6 and part.solver_material == "metal" else 0.35
            elif any(k in name for k in ("rubber", "sole", "tire", "wheel", "foam", "cushion")):
                score += 1.0 if part.raw_E <= 1e8 else 0.5
            else:
                score += 0.7
        return score / max(1, len(candidate.parts))

    def _projection_constraint_penalty(self, candidate: CandidateSet) -> float:
        rigid_count = sum(1 for part in candidate.parts if bool(getattr(part, "rigid_project", False)))
        bond_count = sum(1 for part in candidate.parts if bool(getattr(part, "interface_bond", False)))
        return min(0.12, 0.025 * rigid_count + 0.035 * bond_count)

    def _density_score(self, candidate: CandidateSet) -> float:
        if not candidate.parts:
            return 0.5
        values = []
        for part in candidate.parts:
            density = float(part.simulation_density)
            if density < 250.0:
                values.append(0.35)
            elif density > 5000.0:
                values.append(0.55)
            else:
                values.append(1.0)
        return float(sum(values) / len(values))


    def _solver_response_score(self, candidate: CandidateSet) -> float:
        cohesive_parts = [part for part in candidate.parts if is_cohesive_deformable_material(part)]
        if not cohesive_parts:
            return 0.7
        values = []
        for part in cohesive_parts:
            visual = part.visual_material.lower()
            solver = part.solver_material.lower()
            name = part.part_name.lower()
            plastic_role = any(token in name for token in ("clay", "putty", "plasticine", "dough"))
            if visual in {"sand", "snow"}:
                values.append(1.0 if solver in {"sand", "snow"} else 0.7)
            elif visual == "plasticine" and plastic_role:
                values.append(1.0 if solver == "plasticine" else 0.7)
            elif solver == "jelly":
                values.append(1.0)
            elif solver in {"foam", "plasticine"}:
                values.append(0.55)
            else:
                values.append(0.7)
        return float(sum(values) / len(values))

    def _frame_metrics(self, output_path: str | None) -> dict[str, Any]:
        if not output_path:
            return {}
        output = Path(output_path)
        if not output.exists():
            return {}
        frame_paths = self._find_frame_paths(output)
        if len(frame_paths) < 2:
            return {}
        first = self._measure_frame(frame_paths[0])
        last = self._measure_frame(frame_paths[-1])
        if not first or not last or first["foreground_area"] <= 0 or last["foreground_area"] <= 0:
            return {"frame_count": len(frame_paths), "valid": False}
        width_growth = last["bbox_width"] / max(1.0, first["bbox_width"])
        height_ratio = last["bbox_height"] / max(1.0, first["bbox_height"])
        bbox_area_growth = last["bbox_area"] / max(1.0, first["bbox_area"])
        foreground_area_growth = last["foreground_area"] / max(1.0, first["foreground_area"])
        centroid_drop_ratio = (last["centroid_y"] - first["centroid_y"]) / max(1.0, first["image_height"])
        return {
            "valid": True,
            "frame_count": len(frame_paths),
            "first_frame": str(frame_paths[0]),
            "last_frame": str(frame_paths[-1]),
            "bbox_width_growth_ratio": float(width_growth),
            "bbox_height_ratio": float(height_ratio),
            "bbox_area_growth_ratio": float(bbox_area_growth),
            "foreground_area_growth_ratio": float(foreground_area_growth),
            "centroid_drop_ratio": float(centroid_drop_ratio),
            "last_dark_foreground_ratio": float(last["dark_foreground_ratio"]),
            "last_foreground_area_ratio": float(last["foreground_area"] / max(1.0, last["image_width"] * last["image_height"])),
        }

    def _score_frame_metrics(self, metrics: dict[str, Any], reasons: list[str]) -> float:
        if not metrics or not metrics.get("valid"):
            return 0.0
        score = 0.12
        bbox_growth = float(metrics.get("bbox_area_growth_ratio", 1.0))
        width_growth = float(metrics.get("bbox_width_growth_ratio", 1.0))
        height_ratio = float(metrics.get("bbox_height_ratio", 1.0))
        dark_ratio = float(metrics.get("last_dark_foreground_ratio", 0.0))
        if bbox_growth > 2.0:
            score -= 0.25
            reasons.append(f"foreground bbox grew excessively ({bbox_growth:.2f}x)")
        elif bbox_growth > 1.45:
            score -= 0.10
            reasons.append(f"foreground bbox grew moderately ({bbox_growth:.2f}x)")
        else:
            reasons.append(f"foreground bbox growth {bbox_growth:.2f}x")
        if height_ratio < 0.45 and width_growth > 1.25:
            score -= 0.22
            reasons.append(f"foreground flattened (height {height_ratio:.2f}x, width {width_growth:.2f}x)")
        if dark_ratio > 0.18:
            score -= 0.08
            reasons.append(f"high dark foreground ratio {dark_ratio:.3f}")
        return score

    def _find_frame_paths(self, output: Path) -> list[Path]:
        paths = []
        for suffix in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(output.rglob(suffix))
        filtered = []
        for path in paths:
            lower = path.name.lower()
            if any(token in lower for token in ("overlay", "mask", "contact", "debug")):
                continue
            filtered.append(path)
        return sorted(filtered, key=lambda p: (len(str(p)), str(p)))

    def _measure_frame(self, path: Path) -> dict[str, float] | None:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return None
        rgb = np.asarray(image, dtype=np.float32)
        height, width = rgb.shape[:2]
        corners = np.concatenate(
            [
                rgb[: max(1, height // 20), : max(1, width // 20)].reshape(-1, 3),
                rgb[: max(1, height // 20), -max(1, width // 20) :].reshape(-1, 3),
                rgb[-max(1, height // 20) :, : max(1, width // 20)].reshape(-1, 3),
                rgb[-max(1, height // 20) :, -max(1, width // 20) :].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(corners, axis=0)
        diff = np.linalg.norm(rgb - background.reshape(1, 1, 3), axis=2)
        foreground = diff > 30.0
        if not np.any(foreground):
            return None
        ys, xs = np.nonzero(foreground)
        min_x, max_x = float(xs.min()), float(xs.max() + 1)
        min_y, max_y = float(ys.min()), float(ys.max() + 1)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        dark = foreground & (rgb.mean(axis=2) < 45.0)
        return {
            "image_width": float(width),
            "image_height": float(height),
            "foreground_area": float(foreground.sum()),
            "bbox_width": float(bbox_width),
            "bbox_height": float(bbox_height),
            "bbox_area": float(bbox_width * bbox_height),
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "dark_foreground_ratio": float(dark.sum() / max(1, foreground.sum())),
        }
