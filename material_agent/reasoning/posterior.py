from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..constants import (
    PHYSGM_MATERIALS,
    clamp_physical_values,
    default_E_for_material,
    default_nu_for_material,
    density_for_material,
    normalize_material,
    role_material_candidates,
)
from ..schemas import DistributionOutput, PartEvidence, PartPosterior, SceneEvidence


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in scores.values())
    if total <= 1e-12:
        return {"Plastic": 1.0}
    return {normalize_material(k): max(0.0, float(v)) / total for k, v in scores.items() if max(0.0, float(v)) > 0}


def _weighted_mean(values: list[float], weights: list[float], fallback: float) -> float:
    if not values:
        return fallback
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return fallback
    return float(np.average(v[valid], weights=w[valid]))


def _weighted_std(values: list[float], weights: list[float], fallback: float) -> float:
    if len(values) < 2:
        return fallback
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if valid.sum() < 2:
        return fallback
    mean = float(np.average(v[valid], weights=w[valid]))
    return max(fallback, float(np.sqrt(np.average((v[valid] - mean) ** 2, weights=w[valid]))))


class MaterialPosteriorBuilder:
    def __init__(
        self,
        schema_weight: float = 0.25,
        physgm_weight: float = 0.35,
        vlm_weight: float = 0.30,
        role_weight: float = 0.10,
        whole_weight: float = 0.05,
        memory_weight: float = 0.10,
    ):
        self.schema_weight = float(schema_weight)
        self.physgm_weight = float(physgm_weight)
        self.vlm_weight = float(vlm_weight)
        self.role_weight = float(role_weight)
        self.whole_weight = float(whole_weight)
        self.memory_weight = float(memory_weight)

    def build(
        self,
        scene: SceneEvidence,
        distributions: dict[int, list[DistributionOutput]],
        memory_priors: dict[int, list[dict]] | None = None,
        vlm_priors: dict[int, list[dict]] | None = None,
    ) -> dict[int, PartPosterior]:
        memory_priors = memory_priors or {}
        vlm_priors = vlm_priors or {}
        out: dict[int, PartPosterior] = {}
        for part in scene.parts:
            out[part.part_id] = self._build_part(
                scene,
                part,
                distributions.get(part.part_id, []),
                memory_priors.get(part.part_id, []),
                vlm_priors.get(part.part_id, []),
            )
        return out

    def _build_part(
        self,
        scene: SceneEvidence,
        part: PartEvidence,
        dist_outputs: list[DistributionOutput],
        memory: list[dict],
        vlm_priors: list[dict],
    ) -> PartPosterior:
        material_scores: dict[str, float] = defaultdict(float)
        evidence: list[dict] = []
        warnings: list[str] = []

        for material in part.expected_materials:
            material_scores[normalize_material(material)] += self.schema_weight / max(1, len(part.expected_materials))
            evidence.append({"source": "schema_expected_material", "material": normalize_material(material)})

        for material, conf, reason in role_material_candidates(part.name, part.physical_role):
            material_scores[material] += self.role_weight * conf
            evidence.append({"source": "role_prior", "material": material, "confidence": conf, "reason": reason})

        whole_material = normalize_material(scene.whole_physics.get("material"))
        if whole_material:
            material_scores[whole_material] += self.whole_weight
            evidence.append({"source": "whole_physgm", "material": whole_material})

        for item in dist_outputs:
            for material, prob in item.material_probs.items():
                material_scores[normalize_material(material)] += self.physgm_weight * float(prob) / max(1, len(dist_outputs))
            evidence.append({"source": "physgm_distribution", "variant": item.variant, "material": item.material})

        for item in vlm_priors:
            probs = item.get("material_probs") or {}
            if probs:
                for material, prob in probs.items():
                    material_scores[normalize_material(material)] += self.vlm_weight * float(prob) / max(1, len(vlm_priors))
            else:
                material = normalize_material(item.get("material"))
                material_scores[material] += self.vlm_weight * float(item.get("confidence", 0.5)) / max(1, len(vlm_priors))
            evidence.append(
                {
                    "source": "vlm_material_prior",
                    "material": normalize_material(item.get("material")),
                    "confidence": item.get("confidence"),
                    "reason": item.get("reason", ""),
                }
            )

        for item in memory:
            material = normalize_material(item.get("material") or item.get("visual_material"))
            score = float(item.get("score", 0.5))
            material_scores[material] += self.memory_weight * score
            evidence.append({"source": "skill_memory", "material": material, "score": score})

        if not material_scores:
            fallback = "Plastic"
            material_scores[fallback] = 1.0
            warnings.append("No material evidence; used Plastic fallback.")

        material_probs = _normalize_scores(dict(material_scores))
        selected_material = max(material_probs, key=material_probs.get)
        material_conf = float(material_probs[selected_material])

        logE_values: list[float] = []
        nu_values: list[float] = []
        weights: list[float] = []
        sigma_terms: list[float] = []
        for item in dist_outputs:
            if item.E_mean > 0 and math.isfinite(item.E_mean):
                logE_values.append(math.log10(item.E_mean))
                nu_values.append(float(item.nu_mean))
                weights.append(max(0.05, item.material_probs.get(selected_material, 0.1)))
                sigma_terms.append(float(item.E_sigma_log10))

        fallback_E = default_E_for_material(selected_material)
        fallback_nu = default_nu_for_material(selected_material)
        logE_mean = _weighted_mean(logE_values, weights, math.log10(fallback_E))
        nu_mean = _weighted_mean(nu_values, weights, fallback_nu)
        crop_std = _weighted_std(logE_values, weights, 0.25)
        model_std = float(np.mean(sigma_terms)) if sigma_terms else 0.35
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in material_probs.values())
        entropy_scale = min(0.5, entropy / max(1.0, math.log(len(PHYSGM_MATERIALS))))
        logE_std = max(0.10, 0.50 * model_std + 0.35 * crop_std + 0.15 * entropy_scale)
        nu_std_values = [x.nu_sigma for x in dist_outputs if math.isfinite(x.nu_sigma)]
        nu_std = max(0.02, float(np.mean(nu_std_values)) if nu_std_values else 0.06)

        E_value, nu_value, clamp_warnings = clamp_physical_values(selected_material, 10 ** logE_mean, nu_mean)
        warnings.extend(clamp_warnings)
        logE_mean = math.log10(E_value)
        nu_mean = nu_value
        consistency = 1.0 / (1.0 + logE_std)
        confidence = max(0.0, min(1.0, material_conf * consistency * max(0.2, float(part.confidence or 0.7))))

        return PartPosterior(
            part_id=part.part_id,
            part_name=part.name,
            material_probs=material_probs,
            selected_material=selected_material,
            material_confidence=material_conf,
            logE_mean=logE_mean,
            logE_std=logE_std,
            nu_mean=nu_mean,
            nu_std=nu_std,
            density=density_for_material(selected_material),
            confidence=confidence,
            evidence=evidence,
            warnings=warnings,
        )
