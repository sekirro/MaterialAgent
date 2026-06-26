from __future__ import annotations

import copy
import math
from pathlib import Path

from ..constants import solver_material_for_visual
from ..io_utils import read_json, write_json
from ..schemas import CandidatePartMaterial, CandidateSet, SceneEvidence


DEFAULT_SOLVER_STABILITY = {
    "enabled": True,
    "reference_E": 5.0e6,
    "reference_nu": 0.35,
    "reference_density": 2500.0,
    "safety_factor": 0.70,
    "min_substep_dt": 5.0e-5,
    "max_substeps_per_frame": 800,
}


def _is_composite_regularized_part(mat: CandidatePartMaterial) -> bool:
    if "solver_compatible" in str(mat.source):
        return True
    return any("composite solver regularization" in warning.lower() for warning in mat.warnings)


class SimulationConfigCompiler:
    def __init__(self, template_config: str | Path, backend: str = "auto", solver_stability: dict | None = None):
        self.template_config = Path(template_config).expanduser()
        self.backend = backend
        self.solver_stability = copy.deepcopy(DEFAULT_SOLVER_STABILITY)
        if solver_stability:
            self.solver_stability.update(solver_stability)

    def compile_candidate(self, scene: SceneEvidence, candidate: CandidateSet, output_dir: str | Path) -> dict:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        backend = self.resolve_backend(scene)
        config = self._base_config(candidate)
        config_path = output / f"{candidate.candidate_id}.json"
        materials_path = None
        if backend == "aabb":
            self._add_aabb_params(config, scene, candidate)
        elif backend == "part_id":
            materials_path = output / f"{candidate.candidate_id}_part_materials.json"
            self._write_part_id_materials(materials_path, candidate)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        stability = self._apply_solver_stability(config, candidate)
        write_json(config_path, config)
        return {
            "backend": backend,
            "config_path": str(config_path),
            "part_materials_json": str(materials_path) if materials_path else None,
            "solver_stability": stability,
        }

    def resolve_backend(self, scene: SceneEvidence) -> str:
        if self.backend != "auto":
            return self.backend
        if scene.gaussian_part_ids_path and Path(scene.gaussian_part_ids_path).exists():
            return "part_id"
        return "aabb"

    def _base_config(self, candidate: CandidateSet) -> dict:
        config = read_json(self.template_config, {}) or {}
        config = copy.deepcopy(config)
        config["material"] = candidate.global_material
        config["E"] = float(candidate.global_E)
        config["nu"] = float(candidate.global_nu)
        config["density"] = float(candidate.global_density)
        return config

    def _add_aabb_params(self, config: dict, scene: SceneEvidence, candidate: CandidateSet) -> None:
        aabb_by_id = {p.part_id: p for p in scene.parts}
        additional = []
        metadata = []
        for mat in candidate.parts:
            part = aabb_by_id.get(mat.part_id)
            if not part or not part.aabb_center or not part.aabb_half_size:
                continue
            additional.append(
                {
                    "point": [float(x) for x in part.aabb_center],
                    "size": [float(x) for x in part.aabb_half_size],
                    "E": float(mat.simulation_E),
                    "nu": float(mat.simulation_nu),
                    "density": float(mat.simulation_density),
                }
            )
            metadata.append({**mat.to_dict(), "point": part.aabb_center, "size": part.aabb_half_size})
        config["additional_material_params"] = additional
        config.setdefault("material_agent_metadata", {}).update({"backend": "aabb", "parts": metadata})

    def _write_part_id_materials(self, path: Path, candidate: CandidateSet) -> None:
        fallback = {
            "name": "fallback_global",
            "material": candidate.global_material,
            "E": float(candidate.global_E),
            "nu": float(candidate.global_nu),
            "density": float(candidate.global_density),
        }
        parts = {}
        for mat in candidate.parts:
            parts[str(mat.part_id)] = {
                "name": mat.part_name,
                "material": mat.solver_material,
                "visual_material": mat.visual_material,
                "E": float(mat.simulation_E),
                "nu": float(mat.simulation_nu),
                "density": float(mat.simulation_density),
                "raw_E": float(mat.raw_E),
                "raw_nu": float(mat.raw_nu),
                "raw_density": float(mat.raw_density),
            }
        write_json(path, {"fallback": fallback, "parts": parts})

    def _apply_solver_stability(self, config: dict, candidate: CandidateSet) -> dict:
        stability_config = self.solver_stability
        metadata = {
            "enabled": bool(stability_config.get("enabled", True)),
            "adjusted": False,
        }
        if not metadata["enabled"]:
            self._merge_stability_metadata(config, metadata)
            return metadata

        try:
            frame_dt = float(config["frame_dt"])
            base_substep_dt = float(config["substep_dt"])
        except (KeyError, TypeError, ValueError):
            metadata["reason"] = "missing_frame_or_substep_dt"
            self._merge_stability_metadata(config, metadata)
            return metadata
        if frame_dt <= 0.0 or base_substep_dt <= 0.0:
            metadata["reason"] = "non_positive_time_step"
            self._merge_stability_metadata(config, metadata)
            return metadata

        reference_score = self._elastic_stability_score(
            float(stability_config.get("reference_E", 5.0e6)),
            float(stability_config.get("reference_nu", 0.35)),
            float(stability_config.get("reference_density", 2500.0)),
        )
        max_item = self._max_stability_item(candidate)
        metadata.update(
            {
                "reference_score": reference_score,
                "max_score": max_item["score"],
                "max_part": max_item["part_name"],
                "max_E": max_item["E"],
                "max_nu": max_item["nu"],
                "max_density": max_item["density"],
                "original_substep_dt": base_substep_dt,
                "frame_dt": frame_dt,
            }
        )
        if reference_score <= 0.0 or max_item["score"] <= reference_score:
            metadata["reason"] = "within_reference_stability"
            self._merge_stability_metadata(config, metadata)
            return metadata

        safety = max(0.05, min(1.0, float(stability_config.get("safety_factor", 0.70))))
        target_dt = base_substep_dt * safety * math.sqrt(reference_score / max_item["score"])
        base_steps = max(1, int(math.ceil(frame_dt / base_substep_dt)))
        target_steps = max(base_steps, int(math.ceil(frame_dt / target_dt)))

        min_dt = float(stability_config.get("min_substep_dt", 5.0e-5))
        max_steps_by_min_dt = int(math.floor(frame_dt / min_dt)) if min_dt > 0.0 else target_steps
        configured_max_steps = int(stability_config.get("max_substeps_per_frame", max_steps_by_min_dt or target_steps))
        max_steps = max(base_steps, min(configured_max_steps, max_steps_by_min_dt or configured_max_steps))
        step_per_frame = min(target_steps, max_steps)
        new_substep_dt = frame_dt / float(step_per_frame)

        if new_substep_dt < base_substep_dt:
            config["substep_dt"] = new_substep_dt
            metadata.update(
                {
                    "adjusted": True,
                    "new_substep_dt": new_substep_dt,
                    "step_per_frame": step_per_frame,
                    "target_substep_dt": target_dt,
                    "safety_factor": safety,
                    "reason": "high_elastic_stiffness_requires_smaller_explicit_mpm_dt",
                }
            )
            if target_steps > max_steps:
                metadata["warning"] = "target time step was limited by min_substep_dt/max_substeps_per_frame"
        else:
            metadata["reason"] = "computed_step_not_smaller_than_base"
        self._merge_stability_metadata(config, metadata)
        return metadata

    def _merge_stability_metadata(self, config: dict, metadata: dict) -> None:
        config.setdefault("material_agent_metadata", {})["solver_stability"] = metadata

    def _max_stability_item(self, candidate: CandidateSet) -> dict:
        items = [
            {
                "part_name": "fallback_global",
                "E": float(candidate.global_E),
                "nu": float(candidate.global_nu),
                "density": float(candidate.global_density),
            }
        ]
        for part in candidate.parts:
            items.append(
                {
                    "part_name": part.part_name,
                    "E": float(part.simulation_E),
                    "nu": float(part.simulation_nu),
                    "density": float(part.simulation_density),
                }
            )
        for item in items:
            item["score"] = self._elastic_stability_score(item["E"], item["nu"], item["density"])
        return max(items, key=lambda item: item["score"])

    @staticmethod
    def _elastic_stability_score(E: float, nu: float, density: float) -> float:
        rho = max(float(density), 1.0)
        young = max(float(E), 0.0)
        poisson = min(max(float(nu), -0.95), 0.49)
        mu = young / (2.0 * (1.0 + poisson))
        denom = max((1.0 + poisson) * (1.0 - 2.0 * poisson), 1.0e-6)
        lam = young * poisson / denom
        return (lam + 2.0 * mu) / rho
