from __future__ import annotations

import copy
from pathlib import Path

from ..constants import solver_material_for_visual
from ..io_utils import read_json, write_json
from ..schemas import CandidateSet, SceneEvidence


class SimulationConfigCompiler:
    def __init__(self, template_config: str | Path, backend: str = "auto"):
        self.template_config = Path(template_config).expanduser()
        self.backend = backend

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
        write_json(config_path, config)
        return {
            "backend": backend,
            "config_path": str(config_path),
            "part_materials_json": str(materials_path) if materials_path else None,
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
        config["material_agent_metadata"] = {"backend": "aabb", "parts": metadata}

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

