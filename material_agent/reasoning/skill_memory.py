from __future__ import annotations

from pathlib import Path

from ..io_utils import read_yaml, write_yaml
from ..schemas import CandidateSet, PartEvidence, SceneEvidence


class SkillMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.data = read_yaml(self.path, {"entries": []}) or {"entries": []}
        self.data.setdefault("entries", [])

    def retrieve(self, scene: SceneEvidence) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        for part in scene.parts:
            matches = []
            for entry in self.data.get("entries", []):
                key = entry.get("key", {})
                score = 0.0
                if str(key.get("object", "")).lower() == scene.object_name.lower():
                    score += 0.4
                if str(key.get("part_name", "")).lower() == part.name.lower():
                    score += 0.4
                if key.get("physical_role") and str(key.get("physical_role", "")).lower() in part.physical_role.lower():
                    score += 0.2
                if score > 0:
                    item = dict(entry)
                    item["score"] = max(float(item.get("score", 0.5)), score)
                    matches.append(item)
            out[part.part_id] = sorted(matches, key=lambda x: float(x.get("score", 0)), reverse=True)[:5]
        return out

    def update_from_selection(self, scene: SceneEvidence, candidate: CandidateSet, score: float, output_delta_path: str | Path | None = None) -> list[dict]:
        updates = []
        part_by_id = {p.part_id: p for p in scene.parts}
        for mat in candidate.parts:
            part = part_by_id.get(mat.part_id)
            updates.append(
                {
                    "key": {
                        "object": scene.object_name,
                        "part_name": mat.part_name,
                        "physical_role": part.physical_role if part else "",
                    },
                    "material": mat.visual_material,
                    "solver_material": mat.solver_material,
                    "raw_E": mat.raw_E,
                    "raw_nu": mat.raw_nu,
                    "simulation_E": mat.simulation_E,
                    "simulation_nu": mat.simulation_nu,
                    "density": mat.simulation_density,
                    "source_scene": scene.scene_dir,
                    "score": float(score),
                    "outcome": "success",
                }
            )
        if output_delta_path:
            write_yaml(output_delta_path, {"entries": updates})
        self.data.setdefault("entries", []).extend(updates)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(self.path, self.data)
        return updates

