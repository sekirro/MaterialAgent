from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from material_agent.loaders.partphys_outputs import PartPhysSceneLoader
from material_agent.reasoning.candidate_sampler import CandidateSetSampler
from material_agent.reasoning.posterior import MaterialPosteriorBuilder


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_scene(tmp_path: Path) -> Path:
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(scene / "input" / "input.png")
    Image.new("RGB", (32, 32), "white").save(scene / "input" / "object_isolated_full.png")
    _write_json(
        scene / "schema" / "part_schema.json",
        {
            "object": "hammer",
            "parts": [
                {"name": "head", "expected_materials": ["Metal"], "physical_role": "impact part"},
                {"name": "handle", "expected_materials": ["Wood"], "physical_role": "grip"},
            ],
        },
    )
    parts = []
    for idx, name in enumerate(["head", "handle"]):
        part_dir = scene / "parts" / f"part_{idx:03d}_{name}"
        part_dir.mkdir(parents=True)
        Image.new("L", (32, 32), 255).save(part_dir / "mask.png")
        part = {
            "part_id": idx,
            "name": name,
            "mask_path": str(part_dir / "mask.png"),
            "area": 100,
            "confidence": 0.9,
            "expected_materials": ["Metal" if name == "head" else "Wood"],
            "physics_group": name,
        }
        _write_json(part_dir / "part_summary.json", {"part": part})
        parts.append(part)
    _write_json(scene / "partphys_summary.json", {"object_name": "hammer", "parts": parts})
    _write_json(scene / "physgm_whole" / "predicted_phys.json", {"material": "Metal", "E": 1e6, "nu": 0.3})
    (scene / "physgm_whole").mkdir(exist_ok=True)
    (scene / "physgm_whole" / "point_clouds.ply").write_text("ply\n", encoding="utf-8")
    _write_json(
        scene / "assignment" / "per_part_aabb.json",
        [
            {"part_id": 0, "center": [1, 1, 1], "half_size": [0.1, 0.1, 0.1], "count": 50},
            {"part_id": 1, "center": [1, 1, 1], "half_size": [0.1, 0.1, 0.1], "count": 50},
        ],
    )
    _write_json(scene / "assignment" / "assignment_summary.json", {"per_part_counts": {"0": 50, "1": 50}})
    return scene


def test_loader_and_candidate_sampler(tmp_path):
    scene = PartPhysSceneLoader(_make_scene(tmp_path)).load()
    assert scene.object_name == "hammer"
    assert len(scene.parts) == 2
    posteriors = MaterialPosteriorBuilder().build(scene, {}, {})
    candidates = CandidateSetSampler(budget=3).sample(scene, posteriors)
    assert candidates
    head = candidates[0].parts[0]
    assert head.visual_material == "Metal"

