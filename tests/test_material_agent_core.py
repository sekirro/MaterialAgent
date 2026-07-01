from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from material_agent.agent import MaterialAgentController
from material_agent.evaluation.critic import MaterialCritic
from material_agent.loaders.partphys_outputs import PartPhysSceneLoader
from material_agent.reasoning.candidate_sampler import CandidateSetSampler
from material_agent.reasoning.posterior import MaterialPosteriorBuilder
from material_agent.simulation.config_compiler import SimulationConfigCompiler
from material_agent.simulation.runner import SimulationRunner
from material_agent.schemas import CandidatePartMaterial, CandidateSet, PartEvidence, PartPosterior, SceneEvidence


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



def test_alternative_candidate_does_not_jump_to_unsupported_high_stiffness():
    scene = SceneEvidence(
        scene_dir="/tmp/scene",
        parts=[PartEvidence(part_id=0, name="body", physics_group="body", confidence=0.8, gaussian_count=100)],
    )
    posterior = PartPosterior(
        part_id=0,
        part_name="body",
        material_probs={"Plasticine": 0.57, "Plastic": 0.41},
        selected_material="Plasticine",
        material_confidence=0.57,
        logE_mean=5.7,
        logE_std=0.2,
        nu_mean=0.47,
        nu_std=0.04,
        density=2000.0,
        confidence=0.25,
    )

    candidates = CandidateSetSampler(budget=8).sample(scene, {0: posterior})

    assert all(candidate.candidate_id != "alternative_material" for candidate in candidates)


def test_target_impact_sampler_generates_soft_target_response():
    scene = SceneEvidence(
        scene_dir="/tmp/scene",
        object_name="cake",
        parts=[
            PartEvidence(part_id=0, name="icing", physics_group="icing", confidence=0.8, gaussian_count=100),
            PartEvidence(part_id=1, name="cake_body", physics_group="body", confidence=0.8, gaussian_count=100),
            PartEvidence(part_id=2, name="unknown_body", physics_group="global_body", confidence=0.4, gaussian_count=100),
        ],
    )
    posteriors = {
        0: PartPosterior(
            part_id=0,
            part_name="icing",
            material_probs={"Plasticine": 0.7, "Foam": 0.3},
            selected_material="Plasticine",
            material_confidence=0.7,
            logE_mean=5.7,
            logE_std=0.2,
            nu_mean=0.47,
            nu_std=0.04,
            density=2000.0,
            confidence=0.7,
        ),
        1: PartPosterior(
            part_id=1,
            part_name="cake_body",
            material_probs={"Foam": 0.7, "Plasticine": 0.3},
            selected_material="Foam",
            material_confidence=0.7,
            logE_mean=5.2,
            logE_std=0.2,
            nu_mean=0.30,
            nu_std=0.04,
            density=300.0,
            confidence=0.7,
        ),
        2: PartPosterior(
            part_id=2,
            part_name="unknown_body",
            material_probs={"Plasticine": 0.7, "Plastic": 0.3},
            selected_material="Plasticine",
            material_confidence=0.7,
            logE_mean=5.7,
            logE_std=0.2,
            nu_mean=0.47,
            nu_std=0.04,
            density=2000.0,
            confidence=0.4,
        ),
    }

    candidates = CandidateSetSampler(budget=4, interaction_role="target", interaction_intent="impact").sample(scene, posteriors)
    target = next(candidate for candidate in candidates if candidate.candidate_id == "target_impact_soft_response")

    assert target.score_prior >= 0.82
    semantic = [part for part in target.parts if part.part_name != "unknown_body"]
    residual = next(part for part in target.parts if part.part_name == "unknown_body")
    assert all(part.simulation_E <= 2.5e5 for part in semantic)
    assert residual.simulation_E > 2.5e5
    assert any("Target impact response" in warning for part in semantic for warning in part.warnings)
    assert any("softening skipped" in warning for warning in residual.warnings)



def test_controller_runs_repair_loop_with_mock_sim(tmp_path):
    scene = PartPhysSceneLoader(_make_scene(tmp_path)).load()
    posteriors = MaterialPosteriorBuilder().build(scene, {}, {})
    template = tmp_path / "template.json"
    _write_json(template, {"material": "plasticine", "E": 1e5, "nu": 0.3, "density": 1000.0})
    output = tmp_path / "agent_output"
    controller = MaterialAgentController(
        sampler=CandidateSetSampler(budget=2),
        compiler=SimulationConfigCompiler(template, backend="aabb"),
        runner=SimulationRunner(physgm_root=tmp_path, partphys_root=tmp_path, mock=True),
        critic=MaterialCritic(acceptance_score=0.99),
        max_rounds=2,
        repair_budget=2,
        simulate=True,
    )
    result = controller.run(scene, posteriors, output)
    assert (output / "agent_trace.json").exists()
    assert result.selected.candidate_id
    assert any(candidate.candidate_id.startswith("repair_") for candidate in result.candidates)


def test_compiler_reduces_dt_for_high_stiffness_part(tmp_path):
    template = tmp_path / "template.json"
    _write_json(template, {"material": "plasticine", "E": 1e5, "nu": 0.3, "density": 1000.0, "frame_dt": 0.04, "substep_dt": 0.0002})
    candidate = CandidateSet(
        candidate_id="high_stiffness",
        description="test high stiffness support",
        parts=[
            CandidatePartMaterial(
                part_id=4,
                part_name="plate",
                visual_material="Ceramic",
                solver_material="metal",
                raw_E=1.0e7,
                raw_nu=0.35,
                raw_density=2500.0,
                simulation_E=1.0e7,
                simulation_nu=0.35,
                simulation_density=2500.0,
                confidence=0.9,
                source="test",
            )
        ],
        global_material="plasticine",
        global_E=1e5,
        global_nu=0.3,
        global_density=1000.0,
    )
    compiled = SimulationConfigCompiler(template, backend="part_id").compile_candidate(SceneEvidence(scene_dir=str(tmp_path)), candidate, tmp_path / "out")
    config = json.loads(Path(compiled["config_path"]).read_text())
    materials = json.loads(Path(compiled["part_materials_json"]).read_text())

    assert config["substep_dt"] < 0.0002
    assert config["substep_dt"] <= 0.0001
    assert config["material_agent_metadata"]["solver_stability"]["adjusted"] is True
    assert materials["parts"]["4"]["E"] == 1.0e7
    assert materials["parts"]["4"]["material"] == "metal"
