from __future__ import annotations

import argparse
from pathlib import Path

from .agent import MaterialAgentController
from .evaluation.critic import MaterialCritic
from .evidence.physgm_distribution import PhysGMDistributionExtractor
from .evidence.vlm_material import VLMPartMaterialPriorExtractor
from .io_utils import ensure_dir, read_yaml, write_json
from .loaders.partphys_outputs import PartPhysSceneLoader
from .reasoning.candidate_sampler import CandidateSetSampler
from .reasoning.posterior import MaterialPosteriorBuilder
from .reasoning.skill_memory import SkillMemory
from .report import write_report
from .simulation.config_compiler import SimulationConfigCompiler
from .simulation.runner import SimulationRunner


def _default_output(scene_dir: Path) -> Path:
    return scene_dir / "material_agent"


def _solver_ranges(config: dict) -> dict:
    clamp = config.get("solver_clamp", {}) if isinstance(config, dict) else {}
    out = {
        key: clamp[key]
        for key in ("local_E_range", "local_nu_range", "local_density_range", "rigid_support_E_floor", "rigid_support_E_cap")
        if key in clamp
    }
    regularization = config.get("solver_regularization", {}) if isinstance(config, dict) else {}
    if isinstance(regularization, dict):
        out.update(regularization)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaterialAgent for part-level PhysGM material selection.")
    parser.add_argument("--partphys-scene", required=True, help="PartPhysAgent scene output directory.")
    parser.add_argument("--output-dir", default=None, help="MaterialAgent output directory. Defaults to <scene>/material_agent.")
    parser.add_argument("--physgm-root", default="/root/PhysGM")
    parser.add_argument("--partphys-root", default="/root/PartPhysAgent")
    parser.add_argument("--physgm-config", default="/root/PhysGM/configs/infer.yaml")
    parser.add_argument("--checkpoint", default="/root/PhysGM/checkpoints/checkpoint.pt")
    parser.add_argument("--template-config", default="/root/PhysGM/configs/physical/down_template.json")
    parser.add_argument("--config", default=None, help="Optional MaterialAgent YAML config.")
    parser.add_argument("--candidate-budget", type=int, default=None)
    parser.add_argument("--agent-rounds", type=int, default=None, help="Maximum plan-act-observe-repair rounds when simulation is enabled.")
    parser.add_argument("--repair-budget", type=int, default=None, help="Maximum repair candidates generated per agent round.")
    parser.add_argument("--agent-acceptance-score", type=float, default=None, help="Minimum score accepted before repair stops.")
    parser.add_argument("--backend", choices=["auto", "aabb", "part_id"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--simulate", action="store_true", help="Run candidate simulations.")
    parser.add_argument("--mock-sim", action="store_true", help="Do not run PhysGM simulation; write mock run results.")
    parser.add_argument("--mock-physgm", action="store_true", help="Do not run PhysGM model; use fallback distributions.")
    parser.add_argument("--skip-physgm-distribution", action="store_true", help="Use schema/default priors only.")
    parser.add_argument("--physgm-crop-variants", default=None, help="Comma-separated crop variants for per-part PhysGM evidence. Default is context_dim; use all for every crop.")
    parser.add_argument("--vlm-provider", choices=["none", "openai_compatible"], default="none")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--vlm-api-base", default=None)
    parser.add_argument("--vlm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument("--vlm-weight", type=float, default=None)
    parser.add_argument("--selection", choices=["auto", "human"], default="auto")
    parser.add_argument("--select-candidate", default=None, help="Candidate id to select manually.")
    parser.add_argument("--white-bg", action="store_true", default=True)
    parser.add_argument("--no-white-bg", dest="white_bg", action="store_false")
    parser.add_argument("--render-img", action="store_true", default=True)
    parser.add_argument("--no-render-img", dest="render_img", action="store_false")
    parser.add_argument("--compile-video", action="store_true", default=True)
    parser.add_argument("--no-compile-video", dest="compile_video", action="store_false")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    return parser




def _crop_variants(args_value, config: dict) -> list[str] | None:
    value = args_value
    if value is None and isinstance(config, dict):
        value = config.get("evidence", {}).get("physgm_crop_variants")
    if value is None:
        return ["context_dim"]
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return None
        items = [x.strip() for x in value.split(",")]
    else:
        items = [str(x).strip() for x in value]
    items = [x for x in items if x]
    if any(x.lower() == "all" for x in items):
        return None
    return items


def _write_agent_input_evidence(output_dir: Path, scene, distributions: dict, vlm_priors: dict, memory_priors: dict) -> None:
    parts = []
    for part in scene.parts:
        physgm_items = []
        for item in distributions.get(part.part_id, []):
            top_probs = sorted(item.material_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
            physgm_items.append(
                {
                    "source": "per_part_physgm",
                    "variant": item.variant,
                    "image_path": item.image_path,
                    "material": item.material,
                    "top_material_probs": [{"material": k, "probability": float(v)} for k, v in top_probs],
                    "E": float(item.E_mean),
                    "E_sigma_log10": float(item.E_sigma_log10),
                    "nu": float(item.nu_mean),
                    "nu_sigma": float(item.nu_sigma),
                    "predicted_phys_path": item.predicted_phys_path,
                    "warnings": list(item.warnings),
                }
            )
        parts.append(
            {
                "part_id": part.part_id,
                "part_name": part.name,
                "physics_group": part.physics_group,
                "physical_role": part.physical_role,
                "expected_materials": part.expected_materials,
                "gaussian_count": part.gaussian_count,
                "mask_path": part.mask_path,
                "physgm_part_predictions": physgm_items,
                "vlm_material_priors": vlm_priors.get(part.part_id, []),
                "memory_priors": memory_priors.get(part.part_id, []),
            }
        )
    write_json(
        output_dir / "agent_input_evidence.json",
        {
            "object_name": scene.object_name,
            "scene_dir": scene.scene_dir,
            "material_classes_14": [
                "Wood", "Metal", "Plastic", "Glass", "Fabric", "Leather", "Ceramic",
                "Stone", "Rubber", "Paper", "Sand", "Snow", "Plasticine", "Foam",
            ],
            "mpm_solver_classes_6": ["jelly", "metal", "sand", "foam", "snow", "plasticine"],
            "parts": parts,
        },
    )

def _agent_value(args_value, config: dict, key: str, default):
    if args_value is not None:
        return args_value
    agent_config = config.get("agent", {}) if isinstance(config, dict) else {}
    return agent_config.get(key, default)


def run(args: argparse.Namespace) -> int:
    config = read_yaml(args.config, {}) if args.config else {}
    scene_dir = Path(args.partphys_scene).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output(scene_dir)
    ensure_dir(output_dir)

    state = {
        "partphys_scene": str(scene_dir),
        "output_dir": str(output_dir),
        "warnings": [],
        "steps": [],
    }
    scene = PartPhysSceneLoader(scene_dir).load()
    state["scene"] = scene.to_dict()
    state["warnings"].extend(scene.warnings)
    write_json(output_dir / "part_evidence.json", scene.to_dict())

    memory_path = output_dir / "material_skills.yaml"
    if config.get("evidence", {}).get("skill_memory_path"):
        configured = Path(config["evidence"]["skill_memory_path"]).expanduser()
        memory_path = configured if configured.is_absolute() else output_dir / configured
    memory = SkillMemory(memory_path)
    memory_priors = memory.retrieve(scene)

    distributions = {}
    if not args.skip_physgm_distribution:
        extractor = PhysGMDistributionExtractor(
            physgm_root=args.physgm_root,
            partphys_root=args.partphys_root,
            config_path=args.physgm_config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            amp_dtype=args.amp_dtype,
            mock=args.mock_physgm,
            crop_variants=_crop_variants(args.physgm_crop_variants, config),
        )
        if extractor.available() or args.mock_physgm:
            distributions = extractor.extract_for_scene(scene, output_dir / "distribution")
        else:
            state["warnings"].append("PhysGM distribution extractor unavailable; using priors only.")
    state["steps"].append("distribution")

    vlm_priors = {}
    if args.vlm_provider != "none":
        vlm_extractor = VLMPartMaterialPriorExtractor(
            provider=args.vlm_provider,
            model=args.vlm_model,
            api_base=args.vlm_api_base,
            api_key_env=args.vlm_api_key_env,
            timeout=args.vlm_timeout,
        )
        if vlm_extractor.available():
            vlm_priors = vlm_extractor.extract_for_scene(scene, output_dir / "vlm_material")
        else:
            state["warnings"].append(f"VLM material prior unavailable; env var {args.vlm_api_key_env} is not set.")
    state["steps"].append("vlm_material")

    _write_agent_input_evidence(output_dir, scene, distributions, vlm_priors, memory_priors)

    posterior_config = config.get("posterior", {}) if isinstance(config, dict) else {}
    posterior_builder = MaterialPosteriorBuilder(
        schema_weight=posterior_config.get("schema_expected_material_weight", 0.25),
        physgm_weight=posterior_config.get("physgm_crop_weight", 0.35),
        vlm_weight=args.vlm_weight if args.vlm_weight is not None else posterior_config.get("vlm_material_weight", 0.30),
        role_weight=0.10,
        whole_weight=posterior_config.get("global_physgm_weight_for_multi_part", 0.05),
        memory_weight=posterior_config.get("skill_memory_weight", 0.10),
        vlm_param_weight=posterior_config.get("vlm_param_weight", 0.05),
    )
    posteriors = posterior_builder.build(scene, distributions, memory_priors, vlm_priors)
    write_json(output_dir / "part_posteriors.json", {str(k): v.to_dict() for k, v in posteriors.items()})
    state["steps"].append("posterior")

    budget = args.candidate_budget or config.get("sampling", {}).get("candidate_budget", 5)
    sampler = CandidateSetSampler(budget=budget, solver_ranges=_solver_ranges(config))
    compiler = SimulationConfigCompiler(args.template_config, backend=args.backend, solver_stability=config.get("solver_stability", {}))
    simulate = bool(args.simulate or args.mock_sim)
    runner = None
    if simulate:
        runner = SimulationRunner(
            physgm_root=args.physgm_root,
            partphys_root=args.partphys_root,
            render_img=args.render_img,
            compile_video=args.compile_video,
            white_bg=args.white_bg,
            timeout_sec=args.timeout_sec,
            mock=args.mock_sim,
        )
    agent_config = config.get("agent", {}) if isinstance(config, dict) else {}
    critic = MaterialCritic(
        acceptance_score=float(_agent_value(args.agent_acceptance_score, config, "acceptance_score", 0.72)),
        support_E_min=float(agent_config.get("support_E_min", 1.0e7)),
        spread_growth_limit=float(agent_config.get("spread_growth_limit", 2.0)),
        height_ratio_min=float(agent_config.get("height_ratio_min", 0.45)),
    )
    controller = MaterialAgentController(
        sampler=sampler,
        compiler=compiler,
        runner=runner,
        critic=critic,
        max_rounds=int(_agent_value(args.agent_rounds, config, "max_rounds", 3)),
        repair_budget=int(_agent_value(args.repair_budget, config, "repair_budget", 3)),
        simulate=simulate,
        support_E_floor=float(agent_config.get("support_E_floor", 1.0e7)),
        support_E_cap=float(agent_config.get("support_E_cap", agent_config.get("support_E_floor", 1.0e7))),
        cohesive_E_floor=float(agent_config.get("cohesive_E_floor", 5.0e5)),
        cohesive_density_floor=float(agent_config.get("cohesive_density_floor", 500.0)),
        stability_density_floor=float(agent_config.get("stability_density_floor", 300.0)),
        stability_nu_cap=float(agent_config.get("stability_nu_cap", 0.40)),
    )
    manual = args.select_candidate if args.selection == "human" or args.select_candidate else None
    agent_result = controller.run(scene, posteriors, output_dir, manual_candidate=manual)
    candidates = agent_result.candidates
    compiled_by_id = agent_result.compiled_by_id
    run_results = agent_result.run_results
    scores = agent_result.scores
    selected = agent_result.selected
    selection = agent_result.selection
    write_json(output_dir / "candidate_sets.json", [candidate.to_dict() for candidate in candidates])
    write_json(output_dir / "compiled_candidates.json", compiled_by_id)
    write_json(output_dir / "simulation_results.json", run_results)
    write_json(output_dir / "video_scores.json", scores)
    write_json(output_dir / "selection.json", selection)
    write_json(output_dir / "selected_materials.json", {"selection": selection, "parts": [p.to_dict() for p in selected.parts]})
    state["steps"].append("agent_loop")

    selected_compiled = compiler.compile_candidate(scene, selected, output_dir)
    selected_config = Path(selected_compiled["config_path"])
    final_config = output_dir / "sim_config_materialagent_selected.json"
    final_config.write_text(selected_config.read_text(encoding="utf-8"), encoding="utf-8")
    if selected_compiled.get("part_materials_json"):
        src = Path(selected_compiled["part_materials_json"])
        dst = output_dir / "selected_part_materials.json"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        selected_compiled["part_materials_json"] = str(dst)
    write_json(output_dir / "selected_compiled.json", selected_compiled)

    memory_updates = memory.update_from_selection(scene, selected, float(selection.get("score", 0.0)), output_dir / "material_skill_memory_delta.yaml")
    state["memory_updates"] = memory_updates
    state["selected_candidate"] = selected.candidate_id
    state["selection"] = selection
    state["agent_trace"] = agent_result.trace
    state["steps"].append("selection")
    write_report(output_dir / "report.md", scene, selected, selection, scores)
    write_json(output_dir / "material_state.json", state)

    print(f"MaterialAgent finished: {output_dir}")
    print(f"Selected candidate: {selected.candidate_id}")
    print(f"Selected config: {final_config}")
    if selected_compiled.get("backend") == "part_id":
        print(f"Selected part materials: {selected_compiled.get('part_materials_json')}")
    print(f"Agent trace: {output_dir / 'agent_trace.json'}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
