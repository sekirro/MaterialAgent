from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .evaluation.critic import (
    MaterialCritic,
    is_cohesive_deformable_material,
    is_support_like_material,
)
from .evaluation.selector import CandidateSelector
from .evaluation.video_metrics import VideoEvaluator
from .io_utils import write_json
from .schemas import CandidatePartMaterial, CandidateSet, PartPosterior, SceneEvidence
from .simulation.config_compiler import SimulationConfigCompiler
from .simulation.runner import SimulationRunner


@dataclass
class MaterialAgentResult:
    candidates: list[CandidateSet]
    compiled_by_id: dict[str, dict]
    run_results: list[dict]
    scores: list[dict]
    selected: CandidateSet
    selection: dict
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "compiled_by_id": self.compiled_by_id,
            "run_results": self.run_results,
            "scores": self.scores,
            "selected": self.selected.to_dict(),
            "selection": self.selection,
            "trace": self.trace,
        }


@dataclass
class MaterialAgentController:
    sampler: Any
    compiler: SimulationConfigCompiler
    runner: SimulationRunner | None = None
    evaluator: VideoEvaluator = field(default_factory=VideoEvaluator)
    selector: CandidateSelector = field(default_factory=CandidateSelector)
    critic: MaterialCritic = field(default_factory=MaterialCritic)
    max_rounds: int = 3
    repair_budget: int = 3
    simulate: bool = False
    support_E_floor: float = 1.0e7
    support_E_cap: float = 1.0e7
    cohesive_E_floor: float = 5.0e5
    cohesive_density_floor: float = 500.0
    stability_density_floor: float = 300.0
    stability_nu_cap: float = 0.40

    def run(
        self,
        scene: SceneEvidence,
        posteriors: dict[int, PartPosterior],
        output_dir: str | Path,
        manual_candidate: str | None = None,
    ) -> MaterialAgentResult:
        output = Path(output_dir)
        all_candidates: list[CandidateSet] = []
        compiled_by_id: dict[str, dict] = {}
        run_result_by_id: dict[str, dict] = {}
        score_by_id: dict[str, dict] = {}
        trace: dict[str, Any] = {
            "plan": self._plan(scene),
            "rounds": [],
        }

        pending = self.sampler.sample(scene, posteriors)
        max_rounds = max(1, int(self.max_rounds if self.simulate and not manual_candidate else 1))
        for round_idx in range(max_rounds):
            round_candidates = self._unique_new_candidates(pending, all_candidates)
            if not round_candidates:
                break
            all_candidates.extend(round_candidates)
            round_trace = {
                "round": round_idx,
                "actions": ["generate_initial_candidates" if round_idx == 0 else "generate_repair_candidates"],
                "candidate_ids": [candidate.candidate_id for candidate in round_candidates],
                "observations": [],
            }

            for candidate in round_candidates:
                compiled = self.compiler.compile_candidate(scene, candidate, output / "candidate_configs")
                compiled_by_id[candidate.candidate_id] = compiled
                run_result = self._run_or_stub(scene, candidate, compiled, output)
                run_result_by_id[candidate.candidate_id] = run_result
                score = self.evaluator.evaluate(candidate, run_result)
                score_by_id[candidate.candidate_id] = score
                round_trace["observations"].append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "run_status": run_result.get("status"),
                        "score": score,
                    }
                )

            scores = [score_by_id[candidate.candidate_id] for candidate in all_candidates if candidate.candidate_id in score_by_id]
            selected, selection = self.selector.select(all_candidates, scores, manual_candidate=manual_candidate)
            critique = self.critic.critique(scene, selected, selection, can_repair=self.simulate and not manual_candidate)
            round_trace["selection"] = selection
            round_trace["critic"] = critique.to_dict()
            trace["rounds"].append(round_trace)
            write_json(output / "agent_trace.json", trace)

            if manual_candidate or critique.accepted or round_idx == max_rounds - 1:
                break
            pending = self._repair_candidates(selected, critique.repairs, round_idx + 1)
            if not pending:
                break

        scores = [score_by_id[candidate.candidate_id] for candidate in all_candidates if candidate.candidate_id in score_by_id]
        selected, selection = self.selector.select(all_candidates, scores, manual_candidate=manual_candidate)
        final_critique = self.critic.critique(scene, selected, selection, can_repair=False)
        trace["final"] = {
            "selected_candidate_id": selected.candidate_id,
            "selection": selection,
            "critic": final_critique.to_dict(),
        }
        write_json(output / "agent_trace.json", trace)
        return MaterialAgentResult(
            candidates=all_candidates,
            compiled_by_id=compiled_by_id,
            run_results=[run_result_by_id[candidate.candidate_id] for candidate in all_candidates if candidate.candidate_id in run_result_by_id],
            scores=scores,
            selected=selected,
            selection=selection,
            trace=trace,
        )

    def _plan(self, scene: SceneEvidence) -> dict[str, Any]:
        support_parts = []
        deformable_parts = []
        for part in scene.parts:
            proxy = CandidatePartMaterial(
                part_id=part.part_id,
                part_name=part.name,
                visual_material=(part.expected_materials[0] if part.expected_materials else "Plastic"),
                solver_material="plasticine",
                raw_E=0.0,
                raw_nu=0.0,
                raw_density=0.0,
                simulation_E=0.0,
                simulation_nu=0.0,
                simulation_density=0.0,
                confidence=part.confidence,
                source="planner_proxy",
            )
            if is_support_like_material(proxy):
                support_parts.append(part.name)
            elif is_cohesive_deformable_material(proxy):
                deformable_parts.append(part.name)
        return {
            "objective": "select per-part solver materials and E/nu/density through plan-act-observe-critique-repair loops",
            "inputs": {
                "scene_dir": scene.scene_dir,
                "object_name": scene.object_name,
                "part_count": len(scene.parts),
                "backend": self.compiler.resolve_backend(scene),
            },
            "success_criteria": [
                "candidate simulation completes without numerical/runtime errors",
                "support or rigid-like parts are stiffer than cohesive deformable parts",
                "rendered foreground does not spread or flatten excessively",
                "selected materials keep visual material evidence but may choose solver branches that fit observed dynamics",
            ],
            "support_like_parts": support_parts,
            "cohesive_deformable_parts": deformable_parts,
            "max_rounds": max(1, int(self.max_rounds)),
            "repair_budget": max(1, int(self.repair_budget)),
        }

    def _run_or_stub(self, scene: SceneEvidence, candidate: CandidateSet, compiled: dict, output: Path) -> dict:
        if self.simulate:
            if self.runner is None:
                raise RuntimeError("MaterialAgentController simulate=True requires a SimulationRunner.")
            return self.runner.run_candidate(scene, candidate, compiled, output / "candidate_outputs" / candidate.candidate_id)
        return {
            "candidate_id": candidate.candidate_id,
            "status": "not_run",
            "returncode": 0,
            "command": None,
            "backend": compiled.get("backend"),
            "output_path": None,
            "video_path": None,
            "runtime_sec": 0.0,
            "stdout": None,
            "stderr": None,
        }

    def _unique_new_candidates(self, pending: list[CandidateSet], existing: list[CandidateSet]) -> list[CandidateSet]:
        seen = {candidate.candidate_id for candidate in existing}
        out = []
        for candidate in pending:
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            out.append(candidate)
        return out

    def _repair_candidates(self, base: CandidateSet, repairs: list[str], round_index: int) -> list[CandidateSet]:
        out: list[CandidateSet] = []
        for repair in repairs:
            if repair == "raise_support_stiffness":
                out.append(self._with_support_stiffness(base, round_index))
            elif repair == "elastic_cohesive_response":
                out.append(self._with_elastic_cohesion(base, round_index))
            elif repair == "increase_deformable_cohesion":
                out.append(self._with_higher_cohesion(base, round_index))
            elif repair == "stabilize_numerics":
                out.append(self._with_solver_stability(base, round_index))
            elif repair == "explore_role_balanced_candidate":
                out.append(self._with_role_balanced_params(base, round_index))
        unique: list[CandidateSet] = []
        seen: set[str] = set()
        for candidate in out:
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            unique.append(candidate)
        return unique[: max(1, int(self.repair_budget))]

    def _clone(self, base: CandidateSet, candidate_id: str, description: str) -> CandidateSet:
        return CandidateSet(
            candidate_id=candidate_id,
            description=description,
            parts=[replace(part, warnings=list(part.warnings)) for part in base.parts],
            global_material=base.global_material,
            global_E=float(base.global_E),
            global_nu=float(base.global_nu),
            global_density=float(base.global_density),
            score_prior=float(base.score_prior),
            warnings=list(base.warnings),
        )

    def _with_support_stiffness(self, base: CandidateSet, round_index: int) -> CandidateSet:
        candidate = self._clone(base, f"repair_r{round_index}_support_stiffness", "Repair: raise stiffness for support/rigid-like parts")
        for part in candidate.parts:
            if is_support_like_material(part):
                self._apply_support_prior(part)
        self._refresh_global(candidate)
        return candidate

    def _with_elastic_cohesion(self, base: CandidateSet, round_index: int) -> CandidateSet:
        candidate = self._clone(base, f"repair_r{round_index}_elastic_cohesive", "Repair: use elastic solver branch for cohesive deformable parts")
        for part in candidate.parts:
            if is_support_like_material(part):
                self._apply_support_prior(part)
            elif is_cohesive_deformable_material(part):
                self._apply_elastic_cohesion(part)
        self._refresh_global(candidate)
        return candidate

    def _with_higher_cohesion(self, base: CandidateSet, round_index: int) -> CandidateSet:
        candidate = self._clone(base, f"repair_r{round_index}_higher_cohesion", "Repair: increase cohesive deformable stiffness while preserving part identity")
        for part in candidate.parts:
            if is_support_like_material(part):
                self._apply_support_prior(part)
            elif is_cohesive_deformable_material(part):
                old_E = float(part.simulation_E)
                part.raw_E = max(float(part.raw_E), self.cohesive_E_floor)
                part.simulation_E = max(float(part.simulation_E), self.cohesive_E_floor)
                part.simulation_density = max(float(part.simulation_density), self.cohesive_density_floor)
                if part.simulation_E != old_E:
                    part.warnings.append(f"Cohesive stiffness repair raised E from {old_E:g} to {part.simulation_E:g}.")
        self._refresh_global(candidate)
        return candidate

    def _with_solver_stability(self, base: CandidateSet, round_index: int) -> CandidateSet:
        candidate = self._clone(base, f"repair_r{round_index}_solver_stability", "Repair: reduce near-incompressibility and raise low densities for numerical stability")
        for part in candidate.parts:
            old_nu = float(part.simulation_nu)
            old_density = float(part.simulation_density)
            part.simulation_nu = min(old_nu, self.stability_nu_cap)
            part.raw_nu = min(float(part.raw_nu), self.stability_nu_cap)
            part.simulation_density = max(old_density, self.stability_density_floor)
            if part.simulation_nu != old_nu:
                part.warnings.append(f"Stability repair lowered nu from {old_nu:g} to {part.simulation_nu:g}.")
            if part.simulation_density != old_density:
                part.warnings.append(f"Stability repair raised density from {old_density:g} to {part.simulation_density:g}.")
        self._refresh_global(candidate)
        return candidate

    def _with_role_balanced_params(self, base: CandidateSet, round_index: int) -> CandidateSet:
        candidate = self._clone(base, f"repair_r{round_index}_role_balanced", "Repair: enforce role-level separation between rigid/support and soft/cohesive parts")
        for part in candidate.parts:
            if is_support_like_material(part):
                self._apply_support_prior(part)
            elif is_cohesive_deformable_material(part) and part.solver_material in {"foam", "plasticine"}:
                self._apply_elastic_cohesion(part)
        self._refresh_global(candidate)
        return candidate

    def _apply_support_prior(self, part: CandidatePartMaterial) -> None:
        old_E = float(part.simulation_E)
        part.solver_material = "metal"
        stable_cap = max(float(self.support_E_floor), float(self.support_E_cap))
        part.raw_E = max(float(part.raw_E), self.support_E_floor)
        part.simulation_E = min(max(old_E, self.support_E_floor), stable_cap)
        part.raw_nu = min(float(part.raw_nu), 0.35)
        part.simulation_nu = min(float(part.simulation_nu), 0.35)
        part.raw_density = max(float(part.raw_density), 1000.0)
        part.simulation_density = max(float(part.simulation_density), 1000.0)
        if part.simulation_E != old_E:
            part.warnings.append(f"Support solver calibration changed simulation E from {old_E:g} to {part.simulation_E:g}.")
        part.warnings.append("Support/rigid-like part uses elastic solver branch for shape preservation.")

    def _apply_elastic_cohesion(self, part: CandidatePartMaterial) -> None:
        old_solver = part.solver_material
        old_E = float(part.simulation_E)
        part.solver_material = "jelly"
        part.raw_E = max(float(part.raw_E), self.cohesive_E_floor)
        part.simulation_E = max(float(part.simulation_E), self.cohesive_E_floor)
        part.raw_nu = min(max(float(part.raw_nu), 0.20), 0.42)
        part.simulation_nu = min(max(float(part.simulation_nu), 0.20), 0.42)
        part.raw_density = max(float(part.raw_density), self.cohesive_density_floor)
        part.simulation_density = max(float(part.simulation_density), self.cohesive_density_floor)
        if old_solver != part.solver_material:
            part.warnings.append(f"Cohesive response repair changed solver branch from {old_solver} to jelly.")
        if part.simulation_E != old_E:
            part.warnings.append(f"Cohesive response repair raised E from {old_E:g} to {part.simulation_E:g}.")

    def _refresh_global(self, candidate: CandidateSet) -> None:
        if not candidate.parts:
            return
        anchor = max(candidate.parts, key=lambda part: float(part.confidence))
        candidate.global_material = anchor.solver_material
        candidate.global_E = float(anchor.simulation_E)
        candidate.global_nu = float(anchor.simulation_nu)
        candidate.global_density = float(anchor.simulation_density)
