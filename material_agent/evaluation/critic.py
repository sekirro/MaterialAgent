from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..schemas import CandidatePartMaterial, CandidateSet, SceneEvidence


SUPPORT_TOKENS = (
    "plate",
    "dish",
    "tray",
    "bowl",
    "base support",
    "support",
    "stand",
    "holder",
    "case",
    "shell",
    "frame",
)

COHESIVE_DEFORMABLE_TOKENS = (
    "body",
    "base",
    "core",
    "filling",
    "cream",
    "frosting",
    "icing",
    "soft",
    "cushion",
    "pad",
    "foam",
    "sponge",
    "rubber",
    "gel",
    "fruit",
    "food",
)

RIGID_VISUAL_MATERIALS = {"Ceramic", "Metal", "Glass", "Stone", "Wood"}
PLASTIC_SOLVER_MATERIALS = {"foam", "plasticine"}


def _text(*values: object) -> str:
    return " ".join(str(v or "") for v in values).lower()


def is_support_like_material(part: CandidatePartMaterial) -> bool:
    text = _text(part.part_name)
    return any(token in text for token in SUPPORT_TOKENS) or part.visual_material in RIGID_VISUAL_MATERIALS


def is_cohesive_deformable_material(part: CandidatePartMaterial) -> bool:
    if is_support_like_material(part):
        return False
    text = _text(part.part_name, part.visual_material, part.source)
    if part.solver_material in {"sand", "snow"}:
        return False
    if any(token in text for token in COHESIVE_DEFORMABLE_TOKENS):
        return True
    return part.solver_material in {"foam", "plasticine", "jelly"} and part.visual_material not in RIGID_VISUAL_MATERIALS


@dataclass
class MaterialCritique:
    accepted: bool
    selected_candidate_id: str
    selected_score: float
    issues: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MaterialCritic:
    def __init__(
        self,
        acceptance_score: float = 0.72,
        support_E_min: float = 1.0e7,
        spread_growth_limit: float = 2.0,
        height_ratio_min: float = 0.45,
    ):
        self.acceptance_score = float(acceptance_score)
        self.support_E_min = float(support_E_min)
        self.spread_growth_limit = float(spread_growth_limit)
        self.height_ratio_min = float(height_ratio_min)

    def critique(
        self,
        scene: SceneEvidence,
        selected: CandidateSet,
        selection: dict,
        can_repair: bool,
    ) -> MaterialCritique:
        score = float(selection.get("score", -1.0))
        ok = bool(selection.get("ok", False))
        reasons = [str(x) for x in selection.get("reasons", [])]
        metrics = selection.get("frame_metrics") or {}
        issues: list[str] = []
        repairs: list[str] = []

        if not ok:
            issues.append("simulation_failed")
        if score < self.acceptance_score:
            issues.append("low_selection_score")

        bbox_growth = float(metrics.get("bbox_area_growth_ratio", 1.0) or 1.0)
        width_growth = float(metrics.get("bbox_width_growth_ratio", 1.0) or 1.0)
        height_ratio = float(metrics.get("bbox_height_ratio", 1.0) or 1.0)
        if bbox_growth > self.spread_growth_limit:
            issues.append("excessive_spread")
        if height_ratio < self.height_ratio_min and width_growth > 1.25:
            issues.append("flattened_response")
        if any("numerical/runtime" in item.lower() or "non-zero" in item.lower() for item in reasons):
            issues.append("numerical_or_runtime_error")

        composite_regularized = self._is_composite_regularized_candidate(selected)
        support_soft = [
            part.part_name
            for part in selected.parts
            if is_support_like_material(part) and float(part.simulation_E) < self.support_E_min
        ]
        if support_soft and not composite_regularized:
            issues.append("support_too_soft")
            reasons.append("support/rigid parts below solver stiffness floor: " + ", ".join(support_soft))
        elif support_soft and composite_regularized:
            reasons.append(
                "support stiffness floor skipped because candidate uses composite-compatible solver regularization: "
                + ", ".join(support_soft)
            )

        plastic_cohesive = [
            part.part_name
            for part in selected.parts
            if is_cohesive_deformable_material(part) and part.solver_material in PLASTIC_SOLVER_MATERIALS
        ]
        if plastic_cohesive and any(x in issues for x in ("excessive_spread", "flattened_response", "low_selection_score")):
            issues.append("cohesive_parts_use_plastic_solver")
            reasons.append("cohesive deformable parts use irreversible solver branches: " + ", ".join(plastic_cohesive))

        if "support_too_soft" in issues:
            repairs.append("raise_support_stiffness")
            repairs.append("enable_rigid_support")
        if any(x in issues for x in ("excessive_spread", "flattened_response", "cohesive_parts_use_plastic_solver")):
            repairs.append("elastic_cohesive_response")
            repairs.append("increase_deformable_cohesion")
        if any(x in issues for x in ("simulation_failed", "numerical_or_runtime_error")):
            repairs.append("stabilize_numerics")
        if not repairs and "low_selection_score" in issues:
            repairs.append("explore_role_balanced_candidate")

        hard_issues = {"simulation_failed", "numerical_or_runtime_error", "support_too_soft", "excessive_spread", "flattened_response"}
        accepted = ok and score >= self.acceptance_score and not (hard_issues & set(issues))

        return MaterialCritique(
            accepted=accepted,
            selected_candidate_id=selected.candidate_id,
            selected_score=score,
            issues=list(dict.fromkeys(issues)),
            repairs=list(dict.fromkeys(repairs)),
            reasons=reasons,
            metrics=dict(metrics),
        )

    @staticmethod
    def _is_composite_regularized_candidate(candidate: CandidateSet) -> bool:
        if "solver_compatible" in candidate.candidate_id:
            return True
        return any(
            "composite solver regularization" in warning.lower()
            for part in candidate.parts
            for warning in part.warnings
        )
