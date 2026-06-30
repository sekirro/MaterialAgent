from __future__ import annotations

import math
from typing import Iterable

from ..constants import (
    clamp_physical_values,
    clamp_solver_values,
    default_E_for_material,
    default_nu_for_material,
    density_for_material,
    normalize_material,
    solver_material_for_visual,
)
from ..loaders.partphys_outputs import is_residual_part
from ..schemas import CandidatePartMaterial, CandidateSet, PartEvidence, PartPosterior, SceneEvidence


def _quantile_value(mean: float, std: float, q: float) -> float:
    z = {0.15: -1.04, 0.25: -0.67, 0.50: 0.0, 0.75: 0.67, 0.85: 1.04}.get(round(float(q), 2), 0.0)
    return mean + z * std


def _part_material(
    part: PartEvidence,
    posterior: PartPosterior,
    material: str | None,
    logE_quantile: float,
    nu_quantile: float,
    source: str,
    solver_ranges: dict | None = None,
) -> CandidatePartMaterial:
    material = normalize_material(material or posterior.selected_material)
    logE = _quantile_value(posterior.logE_mean, posterior.logE_std, logE_quantile)
    nu = _quantile_value(posterior.nu_mean, posterior.nu_std, nu_quantile)
    E = 10 ** logE
    if material != posterior.selected_material:
        E = default_E_for_material(material)
        nu = default_nu_for_material(material)
    E, nu, warnings = clamp_physical_values(material, E, nu)
    density = density_for_material(material)
    solver_ranges = solver_ranges or {}
    sim_E, sim_nu, sim_density, sim_warnings = clamp_solver_values(
        E,
        nu,
        density,
        E_range=tuple(solver_ranges["local_E_range"]) if "local_E_range" in solver_ranges else None,
        nu_range=tuple(solver_ranges.get("local_nu_range", (0.05, 0.45))),
        density_range=tuple(solver_ranges.get("local_density_range", (300.0, 3000.0))),
    )
    return CandidatePartMaterial(
        part_id=part.part_id,
        part_name=part.name,
        visual_material=material,
        solver_material=solver_material_for_visual(material),
        raw_E=E,
        raw_nu=nu,
        raw_density=density,
        simulation_E=sim_E,
        simulation_nu=sim_nu,
        simulation_density=sim_density,
        confidence=posterior.confidence,
        source=source,
        warnings=warnings + sim_warnings,
    )


class CandidateSetSampler:
    def __init__(self, budget: int = 5, solver_ranges: dict | None = None):
        self.budget = max(1, int(budget))
        self.solver_ranges = solver_ranges or {}

    def sample(self, scene: SceneEvidence, posteriors: dict[int, PartPosterior]) -> list[CandidateSet]:
        active_parts = self._active_parts(scene, posteriors)
        candidates: list[CandidateSet] = []
        candidates.append(self._build("posterior_map", "MAP material and median E/nu without projection constraints", active_parts, posteriors, 0.50, 0.50))
        support_stiff_mpm = self._support_stiff_mpm_response(active_parts, posteriors)
        if support_stiff_mpm:
            candidates.append(support_stiff_mpm)
        solver_compatible = self._solver_compatible_response(scene, active_parts, posteriors)
        if solver_compatible:
            candidates.append(solver_compatible)
        rigid_support = self._rigid_support_response(active_parts, posteriors)
        if rigid_support:
            candidates.append(rigid_support)
        hard_bonded = self._hard_support_bonded_response(scene, active_parts, posteriors)
        if hard_bonded:
            candidates.append(hard_bonded)
        candidates.append(self._build("soft_response", "Lower E for deformable response without projection constraints", active_parts, posteriors, 0.25, 0.75))
        candidates.append(self._build("stiff_response", "Higher E for rigid/support response", active_parts, posteriors, 0.75, 0.50))
        sweep = self._uncertain_sweep(active_parts, posteriors)
        if sweep:
            candidates.append(sweep)
        alternative = self._alternative_material(active_parts, posteriors)
        if alternative:
            candidates.append(alternative)
        return candidates[: self.budget]

    def _active_parts(self, scene: SceneEvidence, posteriors: dict[int, PartPosterior]) -> list[PartEvidence]:
        non_residual = [p for p in scene.parts if p.part_id in posteriors and not is_residual_part(p)]
        residual = [p for p in scene.parts if p.part_id in posteriors and is_residual_part(p) and int(p.gaussian_count or 0) > 0]
        active = non_residual + residual
        if active:
            return active
        return [p for p in scene.parts if p.part_id in posteriors]

    def _build(
        self,
        candidate_id: str,
        description: str,
        parts: Iterable[PartEvidence],
        posteriors: dict[int, PartPosterior],
        logE_q: float,
        nu_q: float,
    ) -> CandidateSet:
        mats = [
            _part_material(part, posteriors[part.part_id], None, logE_q, nu_q, candidate_id, self.solver_ranges)
            for part in parts
        ]
        return self._candidate(candidate_id, description, mats)

    def _uncertain_sweep(self, parts: list[PartEvidence], posteriors: dict[int, PartPosterior]) -> CandidateSet | None:
        if not parts:
            return None
        target = min(parts, key=lambda p: posteriors[p.part_id].confidence)
        mats = []
        for part in parts:
            q = 0.15 if part.part_id == target.part_id else 0.50
            mats.append(_part_material(part, posteriors[part.part_id], None, q, 0.75, "uncertain_part_sweep", self.solver_ranges))
        return self._candidate("uncertain_part_sweep", f"Lower stiffness sweep for uncertain part {target.name}", mats)

    def _alternative_material(self, parts: list[PartEvidence], posteriors: dict[int, PartPosterior]) -> CandidateSet | None:
        target = None
        alt_material = None
        alt_prob = 0.0
        for part in parts:
            posterior = posteriors[part.part_id]
            sorted_probs = sorted(posterior.material_probs.items(), key=lambda item: item[1], reverse=True)
            if len(sorted_probs) < 2:
                continue
            candidate_material = sorted_probs[1][0]
            candidate_prob = float(sorted_probs[1][1])
            posterior_E = 10 ** float(posterior.logE_mean)
            candidate_E = default_E_for_material(candidate_material)
            if candidate_E > max(posterior_E * 10.0, 1.0e7):
                continue
            if candidate_prob > alt_prob:
                target = part
                alt_material = candidate_material
                alt_prob = candidate_prob
        if target is None or alt_material is None or alt_prob < 0.12:
            return None
        mats = []
        for part in parts:
            material = alt_material if part.part_id == target.part_id else None
            mats.append(_part_material(part, posteriors[part.part_id], material, 0.50, 0.50, "alternative_material", self.solver_ranges))
        return self._candidate("alternative_material", f"Try second material {alt_material} for {target.name}", mats)

    @staticmethod
    def _enable_rigid_projection(mat: CandidatePartMaterial, strength: float = 1.0) -> None:
        mat.rigid_project = True
        mat.rigid_project_strength = min(max(float(strength), 0.0), 1.0)

    @staticmethod
    def _enable_interface_bond(
        mat: CandidatePartMaterial,
        radius: float = 0.045,
        strength: float = 0.95,
        velocity_blend: float = 0.95,
        max_particles: int = 26000,
    ) -> None:
        mat.interface_bond = True
        mat.interface_bond_radius = float(radius)
        mat.interface_bond_strength = min(max(float(strength), 0.0), 1.0)
        mat.interface_bond_velocity_blend = min(max(float(velocity_blend), 0.0), 1.0)
        mat.interface_bond_max_particles = int(max_particles)

    def _support_stiff_mpm_response(self, parts: list[PartEvidence], posteriors: dict[int, PartPosterior]) -> CandidateSet | None:
        if not parts:
            return None
        mats = []
        touched = False
        for part in parts:
            mat = _part_material(part, posteriors[part.part_id], None, 0.50, 0.50, "support_stiff_mpm_response", self.solver_ranges)
            if self._is_rigid_or_support_part(part, mat):
                touched = True
                floor = float(self.solver_ranges.get("rigid_support_E_floor", 1.0e7))
                cap = max(floor, float(self.solver_ranges.get("rigid_support_E_cap", floor)))
                old_solver = mat.solver_material
                old_E = float(mat.simulation_E)
                mat.solver_material = "metal"
                mat.raw_E = max(float(mat.raw_E), floor)
                mat.simulation_E = min(max(old_E, floor), cap)
                mat.raw_nu = min(float(mat.raw_nu), 0.35)
                mat.simulation_nu = min(float(mat.simulation_nu), 0.35)
                mat.raw_density = max(float(mat.raw_density), 1000.0)
                mat.simulation_density = max(float(mat.simulation_density), 1000.0)
                mat.warnings.append(
                    "Support stiff MPM candidate raises support stiffness but deliberately leaves rigid_project disabled "
                    f"({old_solver} -> metal, E >= {floor:g})."
                )
            mats.append(mat)
        if not touched:
            return None
        return self._candidate("support_stiff_mpm_response", "Hard support material with normal MPM dynamics and no projection", mats)

    def _hard_support_bonded_response(
        self,
        scene: SceneEvidence,
        parts: list[PartEvidence],
        posteriors: dict[int, PartPosterior],
    ) -> CandidateSet | None:
        if not parts:
            return None
        prototype = [
            _part_material(part, posteriors[part.part_id], None, 0.50, 0.50, "hard_support_bonded_response", self.solver_ranges)
            for part in parts
        ]
        has_support = any(self._is_rigid_or_support_part(part, mat) for part, mat in zip(parts, prototype))
        has_deformable = any(not self._is_rigid_or_support_part(part, mat) and self._is_cohesive_or_food_part(part, mat) for part, mat in zip(parts, prototype))
        if not (has_support and has_deformable):
            return None

        whole = scene.whole_physics or {}
        baseline_E = float(whole.get("E") or self.solver_ranges.get("hard_bonded_body_E", 5.0e5))
        baseline_nu = float(whole.get("nu") or self.solver_ranges.get("hard_bonded_body_nu", 0.476))
        baseline_density = float(whole.get("density") or self.solver_ranges.get("hard_bonded_body_density", 5000.0))
        support_floor = float(self.solver_ranges.get("hard_bonded_support_E", self.solver_ranges.get("rigid_support_E_floor", 1.0e7)))
        support_cap = float(self.solver_ranges.get("hard_bonded_support_E_cap", support_floor))
        support_density = float(self.solver_ranges.get("hard_bonded_support_density", 2500.0))
        support_nu = float(self.solver_ranges.get("hard_bonded_support_nu", 0.35))

        mats: list[CandidatePartMaterial] = []
        for part, mat in zip(parts, prototype):
            if self._is_rigid_or_support_part(part, mat):
                old_solver = mat.solver_material
                mat.solver_material = "metal"
                mat.simulation_E = min(max(float(mat.simulation_E), support_floor), max(support_floor, support_cap))
                mat.raw_E = min(max(float(mat.raw_E), support_floor), max(support_floor, support_cap))
                mat.simulation_nu = support_nu
                mat.raw_nu = min(float(mat.raw_nu), support_nu)
                mat.simulation_density = support_density
                mat.raw_density = max(float(mat.raw_density), support_density)
                self._enable_rigid_projection(mat, strength=0.85)
                self._enable_interface_bond(mat, radius=0.045, strength=0.95, velocity_blend=0.95, max_particles=26000)
                mat.warnings.append(
                    "Hard support bonded response keeps this support/rigid part stiff and explicitly enables "
                    "rigid_project plus bonded interface constraints "
                    f"({old_solver} -> metal, E in [{support_floor:g}, {max(support_floor, support_cap):g}])."
                )
            else:
                old_solver = mat.solver_material
                role_scale = self._composite_role_scale(part, mat)
                mat.solver_material = "plasticine"
                mat.simulation_E = max(float(mat.simulation_E), baseline_E * role_scale)
                mat.simulation_nu = min(max(baseline_nu, 0.20), 0.485)
                mat.simulation_density = baseline_density
                mat.warnings.append(
                    "Hard support bonded response uses a cohesive plasticine-family solver for the object body/contact layer "
                    f"while preserving part-specific stiffness evidence ({old_solver} -> plasticine)."
                )
            mats.append(mat)
        candidate = self._candidate(
            "hard_support_bonded_response",
            "Keep support parts hard while bonding initial support-body contact layers to prevent contact artifacts",
            mats,
        )
        candidate.warnings.append("Generated because the scene contains support-like parts in contact with cohesive/deformable parts.")
        return candidate

    def _solver_compatible_response(
        self,
        scene: SceneEvidence,
        parts: list[PartEvidence],
        posteriors: dict[int, PartPosterior],
    ) -> CandidateSet | None:
        if not parts:
            return None
        prototype = [
            _part_material(part, posteriors[part.part_id], None, 0.50, 0.50, "solver_compatible_response", self.solver_ranges)
            for part in parts
        ]
        has_support = any(self._is_rigid_or_support_part(part, mat) for part, mat in zip(parts, prototype))
        has_deformable = any(not self._is_rigid_or_support_part(part, mat) and self._is_cohesive_or_food_part(part, mat) for part, mat in zip(parts, prototype))
        if not (has_support and has_deformable):
            return None

        whole = scene.whole_physics or {}
        baseline_material = normalize_material(whole.get("material") or self.solver_ranges.get("composite_visual_material", "Plasticine"))
        baseline_solver = solver_material_for_visual(baseline_material)
        if baseline_solver not in {"plasticine", "jelly", "foam"}:
            baseline_solver = "plasticine"
        baseline_E = float(whole.get("E") or self.solver_ranges.get("composite_E", 5.0e5))
        baseline_nu = float(whole.get("nu") or self.solver_ranges.get("composite_nu", 0.476))
        baseline_density = float(whole.get("density") or self.solver_ranges.get("composite_density", 5000.0))
        min_scale = float(self.solver_ranges.get("composite_E_min_scale", 0.85))
        max_scale = float(self.solver_ranges.get("composite_E_max_scale", 1.7))

        mats: list[CandidatePartMaterial] = []
        for part, mat in zip(parts, prototype):
            old_solver = mat.solver_material
            role_scale = self._composite_role_scale(part, mat)
            target_E = baseline_E * role_scale
            mat.solver_material = baseline_solver
            mat.simulation_E = min(max(target_E, baseline_E * min_scale), baseline_E * max_scale)
            mat.simulation_nu = min(max(baseline_nu, 0.20), 0.485)
            mat.simulation_density = baseline_density
            mat.warnings.append(
                "Composite solver regularization keeps visual material evidence but uses a compatible solver family "
                f"({old_solver} -> {mat.solver_material}) and disables rigid projection for bonded support contacts."
            )
            mats.append(mat)
        candidate = self._candidate(
            "solver_compatible_response",
            "Regularize bonded composite parts into a compatible solver family to avoid support/contact artifacts",
            mats,
        )
        candidate.global_material = baseline_solver
        candidate.global_E = baseline_E
        candidate.global_nu = min(max(baseline_nu, 0.20), 0.485)
        candidate.global_density = baseline_density
        candidate.warnings.append("Generated because the scene contains both support-like and cohesive/deformable parts.")
        return candidate

    def _is_cohesive_or_food_part(self, part: PartEvidence, mat: CandidatePartMaterial) -> bool:
        text = f"{part.name} {part.physics_group} {part.physical_role} {mat.visual_material}".lower()
        contact_body_tokens = (
            "body", "base", "core", "main", "bulk", "filling", "coating", "layer",
            "cream", "frosting", "icing", "fruit", "organic", "soft", "deformable",
            "foam", "sponge", "rubber", "gel", "cloth", "fabric", "leather", "pad",
        )
        if any(token in text for token in contact_body_tokens):
            return True
        return mat.solver_material in {"foam", "plasticine", "jelly"} and mat.visual_material not in {"Ceramic", "Metal", "Glass", "Stone"}

    def _composite_role_scale(self, part: PartEvidence, mat: CandidatePartMaterial) -> float:
        text = f"{part.name} {part.physics_group} {part.physical_role} {mat.visual_material}".lower()
        if self._is_rigid_or_support_part(part, mat):
            return float(self.solver_ranges.get("composite_support_E_scale", 1.6))
        if any(token in text for token in ("candle", "stem", "stick", "handle")):
            return float(self.solver_ranges.get("composite_detail_E_scale", 1.45))
        if any(token in text for token in ("fruit", "strawberry", "berry", "decoration")):
            return float(self.solver_ranges.get("composite_decoration_E_scale", 1.25))
        if any(token in text for token in ("cream", "frosting", "icing")):
            return float(self.solver_ranges.get("composite_coating_E_scale", 1.05))
        return 1.0

    def _rigid_support_response(self, parts: list[PartEvidence], posteriors: dict[int, PartPosterior]) -> CandidateSet | None:
        if not parts:
            return None
        mats = []
        touched = False
        for part in parts:
            mat = _part_material(part, posteriors[part.part_id], None, 0.25, 0.75, "rigid_support_response", self.solver_ranges)
            if self._is_rigid_or_support_part(part, mat):
                touched = True
                floor = float(self.solver_ranges.get("rigid_support_E_floor", 1.0e7))
                cap = max(floor, float(self.solver_ranges.get("rigid_support_E_cap", floor)))
                old_E = float(mat.simulation_E)
                mat.raw_E = max(float(mat.raw_E), floor)
                mat.simulation_E = min(max(old_E, floor), cap)
                self._enable_rigid_projection(mat, strength=1.0)
                if mat.simulation_E != old_E:
                    mat.warnings.append(f"Rigid/support solver calibration changed simulation E from {old_E:g} to {mat.simulation_E:g}.")
                mat.warnings.append("Rigid support candidate explicitly enables rigid_project as a simulation action.")
            mats.append(mat)
        if not touched:
            return None
        return self._candidate("rigid_support_response", "Soft deformables with rigid/support stiffness prior", mats)

    def _is_rigid_or_support_part(self, part: PartEvidence, mat: CandidatePartMaterial) -> bool:
        text = f"{part.name} {part.physics_group} {part.physical_role}".lower()
        if any(token in text for token in ("plate", "dish", "tray", "stand", "support", "holder", "base support")):
            return True
        if mat.visual_material in {"Ceramic", "Metal", "Glass", "Stone"}:
            return True
        return mat.solver_material == "metal"

    def _candidate(self, candidate_id: str, description: str, parts: list[CandidatePartMaterial]) -> CandidateSet:
        if parts:
            largest = max(parts, key=lambda p: p.confidence)
            global_material = largest.solver_material
            global_E = largest.simulation_E
            global_nu = largest.simulation_nu
            global_density = largest.simulation_density
        else:
            material = "Plastic"
            global_material = solver_material_for_visual(material)
            global_E = default_E_for_material(material)
            global_nu = default_nu_for_material(material)
            global_density = density_for_material(material)
        return CandidateSet(
            candidate_id=candidate_id,
            description=description,
            parts=parts,
            global_material=global_material,
            global_E=float(global_E),
            global_nu=float(global_nu),
            global_density=float(global_density),
            score_prior=float(sum(p.confidence for p in parts) / max(1, len(parts))),
        )
