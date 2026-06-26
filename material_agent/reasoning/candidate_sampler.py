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
        E_range=tuple(solver_ranges.get("local_E_range", (1.0e3, 2.0e6))),
        nu_range=tuple(solver_ranges.get("local_nu_range", (0.05, 0.45))),
        density_range=tuple(solver_ranges.get("local_density_range", (50.0, 3000.0))),
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
        active_parts = [p for p in scene.parts if p.part_id in posteriors and not is_residual_part(p)]
        if not active_parts:
            active_parts = [p for p in scene.parts if p.part_id in posteriors]
        candidates: list[CandidateSet] = []
        candidates.append(self._build("posterior_map", "MAP material and median E/nu", active_parts, posteriors, 0.50, 0.50))
        food_stable = self._food_stable_candidate(scene, active_parts, posteriors)
        if food_stable:
            candidates.append(food_stable)
        candidates.append(self._build("soft_response", "Lower E for deformable response", active_parts, posteriors, 0.25, 0.75))
        candidates.append(self._build("stiff_response", "Higher E for rigid/support response", active_parts, posteriors, 0.75, 0.50))
        sweep = self._uncertain_sweep(active_parts, posteriors)
        if sweep:
            candidates.append(sweep)
        alternative = self._alternative_material(active_parts, posteriors)
        if alternative:
            candidates.append(alternative)
        return candidates[: self.budget]

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
            if len(sorted_probs) >= 2 and sorted_probs[1][1] > alt_prob:
                target = part
                alt_material = sorted_probs[1][0]
                alt_prob = sorted_probs[1][1]
        if target is None or alt_material is None or alt_prob < 0.12:
            return None
        mats = []
        for part in parts:
            material = alt_material if part.part_id == target.part_id else None
            mats.append(_part_material(part, posteriors[part.part_id], material, 0.50, 0.50, "alternative_material", self.solver_ranges))
        return self._candidate("alternative_material", f"Try second material {alt_material} for {target.name}", mats)

    def _food_stable_candidate(
        self,
        scene: SceneEvidence,
        parts: list[PartEvidence],
        posteriors: dict[int, PartPosterior],
    ) -> CandidateSet | None:
        text = " ".join([scene.object_name] + [p.name for p in parts]).lower()
        if not any(token in text for token in ("cake", "icing", "frosting", "cream", "strawberr", "dessert")):
            return None
        mats = []
        for part in parts:
            name = part.name.lower()
            if any(token in name for token in ("plate", "dish", "tray")):
                mats.append(
                    self._manual_part(
                        part,
                        posteriors[part.part_id],
                        visual_material="Ceramic",
                        solver_material="metal",
                        E=1.0e8,
                        nu=0.25,
                        density=2800.0,
                        source="food_stable_particle",
                    )
                )
            elif any(token in name for token in ("straw", "berry", "topping", "fruit")):
                mats.append(
                    self._manual_part(
                        part,
                        posteriors[part.part_id],
                        visual_material="Rubber",
                        solver_material="jelly",
                        E=9.0e4,
                        nu=0.42,
                        density=650.0,
                        source="food_stable_particle",
                    )
                )
            elif any(token in name for token in ("frost", "cream", "icing", "swirl", "drip")):
                mats.append(
                    self._manual_part(
                        part,
                        posteriors[part.part_id],
                        visual_material="Plasticine",
                        solver_material="plasticine",
                        E=1.2e4,
                        nu=0.45,
                        density=180.0,
                        source="food_stable_particle",
                    )
                )
            elif "cake" in name or "body" in name or "base" in name:
                mats.append(
                    self._manual_part(
                        part,
                        posteriors[part.part_id],
                        visual_material="Plasticine",
                        solver_material="plasticine",
                        E=3.0e4,
                        nu=0.43,
                        density=300.0,
                        source="food_stable_particle",
                    )
                )
            else:
                mats.append(_part_material(part, posteriors[part.part_id], None, 0.25, 0.75, "food_stable_particle", self.solver_ranges))
        return self._candidate("food_stable_particle", "Food-object stable per-particle material proxy", mats)

    def _manual_part(
        self,
        part: PartEvidence,
        posterior: PartPosterior,
        visual_material: str,
        solver_material: str,
        E: float,
        nu: float,
        density: float,
        source: str,
    ) -> CandidatePartMaterial:
        sim_E, sim_nu, sim_density, sim_warnings = clamp_solver_values(
            E,
            nu,
            density,
            E_range=tuple(self.solver_ranges.get("local_E_range", (1.0e3, 2.0e6))),
            nu_range=tuple(self.solver_ranges.get("local_nu_range", (0.05, 0.45))),
            density_range=tuple(self.solver_ranges.get("local_density_range", (50.0, 3000.0))),
        )
        return CandidatePartMaterial(
            part_id=part.part_id,
            part_name=part.name,
            visual_material=visual_material,
            solver_material=solver_material,
            raw_E=float(E),
            raw_nu=float(nu),
            raw_density=float(density),
            simulation_E=sim_E,
            simulation_nu=sim_nu,
            simulation_density=sim_density,
            confidence=posterior.confidence,
            source=source,
            warnings=["Food stable proxy parameters for per-particle MPM."] + sim_warnings,
        )

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
