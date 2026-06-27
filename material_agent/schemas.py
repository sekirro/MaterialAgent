from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PartEvidence:
    part_id: int
    name: str
    physics_group: str
    mask_path: str | None = None
    area: int = 0
    confidence: float = 0.0
    expected_materials: list[str] = field(default_factory=list)
    physical_role: str = ""
    view_masks: dict[str, str] = field(default_factory=dict)
    gaussian_count: int = 0
    aabb_center: list[float] | None = None
    aabb_half_size: list[float] | None = None
    part_dir: str | None = None
    part_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneEvidence:
    scene_dir: str
    object_name: str = "object"
    input_image: str | None = None
    object_image: str | None = None
    part_schema_path: str | None = None
    parts: list[PartEvidence] = field(default_factory=list)
    whole_physgm_dir: str | None = None
    point_cloud_path: str | None = None
    predicted_phys_path: str | None = None
    assignment_dir: str | None = None
    gaussian_part_ids_path: str | None = None
    part_aabbs_path: str | None = None
    assignment_summary: dict[str, Any] = field(default_factory=dict)
    whole_physics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def path(self) -> Path:
        return Path(self.scene_dir)


@dataclass
class DistributionOutput:
    part_id: int
    part_name: str
    variant: str
    image_path: str
    material_probs: dict[str, float]
    material: str
    E_mu_norm: float
    E_var_norm: float
    nu_mu_norm: float
    nu_var_norm: float
    E_mean: float
    E_sigma_log10: float
    nu_mean: float
    nu_sigma: float
    predicted_phys_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PartPosterior:
    part_id: int
    part_name: str
    material_probs: dict[str, float]
    selected_material: str
    material_confidence: float
    logE_mean: float
    logE_std: float
    nu_mean: float
    nu_std: float
    density: float
    confidence: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePartMaterial:
    part_id: int
    part_name: str
    visual_material: str
    solver_material: str
    raw_E: float
    raw_nu: float
    raw_density: float
    simulation_E: float
    simulation_nu: float
    simulation_density: float
    confidence: float
    source: str
    rigid_project: bool = False
    rigid_project_strength: float = 1.0
    interface_bond: bool = False
    interface_bond_radius: float = 0.035
    interface_bond_strength: float = 0.75
    interface_bond_velocity_blend: float = 0.75
    interface_bond_max_particles: int = 25000
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateSet:
    candidate_id: str
    description: str
    parts: list[CandidatePartMaterial]
    global_material: str
    global_E: float
    global_nu: float
    global_density: float
    score_prior: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

