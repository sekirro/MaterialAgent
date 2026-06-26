from __future__ import annotations

import math


PHYSGM_MATERIALS = [
    "Wood",
    "Metal",
    "Plastic",
    "Glass",
    "Fabric",
    "Leather",
    "Ceramic",
    "Stone",
    "Rubber",
    "Paper",
    "Sand",
    "Snow",
    "Plasticine",
    "Foam",
]

CLASS_TO_MATERIAL = {idx: name for idx, name in enumerate(PHYSGM_MATERIALS)}

E_MEAN = 7.387210
E_STD = 2.456477
NU_MEAN = 0.398
NU_STD = 0.111

MATERIAL_TO_DENSITY = {
    "Wood": 700.0,
    "Metal": 7800.0,
    "Plastic": 1200.0,
    "Glass": 2500.0,
    "Fabric": 500.0,
    "Leather": 900.0,
    "Ceramic": 2500.0,
    "Stone": 2600.0,
    "Rubber": 1100.0,
    "Paper": 800.0,
    "Sand": 1600.0,
    "Snow": 300.0,
    "Plasticine": 2000.0,
    "Foam": 100.0,
}

MATERIAL_TO_E_RANGE = {
    "Wood": (1e8, 2e10),
    "Metal": (1e9, 3e11),
    "Plastic": (1e6, 5e9),
    "Glass": (1e9, 1e11),
    "Fabric": (1e4, 1e8),
    "Leather": (1e5, 1e9),
    "Ceramic": (1e8, 1e11),
    "Stone": (1e8, 1e11),
    "Rubber": (1e4, 1e8),
    "Paper": (1e5, 1e9),
    "Sand": (1e3, 1e7),
    "Snow": (1e3, 1e7),
    "Plasticine": (1e3, 1e7),
    "Foam": (1e3, 1e7),
}

MATERIAL_TO_NU_RANGE = {
    "Wood": (0.25, 0.45),
    "Metal": (0.20, 0.35),
    "Plastic": (0.30, 0.45),
    "Glass": (0.18, 0.30),
    "Fabric": (0.20, 0.45),
    "Leather": (0.30, 0.49),
    "Ceramic": (0.15, 0.35),
    "Stone": (0.10, 0.35),
    "Rubber": (0.40, 0.499),
    "Paper": (0.20, 0.45),
    "Sand": (0.20, 0.45),
    "Snow": (0.10, 0.35),
    "Plasticine": (0.25, 0.49),
    "Foam": (0.10, 0.45),
}

VISUAL_TO_SOLVER_MATERIAL = {
    "Wood": "metal",
    "Metal": "metal",
    "Plastic": "plasticine",
    "Glass": "metal",
    "Fabric": "foam",
    "Leather": "foam",
    "Ceramic": "metal",
    "Stone": "metal",
    "Rubber": "jelly",
    "Paper": "foam",
    "Sand": "sand",
    "Snow": "snow",
    "Plasticine": "plasticine",
    "Foam": "foam",
}

SOLVER_MATERIAL_TO_ID = {
    "jelly": 0,
    "metal": 1,
    "sand": 2,
    "foam": 3,
    "snow": 4,
    "plasticine": 5,
}

ALIASES = {
    "wooden": "Wood",
    "timber": "Wood",
    "steel": "Metal",
    "iron": "Metal",
    "aluminum": "Metal",
    "aluminium": "Metal",
    "metallic": "Metal",
    "cloth": "Fabric",
    "textile": "Fabric",
    "rubbery": "Rubber",
    "rubber": "Rubber",
    "paperboard": "Paper",
    "cardboard": "Paper",
    "ceramics": "Ceramic",
    "clay": "Plasticine",
    "jelly": "Rubber",
}


def normalize_material(name: str | None) -> str:
    if not name:
        return "Plastic"
    text = str(name).strip()
    if not text:
        return "Plastic"
    key = text.lower().replace("_", " ").replace("-", " ").strip()
    if key in ALIASES:
        return ALIASES[key]
    for material in PHYSGM_MATERIALS:
        if key == material.lower():
            return material
    for material in PHYSGM_MATERIALS:
        if material.lower() in key:
            return material
    return "Plastic"


def density_for_material(material: str | None) -> float:
    return MATERIAL_TO_DENSITY[normalize_material(material)]


def default_E_for_material(material: str | None) -> float:
    lo, hi = MATERIAL_TO_E_RANGE[normalize_material(material)]
    return math.sqrt(lo * hi)


def default_nu_for_material(material: str | None) -> float:
    lo, hi = MATERIAL_TO_NU_RANGE[normalize_material(material)]
    return (lo + hi) * 0.5


def solver_material_for_visual(material: str | None) -> str:
    return VISUAL_TO_SOLVER_MATERIAL[normalize_material(material)]


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def clamp_physical_values(material: str | None, E: float, nu: float) -> tuple[float, float, list[str]]:
    material = normalize_material(material)
    warnings: list[str] = []
    e_lo, e_hi = MATERIAL_TO_E_RANGE[material]
    nu_lo, nu_hi = MATERIAL_TO_NU_RANGE[material]
    if not math.isfinite(float(E)) or float(E) <= 0:
        E = default_E_for_material(material)
        warnings.append(f"Invalid E replaced by {E:g} for {material}.")
    if not math.isfinite(float(nu)):
        nu = default_nu_for_material(material)
        warnings.append(f"Invalid nu replaced by {nu:g} for {material}.")
    new_E = clamp(float(E), e_lo, e_hi)
    new_nu = clamp(float(nu), nu_lo, min(nu_hi, 0.499))
    if new_E != float(E):
        warnings.append(f"E clamped from {float(E):g} to {new_E:g} for {material}.")
    if new_nu != float(nu):
        warnings.append(f"nu clamped from {float(nu):g} to {new_nu:g} for {material}.")
    return new_E, new_nu, warnings


def clamp_solver_values(
    E: float,
    nu: float,
    density: float,
    E_range=(1.0e3, 2.0e6),
    nu_range=(0.05, 0.45),
    density_range=(50.0, 3000.0),
) -> tuple[float, float, float, list[str]]:
    warnings: list[str] = []
    new_E = clamp(float(E), E_range[0], E_range[1])
    new_nu = clamp(float(nu), nu_range[0], min(nu_range[1], 0.49))
    new_density = clamp(float(density), density_range[0], density_range[1])
    if new_E != float(E):
        warnings.append(f"Simulation E clamped from {float(E):g} to {new_E:g}.")
    if new_nu != float(nu):
        warnings.append(f"Simulation nu clamped from {float(nu):g} to {new_nu:g}.")
    if new_density != float(density):
        warnings.append(f"Simulation density clamped from {float(density):g} to {new_density:g}.")
    return new_E, new_nu, new_density, warnings


def role_material_candidates(part_name: str, physical_role: str = "") -> list[tuple[str, float, str]]:
    text = f"{part_name} {physical_role}".lower()
    candidates: list[tuple[str, float, str]] = []
    if any(k in text for k in ("head", "blade", "tip", "edge", "impact", "metal")):
        candidates.append(("Metal", 0.75, "impact/head/blade role"))
    if any(k in text for k in ("handle", "grip", "shaft")):
        candidates.extend([("Wood", 0.55, "handle/grip role"), ("Plastic", 0.45, "handle/grip role")])
    if any(k in text for k in ("wheel", "tire", "tyre", "sole")):
        candidates.append(("Rubber", 0.8, "wheel/tire/sole role"))
    if any(k in text for k in ("cushion", "pad", "pillow", "foam")):
        candidates.append(("Foam", 0.75, "cushion/padding role"))
    if any(k in text for k in ("cloth", "fabric", "lace", "upper")):
        candidates.append(("Fabric", 0.65, "fabric-like role"))
    if any(k in text for k in ("plate", "bowl", "ceramic")):
        candidates.extend([("Ceramic", 0.55, "plate/support role"), ("Plastic", 0.35, "plate/support role")])
    if any(k in text for k in ("frosting", "cream", "cake", "soft")):
        candidates.extend([("Foam", 0.55, "soft food/coating role"), ("Plasticine", 0.35, "soft food/coating role")])
    return candidates

