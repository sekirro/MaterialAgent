from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..constants import normalize_material
from ..io_utils import read_json
from ..schemas import PartEvidence, SceneEvidence


def _part_name_key(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _part_id_from_dir(path: Path) -> int | None:
    import re

    match = re.search(r"part_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _existing_path(path: str | Path | None, base: Path | None = None) -> Path | None:
    if not path:
        return None
    p = Path(path).expanduser()
    candidates = [p]
    if base is not None and not p.is_absolute():
        candidates.insert(0, base / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


class PartPhysSceneLoader:
    def __init__(self, scene_dir: str | Path):
        self.scene_dir = Path(scene_dir).expanduser().resolve()

    def load(self) -> SceneEvidence:
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"PartPhys scene not found: {self.scene_dir}")

        summary = read_json(self.scene_dir / "partphys_summary.json", {}) or {}
        schema = read_json(self.scene_dir / "schema" / "part_schema.json", {}) or {}
        assignment_summary = read_json(self.scene_dir / "assignment" / "assignment_summary.json", {}) or {}
        whole_info = summary.get("whole_physgm") or {}
        local_whole_dir = self.scene_dir / "physgm_whole"
        point_cloud = _existing_path(local_whole_dir / "point_clouds.ply") or _existing_path(whole_info.get("point_cloud_path"), self.scene_dir)
        whole_dir = _existing_path(whole_info.get("scene_dir"), self.scene_dir) or _existing_path(local_whole_dir)
        if point_cloud and point_cloud.parent.exists():
            whole_dir = point_cloud.parent
        predicted_phys = _existing_path(local_whole_dir / "predicted_phys.json") or _existing_path(
            whole_info.get("predicted_phys_path"),
            self.scene_dir,
        )
        whole_pred = read_json(predicted_phys, {}) if predicted_phys else {}
        if not whole_pred and isinstance(whole_info.get("raw"), dict):
            whole_pred = dict(whole_info["raw"])
        if not whole_pred and isinstance(whole_info, dict):
            whole_pred = {k: whole_info[k] for k in ["material", "E", "nu", "density"] if k in whole_info}
        aabbs = read_json(self.scene_dir / "assignment" / "per_part_aabb.json", []) or []
        aabb_by_id = {int(x["part_id"]): x for x in aabbs if "part_id" in x}
        count_by_id = self._counts_from_assignment(assignment_summary)

        object_name = summary.get("object_name") or schema.get("object") or "object"
        parts = self._load_parts(summary, schema, aabb_by_id, count_by_id)
        warnings: list[str] = []
        if not parts:
            warnings.append("No parts found in PartPhys output.")
        if not point_cloud:
            warnings.append("Missing physgm_whole/point_clouds.ply.")
        if not (self.scene_dir / "assignment" / "gaussian_part_ids.npy").exists():
            warnings.append("Missing assignment/gaussian_part_ids.npy.")
        if not aabbs:
            warnings.append("Missing or empty assignment/per_part_aabb.json.")

        return SceneEvidence(
            scene_dir=str(self.scene_dir),
            object_name=object_name,
            input_image=self._first_existing(["input/input.png", "input/object_crop.png"]),
            object_image=self._first_existing(["input/object_isolated_full.png", "input/object_crop_white_bg.png", "input/input.png"]),
            part_schema_path=str(self.scene_dir / "schema" / "part_schema.json") if (self.scene_dir / "schema" / "part_schema.json").exists() else None,
            parts=parts,
            whole_physgm_dir=str(whole_dir) if whole_dir else None,
            point_cloud_path=str(point_cloud) if point_cloud else None,
            predicted_phys_path=str(predicted_phys) if predicted_phys else None,
            assignment_dir=str(self.scene_dir / "assignment") if (self.scene_dir / "assignment").exists() else None,
            gaussian_part_ids_path=str(self.scene_dir / "assignment" / "gaussian_part_ids.npy") if (self.scene_dir / "assignment" / "gaussian_part_ids.npy").exists() else None,
            part_aabbs_path=str(self.scene_dir / "assignment" / "per_part_aabb.json") if (self.scene_dir / "assignment" / "per_part_aabb.json").exists() else None,
            assignment_summary=assignment_summary,
            whole_physics=whole_pred,
            warnings=warnings,
        )

    def _first_existing(self, rel_paths: list[str]) -> str | None:
        for rel in rel_paths:
            path = self.scene_dir / rel
            if path.exists():
                return str(path)
        return None

    def _counts_from_assignment(self, assignment_summary: dict[str, Any]) -> dict[int, int]:
        counts: dict[int, int] = {}
        for key, value in (assignment_summary.get("per_part_counts") or {}).items():
            try:
                counts[int(key)] = int(value)
            except Exception:
                continue
        index_path = assignment_summary.get("part_gaussian_index")
        if index_path and Path(index_path).exists():
            index = read_json(index_path, {}) or {}
            for key, values in index.items():
                try:
                    counts[int(key)] = len(values)
                except Exception:
                    continue
        return counts

    def _schema_parts_by_name(self, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for part in schema.get("parts", []) or []:
            name = part.get("name")
            if name:
                out[_part_name_key(name)] = part
        return out

    def _load_parts(
        self,
        summary: dict[str, Any],
        schema: dict[str, Any],
        aabb_by_id: dict[int, dict[str, Any]],
        count_by_id: dict[int, int],
    ) -> list[PartEvidence]:
        schema_by_name = self._schema_parts_by_name(schema)
        raw_parts = summary.get("parts") or read_json(self.scene_dir / "parts" / "selection_summary.json", {}).get("parts", [])
        if not raw_parts:
            raw_parts = []
            for part_dir in sorted((self.scene_dir / "parts").glob("part_*")):
                if not part_dir.is_dir():
                    continue
                part_summary = read_json(part_dir / "part_summary.json", {}) or {}
                part = part_summary.get("part")
                if part:
                    raw_parts.append(part)
                else:
                    pid = _part_id_from_dir(part_dir)
                    if pid is not None:
                        raw_parts.append({"part_id": pid, "name": part_dir.name, "mask_path": str(part_dir / "mask.png")})

        parts: list[PartEvidence] = []
        seen: set[int] = set()
        for raw in raw_parts:
            try:
                pid = int(raw.get("part_id"))
            except Exception:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            name = str(raw.get("name") or raw.get("part_name") or f"part_{pid}")
            spec = schema_by_name.get(_part_name_key(name), {})
            part_dir = Path(raw.get("mask_path", "")).parent if raw.get("mask_path") else self.scene_dir / "parts" / f"part_{pid:03d}_{name}"
            if not part_dir.is_absolute():
                part_dir = (self.scene_dir / part_dir).resolve()
            part_summary = read_json(part_dir / "part_summary.json", {}) or {}
            aabb = aabb_by_id.get(pid, {})
            mask_path = raw.get("mask_path") or str(part_dir / "mask.png")
            mask = Path(mask_path)
            if not mask.is_absolute():
                mask = (self.scene_dir / mask).resolve()
            expected = raw.get("expected_materials") or spec.get("expected_materials") or []
            parts.append(
                PartEvidence(
                    part_id=pid,
                    name=name,
                    physics_group=str(raw.get("physics_group") or spec.get("physics_group") or name),
                    mask_path=str(mask) if mask.exists() else None,
                    area=int(raw.get("area") or 0),
                    confidence=float(raw.get("confidence") or 0.0),
                    expected_materials=[normalize_material(x) for x in expected],
                    physical_role=str(spec.get("physical_role") or raw.get("physical_role") or ""),
                    view_masks={str(k): str(v) for k, v in ((raw.get("metadata") or {}).get("view_masks") or {}).items()},
                    gaussian_count=int(aabb.get("count") or count_by_id.get(pid, 0)),
                    aabb_center=[float(x) for x in aabb.get("center", [])] or None,
                    aabb_half_size=[float(x) for x in aabb.get("half_size", [])] or None,
                    part_dir=str(part_dir),
                    part_summary=part_summary,
                    metadata={"schema": spec, "raw": raw, "aabb": aabb},
                )
            )
        return sorted(parts, key=lambda p: p.part_id)


def is_residual_part(part: PartEvidence) -> bool:
    text = f"{part.name} {part.physics_group}".lower()
    return "unknown" in text or "residual" in text or part.physics_group.lower() in {"global_body", "unknown", "residual"}


def load_gaussian_part_ids(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return np.load(p).astype(np.int32)
