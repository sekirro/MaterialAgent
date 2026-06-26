from __future__ import annotations

import copy
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import CLASS_TO_MATERIAL, E_MEAN, E_STD, NU_MEAN, NU_STD, density_for_material, normalize_material
from ..io_utils import ensure_dir, write_json
from ..schemas import DistributionOutput, PartEvidence, SceneEvidence
from .crops import build_part_crops


def decode_E(E_mu_norm: float) -> float:
    return float((10 ** (float(E_mu_norm) * E_STD + E_MEAN)) * 0.1)


def decode_nu(nu_mu_norm: float) -> float:
    return float(float(nu_mu_norm) * NU_STD + NU_MEAN)


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / max(float(exp.sum()), 1e-12)


class PhysGMDistributionExtractor:
    def __init__(
        self,
        physgm_root: str | Path,
        partphys_root: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        amp_dtype: str = "bf16",
        mock: bool = False,
        crop_variants: list[str] | None = None,
    ):
        self.physgm_root = Path(physgm_root).expanduser().resolve()
        self.partphys_root = Path(partphys_root).expanduser().resolve()
        self.config_path = self._resolve(config_path)
        self.checkpoint_path = self._resolve(checkpoint_path)
        self.device = device
        self.amp_dtype = amp_dtype
        self.mock = bool(mock)
        self.crop_variants = tuple(crop_variants or [])
        self._loaded = False
        self.model = None
        self.base_config = None
        self.Dataset = None
        self.DataLoader = None
        self.torch = None
        self.build_scene = None

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        if p.exists():
            return p.resolve()
        q = self.physgm_root / p
        if q.exists():
            return q.resolve()
        return p

    def available(self) -> bool:
        return self.mock or (self.physgm_root.exists() and self.config_path.exists() and self.checkpoint_path.exists())

    def _load(self) -> None:
        if self._loaded or self.mock:
            return
        if not self.available():
            raise FileNotFoundError(f"PhysGM config/checkpoint unavailable: {self.config_path}, {self.checkpoint_path}")
        for path in [str(self.partphys_root), str(self.physgm_root)]:
            if path not in sys.path:
                sys.path.insert(0, path)
        import torch  # type: ignore
        import yaml  # type: ignore
        from easydict import EasyDict as edict  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
        from data.dataset_infer import Dataset  # type: ignore
        from model.physgm import PhysGM  # type: ignore
        from partphys.scene_builder import build_physgm_input_scene  # type: ignore

        with self.config_path.open("r", encoding="utf-8") as f:
            self.base_config = edict(yaml.safe_load(f))
        self.base_config.evaluation = True
        self.torch = torch
        self.Dataset = Dataset
        self.DataLoader = DataLoader
        self.build_scene = build_physgm_input_scene
        self.model = PhysGM(self.base_config, device=self.device).to(self.device)
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        self.model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        self.model.eval()
        self._loaded = True

    def extract_for_scene(self, scene: SceneEvidence, output_dir: str | Path) -> dict[int, list[DistributionOutput]]:
        output = ensure_dir(output_dir)
        all_outputs: dict[int, list[DistributionOutput]] = {}
        for part in scene.parts:
            if part.mask_path is None:
                continue
            part_output = ensure_dir(output / f"part_{part.part_id:03d}_{part.name}")
            existing_crops = self._existing_crops(part)
            crops = existing_crops or build_part_crops(scene, part, part_output / "crops")
            crops = self._filter_crops(crops)
            outputs: list[DistributionOutput] = []
            for variant, crop_path in crops.items():
                run_dir = ensure_dir(part_output / "physgm_outputs" / f"{variant}")
                try:
                    outputs.append(self.extract_one(part, variant, crop_path, run_dir))
                except Exception as exc:
                    outputs.append(self._mock_or_error(part, variant, crop_path, str(exc)))
            all_outputs[part.part_id] = outputs
            write_json(part_output / "distribution_outputs.json", [x.to_dict() for x in outputs])
        write_json(output / "part_distribution_outputs.json", {str(k): [x.to_dict() for x in v] for k, v in all_outputs.items()})
        return all_outputs


    def _filter_crops(self, crops: dict[str, str]) -> dict[str, str]:
        if not self.crop_variants:
            return crops
        selected = {name: crops[name] for name in self.crop_variants if name in crops}
        if selected:
            return selected
        return crops

    def _existing_crops(self, part: PartEvidence) -> dict[str, str]:
        if not part.part_dir:
            return {}
        root = Path(part.part_dir)
        mapping = {}
        for variant in ["tight", "padded", "context_dim", "isolated_full"]:
            path = root / f"crop_{variant}.png"
            if path.exists():
                mapping[variant] = str(path)
        return mapping

    def _mock_or_error(self, part: PartEvidence, variant: str, image_path: str, warning: str) -> DistributionOutput:
        material = normalize_material(part.expected_materials[0] if part.expected_materials else None)
        E = 1e5
        nu = 0.35
        probs = {m: 0.0 for m in CLASS_TO_MATERIAL.values()}
        probs[material] = 1.0
        return DistributionOutput(
            part_id=part.part_id,
            part_name=part.name,
            variant=variant,
            image_path=str(image_path),
            material_probs=probs,
            material=material,
            E_mu_norm=(math.log10(max(E / 0.1, 1.0)) - E_MEAN) / E_STD,
            E_var_norm=1.0,
            nu_mu_norm=(nu - NU_MEAN) / NU_STD,
            nu_var_norm=1.0,
            E_mean=E,
            E_sigma_log10=E_STD,
            nu_mean=nu,
            nu_sigma=NU_STD,
            warnings=[warning],
        )

    def extract_one(self, part: PartEvidence, variant: str, image_path: str, output_dir: str | Path) -> DistributionOutput:
        if self.mock:
            return self._mock_or_error(part, variant, image_path, "mock PhysGM distribution")
        self._load()
        assert self.torch is not None and self.model is not None and self.base_config is not None
        assert self.Dataset is not None and self.DataLoader is not None and self.build_scene is not None
        torch = self.torch
        output = ensure_dir(output_dir)
        pose = self.physgm_root / "example_data" / "cake" / "pose.json"
        scene_info = self.build_scene(
            image_path,
            output / "input_scene",
            f"material_agent_part_{part.part_id:03d}_{variant}",
            template_pose_json=str(pose) if pose.exists() else None,
            duplicate_single_image=True,
            size=512,
        )
        config = copy.deepcopy(self.base_config)
        config.data.data_path = scene_info["data_txt"]
        dataset = self.Dataset(config)
        dataloader = self.DataLoader(dataset, batch_size=1, shuffle=False)
        amp = torch.float16 if self.amp_dtype == "fp16" else torch.bfloat16
        autocast_cm = torch.autocast(device_type="cuda", dtype=amp) if "cuda" in str(self.device) else nullcontext()
        with torch.no_grad():
            batch = next(iter(dataloader))
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            with autocast_cm:
                ret = self.model(batch)
            E_mu_norm = float(ret["E_mu"][0].float().item())
            E_var_norm = float(ret.get("E_var", torch.ones_like(ret["E_mu"]))[0].float().item())
            nu_mu_norm = float(ret["nu_mu"][0].float().item())
            nu_var_norm = float(ret.get("nu_var", torch.ones_like(ret["nu_mu"]))[0].float().item())
            logits = ret["phys_logits"][0].float().detach().cpu().numpy()
        probs_arr = softmax(logits)
        material_probs = {CLASS_TO_MATERIAL[idx]: float(probs_arr[idx]) for idx in range(min(len(probs_arr), len(CLASS_TO_MATERIAL)))}
        material = normalize_material(max(material_probs, key=material_probs.get))
        E_mean = decode_E(E_mu_norm)
        nu_mean = decode_nu(nu_mu_norm)
        E_sigma_log10 = float(max(0.05, math.sqrt(max(E_var_norm, 1e-8)) * E_STD))
        nu_sigma = float(max(0.005, math.sqrt(max(nu_var_norm, 1e-8)) * NU_STD))
        predicted = output / "predicted_phys_distribution.json"
        data = {
            "material": material,
            "material_probs": material_probs,
            "E": E_mean,
            "nu": nu_mean,
            "density": density_for_material(material),
            "E_mu_norm": E_mu_norm,
            "E_var_norm": E_var_norm,
            "nu_mu_norm": nu_mu_norm,
            "nu_var_norm": nu_var_norm,
            "E_sigma_log10": E_sigma_log10,
            "nu_sigma": nu_sigma,
            "input_scene": scene_info,
        }
        write_json(predicted, data)
        return DistributionOutput(
            part_id=part.part_id,
            part_name=part.name,
            variant=variant,
            image_path=str(image_path),
            material_probs=material_probs,
            material=material,
            E_mu_norm=E_mu_norm,
            E_var_norm=E_var_norm,
            nu_mu_norm=nu_mu_norm,
            nu_var_norm=nu_var_norm,
            E_mean=E_mean,
            E_sigma_log10=E_sigma_log10,
            nu_mean=nu_mean,
            nu_sigma=nu_sigma,
            predicted_phys_path=str(predicted),
        )

