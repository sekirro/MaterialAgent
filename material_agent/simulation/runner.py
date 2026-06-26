from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from ..io_utils import ensure_dir, write_json
from ..schemas import CandidateSet, SceneEvidence


class SimulationRunner:
    def __init__(
        self,
        physgm_root: str | Path,
        partphys_root: str | Path,
        render_img: bool = True,
        compile_video: bool = True,
        white_bg: bool = True,
        timeout_sec: int = 1800,
        mock: bool = False,
    ):
        self.physgm_root = Path(physgm_root).expanduser().resolve()
        self.partphys_root = Path(partphys_root).expanduser().resolve()
        self.render_img = bool(render_img)
        self.compile_video = bool(compile_video)
        self.white_bg = bool(white_bg)
        self.timeout_sec = int(timeout_sec)
        self.mock = bool(mock)

    def run_candidate(self, scene: SceneEvidence, candidate: CandidateSet, compiled: dict, output_dir: str | Path) -> dict:
        output = ensure_dir(output_dir)
        if self.mock:
            result = {
                "candidate_id": candidate.candidate_id,
                "status": "mock",
                "returncode": 0,
                "command": "mock simulation",
                "output_path": str(output),
                "video_path": None,
                "runtime_sec": 0.0,
                "stdout": None,
                "stderr": None,
            }
            write_json(output / "run_result.json", result)
            return result
        if not scene.whole_physgm_dir:
            raise RuntimeError("Scene has no whole PhysGM directory.")
        backend = compiled["backend"]
        cmd = self._command(scene, backend, compiled, output)
        env = os.environ.copy()
        env["PHYSGM_ROOT"] = str(self.physgm_root)
        env["PYTHONPATH"] = f"{self.physgm_root}:{self.partphys_root}:{env.get('PYTHONPATH', '')}"
        stdout_path = output / "stdout.txt"
        stderr_path = output / "stderr.txt"
        start = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(self.physgm_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_sec,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        video_path = self._find_video(output)
        result = {
            "candidate_id": candidate.candidate_id,
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": int(proc.returncode),
            "command": " ".join(cmd),
            "backend": backend,
            "output_path": str(output),
            "video_path": str(video_path) if video_path else None,
            "runtime_sec": time.time() - start,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        write_json(output / "run_result.json", result)
        return result

    def _command(self, scene: SceneEvidence, backend: str, compiled: dict, output: Path) -> list[str]:
        if backend == "part_id":
            script = self.partphys_root / "tools" / "gs_simulation_partid_materials.py"
            cmd = [
                sys.executable,
                str(script),
                "--model_path",
                str(scene.whole_physgm_dir),
                "--output_path",
                str(output),
                "--config",
                compiled["config_path"],
                "--part_ids",
                str(scene.gaussian_part_ids_path),
                "--part_materials_json",
                str(compiled["part_materials_json"]),
            ]
        else:
            script = self.physgm_root / "gs_simulation.py"
            cmd = [
                sys.executable,
                str(script),
                "--model_path",
                str(scene.whole_physgm_dir),
                "--output_path",
                str(output),
                "--config",
                compiled["config_path"],
            ]
        if self.render_img:
            cmd.append("--render_img")
        if self.compile_video:
            cmd.append("--compile_video")
        if self.white_bg:
            cmd.append("--white_bg")
        return cmd

    def _find_video(self, output: Path) -> Path | None:
        videos = sorted(list(output.rglob("*.mp4")) + list(output.rglob("*.avi")) + list(output.rglob("*.mov")))
        return videos[0] if videos else None

