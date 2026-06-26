from __future__ import annotations

from pathlib import Path

from .schemas import CandidateSet, SceneEvidence


def write_report(path: str | Path, scene: SceneEvidence, selected: CandidateSet, selection: dict, scores: list[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MaterialAgent Report",
        "",
        f"Scene: `{scene.scene_dir}`",
        f"Object: `{scene.object_name}`",
        f"Selected candidate: `{selected.candidate_id}`",
        f"Score: `{selection.get('score')}`",
        "",
        "## Selected Parts",
        "",
    ]
    for part in selected.parts:
        lines.append(
            f"- part {part.part_id} `{part.part_name}`: {part.visual_material} "
            f"(solver {part.solver_material}), raw E={part.raw_E:.4g}, raw nu={part.raw_nu:.4g}, "
            f"sim E={part.simulation_E:.4g}, sim nu={part.simulation_nu:.4g}"
        )
    lines.extend(["", "## Candidate Scores", ""])
    for score in scores:
        lines.append(f"- `{score.get('candidate_id')}`: score={score.get('score')}, ok={score.get('ok')}, reasons={score.get('reasons')}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p

