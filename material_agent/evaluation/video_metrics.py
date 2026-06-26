from __future__ import annotations

from pathlib import Path

from ..schemas import CandidateSet


BAD_STDERR_TOKENS = ["nan", "inf", "cuda error", "traceback", "runtimeerror"]


class VideoEvaluator:
    def evaluate(self, candidate: CandidateSet, run_result: dict) -> dict:
        score = 0.0
        reasons: list[str] = []
        failed = False
        if run_result.get("returncode", 1) != 0:
            failed = True
            reasons.append("simulation returned non-zero")
        stderr_path = run_result.get("stderr")
        stderr = ""
        if stderr_path and Path(stderr_path).exists():
            stderr = Path(stderr_path).read_text(encoding="utf-8", errors="ignore").lower()
        if any(token in stderr for token in BAD_STDERR_TOKENS):
            failed = True
            reasons.append("stderr contains numerical/runtime error token")
        video_path = run_result.get("video_path")
        if run_result.get("status") != "mock" and (not video_path or not Path(video_path).exists()):
            reasons.append("video missing")
            score -= 0.15
        elif video_path and Path(video_path).exists():
            size = Path(video_path).stat().st_size
            if size < 4096:
                reasons.append("video file is too small")
                score -= 0.2
            else:
                score += 0.2
                reasons.append("video exists")
        if failed:
            return {"candidate_id": candidate.candidate_id, "ok": False, "score": -1.0, "reasons": reasons}

        avg_conf = sum(part.confidence for part in candidate.parts) / max(1, len(candidate.parts))
        diversity = len({p.visual_material for p in candidate.parts}) / max(1, len(candidate.parts))
        stiffness_order = self._stiffness_role_score(candidate)
        score += 0.35 * avg_conf + 0.15 * diversity + 0.30 * stiffness_order + 0.20 * candidate.score_prior
        reasons.append(f"avg material confidence {avg_conf:.3f}")
        reasons.append(f"material diversity {diversity:.3f}")
        reasons.append(f"role stiffness score {stiffness_order:.3f}")
        return {"candidate_id": candidate.candidate_id, "ok": True, "score": float(score), "reasons": reasons}

    def _stiffness_role_score(self, candidate: CandidateSet) -> float:
        if not candidate.parts:
            return 0.5
        score = 0.0
        for part in candidate.parts:
            name = part.part_name.lower()
            if any(k in name for k in ("head", "blade", "tip", "plate", "support")):
                score += 1.0 if part.raw_E >= 1e8 else 0.4
            elif any(k in name for k in ("rubber", "sole", "tire", "wheel", "foam", "cushion")):
                score += 1.0 if part.raw_E <= 1e8 else 0.5
            else:
                score += 0.7
        return score / max(1, len(candidate.parts))

