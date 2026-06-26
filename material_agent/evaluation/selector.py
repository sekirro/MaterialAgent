from __future__ import annotations

from ..schemas import CandidateSet


class CandidateSelector:
    def select(self, candidates: list[CandidateSet], scores: list[dict], manual_candidate: str | None = None) -> tuple[CandidateSet, dict]:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        score_by_id = {item["candidate_id"]: item for item in scores}
        if manual_candidate:
            if manual_candidate not in by_id:
                raise ValueError(f"Manual candidate not found: {manual_candidate}")
            score = score_by_id.get(manual_candidate, {"candidate_id": manual_candidate, "score": 1.0, "ok": True, "reasons": ["manual selection"]})
            return by_id[manual_candidate], {**score, "selection_mode": "manual"}
        valid = [item for item in scores if item.get("ok", False)]
        if not valid:
            best = max(scores, key=lambda item: float(item.get("score", -999))) if scores else {"candidate_id": candidates[0].candidate_id, "score": -1.0}
        else:
            best = max(valid, key=lambda item: float(item.get("score", -999)))
        return by_id[best["candidate_id"]], {**best, "selection_mode": "auto"}

