"""Human review decisions with durable history.

Decisions are stored per analysis session. They never mutate the uploaded file
or the in-memory original dataframe; sanitisation is applied only to a fresh
copy (or a new ZIP archive) at export time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import read_json, utc_now_iso, write_json

ACTIONS = {
    "approve_correction": "Approve suggested correction",
    "reject_correction": "Reject suggested correction",
    "retain_original": "Retain original label",
    "remove": "Remove sample from cleaned dataset",
    "uncertain": "Mark as uncertain",
    "clear": "Clear decision",
}


@dataclass
class ReviewDecision:
    sample_id: str
    action: str
    corrected_label: str | None = None
    original_label: str | None = None
    suggested_label: str | None = None
    notes: str = ""
    reviewer: str = ""
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReviewStore:
    """Session-scoped review decisions backed by JSON + JSONL history."""

    def __init__(self, session_id: str, workspace: Path, reviewer: str = "") -> None:
        self.session_id = session_id
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.workspace / "reviews.json"
        self.history_path = self.workspace / "reviews_history.jsonl"
        self.reviewer = reviewer
        self.decisions: dict[str, ReviewDecision] = {}
        self.history: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        payload = read_json(self.decisions_path, default={}) or {}
        for sample_id, item in (payload.get("decisions") or {}).items():
            try:
                self.decisions[str(sample_id)] = ReviewDecision(**item)
            except TypeError:
                continue
        if self.history_path.exists():
            import json

            try:
                with open(self.history_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            try:
                                self.history.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError:
                pass

    def _save(self) -> None:
        write_json(
            self.decisions_path,
            {
                "session_id": self.session_id,
                "updated_at": utc_now_iso(),
                "decisions": {sid: decision.to_dict() for sid, decision in self.decisions.items()},
            },
        )

    def set_decision(
        self,
        sample_id: str,
        action: str,
        corrected_label: str | None = None,
        original_label: str | None = None,
        suggested_label: str | None = None,
        notes: str = "",
        reviewer: str | None = None,
    ) -> ReviewDecision | None:
        if action not in ACTIONS:
            raise ValueError(f"Unknown review action '{action}'.")
        if action == "clear":
            self.decisions.pop(sample_id, None)
            self._append_history(
                {
                    "sample_id": sample_id,
                    "action": "clear",
                    "timestamp": utc_now_iso(),
                    "reviewer": reviewer if reviewer is not None else self.reviewer,
                }
            )
            self._save()
            return None
        if action == "approve_correction" and not corrected_label:
            raise ValueError("Approving a correction requires the corrected label.")

        decision = ReviewDecision(
            sample_id=sample_id,
            action=action,
            corrected_label=corrected_label if action == "approve_correction" else None,
            original_label=original_label,
            suggested_label=suggested_label,
            notes=notes.strip(),
            reviewer=(reviewer if reviewer is not None else self.reviewer).strip(),
        )
        self.decisions[sample_id] = decision
        self._append_history(decision.to_dict())
        self._save()
        return decision

    def _append_history(self, record: dict[str, Any]) -> None:
        import json

        self.history.append(record)
        with open(self.history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def get(self, sample_id: str) -> ReviewDecision | None:
        return self.decisions.get(sample_id)

    def count(self, action: str) -> int:
        return sum(1 for decision in self.decisions.values() if decision.action == action)

    def summary(self) -> dict[str, int]:
        result = {action: 0 for action in ACTIONS if action != "clear"}
        for decision in self.decisions.values():
            result[decision.action] = result.get(decision.action, 0) + 1
        result["reviewed"] = len(self.decisions)
        result["unreviewed_pending"] = 0  # filled by the caller when a queue is known
        return result

    def changes_frame(self) -> pd.DataFrame:
        if not self.decisions:
            return pd.DataFrame(
                columns=[
                    "sample_id", "action", "original_label", "effective_label",
                    "suggested_label", "notes", "reviewer", "timestamp",
                ]
            )
        rows = []
        for decision in self.decisions.values():
            if decision.action == "approve_correction":
                effective = decision.corrected_label
            elif decision.action == "remove":
                effective = None
            else:
                effective = decision.original_label
            rows.append(
                {
                    "sample_id": decision.sample_id,
                    "action": decision.action,
                    "original_label": decision.original_label,
                    "effective_label": effective,
                    "suggested_label": decision.suggested_label,
                    "notes": decision.notes,
                    "reviewer": decision.reviewer,
                    "timestamp": decision.timestamp,
                }
            )
        frame = pd.DataFrame(rows)
        return frame.sort_values("timestamp").reset_index(drop=True)

    def history_frame(self, limit: int = 1000) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame(columns=["sample_id", "action", "timestamp", "reviewer", "notes"])
        frame = pd.DataFrame(self.history)
        return frame.tail(limit).iloc[::-1].reset_index(drop=True)

    def effective_label(self, sample_id: str, original_label: str | None) -> str | None:
        decision = self.decisions.get(sample_id)
        if decision is None:
            return original_label
        if decision.action == "approve_correction":
            return decision.corrected_label
        return original_label

    def is_removed(self, sample_id: str) -> bool:
        decision = self.decisions.get(sample_id)
        return decision is not None and decision.action == "remove"
