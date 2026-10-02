"""Data models for predictions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass
class Prediction:
    """A single forecast prediction."""

    id: int
    question: str
    probability: float  # 0..1 — your P(YES)
    created_at: str  # ISO 8601
    category: str | None = None
    resolved: bool = False
    outcome: bool | None = None  # True = YES happened, False = NO
    resolved_at: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be finite and between 0 and 1")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.resolved and not isinstance(self.outcome, bool):
            raise ValueError("a resolved prediction must have a boolean outcome")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Prediction:
        return cls(**data)


@dataclass
class PredictionStore:
    """Container for all predictions."""

    next_id: int = 1
    predictions: list[Prediction] = field(default_factory=list)

    def add(
        self,
        question: str,
        probability: float,
        category: str | None = None,
    ) -> Prediction:
        pred = Prediction(
            id=self.next_id,
            question=question,
            probability=probability,
            created_at=datetime.now(UTC).isoformat(),
            category=category,
        )
        self.predictions.append(pred)
        self.next_id += 1
        return pred

    def get(self, pred_id: int) -> Prediction | None:
        for p in self.predictions:
            if p.id == pred_id:
                return p
        return None

    def resolved_predictions(self) -> list[Prediction]:
        return [p for p in self.predictions if p.resolved]

    def open_predictions(self) -> list[Prediction]:
        return [p for p in self.predictions if not p.resolved]

    def to_dict(self) -> dict:
        return {
            "next_id": self.next_id,
            "predictions": [p.to_dict() for p in self.predictions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PredictionStore:
        predictions = [Prediction.from_dict(p) for p in data.get("predictions", [])]
        if len({p.id for p in predictions}) != len(predictions):
            raise ValueError("duplicate prediction IDs")
        return cls(next_id=max(data.get("next_id", 1), max((p.id for p in predictions), default=0) + 1),
                   predictions=predictions)
