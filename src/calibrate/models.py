"""Data models for predictions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


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
            created_at=datetime.now(timezone.utc).isoformat(),
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
        return cls(
            next_id=data.get("next_id", 1),
            predictions=[Prediction.from_dict(p) for p in data.get("predictions", [])],
        )
