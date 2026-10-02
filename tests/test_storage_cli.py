import json

import pytest
from typer.testing import CliRunner

from calibrate import storage
from calibrate.cli import app
from calibrate.models import PredictionStore
from calibrate.scoring import calibration_bins


def test_add_resolve_score_and_seed_preserves_store(tmp_path):
    runner = CliRunner()
    data = str(tmp_path / "predictions.json")
    assert runner.invoke(app, ["add", "Will ИИ ship?", "1", "--data", data]).exit_code == 0
    assert runner.invoke(app, ["resolve", "1", "yes", "--data", data]).exit_code == 0
    score = runner.invoke(app, ["score", "--data", data])
    assert score.exit_code == 0 and "0.000000" in score.output
    before = (tmp_path / "predictions.json").read_bytes()
    assert runner.invoke(app, ["seed", "--data", data]).exit_code == 1
    assert (tmp_path / "predictions.json").read_bytes() == before


def test_failed_atomic_replace_keeps_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "predictions.json"
    store = PredictionStore()
    store.add("Question?", 0.5)
    storage.save(store, str(path))
    before = path.read_bytes()
    store.add("Another?", 0.7)
    def fail_replace(*args):
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(OSError):
        storage.save(store, str(path))
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_missing_next_id_cannot_reuse_prediction_ids():
    store = PredictionStore()
    store.add("Question?", 0.5)
    data = store.to_dict()
    data.pop("next_id")
    loaded = PredictionStore.from_dict(data)
    assert loaded.add("Another?", 0.7).id == 2


@pytest.mark.parametrize("bins", [0, -1])
def test_invalid_bin_count(bins):
    with pytest.raises(ValueError):
        calibration_bins([(0.5, 1)], bins)


def test_bin_boundary_and_unknown_outcome():
    assert calibration_bins([(0.3, 1)])[3].n == 1
    with pytest.raises(ValueError):
        PredictionStore.from_dict({"predictions": [{"id": 1, "question": "Q", "probability": 0.5,
                                                    "created_at": "", "resolved": True}]})
