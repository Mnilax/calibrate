"""Tests for scoring functions."""

import math

import pytest

from calibrate.scoring import (
    base_rate,
    brier_score,
    calibration_bins,
    log_loss,
    murphy_decomposition,
)


class TestBrierScore:
    def test_perfect_predictions(self):
        """Perfect forecaster: p=o always -> Brier=0."""
        preds = [(1.0, 1), (0.0, 0), (1.0, 1), (0.0, 0)]
        assert brier_score(preds) == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Worst forecaster: p=1-o always -> Brier=1."""
        preds = [(1.0, 0), (0.0, 1), (1.0, 0), (0.0, 1)]
        assert brier_score(preds) == pytest.approx(1.0)

    def test_always_fifty_fifty(self):
        """Always 0.5 on 50/50 -> Brier=0.25."""
        preds = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
        assert brier_score(preds) == pytest.approx(0.25)

    def test_single_prediction(self):
        preds = [(0.7, 1)]
        assert brier_score(preds) == pytest.approx(0.09)

    def test_empty(self):
        assert brier_score([]) == 0.0


class TestLogLoss:
    def test_confident_correct(self):
        """High-confidence correct predictions -> low log loss."""
        preds = [(0.99, 1), (0.01, 0)]
        ll = log_loss(preds)
        assert ll < 0.02

    def test_always_fifty(self):
        """Always 0.5 -> log loss = ln(2) ≈ 0.693."""
        preds = [(0.5, 1), (0.5, 0)]
        assert log_loss(preds) == pytest.approx(math.log(2), rel=1e-10)

    def test_manual_calculation(self):
        """Manual verification: p=0.8, o=1 -> -ln(0.8) ≈ 0.2231."""
        preds = [(0.8, 1)]
        expected = -math.log(0.8)
        assert log_loss(preds) == pytest.approx(expected, rel=1e-10)

    def test_clipping_prevents_inf(self):
        """Extreme probabilities don't cause inf."""
        preds = [(0.0, 1), (1.0, 0)]
        ll = log_loss(preds)
        assert math.isfinite(ll)

    def test_empty(self):
        assert log_loss([]) == 0.0


class TestCalibrationBins:
    def test_bin_count(self):
        preds = [(0.1, 0), (0.5, 1), (0.9, 1)]
        bins = calibration_bins(preds, num_bins=10)
        assert len(bins) == 10

    def test_all_in_one_bin(self):
        preds = [(0.55, 1), (0.52, 0), (0.58, 1)]
        bins = calibration_bins(preds, num_bins=10)
        bin_5 = bins[5]  # [0.5, 0.6)
        assert bin_5.n == 3
        assert bin_5.mean_prob == pytest.approx(0.55, rel=1e-5)
        assert bin_5.observed_freq == pytest.approx(2 / 3, rel=1e-5)

    def test_empty_bins(self):
        preds = [(0.1, 0)]
        bins = calibration_bins(preds, num_bins=10)
        assert bins[1].n == 1
        assert bins[0].n == 0


class TestMurphyDecomposition:
    def test_identity_holds(self):
        """Murphy identity: Brier ≈ Reliability - Resolution + Uncertainty."""
        preds = [
            (0.1, 0), (0.2, 0), (0.3, 1), (0.4, 0),
            (0.5, 1), (0.6, 1), (0.7, 0), (0.8, 1),
            (0.9, 1), (0.15, 0), (0.85, 1), (0.55, 0),
            (0.25, 0), (0.75, 1), (0.65, 1), (0.35, 0),
        ]
        bs = brier_score(preds)
        murphy = murphy_decomposition(preds)
        assert murphy.brier_check == pytest.approx(bs, abs=1e-9)

    def test_perfect_calibration(self):
        """Perfect predictions -> Brier=0, check decomposition."""
        preds = [(1.0, 1), (0.0, 0), (1.0, 1), (0.0, 0)]
        murphy = murphy_decomposition(preds)
        # Perfect calibration: reliability=0
        # Brier=0, so reliability - resolution + uncertainty = 0
        assert murphy.brier_check == pytest.approx(0.0, abs=1e-9)

    def test_always_fifty_murphy(self):
        """Always 0.5 on 50/50 data."""
        preds = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
        murphy = murphy_decomposition(preds)
        bs = brier_score(preds)
        assert murphy.brier_check == pytest.approx(bs, abs=1e-9)
        assert murphy.uncertainty == pytest.approx(0.25)

    def test_empty(self):
        murphy = murphy_decomposition([])
        assert murphy.reliability == 0.0
        assert murphy.resolution == 0.0
        assert murphy.uncertainty == 0.0


class TestBaseRate:
    def test_fifty_fifty(self):
        preds = [(0.5, 1), (0.5, 0)]
        assert base_rate(preds) == pytest.approx(0.5)

    def test_all_yes(self):
        preds = [(0.5, 1), (0.5, 1)]
        assert base_rate(preds) == pytest.approx(1.0)

    def test_empty(self):
        assert base_rate([]) == 0.0
