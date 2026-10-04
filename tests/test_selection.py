import pytest

from btc_predictor.training.select_model import score


def test_lower_selection_score_is_better():
    good = {
        "mae_bps": 1,
        "rmse_bps": 1,
        "path_mae_bps": 1,
        "direction_error": 0.4,
        "instability_bps": 0.2,
        "calibration_error": 0.1,
    }
    bad = {**good, "mae_bps": 10}
    assert score(good) < score(bad)


def test_missing_metrics_fail_closed():
    with pytest.raises(ValueError):
        score({"mae_bps": 1})
