import numpy as np
import pytest

from btc_predictor.features.market_analyser import (
    FEATURE_NAMES,
    MarketAnalyser,
    analyse_history,
    normalize_history,
)


def bars(n=120):
    close = 100.0 + np.arange(n) * 0.01
    return np.column_stack(
        [
            close - 0.01,
            close + 0.02,
            close - 0.03,
            close,
            np.ones(n),
            close,
            np.full(n, 10),
            np.full(n, 0.6),
            close * 0.6,
        ]
    )


def test_analyser_is_fixed_size_and_finite():
    values = analyse_history(bars())
    assert values.shape == (96,)
    assert len(FEATURE_NAMES) == 96
    assert np.isfinite(values).all()


def test_normalization_anchors_last_close():
    normalized = normalize_history(bars())
    assert normalized.shape == (120, 9)
    assert normalized[-1, 3] == pytest.approx(0.0, abs=1e-6)


def test_incremental_analyser_rejects_time_reversal():
    analyser = MarketAnalyser(5)
    analyser.update(1000, bars(1)[0])
    with pytest.raises(ValueError):
        analyser.update(1000, bars(1)[0])
