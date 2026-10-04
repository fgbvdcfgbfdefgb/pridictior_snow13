import numpy as np

from btc_predictor.training.continual import ContinualReplayBuffer


def row(price):
    return np.array([price, price, price, price, 1.0, price, 1.0, 0.5, price * 0.5])


def test_reward_is_released_only_after_full_horizon():
    replay = ContinualReplayBuffer(history_seconds=4, horizon_seconds=3)
    outputs = [replay.append(i * 1000, row(100 + i)) for i in range(7)]
    assert outputs[:-1] == [None] * 6
    matured = outputs[-1]
    assert matured is not None
    assert matured.issued_timestamp_ms == 3_000
    assert matured.history.shape == (4, 9)
    assert matured.target.shape == (3,)
    expected = np.log(np.array([104, 105, 106]) / 103)
    np.testing.assert_allclose(matured.target, expected, rtol=1e-6)
