import pytest

torch = pytest.importorskip("torch")

from btc_predictor.models import build_model


@pytest.mark.parametrize("architecture", ["patch_transformer", "dilated_tcn", "gru_attention"])
def test_candidate_output_shape(architecture):
    common = {
        "architecture": architecture,
        "input_dim": 9,
        "feature_dim": 96,
        "horizon": 30,
        "quantiles": 3,
        "output_rank": 8,
    }
    if architecture == "patch_transformer":
        common.update(d_model=48, layers=1, heads=4)
    elif architecture == "dilated_tcn":
        common.update(channels=16, blocks=2, kernel_size=3)
    else:
        common.update(hidden_size=24, layers=1, heads=4)
    model = build_model(common)
    history = torch.randn(1, 43_200, 9)
    features = torch.randn(1, 96)
    output = model(history, features)
    assert output.shape == (1, 30, 3)
