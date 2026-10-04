from datetime import date
from pathlib import Path

from btc_predictor.data.binance_download import plan_archives


def test_closed_month_and_partial_month_plan():
    plan = plan_archives(date(2020, 1, 1), date(2020, 2, 2), Path("data"), "BTCUSDT")
    assert plan[0].period == "2020-01"
    assert "/monthly/" in plan[0].url
    assert [x.period for x in plan[1:]] == ["2020-02-01", "2020-02-02"]
    assert all("/daily/" in x.url for x in plan[1:])
