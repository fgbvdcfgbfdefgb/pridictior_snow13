import csv
import zipfile
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from btc_predictor.data.binance_download import convert_zip_to_parquet, plan_archives


def test_closed_month_and_partial_month_plan():
    plan = plan_archives(date(2020, 1, 1), date(2020, 2, 2), Path("data"), "BTCUSDT")
    assert plan[0].period == "2020-01"
    assert "/monthly/" in plan[0].url
    assert [x.period for x in plan[1:]] == ["2020-02-01", "2020-02-02"]
    assert all("/daily/" in x.url for x in plan[1:])


def test_converter_makes_missing_seconds_explicit(tmp_path):
    archive = tmp_path / "source.zip"
    csv_path = tmp_path / "source.csv"
    base = 1_577_836_800_000
    rows = [
        [base, 100, 101, 99, 100, 1, base + 999, 100, 1, 0.5, 50, 0],
        [base + 2000, 102, 103, 101, 102, 2, base + 2999, 204, 2, 1, 102, 0],
    ]
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(csv_path, "source.csv")
    output = tmp_path / "output.parquet"
    count, _first, _last, imputed = convert_zip_to_parquet(archive, output)
    table = pq.read_table(output)
    assert count == 3
    assert imputed == 1
    assert table.column("is_imputed").to_pylist() == [False, True, False]
    assert table.column("close").to_pylist() == [100.0, 100.0, 102.0]
