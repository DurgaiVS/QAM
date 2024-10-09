from typing import Optional

import yfinance as yf
from filterpy.kalman import KalmanFilter
from pandas import DataFrame

from ..utils import QAMFileWriter


def yf_downloader(
    symbol: str,
    base_dir: str,
    interval: str = "1h",
    range: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    dataset_dir = f"{base_dir}/{symbol}"
    df: DataFrame = yf.Ticker(symbol).history(
        period=range, interval=interval, start=start, end=end
    )
    # df = df.drop(["Dividends", "Stock Splits", "Datetime"], axis=1)
    kf = KalmanFilter()

    # TODO: read sample wise data and apply kalman filter and then write to output file
    with QAMFileWriter(
        base_dir=dataset_dir, filename_stem="raw", extension="jsonl.gz"
    ) as f:
        df.to_csv(f)
