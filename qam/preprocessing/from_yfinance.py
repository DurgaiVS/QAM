import glob
import gzip
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas
import yfinance as yf
from filterpy.kalman import KalmanFilter
from pandas import DataFrame
from tqdm import tqdm

from ..utils import QAMFileWriter, QAMTimePoint
from . import Source


class YFinanceSource(Source):
    def __init__(
        self,
        symbols: List[str],
        interval: str,
        start: str,
        end: str,
        split: Dict[str, float],
        worker_count: Optional[int],
        range: Optional[str] = None,
        skip_download: bool = False,
    ):
        super().__init__(
            symbols,
            interval,
            start,
            end,
            range,
            split,
            worker_count,
            skip_download=skip_download,
        )

        self.start_td = datetime.strptime(self.start, "%Y-%m-%d")
        self.end_td = datetime.strptime(self.end, "%Y-%m-%d")

    async def download(self, symbol):
        w_dir = os.path.join(self.base_dir, symbol, "raw")
        os.makedirs(w_dir, exist_ok=True)

        symbol_ticker = yf.Ticker(symbol)
        from_time = self.start_td
        to_time = from_time + timedelta(days=8)

        with QAMFileWriter(
            base_dir=w_dir,
            filename_stem="raw",
            extension="csv.gz",
            size_per_file=float("inf"),
        ) as f:
            while from_time < self.end_td:
                df: DataFrame = symbol_ticker.history(
                    period=self.range,
                    interval=self.interval,
                    start=from_time.strftime("%Y-%m-%d"),
                    end=to_time.strftime("%Y-%m-%d"),
                )
                # df = df.drop(["Dividends", "Stock Splits", "Datetime"], axis=1)
                # kf = KalmanFilter()

                # TODO: read sample wise data and apply kalman filter and then write to output file
                df.to_csv(f)
                f._wrap_up_and_open_new()

                from_time = to_time
                to_time = from_time + timedelta(days=8)
                if to_time > self.end_td:
                    to_time = self.end_td

    def prepare_data_and_populate_q(self, symbol, split, data_q):
        r_dir = os.path.join(self.base_dir, symbol, "raw")
        file_paths = glob.glob("raw*csv.gz", root_dir=r_dir)
        file_paths.sort()

        for filepath in tqdm(
            file_paths, desc=f"Reader - {symbol}:{split}", leave=False
        ):
            with gzip.open(os.path.join(r_dir, filepath), "rt") as f:
                for record in pandas.read_csv(f).iloc:
                    td = datetime.strptime(record.Datetime[:-6], "%Y-%m-%d %H:%M:%S")
                    if self.is_falling_before_split(split, td):
                        continue
                    elif self.is_falling_after_split(split, td):
                        break

                    qam_tp = QAMTimePoint(
                        symbol=symbol,
                        open=float(record.Open),
                        close=float(record.Close),
                        high=float(record.High),
                        low=float(record.Low),
                        volume=int(record.Volume),
                        time=td.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                    data_q.put(qam_tp.to_bytes())

        data_q.put(None)
