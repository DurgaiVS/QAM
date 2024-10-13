import glob
import gzip
import json
import logging
import os
from typing import Dict, List, Optional

import requests
import torch.multiprocessing as mp
from tqdm import tqdm

from ..utils import QAMFileWriter, QAMTimePoint, WorkerPool
from . import Source

"https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL&timeframe=1Min&start=2024-01-01T01%3A00%3A00.11Z&end=2024-01-03T01%3A00%3A00.11Z&limit=1000&adjustment=all&feed=sip&currency=USD&page_token=0&sort=asc"


class AlpacaSource(Source):
    def __init__(
        self,
        symbols: List[str],
        interval: str,
        start: str,
        end: str,
        range: str,
        split: Dict[str, float],
        worker_count: Optional[int],
    ):

        super().__init__(symbols, interval, start, end, range, split, worker_count)

        self.base_url = f"https://data.alpaca.markets/v2/stocks/bars?limit=10000&adjustment=raw&feed=sip&currency=USD&sort=asc&timeframe={interval}&start={start}&end={end}"

        with open(os.environ.get("ALPACA_KEY"), "r") as f:
            self.headers = json.load(f)
        self.headers["accept"] = "application/json"

    def download(self, symbol: str):
        next_pg_token = None
        url = self.base_url + f"&symbols={symbol}"
        w_dir = os.path.join(self.base_dir, symbol, "raw")
        os.makedirs(w_dir, exist_ok=True)

        tqdm.set_lock(tqdm.get_lock())
        pbar = tqdm(desc=f"Downloading {symbol}", leave=False)

        with QAMFileWriter(
            base_dir=w_dir, filename_stem="raw", extension="json.gz", size_per_file=10
        ) as f:
            while True:
                response = requests.get(
                    url + (f"&page_token={next_pg_token}" if next_pg_token else ""),
                    headers=self.headers,
                )
                if response.status_code != 200:
                    raise RuntimeError(response.text)

                data = json.loads(response.text)
                next_pg_token = data.pop("next_page_token", None)
                pbar.update()

                f.write(json.dumps(data))
                if next_pg_token is None:
                    break

        with open(os.path.join(w_dir, "meta.json"), "w") as f:
            json.dump({"start": self.start, "end": self.end}, f)

        pbar.close()

    def prepare_data_and_populate_q(self, symbol: str, data_q: "mp.Queue"):
        """
        Properties#
        t   Timestamp in RFC-3339 format with nanosecond precision.
        o   Open price.
        h   High price.
        l   Low price.
        c   Close price.
        v   Volume.
        n   Number of trades.
        vw  Volume weighted average price.
        """

        r_dir = os.path.join(self.base_dir, symbol, "raw")

        for file_path in glob.glob("raw*.json.gz", root_dir=r_dir):
            file_path = os.path.join(r_dir, file_path)
            with gzip.open(file_path, "rt") as f:
                data = json.load(f)["bars"][symbol]

            for tp in data:
                qam_tp = QAMTimePoint(
                    open=data["o"],
                    close=data["c"],
                    high=data["h"],
                    low=data["l"],
                    n_trades=data["n"],
                    volume=data["v"],
                    volume_wa=data["vw"],
                    time=data["t"],
                    symbol=symbol,
                )

                data_q.put(qam_tp)

        data_q.put(None)

    def writer(self, symbol: str, data_q: "mp.Queue"):
        base = os.path.join(self.base_dir, symbol, "processed")
        with QAMFileWriter(
            base_dir=base,
            filename_stem="processed",
            extension="jsonl.gz",
            size_per_file=((2**5) * 1024 * 1024),
        ) as f:
            while True:
                data = data_q.get()
                if data is None:
                    break

                f.write(data)

            logging.info(f"Symbol {symbol} is done.")

    def process_data(self):
        download = WorkerPool(self.download, self.symbols, self.w_count)
        download.start()

        data_q_s = []
        ctx = mp.get_context("fork")
        for _ in range(len(self.symbols)):
            data_q_s.append(ctx.Queue(-1))

        reader = WorkerPool(
            self.prepare_data_and_populate_q,
            {"symbol": self.symbols, "data_q": data_q_s},
            self.w_count,
        )
        writer = WorkerPool(
            self.writer, {"symbol": self.symbols, "data_q": data_q_s}, self.w_count
        )

        writer.start()
        reader.start()

        logging.info("Data Preparation done for Alpaca sources...")
