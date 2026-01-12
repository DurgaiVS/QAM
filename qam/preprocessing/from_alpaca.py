import asyncio
import glob
import gzip
import json
import logging
import math
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp
from tqdm import tqdm

from ..constants import SUB_SPLITS
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
        split: Dict[str, float],
        worker_count: Optional[int],
        range: Optional[str] = None,
        skip_download: bool = False,
    ):

        super().__init__(symbols, interval, start, end, range, split, worker_count)

        self.splitwise_durations: Dict[str, Dict[str, str]] = {}
        self.splitwise_dates: Dict[str, Dict[str, datetime]] = {}
        self.skip_download = skip_download
        self.base_url = (
            "https://data.alpaca.markets/v2/stocks/bars?limit=10000&adjustment=raw"
            f"&feed=sip&currency=USD&sort=asc&timeframe={interval}"
        )
        self.calc_splitwise_range()

        with open(os.environ.get("ALPACA_KEY"), "r") as f:
            self.headers = json.load(f)
        self.headers["accept"] = "application/json"

    def calc_splitwise_range(self):
        """
        Used to save the duration 'start' and 'end' splitwise using the 'split' attribute.
        """
        s_date = datetime.strptime(self.start, "%Y-%m-%d")
        e_date = datetime.strptime(self.end, "%Y-%m-%d")

        total_days = (e_date - s_date).days + 1

        train_days = math.ceil(total_days * self.split["train"])
        dev_days = math.ceil(total_days * self.split["dev"])
        test_days = math.ceil(total_days * self.split["test"])

        if (train_days + dev_days + test_days) > total_days:
            diff_days = (train_days + dev_days + test_days) - total_days
            train_days -= diff_days

        self.splitwise_dates["train"] = {
            "start": s_date,
            "end": s_date + timedelta(train_days),
        }
        self.splitwise_dates["dev"] = {
            "start": s_date + timedelta(train_days + 1),
            "end": s_date + timedelta(train_days + dev_days),
        }
        self.splitwise_dates["test"] = {
            "start": s_date + timedelta(train_days + dev_days + 1),
            "end": s_date + timedelta(train_days + dev_days + test_days),
        }

        for ss in SUB_SPLITS:
            for k, v in self.splitwise_dates[ss].items():
                if ss not in self.splitwise_durations:
                    self.splitwise_durations[ss] = {k: v.strftime("%Y-%m-%d")}

                else:
                    self.splitwise_durations[ss][k] = v.strftime("%Y-%m-%d")

    async def download(self, symbol: str):
        next_pg_token = None
        url = self.base_url + f"&symbols={symbol}&start={self.start}&end={self.end}"
        w_dir = os.path.join(self.base_dir, symbol, "raw")
        os.makedirs(w_dir, exist_ok=True)

        tqdm.set_lock(tqdm.get_lock())
        pbar = tqdm(desc=f"Downloading {symbol}...", leave=False)

        with QAMFileWriter(
            base_dir=w_dir, filename_stem="raw", extension="json.gz", size_per_file=10
        ) as f:
            # NOTE: Since we want to write every responce into seperate file, we've
            #       very low `size_per_file`...

            connector = aiohttp.TCPConnector()
            async with aiohttp.ClientSession(connector=connector) as session:
                while True:
                    response = await session.get(
                        url=url
                        + (
                            f"&page_token={next_pg_token}"
                            if next_pg_token is not None
                            else ""
                        ),
                        headers=self.headers,
                        ssl=False,
                    )

                    if response.status != 200:
                        raise RuntimeError(response.content)

                    data = json.loads(await response.content.read())
                    next_pg_token = data.pop("next_page_token", None)
                    pbar.update()

                    f.write(json.dumps(data))
                    if next_pg_token is None:
                        break

        with open(os.path.join(w_dir, "meta.json"), "w") as f:
            json.dump({"start": self.start, "end": self.end}, f)

        pbar.close()
        logging.info(f"Downloading done for '{symbol}'...")

    def download_async(self, symbol: str):
        asyncio.run(self.download(symbol))

    def is_falling_after_split(self, split: str, timestamp: datetime) -> bool:
        return timestamp > self.splitwise_dates[split]["end"]

    def is_falling_before_split(self, split: str, timestamp: datetime) -> bool:
        return timestamp < self.splitwise_dates[split]["start"]

    def prepare_data_and_populate_q(self, symbol: str, split: str, data_q: "mp.Queue"):
        """
        Properties
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
        file_paths = glob.glob("raw*json.gz", root_dir=r_dir)
        file_paths.sort()

        for file_path in tqdm(
            file_paths,
            desc=f"Reader - {symbol}:{split}",
            leave=False,
        ):
            file_path = os.path.join(r_dir, file_path)
            with gzip.open(file_path, "rt") as f:
                data = json.load(f)["bars"][symbol]

            for tp in data:
                _t = tp["t"].rfind(":")
                td = datetime.strptime(
                    (tp["t"][:_t] + tp["t"][_t + 1 :]), "%Y-%m-%d %H:%M:%S%z"
                )

                if self.is_falling_before_split(split, td):
                    continue

                elif self.is_falling_after_split(split, td):
                    break

                qam_tp = QAMTimePoint(
                    open=tp["o"],
                    close=tp["c"],
                    high=tp["h"],
                    low=tp["l"],
                    n_trades=tp["n"],
                    volume=tp["v"],
                    volume_wa=tp["vw"],
                    time=tp["t"],
                    symbol=symbol,
                )

                data_q.put(qam_tp.to_bytes())

        data_q.put(None)

    def writer(self, symbol: str, split: str, data_q: "mp.Queue"):
        base = os.path.join(self.base_dir, symbol, "processed")
        os.makedirs(base, exist_ok=True)
        p_bar = tqdm(desc=f"Writer - {symbol}:{split}", leave=False)
        with QAMFileWriter(
            base_dir=base,
            filename_stem=split,
            extension="jsonl.gz",
            size_per_file=(1024 * 1024),
            writer_mode="b",
        ) as f:
            while True:
                data = data_q.get()
                if data is None:
                    break

                p_bar.update()
                f.write(data)

            p_bar.close()
            logging.info(f"Preprocessing done for '{symbol}:{split}'.")

    def process_data(self):
        if not self.skip_download:
            download = WorkerPool(self.download_async, self.symbols, self.w_count)
            download.start()
            download.join()

        data_q_s = []
        readers: List[WorkerPool] = []
        writers: List[WorkerPool] = []
        ctx = mp.get_context("fork")

        for split_name in self.splitwise_durations.keys():
            for _ in range(len(self.symbols)):
                data_q_s.append(ctx.Queue(-1))

            reader = WorkerPool(
                self.prepare_data_and_populate_q,
                {"symbol": self.symbols, "data_q": data_q_s},
                self.w_count,
                kwargs={"split": split_name},
                backend="process",
            )
            writer = WorkerPool(
                self.writer,
                {"symbol": self.symbols, "data_q": data_q_s},
                self.w_count,
                kwargs={"split": split_name},
                backend="process",
            )

            writer.start()
            reader.start()

            readers.append(reader)
            writers.append(writer)

        for reader, writer in zip(readers, writers):
            reader.join()
            writer.join()

        logging.info("Data Preparation done for Alpaca sources...")
