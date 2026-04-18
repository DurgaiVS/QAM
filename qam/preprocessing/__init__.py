import asyncio
import logging
import math
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from tqdm import tqdm

from ..constants import DATA_DIR, MAX_SEQ_LEN, SUB_SPLITS
from ..utils import QAMFileWriter, WorkerPool


class Source:
    def __init__(
        self,
        symbols: List[str],
        interval: str,
        start: str,
        end: str,
        range: Optional[str],
        split: Dict[str, float],
        worker_count: Optional[int] = None,
        skip_download: bool = True,
        start_method: str = "fork",
        size_per_file: Optional[int] = 1024 * 1024 * 250,  # 250MB
        count_per_file: Optional[int] = None,
    ):
        self.base_dir = os.path.join(
            DATA_DIR, self.__class__.__name__.lower(), interval
        )
        self.symbols = symbols
        self.interval = interval
        self.start = start
        self.end = end
        self.range = range
        self.w_count = worker_count or len(symbols)
        self.split = split
        self.skip_download = skip_download
        self.start_method = start_method
        self.size_per_file = size_per_file
        self.count_per_file = count_per_file

        self.splitwise_durations: Dict[str, Dict[str, str]] = {}
        self.splitwise_dates: Dict[str, Dict[str, datetime]] = {}
        self.calc_splitwise_range()

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

    def check_tzinfo(self, split: str, timestamp: datetime):
        if timestamp.tzinfo and (not self.splitwise_dates[split]["end"].tzinfo):
            self.splitwise_dates[split]["end"].replace(tzinfo=timestamp.tzinfo)

    def is_falling_after_split(self, split: str, timestamp: datetime) -> bool:
        self.check_tzinfo(split, timestamp)
        return timestamp > self.splitwise_dates[split]["end"]

    def is_falling_before_split(self, split: str, timestamp: datetime) -> bool:
        self.check_tzinfo(split, timestamp)
        return timestamp < self.splitwise_dates[split]["start"]

    def writer(self, symbol: str, split: str, data_q: "mp.Queue"):
        base = os.path.join(self.base_dir, symbol, "processed")
        os.makedirs(base, exist_ok=True)
        with QAMFileWriter(
            base_dir=base,
            filename_stem=split,
            extension="jsonl.gz",
            # size_per_file=self.size_per_file,
            count_per_file=self.count_per_file or MAX_SEQ_LEN,
            writer_mode="b",
        ) as f, tqdm(desc=f"Writer - {symbol}:{split}", leave=False) as p_bar:
            while True:
                data = data_q.get()
                if data is None:
                    break

                p_bar.update()
                f.write(data)

            logging.info(f"Preprocessing done for '{symbol}:{split}'.")

    def download_async(self, symbol: str):
        asyncio.run(self.download(symbol))

    def prepare(self):
        if not self.skip_download:
            download = WorkerPool(
                self.download_async, self.w_count, mappable=self.symbols
            )
            download.start()
            download.join()

        self.process_data()

    def process_data(self):
        data_q_s = []
        readers: List[WorkerPool] = []
        writers: List[WorkerPool] = []
        ctx = mp.get_context("fork")

        for split_name in self.splitwise_durations.keys():
            for _ in range(len(self.symbols)):
                data_q_s.append(ctx.Queue(-1))

            reader = WorkerPool(
                self.prepare_data_and_populate_q,
                self.w_count,
                kwargs={"split": split_name},
                mappable={"symbol": self.symbols, "data_q": data_q_s},
                backend="process",
            )
            writer = WorkerPool(
                self.writer,
                self.w_count,
                kwargs={"split": split_name},
                mappable={"symbol": self.symbols, "data_q": data_q_s},
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

    async def download(self, symbol):
        raise NotImplementedError

    def prepare_data_and_populate_q(self, symbol: str, split: str, data_q: "mp.Queue"):
        raise NotImplementedError
