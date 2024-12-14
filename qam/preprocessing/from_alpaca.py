import asyncio
import glob
import gzip
import json
import logging
import os
from typing import Dict, List, Optional

import aiohttp
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
        split: Dict[str, float],
        worker_count: Optional[int],
        range: Optional[str] = None,
        skip_download: bool = False,
    ):

        super().__init__(symbols, interval, start, end, range, split, worker_count)

        self.skip_download = skip_download
        self.base_url = f"https://data.alpaca.markets/v2/stocks/bars?limit=10000&adjustment=raw&feed=sip&currency=USD&sort=asc&timeframe={interval}&start={start}&end={end}"

        with open(os.environ.get("ALPACA_KEY"), "r") as f:
            self.headers = json.load(f)
        self.headers["accept"] = "application/json"

    async def download(self, symbol: str):
        next_pg_token = None
        url = self.base_url + f"&symbols={symbol}"
        w_dir = os.path.join(self.base_dir, symbol, "raw")
        os.makedirs(w_dir, exist_ok=True)

        tqdm.set_lock(tqdm.get_lock())
        pbar = tqdm(desc=f"Downloading {symbol}", leave=False)

        with QAMFileWriter(
            base_dir=w_dir, filename_stem="raw", extension="json.gz", size_per_file=10
        ) as f:
            connector = aiohttp.TCPConnector()
            async with aiohttp.ClientSession(connector=connector) as session:
                while True:
                    response = await session.get(
                        url=url
                        + (f"&page_token={next_pg_token}" if next_pg_token else ""),
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

    def prepare_data_and_populate_q(self, symbol: str, data_q: "mp.Queue"):
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

        for file_path in tqdm(
            glob.glob("raw*.json.gz", root_dir=r_dir),
            desc=f"Reader {symbol}",
            leave=False,
        ):
            file_path = os.path.join(r_dir, file_path)
            with gzip.open(file_path, "rt") as f:
                data = json.load(f)["bars"][symbol]

            for tp in data:
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

                data_q.put(qam_tp)

        data_q.put(None)

    def writer(self, symbol: str, data_q: "mp.Queue"):
        base = os.path.join(self.base_dir, symbol, "processed")
        os.makedirs(base, exist_ok=True)
        p_bar = tqdm(desc=f"Writer {symbol}", leave=False)
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

                p_bar.update()
                f.write(data)

            p_bar.close()
            logging.info(f"Symbol {symbol} is done.")

    def download_async(self, symbol: str):
        asyncio.run(self.download(symbol))

    def process_data(self):
        if not self.skip_download:
            download = WorkerPool(self.download_async, self.symbols, self.w_count)
            download.start()
            download.join()

        data_q_s = []
        ctx = mp.get_context("fork")
        for _ in range(len(self.symbols)):
            data_q_s.append(ctx.Queue(-1))

        reader = WorkerPool(
            self.prepare_data_and_populate_q,
            {"symbol": self.symbols, "data_q": data_q_s},
            self.w_count,
            backend="process",
        )
        writer = WorkerPool(
            self.writer,
            {"symbol": self.symbols, "data_q": data_q_s},
            self.w_count,
            backend="process",
        )

        writer.start()
        reader.start()

        reader.join()
        writer.join()

        logging.info("Data Preparation done for Alpaca sources...")
