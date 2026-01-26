import glob
import gzip
import json
import logging
import multiprocessing as mp
import os
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
from tqdm import tqdm

from ..utils import QAMFileWriter, QAMTimePoint
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

        super().__init__(
            symbols, interval, start, end, range, split, worker_count, skip_download
        )
        self.base_url = (
            "https://data.alpaca.markets/v2/stocks/bars?limit=10000&adjustment=raw"
            f"&feed=sip&currency=USD&sort=asc&timeframe={interval}"
        )

        with open(os.environ.get("ALPACA_KEY"), "r") as f:
            self.headers = json.load(f)
        self.headers["accept"] = "application/json"

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
            with gzip.open(os.path.join(r_dir, file_path), "rt") as f:
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
                    time=td.strftime("%Y-%m-%d %H:%M:%S"),
                    symbol=symbol,
                )

                data_q.put(qam_tp.to_bytes())

        data_q.put(None)
