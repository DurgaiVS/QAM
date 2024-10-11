import json
import os

import requests
from tqdm import tqdm

from ..utils import QAMFileWriter

"https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL&timeframe=1Min&start=2024-01-01T01%3A00%3A00.11Z&end=2024-01-03T01%3A00%3A00.11Z&limit=1000&adjustment=all&feed=sip&currency=USD&page_token=0&sort=asc"

BASE_URL = "https://data.alpaca.markets/v2/stocks/bars?limit=10000&adjustment=all&feed=sip&currency=USD&sort=asc"


def alpaca_downloader(
    symbol: str,
    base_dir: str,
    interval: str,
    start: str,
    end: str,
):
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
    with open(os.environ.get("ALPACA_KEY"), "r") as f:
        headers = json.load(f)
    headers["accept"] = "application/json"

    pg_token = None
    start = start.replace(":", "%3A")
    end = end.replace(":", "%3A")
    # if isinstance(symbol, (list, tuple)):
    #     symbol = "%2C".join(symbol)

    pbar = tqdm(desc=f"Downloading {symbol}", leave=False)
    url = f"{BASE_URL}&timeframe={interval}&start={start}&end={end}&symbols={symbol}"

    with QAMFileWriter(
        base_dir=base_dir, filename_stem="raw", extension="json.gz", size_per_file=2
    ) as f:
        while True:
            response = requests.get(
                url + (f"&page_token={pg_token}" if pg_token else ""),
                headers=headers,
            )
            if response.status_code != 200:
                raise RuntimeError(response.text)

            data: dict = json.loads(response.text)
            pg_token = data.pop("next_page_token", None)
            pbar.update()

            f.write(json.dumps(data))
            if pg_token is None:
                break

    with open(f"{base_dir}/meta.json", "w") as f:
        json.dump({"start": start, "end": end}, f)

    pbar.close()
