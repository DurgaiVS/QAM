import json
import os
from typing import Optional, Union

import requests

"https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL&timeframe=1Min&start=2024-01-01T01%3A00%3A00.11Z&end=2024-01-03T01%3A00%3A00.11Z&limit=1000&adjustment=all&feed=sip&currency=USD&page_token=0&sort=asc"

BASE_URL = "https://data.alpaca.markets/v2/stocks/bars?limit=10000&adjustment=all&feed=sip&currency=USD&sort=asc"


def alpaca_downloader(
    symbol: Union[str, list[str]],
    base_dir: str,
    interval: str = "1h",
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    with open(os.environ.get("ALPACA_KEY"), "r") as f:
        headers = json.load(f)
    headers["accept"] = "application/json"

    pg_token = 0
    start = start.replace(":", "%3A")
    end = end.replace(":", "%3A")
    if isinstance(symbol, (list, tuple)):
        symbol = "%2C".join(symbol)

    response = requests.get(
        f"{BASE_URL}&page_token={pg_token}&timeframe={interval}&start={start}&end={end}&symbols={symbol}",
        headers=headers,
    )

    with open(f"{base_dir}/data.json", "w") as f:
        json.dump(response.text, f, indent=True)
