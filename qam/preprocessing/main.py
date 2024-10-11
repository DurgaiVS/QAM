import os
import threading
from typing import List

import typer

from ..constants import DATA_DIR
from ..utils import get_cfg
from . import alpaca_downloader, yf_downloader

app = typer.Typer()


@app.command()
def from_yfinance(overrides: list[str] = []):
    """
    range: ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
    start: YYYY-MM-DD
    end: YYYY-MM-DD
    """

    cfg = get_cfg("preprocess", overrides, "data_preparation")

    assert cfg.range or (
        cfg.start and cfg.end
    ), "Either start and end date or range should be provided"
    os.makedirs(DATA_DIR, exist_ok=True)

    threads: list[threading.Thread] = []
    for symbol in cfg.symbols:
        t = threading.Thread(
            target=yf_downloader,
            args=(
                symbol,
                DATA_DIR,
                cfg.interval,
                cfg.range,
                cfg.start,
                cfg.end,
            ),
        )
        t.start()
        threads.append(t)

        if len(threads) == cfg.max_parallel_count:
            threads[0].join()
            threads.pop(0)

    for t in threads:
        t.join()


@app.command()
def from_alpaca(overrides: List[str]):
    """
    interval:
        [1-59]Min / T
        [1-23]Hour / H
        1Day / D
        1Week / W
        [1,2,3,4,6,12]Month / M
    start: YYYY-MM-DD[T00:00:00]
    end: YYYY-MM-DD[T00:00:00]
    """

    cfg = get_cfg("preprocess", overrides, "data_preparation")

    assert cfg.start and cfg.end, "Both start and end dates are required."
    os.makedirs(DATA_DIR, exist_ok=True)

    for sym in cfg.symbols:
        sym_path = os.path.join(DATA_DIR, sym)
        os.makedirs(sym_path)

        alpaca_downloader(
            sym,
            sym_path,
            cfg.interval,
            cfg.start,
            cfg.end,
        )
