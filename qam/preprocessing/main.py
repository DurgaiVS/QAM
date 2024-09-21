import os
import threading

import typer

from ..constants import DATA_DIR
from ..utils import get_cfg
from . import _downloader

app = typer.Typer()


@app.command()
def from_yahoo(overrides: list[str] = []):

    cfg = get_cfg("preprocess", overrides, "data_preparation")

    """
    range: ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
    start: YYYY-MM-DD
    end: YYYY-MM-DD
    """
    for split in cfg.split:
        assert split.range or (
            split.start and split.end
        ), "Either start and end date or range should be provided"

        split_dir = f"{DATA_DIR}/{split.name}"
        os.makedirs(split_dir, exist_ok=True)

        threads: list[threading.Thread] = []
        for symbol in cfg.symbols:
            t = threading.Thread(
                target=_downloader,
                args=(
                    symbol,
                    split_dir,
                    split.name,
                    cfg.interval,
                    split.range,
                    split.start,
                    split.end,
                ),
            )
            t.start()
            threads.append(t)

            if len(threads) == cfg.max_parallel_count:
                threads[0].join()
                threads.pop(0)

        for t in threads:
            t.join()
