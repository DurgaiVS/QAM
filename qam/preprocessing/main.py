from typing import List

import typer

app = typer.Typer()


@app.command()
def from_yfinance(overrides: List[str]):
    """
    intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo]
    range: ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
    start: YYYY-MM-DD
    end: YYYY-MM-DD
    """

    from ..utils import get_cfg
    from .from_yfinance import YFinanceSource

    cfg = get_cfg("preprocess", overrides, "data_preparation")

    assert cfg.start and (
        cfg.range or cfg.end
    ), "'start' and 'range' or 'end' should be provided"
    if cfg.worker_count == -1:
        cfg.worker_count = len(cfg.symbols) * 3
    source = YFinanceSource(**cfg)
    source.prepare()


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

    from ..utils import get_cfg
    from .from_alpaca import AlpacaSource

    cfg = get_cfg("preprocess", overrides, "data_preparation")

    assert cfg.start and (
        cfg.range or cfg.end
    ), "'start' and 'range' or 'end' should be provided"
    if cfg.worker_count == -1:
        cfg.worker_count = len(cfg.symbols) * 3
    source = AlpacaSource(**cfg)
    source.prepare()
