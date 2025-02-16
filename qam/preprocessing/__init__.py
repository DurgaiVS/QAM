import os
from typing import Dict, List, Optional

from ..constants import DATA_DIR


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
    ):
        self.base_dir = os.path.join(DATA_DIR, interval)
        self.symbols = symbols
        self.interval = interval
        self.start = start
        self.end = end
        self.range = range
        self.w_count = worker_count or len(symbols)
        self.split = split

    def process_data(self):
        raise NotImplementedError
