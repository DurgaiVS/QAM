from qam.preprocessing.from_alpaca import AlpacaSource

source = AlpacaSource(
    ["AAPL", "NVDA"],
    "1Min",
    "2024-01-01",
    "2024-10-10",
    "",
    {"train": 0.7, "val": 0.1, "test": 0.2},
    2,
    True,
)

source.process_data()
