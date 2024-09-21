import os

DATA_DIR: str = f"{os.environ['QAM_ROOT']}/dataset"
WINDOW_SIZE: int = 128
SAMPLE_DIM: int = 4
SUBSAMPLING_FACTOR: int = 8
HIGH_ID: int = 1
PAD_ID: int = -100
LABEL_THRESHOLD_VALUE: float = 5.0
CONFIG_PATH: str = "./config"
SAMPLE_COUNT_PER_LABEL: int = 32 * SUBSAMPLING_FACTOR
STRIDE_LENGTH: int = WINDOW_SIZE / 4
