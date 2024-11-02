import os

PAD_ID: int = -100

# for a 512 timestep data, the model will guess the trend direction within the next 512 timestep...
TREND_UPDATE_SEQ_LEN: int = 512

MAX_SEQ_LEN: int = 512
STRIDE_LENGTH: int = 128
SUBSAMPLING_FACTOR: int = 8
SAMPLE_DIM: int = 9

# when a trend is more than or less than this threshold, then it is tagged with `very ...`
LABEL_THRESHOLD_VALUE: float = 5.0

DATA_DIR: str = f"{os.environ['QAM_ROOT']}/dataset"
CONFIG_PATH: str = "./config"
RESHARD_DIR_NAME = "resharded"
SUBSET = {
    "train": ["short", "long"],
    "eval": ["dev", "test"],
}
