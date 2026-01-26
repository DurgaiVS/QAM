import os

PAD_ID: int = -100

MAX_SEQ_LEN: int = 128
# NOTE: For a 'MAX_SEQ_LEN' timestep data, the model will guess
#       the trend direction within the next 'TREND_UPDATE_SEQ_LEN' timestep...
TREND_UPDATE_SEQ_LEN: int = 32

STRIDE_LENGTH: int = 32
SUBSAMPLING_FACTOR: int = 8
SAMPLE_DIM: int = 9

# NOTE: When a trend's absolute percentage diff goes above or below this threshold,
#       it is tagged with `very ...`
LABEL_MAX_DIFF_PERCENT: float = 0.1
# NOTE: Minimum increment percentage to consider for `High` related labels...
#       If below this will be moved to `Low` related, or `NoImp`
LABEL_MIN_INCREMENT_PERCENT: float = 0.05

DATA_DIR: str = f"{os.environ['QAM_ROOT']}/dataset"
CONFIG_PATH: str = os.path.realpath(f"{os.environ['QAM_ROOT']}/qam/config")
RESHARD_DIR_NAME = "resharded"
SUB_SPLITS = ["train", "dev", "test"]
SPLITS = {"train": ["train"], "eval": ["dev", "test"]}
