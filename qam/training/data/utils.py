from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.utils.data

from ...constants import LABEL_THRESHOLD_VALUE, SUBSAMPLING_FACTOR
from ...utils import Classifier, QAMDataBatch, QAMDataSample, QAMTimePoint


def collate_fn(data_batch: List[List[QAMDataSample]]) -> QAMDataBatch:
    collated_batch = QAMDataBatch()

    for batch in data_batch:
        collated_batch.extend(batch)

    return collated_batch.tensorize()


def get_worker_info() -> Tuple[int, int]:
    """
    Returns the data worker id globally, and the global data worker
    counts...

    Returns
    -------
    Tuple[int, int]
        Data worker id globally, Global data worker count
    """
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info is not None, "Couldn't get the worker id..."
    # `num_local_workers`: total number of workers in each local GPU
    # `worker_id`: id of worker within each local GPU
    num_local_workers, worker_id = worker_info.num_workers, worker_info.id
    dist_rank, dist_world_size = 0, 1

    if dist.is_available() and dist.is_initialized():
        dist_world_size = dist.get_world_size()
        dist_rank = dist.get_rank()

    # `num_workers`: total number of workers across all the GPUs
    global_num_workers = dist_world_size * num_local_workers

    # `global_worker_id`: a uniq ID for a worker in global setting
    global_worker_id = worker_id * dist_world_size + dist_rank

    return global_worker_id, global_num_workers
