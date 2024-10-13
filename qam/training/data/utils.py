from typing import List

import torch
import torch.distributed as dist

from ...constants import HIGH_ID, LABEL_THRESHOLD_VALUE, SUBSAMPLING_FACTOR
from ...utils import Classifier, QAMDataBatch, QAMDataSample, QAMTimePoint


def find_label(samples: List[List]) -> int:
    max_val_future, max_val_present = float("-inf"), float("-inf")

    for sample in samples[SUBSAMPLING_FACTOR:]:
        if sample[HIGH_ID] > max_val_future:
            max_val_future = sample[HIGH_ID]

    for sample in samples[:SUBSAMPLING_FACTOR]:
        if sample[HIGH_ID] > max_val_future:
            max_val_present = sample[HIGH_ID]

    graph_diff = max_val_future - max_val_present

    if graph_diff > LABEL_THRESHOLD_VALUE:
        return Classifier.VERY_HIGH.value

    if graph_diff > 0 and graph_diff < LABEL_THRESHOLD_VALUE:
        return Classifier.HIGH.value

    if graph_diff < 0 and graph_diff > -LABEL_THRESHOLD_VALUE:
        return Classifier.LOW.value

    if graph_diff < -LABEL_THRESHOLD_VALUE:
        return Classifier.VERY_LOW.value


def collate_fn(data_batch: List[List[QAMDataSample]]) -> QAMDataBatch:
    collated_batch = QAMDataBatch()

    for batch in data_batch:
        collated_batch.extend(batch)

    return collated_batch.tensorize()


def worker_init_fn(worker_id):
    """
    Parameters
    ----------
    worker_id : ``int``
        ID of the current local worker.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # `num_local_workers`: workers in each local GPU
    num_local_workers: int = worker_info.num_workers

    if dist.is_available() and dist.is_initialized():
        dataset.dist_world_size = dist.get_world_size()
        dataset.dist_rank = dist.get_rank()

    # `num_workers`: total number of workers across all the GPUs
    dataset.num_workers = dataset.dist_world_size * num_local_workers

    # `global_worker_id`: a uniq ID for a worker in global setting
    global_worker_id = worker_id * dataset.dist_world_size + dataset.dist_rank
    dataset.worker_id = global_worker_id
