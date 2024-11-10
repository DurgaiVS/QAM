import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch.multiprocessing as mp
from tqdm import tqdm

from ...constants import DATA_DIR, RESHARD_DIR_NAME, SPLITS, SUB_SPLITS
from ...utils import (
    DatasetMeta,
    QAMFileWriter,
    get_samplescount_per_shard,
    yield_sample_from_serially_sorted_files,
)


def _get_total_samplescount(
    shards: List[str], data_q: Optional["mp.Queue"] = None
) -> int:
    total_samples = 0

    for _ in yield_sample_from_serially_sorted_files(shards):
        total_samples += 1

    if data_q:
        data_q.put(total_samples)
    return total_samples


def _reader(shards: List[str], data_q: "mp.Queue"):
    for sample in yield_sample_from_serially_sorted_files(shards):
        data_q.put(sample)

    data_q.put(None)


def _writer(
    data_q: "mp.Queue",
    readers_count: int,
    total_shards_req: int,
    group_type: str,
    id: Optional[str],
    **kwargs,
):
    tqdm.set_lock(tqdm.get_lock())
    desc = f"{id}-{group_type}" if id else group_type
    pbar = tqdm(total=total_shards_req, desc=desc, leave=False)
    workers_done = 0
    _fc = 0

    with QAMFileWriter(**kwargs) as itn_writer:
        while True:
            grp = data_q.get()

            if grp is None:
                workers_done += 1
                if workers_done == readers_count:
                    break
                else:
                    continue

            itn_writer.write(grp)

            if itn_writer._files_counter != _fc:
                pbar.update()
                _fc = itn_writer._files_counter
                if _fc == total_shards_req:
                    break

    pbar.close()


def _is_reshard_required(
    reshard_dir: str,
    sub_split: str,
    meta: DatasetMeta,
) -> bool:
    metafile_path = os.path.join(reshard_dir, f"meta-{sub_split}.json")

    if not os.path.exists(metafile_path):
        if os.path.exists(reshard_dir):
            logging.info(
                f"Cannot find metadata file, removing existing resharded directory `{reshard_dir}`"
            )
            shutil.rmtree(reshard_dir)
        os.mkdir(reshard_dir)
        return True

    resharded_meta = DatasetMeta.from_file(metafile_path)
    meta.transfer_data_from(resharded_meta)

    if resharded_meta.is_aligning_with(meta):
        return False

    logging.info(
        f"Metadata mismatching, removing existing resharded directory `{reshard_dir}`"
    )
    shutil.rmtree(reshard_dir)
    os.mkdir(reshard_dir)
    return True


def path_resolver(split_name: str, base_path: Union[str, Path]) -> List[Path]:
    if isinstance(base_path, str):
        base_path = Path(base_path).resolve()
    else:
        base_path.resolve()

    return list(base_path.rglob(f"{split_name}-*.jsonl.gz"))


def _resharder_for_eval(
    sub_split: str,
    s_name: str,
    gpus_count: Union[int, List[int]],
    worker_per_gpu_count: int,
    batch_size: int,
    req_total_shards: int,
):

    ctx = mp.get_context("fork")

    reshard_dir = os.path.join(DATA_DIR, s_name, RESHARD_DIR_NAME)
    meta = DatasetMeta(batch_size, [s_name], gpus_count, worker_per_gpu_count)
    reshard_req = _is_reshard_required(reshard_dir, sub_split, meta)

    if not reshard_req:
        return

    data_q = ctx.Queue(-1)
    shards = path_resolver(sub_split, os.path.join(DATA_DIR, s_name, "processed"))
    shards.sort()

    samples_count = (
        _get_total_samplescount(shards)
        if (getattr(meta, f"{sub_split}_samples_count") is None)
        else getattr(meta, f"{sub_split}_samples_count")
    )
    samples_per_shard = get_samplescount_per_shard(samples_count, req_total_shards)

    p = ctx.Process(target=_reader, args=(shards, data_q))
    p.start()

    _writer(
        data_q,
        1,
        req_total_shards,
        sub_split,
        id=s_name,
        base_dir=reshard_dir,
        filename_stem=sub_split,
        extension="jsonl.gz",
        count_per_file=samples_per_shard,
    )

    meta.compute_max_safe_batches_count().write_to(
        os.path.join(reshard_dir, f"meta-{sub_split}.json")
    )
    p.join()


def resharder_for_eval(
    sub_split: str,
    symbols_info: Dict[str, Dict[str, bool]],
    gpus_count: Union[int, List[int]],
    worker_per_gpu_count: int,
    batch_size: int,
):
    assert (
        sub_split in SPLITS["eval"]
    ), f"Only {SPLITS['eval']} sub splits are supported, but got {sub_split}."

    procs: List[mp.Process] = []
    ctx = mp.get_context("fork")
    req_total_shards = gpus_count * worker_per_gpu_count
    for s_name, s_info in symbols_info.items():
        if not s_info[sub_split]:
            continue

        p = ctx.Process(
            target=_resharder_for_eval,
            args=(
                sub_split,
                s_name,
                gpus_count,
                worker_per_gpu_count,
                batch_size,
                req_total_shards,
            ),
        )
        p.start()

        procs.append(p)

    for p in procs:
        p.join()


def _samples_counter_for_train(shards: List[List[Path]]) -> int:
    ctx = mp.get_context("fork")
    data_q = ctx.Queue(len(shards))
    procs: List[mp.Process] = []
    total_samples_count: int = 0

    for sub_shards in shards:
        p = ctx.Process(target=_get_total_samplescount, args=(sub_shards, data_q))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        samples_count = data_q.get()
        total_samples_count += samples_count

    return total_samples_count


def resharder_for_train(
    sub_split: str,
    symbols_info: Dict[str, Dict[str, bool]],
    gpus_count: Union[int, List[int]],
    worker_per_gpu_count: int,
    batch_size: int,
):
    assert (
        sub_split in SPLITS["train"]
    ), f"Only {SPLITS['train']} sub splits are supported, but got {sub_split}."

    reshard_dir = os.path.join(DATA_DIR, RESHARD_DIR_NAME)
    req_total_shards = gpus_count * worker_per_gpu_count
    ctx = mp.get_context("fork")
    procs: List[mp.Process] = []
    meta = DatasetMeta(
        batch_size,
        [s_name for s_name, s_info in symbols_info.items() if s_info["train"]],
        gpus_count,
        worker_per_gpu_count,
    )

    reshard_req = _is_reshard_required(reshard_dir, sub_split, meta)
    if not reshard_req:
        return

    shards: List[List[Path]] = []
    data_q = ctx.Queue(-1)

    for s_name, s_info in symbols_info.items():
        if not s_info["train"]:
            continue

        sub_shards = path_resolver("train", os.path.join(DATA_DIR, s_name, "processed"))
        sub_shards.sort()
        shards.append(sub_shards)

    samples_count = (
        _samples_counter_for_train(shards)
        if (getattr(meta, f"{sub_split}_samples_count") is None)
        else getattr(meta, f"{sub_split}_samples_count")
    )
    samples_per_shard = get_samplescount_per_shard(samples_count, req_total_shards)

    for sub_shards in shards:
        p = ctx.Process(target=_reader, args=(sub_shards, data_q))
        p.start()
        procs.append(p)

    _writer(
        data_q,
        len(shards),
        req_total_shards,
        sub_split,
        base_dir=reshard_dir,
        filename_stem=sub_split,
        extension="jsonl.gz",
        count_per_file=samples_per_shard,
    )

    meta.compute_max_safe_batches_count().write_to(
        os.path.join(reshard_dir, f"meta-{sub_split}.json")
    )
    for p in procs:
        p.join()


def reshard_if_needed(
    symbols_info: Dict[str, Dict[str, bool]],
    gpus_count: Union[int, List[int]],
    worker_per_gpu_count: int,
    batch_size: int,
    only_selective_splits: List[str] = SUB_SPLITS,
) -> DatasetMeta:
    if not isinstance(gpus_count, int):
        gpus_count = len(gpus_count)

    gl = globals()
    ctx = mp.get_context("fork")
    procs: List[mp.Process] = []
    dataset_meta = DatasetMeta(batch_size, list(symbols_info.keys()))

    for split, sub_splits in SPLITS:
        resharder = gl.get(f"resharder_for_{split}")
        for sub_split in sub_splits:
            if sub_split not in only_selective_splits:
                continue

            p = ctx.Process(
                target=resharder,
                args=(
                    sub_split,
                    symbols_info,
                    gpus_count,
                    worker_per_gpu_count,
                    batch_size,
                ),
                daemon=False,
            )
            p.start()

            procs.append(p)

    for p in procs:
        p.join()

    meta = DatasetMeta.from_file(
        os.path.join(DATA_DIR, RESHARD_DIR_NAME, "meta-train.json")
    ).compute_max_safe_batches_count()

    assert (
        meta.train_steps_count is not None
    ), "Expected meta to be updated by train steps count, but not..."
    return meta
