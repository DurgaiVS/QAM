import logging
import multiprocessing as mp
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm

from ...constants import DATA_DIR, RESHARD_DIR_NAME, SPLITS, SUB_SPLITS
from ...utils import (
    DatasetMeta,
    QAMFileWriter,
    get_samples_and_extrarounder_count_per_shard,
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
        data_q.put(sample.to_bytes())

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

    with QAMFileWriter(**kwargs) as writer:
        while True:
            data = data_q.get()

            if data is None:
                workers_done += 1
                if workers_done == readers_count:
                    break
                else:
                    continue

            writer.write(data)

            if writer._files_counter != _fc:
                pbar.update()
                _fc = writer._files_counter
                if _fc == total_shards_req:
                    break

    pbar.close()


def _is_reshard_required(
    reshard_dir: str,
    meta: DatasetMeta,
) -> bool:
    metafile_path = os.path.join(reshard_dir, f"meta.json")

    if not os.path.exists(metafile_path):
        if os.path.exists(reshard_dir):
            logging.info(
                f"Cannot find metadata file, removing existing resharded directory `{reshard_dir}`"
            )
            shutil.rmtree(reshard_dir)
        os.mkdir(reshard_dir)
        return True

    resharded_meta = DatasetMeta.from_file(metafile_path)
    meta.import_sample_counts(resharded_meta)

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
    datasource_name: str,
    datasource_interval: str,
    req_total_shards: int,
):

    ctx = mp.get_context("fork")

    reshard_dir = os.path.join(
        DATA_DIR,
        datasource_name,
        datasource_interval,
        s_name,
        sub_split,
        RESHARD_DIR_NAME,
    )
    meta = DatasetMeta([s_name], batch_size, gpus_count, worker_per_gpu_count)
    reshard_req = _is_reshard_required(reshard_dir, meta)

    if not reshard_req:
        return

    data_q = ctx.Queue(-1)
    shards = path_resolver(
        sub_split,
        os.path.join(
            DATA_DIR, datasource_name, datasource_interval, s_name, "processed"
        ),
    )
    shards.sort()

    samples_count = meta.get_sample_count(sub_split) or _get_total_samplescount(shards)
    samples_per_shard, extra_one_rounder = get_samples_and_extrarounder_count_per_shard(
        samples_count, req_total_shards
    )

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
        extra_one_rounder=extra_one_rounder,
        writer_mode="b",
    )

    meta.compute_max_safe_batches_count().write_to(
        os.path.join(reshard_dir, f"meta.json")
    )
    p.join()


def resharder_for_eval(
    sub_split: str,
    symbols: Dict[str, Optional[Dict[str, bool]]],
    gpus_count: Union[int, List[int]],
    worker_per_gpu_count: int,
    batch_size: int,
    datasource_name: str,
    datasource_interval: str,
):
    assert (
        sub_split in SPLITS["eval"]
    ), f"Only {SPLITS['eval']} sub splits are supported, but got {sub_split}."

    procs: List[mp.Process] = []
    ctx = mp.get_context("fork")
    req_total_shards = gpus_count * worker_per_gpu_count
    for s_name, s_info in symbols.items():
        if (s_info is not None) and (sub_split not in s_info):
            continue

        p = ctx.Process(
            target=_resharder_for_eval,
            args=(
                sub_split,
                s_name,
                gpus_count,
                worker_per_gpu_count,
                batch_size,
                datasource_name,
                datasource_interval,
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
    symbols: Dict[str, Optional[Dict[str, bool]]],
    gpus_count: Union[int, List[int]],
    worker_per_gpu_count: int,
    batch_size: int,
    datasource_name: str,
    datasource_interval: str,
):
    assert (
        sub_split in SPLITS["train"]
    ), f"Only {SPLITS['train']} sub splits are supported, but got {sub_split}."

    reshard_dir = os.path.join(
        DATA_DIR, datasource_name, datasource_interval, RESHARD_DIR_NAME
    )
    req_total_shards = gpus_count * worker_per_gpu_count
    meta = DatasetMeta(
        [
            s_name
            for s_name, s_info in symbols.items()
            if ((s_info is None) or ("train" in s_info))
        ],
        batch_size,
        gpus_count,
        worker_per_gpu_count,
    )

    procs: List[mp.Process] = []
    ctx = mp.get_context("fork")

    reshard_req = _is_reshard_required(reshard_dir, meta)
    if not reshard_req:
        return

    shards: List[List[Path]] = []
    data_q = ctx.Queue(-1)

    for s_name, s_info in symbols.items():
        if (s_info is not None) and ("train" not in s_info):
            continue

        sub_shards = path_resolver(
            "train",
            os.path.join(
                DATA_DIR, datasource_name, datasource_interval, s_name, "processed"
            ),
        )
        sub_shards.sort()
        shards.append(sub_shards)

    samples_count = meta.get_sample_count(sub_split) or _samples_counter_for_train(
        shards
    )
    samples_per_shard, _ = get_samples_and_extrarounder_count_per_shard(
        samples_count, req_total_shards
    )

    for sub_shards in shards:
        p = ctx.Process(target=_reader, args=(sub_shards, data_q))
        p.start()
        procs.append(p)

    _writer(
        data_q,
        len(shards),
        req_total_shards,
        sub_split,
        id=sub_split,
        base_dir=reshard_dir,
        filename_stem=sub_split,
        extension="jsonl.gz",
        count_per_file=samples_per_shard,
        writer_mode="b",
    )

    meta.compute_max_safe_batches_count().write_to(
        os.path.join(reshard_dir, f"meta.json")
    )
    for p in procs:
        p.join()


def reshard_if_needed(
    symbols: Dict[str, Optional[Dict[str, bool]]],
    gpus_count: Union[int, List[int]],
    accelerator: str,
    worker_per_gpu_count: int,
    batch_size: int,
    datasource_name: str,
    datasource_interval: str,
    only_selective_splits: List[str] = SUB_SPLITS,
) -> DatasetMeta:
    if not isinstance(gpus_count, int):
        gpus_count = len(gpus_count)
    elif accelerator == "gpu" and gpus_count == -1:
        gpus_count = torch.cuda.device_count()
    elif accelerator == "cpu":
        gpus_count = 1

    ctx = mp.get_context("fork")
    procs: List[mp.Process] = []
    fn_as_per_splits = {
        "train": resharder_for_train,
        "eval": resharder_for_eval,
    }

    for split, sub_splits in SPLITS.items():
        resharder = fn_as_per_splits[split]
        for sub_split in sub_splits:
            if sub_split not in only_selective_splits:
                continue

            p = ctx.Process(
                target=resharder,
                args=(
                    sub_split,
                    symbols,
                    gpus_count,
                    worker_per_gpu_count,
                    batch_size,
                    datasource_name,
                    datasource_interval,
                ),
            )
            p.start()

            procs.append(p)

    for p in procs:
        p.join()

    meta = DatasetMeta.from_file(
        os.path.join(
            DATA_DIR,
            datasource_name,
            datasource_interval,
            RESHARD_DIR_NAME,
            "meta.json",
        )
    ).compute_max_safe_batches_count()

    return meta
