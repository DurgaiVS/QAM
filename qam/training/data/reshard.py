import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch.multiprocessing as mp
from tqdm import tqdm

from ...utils import QAMFileWriter

# from common_utils import SUBSET, DatasetMeta, QAMFileWriter, yield_groups_from_file

RESHARD_DIR_NAME = "resharded"


def _get_total_groups_from_sources(
    shards: List[str], max_words_per_group: int, workers_count: int
) -> int:
    total_groups = 0

    for shard in shards:
        for _ in yield_groups_from_file(shard, max_words_per_group):
            total_groups += 1

    return total_groups


def _get_total_groups_from_sources_dist(
    shards: List[str], max_words_per_group: int, workers_count: int
) -> int:
    ctx = mp.get_context("fork")
    data_q = ctx.Queue(workers_count)
    total_groups = 0
    procs = []

    def _update_total_groups(
        shards: List[str], max_words_per_group: int, data_q: "mp.Queue"
    ):
        groups = 0
        for shard in shards:
            for _ in yield_groups_from_file(shard, max_words_per_group):
                groups += 1

        data_q.put(groups)

    for i in range(workers_count):
        p = ctx.Process(
            target=_update_total_groups,
            args=(shards[i::workers_count], max_words_per_group, data_q),
            daemon=True,
        )
        p.start()
        procs.append(p)

    for p in procs:
        total_groups += data_q.get()
        p.join()

    return total_groups


def _reader(shards: List[str], max_words_per_group: int, data_q: "mp.Queue"):

    for shard in shards:
        for grp in yield_groups_from_file(shard, max_words_per_group):
            data_q.put(grp)

    data_q.put(None)


def _writer(
    data_q: "mp.Queue",
    readers_count: int,
    total_shards_req: int,
    group_type: str,
    id: int,
    **kwargs,
):

    pbar = tqdm(total=total_shards_req, desc=f"{group_type}-{id}", leave=False)
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


def _parallel_resharding(
    shards: List[str],
    total_shards_req: int,
    group_type: str,
    max_words_per_group: int,
    workers_count: int,
    id: int,
    **kwargs,
):
    ctx = mp.get_context("fork")
    data_q = ctx.Queue(-1)
    procs: List[mp.Process] = []

    for i in range(workers_count):
        p = ctx.Process(
            target=_reader,
            args=(shards[i::workers_count], max_words_per_group, data_q),
            daemon=True,
        )
        p.start()

        procs.append(p)

    _writer(data_q, workers_count, total_shards_req, group_type, id, **kwargs)

    for p in procs:
        p.kill()


def _serial_resharding(
    shards: List[str],
    total_shards_req: int,
    group_type: str,
    max_words_per_group: int,
    workers_count: int,
    id: int,
    **kwargs,
):
    pbar = tqdm(total=total_shards_req, desc=f"{group_type}-{id}", leave=False)
    _fc = 0

    with QAMFileWriter(**kwargs) as itn_writer:
        for shard in shards:
            for grp in yield_groups_from_file(shard, max_words_per_group):
                itn_writer.write(grp)

                if itn_writer._files_counter != _fc:
                    pbar.update()
                    _fc = itn_writer._files_counter

                    if _fc == total_shards_req:
                        break
            if _fc == total_shards_req:
                break

    pbar.close()


def _reshard_datasets(
    shards: List[str],
    reshard_dir: str,
    group_type: str,
    gpus_count: int,
    worker_per_gpu_count: int,
    max_words_per_group: int,
    info_q: "mp.Queue",
    total_groups: Optional[int] = None,
    req_dist_workers: bool = False,
    id: int = 0,
):
    tqdm.set_lock(tqdm.get_lock())
    if req_dist_workers:
        get_total_groups = _get_total_groups_from_sources_dist
        reshard = _parallel_resharding
    else:
        get_total_groups = _get_total_groups_from_sources
        reshard = _serial_resharding
    workers_count = min(worker_per_gpu_count, len(shards))

    total_groups = (
        total_groups
        if total_groups
        else get_total_groups(shards, max_words_per_group, worker_per_gpu_count)
    )
    total_shards_req = gpus_count * worker_per_gpu_count

    if total_groups < gpus_count:
        logging.info(
            f"Groups count {total_groups} is much lesser than GPUs count {gpus_count} for case `{group_type}-{id}`. So, skipping resharding."
        )
        return

    while (total_groups < total_shards_req) and (total_shards_req > gpus_count):
        total_shards_req -= gpus_count

    groups_per_shard = total_groups // total_shards_req
    # extra_one_rounder = 0 # total_groups % total_shards_req
    info_q.put((group_type, groups_per_shard * total_shards_req))

    reshard(
        shards=shards,
        total_shards_req=total_shards_req,
        group_type=group_type,
        max_words_per_group=max_words_per_group,
        workers_count=workers_count,
        id=id,
        base_dir=reshard_dir,
        filename_stem=group_type,
        extension="jsonl.gz",
        count_per_file=groups_per_shard,
    )

    logging.info(f"Resharding done for case `{group_type}-{id}`.")


def _is_reshard_required(
    reshard_dir: str,
    dataset_names: List[str],
    gpus_count: int,
    worker_per_gpu_count: int,
    max_words_per_group: int,
) -> Tuple[bool, Dict[str, str]]:
    metafile_path = os.path.join(reshard_dir, "meta.json")

    if not os.path.exists(metafile_path):
        if os.path.exists(reshard_dir):
            logging.info(
                f"Cannot find metadata file, removing existing resharded directory `{reshard_dir}`"
            )
            shutil.rmtree(reshard_dir)
        os.mkdir(reshard_dir)
        return True, {}

    with open(metafile_path, "r") as f:
        meta = json.load(f)

    if (
        (meta["gpus_count"] % gpus_count == 0)
        and all(
            [True if ds in meta["dataset_names"] else False for ds in dataset_names]
        )
        and (meta["worker_per_gpu_count"] % worker_per_gpu_count == 0)
        and (meta["max_words_per_group"] == max_words_per_group)
    ):
        return False, meta

    logging.info(
        f"Metadata mismatching, removing existing resharded directory `{reshard_dir}`"
    )
    shutil.rmtree(reshard_dir)
    os.mkdir(reshard_dir)
    return True, meta


def _get_category(datasets: Dict[str, Dict[str, str]]) -> str:
    is_train = False
    is_eval = False

    for d_name, d_path in datasets.items():
        path = d_path["base_dir"]

        if "training" in path:
            is_train = True
        elif "evaluation" in path:
            is_eval = True

    assert (
        is_train or is_eval
    ), "Got neither of expected dataset categories, training or evaluation."
    assert not (
        is_train and is_eval
    ), "Cannot perform resharding for mixed dataset categories, evaluation and training categories."

    if is_train:
        return "train"
    elif is_eval:
        return "eval"


def path_resolver(group_type: str, base_path: Union[str, Path]) -> List[Path]:
    if isinstance(base_path, str):
        base_path = Path(base_path).resolve()
    else:
        base_path.resolve()

    return list(base_path.glob(f"{group_type}-*.jsonl.gz"))


def _iterator_as_per_category(
    catg: str, ds: Dict[str, Dict[str, str]]
) -> Generator[Tuple[str, Dict[str, Dict[str, str]]], None, None]:
    if catg == "train":
        reshard_dir = (
            Path(next(iter(ds.values()))["base_dir"]).parent / RESHARD_DIR_NAME
        )
        yield str(reshard_dir), ds

    elif catg == "eval":
        for ds_name, ds_val in ds.items():
            reshard_dir = Path(ds_val["base_dir"]) / RESHARD_DIR_NAME
            yield str(reshard_dir), {ds_name: ds_val}


def reshard_if_needed(
    datasets: List[Dict[str, Dict[str, str]]],
    gpus_count: Union[int, List[int]],
    worker_per_gpu_count: int,
    batch_size: int,
    max_words_per_group: int,
) -> DatasetMeta:
    overall_meta = DatasetMeta()

    if not isinstance(gpus_count, int):
        gpus_count = len(gpus_count)

    ctx = mp.get_context("fork")
    info_q = ctx.Queue(-1)
    procs: List[mp.Process] = []

    for ds in datasets:
        catg = _get_category(ds)

        for i, (reshard_dir, ds_subs) in enumerate(_iterator_as_per_category(catg, ds)):
            count = 0
            resharding_required, meta = None, None
            overall_meta.set_reshard_dir_for_category(catg, reshard_dir)
            for group_type in SUBSET[catg]:
                shards: List[Path] = []
                for d_name, d_path in ds_subs.items():
                    shards.extend(path_resolver(group_type, d_path["base_dir"]))

                if len(shards) == 0:
                    continue

                if not isinstance(resharding_required, bool):
                    resharding_required, meta = _is_reshard_required(
                        reshard_dir,
                        ds_subs.keys(),
                        gpus_count,
                        worker_per_gpu_count,
                        max_words_per_group,
                    )

                if not resharding_required:
                    logging.info(f"Resharding not required for category `{catg}-{i}`")
                    break

                logging.info(f"Resharding {catg}'s `{group_type}-{i}` group")
                req_dist_resharding = len(shards) > worker_per_gpu_count
                random.shuffle(shards)

                p = ctx.Process(
                    target=_reshard_datasets,
                    args=(
                        shards,
                        reshard_dir,
                        group_type,
                        gpus_count,
                        worker_per_gpu_count,
                        max_words_per_group,
                        info_q,
                        (
                            meta.get(group_type, None)
                            if meta.get("max_words_per_group", 0) == max_words_per_group
                            else None
                        ),
                        True if req_dist_resharding else False,
                        i,
                    ),
                    daemon=False if req_dist_resharding else True,
                )
                p.start()
                count += 1
                procs.append(p)

            if not resharding_required:
                overall_meta.update_stats_for_category(catg, meta)
                continue

            meta = {
                "dataset_names": list(ds_subs.keys()),
                "gpus_count": gpus_count,
                "worker_per_gpu_count": worker_per_gpu_count,
                "max_words_per_group": max_words_per_group,
            }
            for _ in range(count):
                group_type, samples_count = info_q.get()
                meta[group_type] = samples_count
                overall_meta.update_stats_for_group(group_type, samples_count)

            with open(os.path.join(reshard_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=4)

    for p in procs:
        p.join()

    return overall_meta.compute_max_safe_batches_count(batch_size, gpus_count)
