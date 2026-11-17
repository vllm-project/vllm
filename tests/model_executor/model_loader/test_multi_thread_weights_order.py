# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time

import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader import weight_utils


def _write_shards(tmp_path, num_files):
    files = []
    for i in range(num_files):
        path = tmp_path / f"model-{i:05d}.safetensors"
        save_file(
            {
                f"w{i}.a": torch.full((4,), float(i)),
                f"w{i}.b": torch.full((2,), float(i)),
            },
            str(path),
        )
        files.append(str(path))
    return files


def test_multi_thread_iterator_yields_in_file_order(tmp_path, monkeypatch):
    files = _write_shards(tmp_path, 6)
    real_load_file = weight_utils.load_file

    # Earlier shards finish last, so completion order is the reverse of file order.
    def slow_load_file(path, device="cpu"):
        time.sleep(0.05 * (len(files) - files.index(path)))
        return real_load_file(path, device=device)

    monkeypatch.setattr(weight_utils, "load_file", slow_load_file)
    iterator = weight_utils.multi_thread_safetensors_weights_iterator(
        files, False, max_workers=4
    )
    shards = [name.split(".")[0] for name, _ in iterator]
    assert shards == [f"w{i}" for i in range(len(files)) for _ in range(2)]


def test_multi_thread_iterator_bounds_in_flight_shards(tmp_path, monkeypatch):
    files = _write_shards(tmp_path, 8)
    real_load_file = weight_utils.load_file
    started = []
    lock = threading.Lock()

    def counting_load_file(path, device="cpu"):
        with lock:
            started.append(path)
        return real_load_file(path, device=device)

    monkeypatch.setattr(weight_utils, "load_file", counting_load_file)
    iterator = weight_utils.multi_thread_safetensors_weights_iterator(
        files, False, max_workers=2
    )
    next(iterator)
    time.sleep(0.2)
    # Only the shard being consumed and the prefetch window may have been read.
    assert len(started) <= 3
    assert len(list(iterator)) == 2 * len(files) - 1
