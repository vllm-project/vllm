# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for checkpoint prefetching."""

import io
import threading
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from vllm.model_executor.model_loader import weight_utils
from vllm.platforms import current_platform

FILES = [f"model-{i:05d}-of-00011.safetensors" for i in range(11)]


class _InlineThread:
    def __init__(self, target, daemon):
        self.target = target

    def start(self):
        self.target()


def _prefetched_files(monkeypatch, files):
    prefetched = []
    monkeypatch.setattr(
        weight_utils, "threading", SimpleNamespace(Thread=_InlineThread)
    )
    monkeypatch.setattr(
        weight_utils, "_prefetch_checkpoint", lambda p, _: prefetched.append(p)
    )
    weight_utils._prefetch_all_checkpoints(files, num_prefetch_threads=1)
    return prefetched


@pytest.mark.parametrize("num_nodes,ranks_per_node", [(1, 1), (1, 4), (2, 4), (4, 2)])
def test_each_node_prefetches_all_files(monkeypatch, num_nodes, ranks_per_node):
    world_size = num_nodes * ranks_per_node
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: world_size > 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(
        weight_utils, "get_node_count", lambda: num_nodes, raising=False
    )
    for node in range(num_nodes):
        node_files = []
        for local_rank in range(ranks_per_node):
            rank = node * ranks_per_node + local_rank
            world = SimpleNamespace(
                world_size=world_size, rank=rank, local_rank=local_rank
            )
            monkeypatch.setattr(torch.distributed, "get_rank", lambda r=rank: r)
            monkeypatch.setattr(weight_utils, "get_world_group", lambda w=world: w)
            node_files += _prefetched_files(monkeypatch, FILES)
        # The page cache is per node: each node must read the whole checkpoint.
        assert sorted(node_files) == FILES, f"node {node}"


def test_auto_prefetch_on_wekafs(tmp_path, monkeypatch, caplog_vllm, vllm_runner):
    """Loading a local checkpoint with the default strategy prefetches on WekaFS."""
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        LlamaForCausalLM(config).save_pretrained(tmp_path)

    def open_file(path, *args, **kwargs):
        if path == "/proc/mounts":
            return io.StringIO(f"server {tmp_path} wekafs rw 0 0\n")
        return open(path, *args, **kwargs)

    monkeypatch.setattr(weight_utils, "open", open_file, raising=False)
    # Keep the simulated mount and log capture visible to the model loader.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    finished = threading.Event()

    def signal_completion(record):
        if record.getMessage().startswith(
            "Prefetching checkpoint files into page cache finished"
        ):
            finished.set()
        return True

    caplog_vllm.handler.addFilter(signal_completion)
    try:
        with vllm_runner(
            str(tmp_path),
            dtype="float32" if current_platform.is_cpu() else "float16",
            skip_tokenizer_init=True,
            enforce_eager=True,
            compilation_config={"mode": 0},
            max_model_len=32,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            kv_cache_memory_bytes=1 << 20,
            gpu_memory_utilization=0.1,
        ):
            assert (
                "Prefetching checkpoint files into page cache started"
                in caplog_vllm.text
            )
            assert finished.wait(timeout=10), (
                "WekaFS checkpoint prefetch did not finish"
            )
            assert "Failed to prefetch checkpoint file" not in caplog_vllm.text
    finally:
        caplog_vllm.handler.removeFilter(signal_completion)
