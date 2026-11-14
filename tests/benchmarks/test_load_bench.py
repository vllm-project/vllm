# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.entrypoints.cli.load_bench import add_arguments, consume_tensor, run


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("shape", [(), (0,), (37,)])
def test_copy_consumes_payload_with_partial_last_chunk(dtype, shape):
    tensor = torch.ones(shape, dtype=dtype)
    buffer = torch.zeros(7, dtype=torch.uint8)
    size = consume_tensor(tensor, buffer)
    raw = tensor.reshape(-1).view(torch.uint8)
    assert size == tensor.numel() * tensor.element_size()
    if size:
        tail = (size - 1) % buffer.numel() + 1
        assert torch.equal(buffer[:tail], raw[-tail:])


@pytest.mark.parametrize(
    "strategy",
    ["lazy", "prefetch", "streaming", "streaming_multithread", "eager", "multithread"],
)
def test_native_loader_consumes_only_indexed_shards_on_cpu(tmp_path, strategy):
    save_file(
        {"weight": torch.ones(37, dtype=torch.bfloat16)}, tmp_path / "a.safetensors"
    )
    save_file({"unused": torch.ones(100)}, tmp_path / "unused.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "a.safetensors"}})
    )
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    argv = ["--model-path", str(tmp_path)]
    if strategy in ("multithread", "streaming_multithread"):
        argv += ["--model-loader-extra-config", '{"enable_multithread_load":true}']
        if strategy == "streaming_multithread":
            argv += ["--safetensors-load-strategy", "streaming"]
    else:
        argv += ["--safetensors-load-strategy", strategy]
    result = run(parser.parse_args(argv))
    assert result["tensors"] == 1
    assert result["tensor_bytes"] == 74
    assert result["load_seconds"] > 0


def test_prefetch_only_joins_selected_checkpoint_reads(tmp_path):
    save_file(
        {"weight": torch.ones(37, dtype=torch.bfloat16)}, tmp_path / "a.safetensors"
    )
    save_file(
        {"unused": torch.ones(100, dtype=torch.bfloat16)},
        tmp_path / "unused.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "a.safetensors"}})
    )
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    result = run(
        parser.parse_args(
            [
                "--model-path",
                str(tmp_path),
                "--prefetch-only",
                "--prefetch-file-window",
                "2",
                "--io-threads-per-file",
                "3",
                "--safetensors-prefetch-block-size",
                "16",
            ]
        )
    )
    assert result["benchmark"] == "checkpoint_prefetch_only"
    assert result["files"] == 1
    assert result["checkpoint_bytes"] == (tmp_path / "a.safetensors").stat().st_size
    assert result["prefetch_file_window"] == 2
    assert result["io_threads_per_file"] == 3
    assert result["file_parallelism"] == 1
    assert result["safetensors_prefetch_block_size"] == 16
    assert result["prefetch_seconds"] > 0
    assert result["prefetch_gib_per_second"] > 0
