# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import logging
import threading

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from vllm import SamplingParams
from vllm.model_executor.model_loader import weight_utils


@pytest.fixture
def checkpoint_filesystem(monkeypatch, tmp_path):
    real_open = open

    def set_filesystem(fs_type):
        mounts = f"/dev/root / ext4 rw 0 0\nserver {tmp_path} {fs_type} rw 0 0\n"

        def open_file(path, *args, **kwargs):
            if str(path) == "/proc/mounts":
                return io.StringIO(mounts)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", open_file)

    return set_filesystem


@pytest.fixture
def prefetch_log(caplog_vllm):
    caplog_vllm.set_level(logging.INFO, logger=weight_utils.__name__)

    class PrefetchLog(logging.Handler):
        def __init__(self):
            super().__init__()
            self.started = False
            self.failed = False
            self.finished = threading.Event()

        def emit(self, record):
            if record.name != weight_utils.__name__:
                return
            message = record.getMessage()
            if message.startswith(
                "Prefetching checkpoint files into page cache started"
            ):
                self.started = True
            if message.startswith(
                "Prefetching checkpoint files into page cache finished"
            ):
                self.finished.set()
            if message.startswith("Failed to prefetch checkpoint file"):
                self.failed = True

    handler = PrefetchLog()
    logger = logging.getLogger()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        try:
            if handler.started:
                assert handler.finished.wait(timeout=10), "Prefetch did not finish"
                assert not handler.failed, (
                    "Checkpoint prefetch failed; see captured logs"
                )
        finally:
            logger.removeHandler(handler)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "fs_type,strategy,fits_in_ram,expected_prefetch",
    [
        ("nfs", None, True, True),
        ("nfs4", None, True, True),
        ("lustre", None, True, True),
        ("wekafs", None, True, True),
        ("wekafs", None, False, False),
        ("wekafs", "lazy", True, False),
        ("wekafs", "eager", True, False),
        ("wekafs", "prefetch", False, True),
        ("ext4", None, True, False),
        ("ext4", "prefetch", True, True),
    ],
)
def test_safetensors_prefetch_respects_filesystem_memory_and_strategy(
    tmp_path,
    monkeypatch,
    checkpoint_filesystem,
    prefetch_log,
    fs_type,
    strategy,
    fits_in_ram,
    expected_prefetch,
):
    """Auto-prefetch recognizes network mounts without changing loaded weights."""
    checkpoint = tmp_path / "model.safetensors"
    weights = {"weight": torch.arange(18, dtype=torch.float32)}
    save_file(weights, checkpoint)
    checkpoint_filesystem(fs_type)
    # Exercise the 90% limit and the immediately smaller RAM budget.
    minimum_ram = (checkpoint.stat().st_size * 10 + 8) // 9
    available = minimum_ram if fits_in_ram else minimum_ram - 1
    monkeypatch.setattr(weight_utils, "get_available_memory", lambda: available)

    loaded = dict(
        weight_utils.safetensors_weights_iterator(
            [str(checkpoint)],
            False,
            safetensors_load_strategy=strategy,
        )
    )

    assert prefetch_log.started is expected_prefetch
    assert loaded.keys() == weights.keys()
    assert torch.equal(loaded["weight"], weights["weight"])


def test_llm_wekafs_auto_prefetch_preserves_greedy_tokens(
    tmp_path,
    monkeypatch,
    vllm_runner,
    checkpoint_filesystem,
    prefetch_log,
):
    """The default WekaFS load prefetches and matches explicit lazy loading."""
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        LlamaForCausalLM(config).save_pretrained(tmp_path)
    checkpoint_filesystem("wekafs")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sampling = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        detokenize=False,
    )
    outputs = []
    for strategy in ("lazy", None):
        with vllm_runner(
            str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float32",
            max_model_len=32,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            kv_cache_memory_bytes=1 << 20,
            enforce_eager=True,
            compilation_config={"mode": 0},
            safetensors_load_strategy=strategy,
        ) as runner:
            result = runner.llm.generate(
                [{"prompt_token_ids": [1, 3, 5]}],
                sampling,
                use_tqdm=False,
            )
            outputs.append(result[0].outputs[0].token_ids)
        assert prefetch_log.started is (strategy is None)
    assert len(outputs[0]) == 4
    assert outputs[0] == outputs[1]
