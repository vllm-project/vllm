# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 mono decode layer runs only where its kernels are exact:
decode-only steps of at most MAX_ROWS rows with causal SWA windows, eager or in
a FULL graph. A PIECEWISE capture is replayed for mixed batches, so it must
never record the mono launches."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v41.amd import mono_decode as md

M = 6


class _Runner:
    def __init__(self):
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ("mono outputs",)


def _layer():
    cache = lambda: torch.zeros(4, 128 * md.RECORD, dtype=torch.uint8)  # noqa: E731
    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=cache()),
        compressed_cache_prefix="comp",
        topk_indices_buffer=torch.zeros(64, 512, dtype=torch.int32),
        compress_ratio=1,
        _compressed_kv_cache=cache,
    )
    return SimpleNamespace(attn=attn)


def _swa(**overrides):
    fields = dict(
        num_prefills=0,
        num_decodes=1,
        num_decode_tokens=M,
        decode_swa_width=md.SWA_WIDTH,
        decode_swa_indices=torch.zeros(64, 1, md.SWA_WIDTH, dtype=torch.int32),
        decode_swa_lens=torch.zeros(64, dtype=torch.int32),
        token_to_req_indices=torch.zeros(64, dtype=torch.int32),
        slot_mapping=torch.zeros(64, dtype=torch.int64),
        block_size=32,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _inputs(rows=M):
    return dict(
        x=torch.zeros(rows, 5120, dtype=torch.bfloat16),
        positions=torch.zeros(rows, dtype=torch.int64),
        residual=torch.zeros(rows, 4, 5120, dtype=torch.bfloat16),
        post_mix=torch.zeros(rows, 4, 1),
        res_mix=torch.zeros(rows, 4, 4),
        pre_mix=torch.zeros(rows, 4),
    )


@pytest.fixture
def run(monkeypatch):
    """Call the mono path for one layer under a forward context built from
    ``mode`` and ``metadata``; returns (result, runner calls)."""

    def _run(mode=CUDAGraphMode.FULL, metadata=None, **inputs):
        if metadata is None:
            metadata = {
                "swa": _swa(),
                "comp": SimpleNamespace(block_size=128, block_table=None),
            }
        fc = SimpleNamespace(cudagraph_runtime_mode=mode, attn_metadata=metadata)
        runner = _Runner()
        monkeypatch.setattr(md, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(md, "get_forward_context", lambda: fc)
        monkeypatch.setattr(md, "_mono_runner", lambda device: runner)
        monkeypatch.setattr(md.MonoDecodeLayer, "weights", lambda self, layer: None)
        args = {**_inputs(), **inputs}
        return md.MonoDecodeLayer()(_layer(), **args), runner.calls

    return _run


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_decode_step_runs_mono(run, mode):
    out, calls = run(mode)
    assert out == ("mono outputs",) and len(calls) == 1
    args, kwargs = calls[0]
    # the caches reach the kernels as [blocks, block, record] bytes
    assert args[8].shape == (4, 32, md.RECORD) and args[8].stride() == (
        128 * md.RECORD,
        md.RECORD,
        1,
    )
    assert kwargs["comp_cache"].shape == (4, 128, md.RECORD)


@pytest.mark.parametrize(
    "case",
    [
        dict(mode=CUDAGraphMode.PIECEWISE),
        dict(
            metadata={
                "swa": _swa(num_prefills=1),
                "comp": SimpleNamespace(block_size=128),
            }
        ),
        dict(
            metadata={
                "swa": _swa(decode_swa_width=256),
                "comp": SimpleNamespace(block_size=128),
            }
        ),
        dict(
            metadata={
                "swa": _swa(num_decode_tokens=M + 1),
                "comp": SimpleNamespace(block_size=128),
            }
        ),
        dict(metadata={}),  # a profile / dummy run: no attention metadata
        dict(**_inputs(md.MAX_ROWS + 6)),
        dict(residual=None),  # the first layer's seam broadcasts the embedding
    ],
    ids=[
        "piecewise",
        "prefill",
        "noncausal-window",
        "padding",
        "no-metadata",
        "rows",
        "first-layer",
    ],
)
def test_other_steps_take_the_original_path(run, case):
    out, calls = run(**case)
    assert out is None and not calls
