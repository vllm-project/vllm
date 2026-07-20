# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@dataclass
class _GrammarOutput:
    structured_output_request_ids: list[str]
    grammar_bitmask: np.ndarray


class _SchedulerOutput:
    pass


class _LRUCache(dict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize


def _async_tensor_h2d(data, device, dtype=None):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=dtype, device="cpu")
    return data.to(device=device, dtype=dtype, non_blocking=True)


def _load_utils_module(monkeypatch: pytest.MonkeyPatch):
    def init_logger(_name: str):
        return SimpleNamespace()

    def lazy_loader(_name: str, _globals: dict, _module_name: str):
        return SimpleNamespace()

    stubs = {
        "cachetools": SimpleNamespace(LRUCache=_LRUCache),
        "vllm": types.ModuleType("vllm"),
        "vllm.envs": SimpleNamespace(VLLM_REGEX_COMPILATION_TIMEOUT_S=0),
        "vllm.logger": SimpleNamespace(init_logger=init_logger),
        "vllm.utils": types.ModuleType("vllm.utils"),
        "vllm.utils.import_utils": SimpleNamespace(LazyLoader=lazy_loader),
        "vllm.utils.torch_utils": SimpleNamespace(async_tensor_h2d=_async_tensor_h2d),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.core": types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.output": SimpleNamespace(
            GrammarOutput=_GrammarOutput, SchedulerOutput=_SchedulerOutput
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = (
        Path(__file__).parents[3] / "vllm" / "v1" / "structured_output" / "utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_vllm_structured_output_utils", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeXGrammar:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_token_bitmask_inplace(self, logits, bitmask, indices=None) -> None:
        if bitmask.is_cuda or (isinstance(indices, torch.Tensor) and indices.is_cuda):
            torch.accelerator.synchronize()
        indices_cpu = (
            indices.detach().cpu().tolist() if torch.is_tensor(indices) else indices
        )
        bitmask_cpu = bitmask.detach().cpu().clone()
        self.calls.append(
            {
                "bitmask": bitmask_cpu,
                "indices": indices_cpu,
                "bitmask_device": bitmask.device.type,
                "bitmask_dtype": bitmask.dtype,
            }
        )
        rows = range(logits.shape[0]) if indices_cpu is None else indices_cpu
        for row in rows:
            logits[row, 0] = bitmask_cpu[row, 0].to(logits.dtype)


def _mask(rows: int, width: int = 4, base: int = 0) -> np.ndarray:
    return np.arange(base, base + rows * width, dtype=np.int32).reshape(rows, width)


def _expected_sorted_bitmask(
    req_ids: list[str],
    structured_ids: list[str],
    grammar_bitmask: np.ndarray,
    spec_tokens: dict[str, list[int]],
    logits_rows: int,
) -> tuple[torch.Tensor, list[int]]:
    structured_id_set = set(structured_ids)
    batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    for batch_index, req_id in enumerate(req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in structured_id_set:
            batch_indices[req_id] = logit_index

    sorted_bitmask = np.full(
        (logits_rows, grammar_bitmask.shape[1]), -1, dtype=grammar_bitmask.dtype
    )
    out_indices: list[int] = []
    cumulative_index = 0
    for req_id in structured_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens
    return torch.from_numpy(sorted_bitmask), out_indices


def _torch_full_reference(
    req_ids: list[str],
    structured_ids: list[str],
    grammar_bitmask: np.ndarray,
    spec_tokens: dict[str, list[int]],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, list[int] | None, torch.Tensor]:
    expected_host, out_indices = _expected_sorted_bitmask(
        req_ids, structured_ids, grammar_bitmask, spec_tokens, logits.shape[0]
    )
    sorted_bitmask_tensor = torch.full(
        expected_host.shape,
        -1,
        dtype=torch.from_numpy(grammar_bitmask[:0]).dtype,
        pin_memory=logits.is_cuda,
    )
    sorted_bitmask_tensor.copy_(expected_host)
    device_bitmask = sorted_bitmask_tensor.to(logits.device, non_blocking=True)

    indices = None if len(out_indices) == logits.shape[0] else out_indices
    expected_logits = torch.zeros_like(logits)
    bitmask_on_device = expected_host.to(logits.device)
    rows = range(logits.shape[0]) if indices is None else indices
    for row in rows:
        expected_logits[row, 0] = bitmask_on_device[row, 0].to(logits.dtype)
    if logits.is_cuda:
        torch.accelerator.synchronize()
    return device_bitmask.detach().cpu(), indices, expected_logits


def _run_and_compare_to_torch_full_reference(
    monkeypatch: pytest.MonkeyPatch,
    *,
    req_ids: list[str],
    structured_ids: list[str],
    grammar_bitmask: np.ndarray,
    spec_tokens: dict[str, list[int]] | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> _FakeXGrammar:
    if device == "cuda" and torch.accelerator.device_count() == 0:
        pytest.skip("CUDA is required for this test")

    spec_tokens = spec_tokens or {}
    logits_rows = len(req_ids) + sum(len(v) for v in spec_tokens.values())
    logits = torch.zeros((logits_rows, 8), dtype=dtype, device=device)
    ref_bitmask, ref_indices, ref_logits = _torch_full_reference(
        req_ids, structured_ids, grammar_bitmask, spec_tokens, logits
    )

    utils = _load_utils_module(monkeypatch)
    fake_xgr = _FakeXGrammar()
    utils.xgr = fake_xgr
    utils.apply_grammar_bitmask(
        SimpleNamespace(scheduled_spec_decode_tokens=spec_tokens),
        _GrammarOutput(structured_ids, grammar_bitmask),
        SimpleNamespace(req_ids=req_ids),
        logits,
    )
    if device == "cuda":
        torch.accelerator.synchronize()

    assert len(fake_xgr.calls) == 1
    call = fake_xgr.calls[0]
    torch.testing.assert_close(call["bitmask"], ref_bitmask, rtol=0, atol=0)
    assert call["indices"] == ref_indices
    assert call["bitmask_dtype"] == torch.int32
    assert call["bitmask_device"] == torch.device(device).type
    torch.testing.assert_close(logits, ref_logits, rtol=0, atol=0)
    return fake_xgr


@pytest.mark.skipif(torch.accelerator.device_count() == 0, reason="CUDA is required")
@pytest.mark.parametrize(
    ("req_ids", "structured_ids", "grammar_bitmask", "spec_tokens", "indices"),
    [
        pytest.param(
            ["a", "b"],
            [],
            np.empty((0, 4), dtype=np.int32),
            {},
            [],
            id="empty",
        ),
        pytest.param(
            ["a", "b", "c"],
            ["a", "b", "c"],
            _mask(3, base=10),
            {},
            None,
            id="all-active",
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            ["d", "a"],
            _mask(2, base=100),
            {},
            [3, 0],
            id="reordered-mixed",
        ),
        pytest.param(
            ["a", "b", "c"],
            ["c", "a"],
            _mask(5, base=200),
            {"a": [11, 12], "c": [13]},
            [4, 5, 0, 1, 2],
            id="speculative-offsets",
        ),
    ],
)
def test_cuda_bitmask_rows_match_torch_full_reference(
    monkeypatch, req_ids, structured_ids, grammar_bitmask, spec_tokens, indices
):
    fake_xgr = _run_and_compare_to_torch_full_reference(
        monkeypatch,
        req_ids=req_ids,
        structured_ids=structured_ids,
        grammar_bitmask=grammar_bitmask,
        spec_tokens=spec_tokens,
    )
    assert fake_xgr.calls[0]["indices"] == indices


@pytest.mark.skipif(torch.accelerator.device_count() == 0, reason="CUDA is required")
def test_cuda_repeated_calls_do_not_reuse_stale_rows(monkeypatch):
    _run_and_compare_to_torch_full_reference(
        monkeypatch,
        req_ids=["a", "b", "c", "d"],
        structured_ids=["a", "c"],
        grammar_bitmask=_mask(2, base=300),
    )
    fake_xgr = _run_and_compare_to_torch_full_reference(
        monkeypatch,
        req_ids=["a", "b", "c", "d"],
        structured_ids=["b"],
        grammar_bitmask=_mask(1, base=500),
    )
    expected = torch.full((4, 4), -1, dtype=torch.int32)
    expected[1] = torch.from_numpy(_mask(1, base=500)[0])
    torch.testing.assert_close(fake_xgr.calls[0]["bitmask"], expected, rtol=0, atol=0)


def test_cpu_path_preserves_logits_dtype_and_matches_torch_full_reference(monkeypatch):
    _run_and_compare_to_torch_full_reference(
        monkeypatch,
        req_ids=["a", "b", "c"],
        structured_ids=["c", "a"],
        grammar_bitmask=_mask(2, base=700),
        device="cpu",
        dtype=torch.float16,
    )
