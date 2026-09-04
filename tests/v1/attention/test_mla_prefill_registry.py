# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend registry."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)
from vllm.v1.attention.backends.mla.prefill.trtllm_ragged import (
    TrtllmRaggedPrefillBackend,
)


class CustomMLAPrefillBackend(MLAPrefillBackend):
    """Mock custom MLA prefill backend for testing."""

    supported_dtypes = [torch.bfloat16, torch.float16]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    def run_prefill_new_tokens(self, q, k, v, return_softmax_lse):
        raise NotImplementedError

    def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
        raise NotImplementedError


def test_prefill_backend_clone_has_isolated_metadata():
    backend = CustomMLAPrefillBackend(
        num_heads=4,
        scale=0.5,
        kv_lora_rank=8,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=32,
        vllm_config=object(),
    )

    clone = backend.clone()

    assert isinstance(clone, CustomMLAPrefillBackend)
    assert clone is not backend
    assert clone.num_heads == backend.num_heads
    assert clone.scale == backend.scale
    backend._prefill_metadata = object()
    clone._prefill_metadata = object()
    assert clone._prefill_metadata is not backend._prefill_metadata


@pytest.mark.parametrize("method", ["new_tokens", "context_chunk"])
def test_trtllm_ragged_uses_trusted_active_row_path(monkeypatch, method):
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    backend = object.__new__(TrtllmRaggedPrefillBackend)
    backend.scale = 0.5
    backend._workspace_buffer = torch.empty(1, dtype=torch.uint8)
    backend._use_pcp = False
    backend.prepare_metadata(
        SimpleNamespace(
            query_start_loc=query_start_loc,
            max_query_len=2,
            output_dtype=torch.float32,
            query_lens_cpu=torch.diff(query_start_loc),
        )
    )
    captured = {}

    def fake_ragged_attention(**kwargs):
        captured.update(kwargs)
        if method == "context_chunk":
            return kwargs["out"], torch.empty(2, 2)
        return kwargs["out"]

    monkeypatch.setattr(
        "flashinfer.prefill.trtllm_ragged_attention_deepseek",
        fake_ragged_attention,
    )
    q = torch.empty(2, 2, 192)
    num_kv_tokens = 3 if method == "context_chunk" else 2
    k = torch.empty(num_kv_tokens, 2, 192)
    v = torch.empty(num_kv_tokens, 2, 128)

    if method == "new_tokens":
        backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=False)
    else:
        seq_lens = torch.tensor([3], dtype=torch.int32)
        cu_seq_lens = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens, dim=0)]
        )
        backend.run_prefill_context_chunk(
            SimpleNamespace(
                seq_lens=seq_lens,
                max_query_len=2,
                max_seq_len=3,
                num_requests=seq_lens.shape[0],
                query_start_loc=query_start_loc,
                cu_seq_lens=cu_seq_lens,
                all_rows_active=True,
            ),
            q,
            k,
            v,
        )

    assert captured["skip_all_rows_active_check"] is True


@pytest.mark.parametrize(("pcp_size", "expected"), [(1, False), (2, True)])
def test_trtllm_ragged_records_pcp_mode(monkeypatch, pcp_size, expected):
    workspace = torch.empty(1, dtype=torch.uint8)
    manager = SimpleNamespace(get_simultaneous=lambda _: (workspace,))
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.prefill.trtllm_ragged."
        "current_workspace_manager",
        lambda: manager,
    )
    backend = TrtllmRaggedPrefillBackend(
        num_heads=2,
        scale=0.5,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=pcp_size,
            )
        ),
    )

    assert backend._use_pcp is expected


def test_trtllm_ragged_rejects_empty_pcp_query_row():
    backend = object.__new__(TrtllmRaggedPrefillBackend)
    backend._use_pcp = True
    query_lens_cpu = torch.tensor([2, 0], dtype=torch.int32)
    query_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), query_lens_cpu.cumsum(0)]
    )

    with pytest.raises(ValueError, match="mixed active and empty query rows"):
        backend.prepare_metadata(
            SimpleNamespace(
                query_start_loc=query_start_loc,
                query_lens_cpu=query_lens_cpu,
            )
        )


@pytest.mark.parametrize(
    ("query_lens_cpu", "message"),
    [
        (None, "requires CPU query lengths"),
        (torch.tensor([], dtype=torch.int32), "non-empty 1D"),
        (torch.tensor([[1]], dtype=torch.int32), "non-empty 1D"),
        (torch.tensor([-1], dtype=torch.int32), "non-negative"),
    ],
)
def test_trtllm_ragged_rejects_invalid_cpu_query_lengths(
    query_lens_cpu, message
):
    backend = object.__new__(TrtllmRaggedPrefillBackend)
    backend._use_pcp = True

    with pytest.raises(ValueError, match=message):
        backend.prepare_metadata(
            SimpleNamespace(
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                query_lens_cpu=query_lens_cpu,
            )
        )


def test_trtllm_ragged_non_pcp_uses_scheduler_active_row_invariant():
    backend = object.__new__(TrtllmRaggedPrefillBackend)
    backend._use_pcp = False

    backend.prepare_metadata(
        SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            query_lens_cpu=None,
        )
    )

    assert backend._has_active_rows


def test_trtllm_ragged_rejects_inactive_context_row():
    backend = object.__new__(TrtllmRaggedPrefillBackend)
    backend._use_pcp = False

    with pytest.raises(ValueError, match="empty query or KV row"):
        backend.run_prefill_context_chunk(
            SimpleNamespace(all_rows_active=False),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        )


@pytest.mark.parametrize("return_softmax_lse", [False, True])
@pytest.mark.parametrize("num_padded_tokens", [0, 4])
def test_trtllm_ragged_handles_empty_pcp_rank_without_kernel_launch(
    monkeypatch, return_softmax_lse, num_padded_tokens
):
    backend = object.__new__(TrtllmRaggedPrefillBackend)
    backend.num_heads = 2
    backend.scale = 0.5
    backend._workspace_buffer = torch.empty(1, dtype=torch.uint8)
    backend._use_pcp = True
    backend.prepare_metadata(
        SimpleNamespace(
            query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
            max_query_len=0,
            output_dtype=torch.float32,
            query_lens_cpu=torch.tensor([0], dtype=torch.int32),
        )
    )

    def fail_ragged_attention(**kwargs):
        pytest.fail("empty PCP rank must not launch ragged attention")

    monkeypatch.setattr(
        "flashinfer.prefill.trtllm_ragged_attention_deepseek",
        fail_ragged_attention,
    )
    q = torch.empty(num_padded_tokens, 2, 192)
    k = torch.empty(num_padded_tokens, 2, 192)
    v = torch.empty(num_padded_tokens, 2, 128)
    result = backend.run_prefill_new_tokens(
        q,
        k,
        v,
        return_softmax_lse=return_softmax_lse,
    )

    if return_softmax_lse:
        out, lse = result
        assert out.shape == (num_padded_tokens, 2, 128)
        assert torch.count_nonzero(out) == 0
        assert lse.shape == (2, num_padded_tokens)
        assert torch.isneginf(lse).all()
    else:
        assert isinstance(result, torch.Tensor)
        assert result.shape == (num_padded_tokens, 2, 128)
        assert torch.count_nonzero(result) == 0


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Clear any overrides after each test."""
    yield
    for member in MLAPrefillBackendEnum:
        member.clear_override()


def test_custom_is_not_alias_of_any_backend():
    all_backends = list(MLAPrefillBackendEnum)

    aliases = []
    for backend in all_backends:
        if backend.name != "CUSTOM" and backend is MLAPrefillBackendEnum.CUSTOM:
            aliases.append(backend.name)

    assert len(aliases) == 0, (
        f"BUG! CUSTOM is an alias of: {', '.join(aliases)}!\n"
        f"CUSTOM.value = {repr(MLAPrefillBackendEnum.CUSTOM.value)}\n"
        f"All MLA prefill backend values:\n"
        + "\n".join(f"  {b.name}: {repr(b.value)}" for b in all_backends)
    )

    assert MLAPrefillBackendEnum.CUSTOM.name == "CUSTOM"


def test_custom_unregistered_raises():
    with pytest.raises(ValueError, match="must be registered before use"):
        MLAPrefillBackendEnum.CUSTOM.get_path()


def test_register_custom_backend_with_class_path():
    register_mla_prefill_backend(
        backend=MLAPrefillBackendEnum.CUSTOM,
        class_path=(
            "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
        ),
    )

    assert MLAPrefillBackendEnum.CUSTOM.is_overridden()

    class_path = MLAPrefillBackendEnum.CUSTOM.get_path()
    assert class_path == (
        "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
    )

    backend_cls = MLAPrefillBackendEnum.CUSTOM.get_class()
    assert backend_cls.get_name() == "CUSTOM"


def test_register_custom_backend_as_decorator():
    @register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM)
    class DecoratedPrefillBackend(MLAPrefillBackend):
        supported_dtypes = [torch.bfloat16]

        @staticmethod
        def get_name() -> str:
            return "DECORATED"

        def run_prefill_new_tokens(self, q, k, v, return_softmax_lse):
            raise NotImplementedError

        def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
            raise NotImplementedError

    assert MLAPrefillBackendEnum.CUSTOM.is_overridden()
    assert "DecoratedPrefillBackend" in MLAPrefillBackendEnum.CUSTOM.get_path()


def test_override_existing_backend():
    original_path = MLAPrefillBackendEnum.FLASH_ATTN.get_path()

    register_mla_prefill_backend(
        backend=MLAPrefillBackendEnum.FLASH_ATTN,
        class_path=(
            "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
        ),
    )

    assert MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()
    assert MLAPrefillBackendEnum.FLASH_ATTN.get_path() != original_path

    backend_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
    assert backend_cls.get_name() == "CUSTOM"


def test_clear_override():
    original_path = MLAPrefillBackendEnum.FLASH_ATTN.get_path()

    register_mla_prefill_backend(
        backend=MLAPrefillBackendEnum.FLASH_ATTN,
        class_path=(
            "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
        ),
    )
    assert MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()

    MLAPrefillBackendEnum.FLASH_ATTN.clear_override()
    assert not MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()
    assert MLAPrefillBackendEnum.FLASH_ATTN.get_path() == original_path


def test_unknown_backend_name_raises():
    with pytest.raises(ValueError, match="Unknown MLA prefill backend"):
        MLAPrefillBackendEnum["NONEXISTENT"]


def test_rocm_aiter_fa_registered():
    """ROCM_AITER_FA is a known backend pointing at the AITER FA class."""
    assert "ROCM_AITER_FA" in MLAPrefillBackendEnum.__members__

    path = MLAPrefillBackendEnum.ROCM_AITER_FA.get_path()
    assert path == (
        "vllm.v1.attention.backends.mla.prefill.aiter_flash_attn."
        "AiterFlashAttnPrefillBackend"
    )

    backend_cls = MLAPrefillBackendEnum.ROCM_AITER_FA.get_class()
    assert backend_cls.get_name() == "ROCM_AITER_FA"
    # The AITER FA path is the fp16/bf16 generic-varlen prefill path.
    assert backend_cls.supports_dtype(torch.bfloat16)
    assert backend_cls.supports_dtype(torch.float16)
