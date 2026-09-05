# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.worker import cp_utils
from vllm.v1.worker.cp_utils import should_skip_dcp_context_attention


def test_skip_gate_only_for_zero_context():
    assert should_skip_dcp_context_attention(torch.zeros(3, dtype=torch.int32))
    assert not should_skip_dcp_context_attention(
        torch.tensor([0, 5, 0], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    "dcp_world_size,interleave_size,context_len",
    [(2, 16, 10), (4, 16, 10), (8, 16, 10), (4, 1, 2)],
)
def test_skip_gate_rank_invariant_with_divergent_local_context(
    dcp_world_size: int, interleave_size: int, context_len: int
):
    """Contexts shorter than a full interleave round land entirely on a
    subset of DCP ranks, so the per-rank local context lengths diverge:
    some ranks hold zero local context while others hold all of it. Ranks
    with zero local context must still take the collective (non-skip) path,
    otherwise the query all-gather in _forward_with_dcp deadlocks across
    ranks. The skip gate must therefore depend only on the rank-invariant
    global context lengths, never on get_dcp_local_seq_lens output.
    """
    context_kv_lens = torch.tensor([context_len], dtype=torch.int32)
    local_maxes = [
        int(
            get_dcp_local_seq_lens(
                context_kv_lens, dcp_world_size, rank, interleave_size
            ).max()
        )
        for rank in range(dcp_world_size)
    ]
    # Precondition: the local view diverges across ranks.
    assert 0 in local_maxes
    assert max(local_maxes) > 0
    # The batch still has context globally, so no rank may skip.
    assert not should_skip_dcp_context_attention(context_kv_lens)


# -- Tests for improved error messages in check_attention_cp_compatibility --


class _DummyBackend:
    @staticmethod
    def get_name():
        return "DummyBackend"

    @staticmethod
    def supports_pcp():
        return False


class _DummyImpl:
    need_to_return_lse_for_decode = False
    supports_pcp = True
    supports_mtp_with_cp_non_trivial_interleave_size = True


class _DummyLayer:
    impl = _DummyImpl()

    @staticmethod
    def get_attn_backend():
        return _DummyBackend()


def test_dcp_error_includes_backend_hint(monkeypatch: pytest.MonkeyPatch):
    """DCP assertion should mention --attention-backend and
    --decode-context-parallel-size for actionable guidance."""
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
    )
    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda *_: {"layer.0": _DummyLayer()},
    )
    with pytest.raises(AssertionError, match="--attention-backend") as exc:
        cp_utils.check_attention_cp_compatibility(vllm_config)
    assert "--decode-context-parallel-size" in str(exc.value)
    assert "DummyImpl" in str(exc.value)


def test_pcp_error_includes_backend_hint(monkeypatch: pytest.MonkeyPatch):
    """PCP assertion should mention --attention-backend and
    --prefill-context-parallel-size for actionable guidance."""
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
    )
    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda *_: {"layer.0": _DummyLayer()},
    )
    with pytest.raises(AssertionError, match="--attention-backend") as exc:
        cp_utils.check_attention_cp_compatibility(vllm_config)
    assert "--prefill-context-parallel-size" in str(exc.value)
    assert "DummyBackend" in str(exc.value)
