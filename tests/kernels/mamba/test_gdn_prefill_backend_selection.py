# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the GDN prefill selector answers, per request and per device."""

import sys
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_module
import vllm.utils.flashinfer as flashinfer_utils
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _resolve_gdn_prefill_backend,
    fi_chunk_gated_delta_rule,
)
from vllm.platforms.interface import DeviceCapability


def _config(requested: str | None, head_k_dim: int = 128):
    additional_config = {} if requested is None else {"gdn_prefill_backend": requested}
    return SimpleNamespace(
        additional_config=additional_config,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=head_k_dim)
        ),
    )


def _platform(major: int, minor: int = 0, cuda_major: int = 13):
    return SimpleNamespace(
        is_cuda=lambda: True,
        is_rocm=lambda: False,
        is_cpu=lambda: False,
        is_xpu=lambda: False,
        is_device_capability=lambda m: (major * 10 + minor) == m,
        is_device_capability_family=lambda m: major * 10 == m,
        get_cuda_runtime_major=lambda: cuda_major,
        get_device_capability=lambda: DeviceCapability(major=major, minor=minor),
    )


@pytest.fixture
def sm80(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gdn_module, "current_platform", _platform(8, 0))


@pytest.fixture
def sm75(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gdn_module, "current_platform", _platform(7, 5))


def test_sm8x_is_runnable_but_not_the_default(sm80) -> None:
    """Runnable and preferred are different questions.

    The SM8x path is new here, so "auto" keeps the fallback until it has been
    measured against it, while a caller who names it still gets it.
    """
    assert _resolve_gdn_prefill_backend(_config("auto"))[1] == "triton"
    assert _resolve_gdn_prefill_backend(_config(None))[1] == "triton"
    assert _resolve_gdn_prefill_backend(_config("flashinfer"))[1] == "flashinfer"


def test_sm8x_needs_the_head_dim_the_kernel_builds(sm80) -> None:
    with pytest.raises(ValueError, match="head_k_dim=64"):
        _resolve_gdn_prefill_backend(_config("flashinfer", head_k_dim=64))


@pytest.mark.parametrize("requested", ["flashinfer", "cutedsl"])
def test_an_unsupported_device_refuses_rather_than_substitutes(
    requested: str, sm75
) -> None:
    """Refusing is not the same as quietly running something else.

    The selector used to log and fall back to Triton, which left a caller who
    asked for a specific kernel with no way to find out they had not got it.
    """
    with pytest.raises(ValueError, match=requested):
        _resolve_gdn_prefill_backend(_config(requested))


def test_auto_falls_back_on_an_unsupported_device(sm75) -> None:
    assert _resolve_gdn_prefill_backend(_config("auto"))[1] == "triton"


def test_triton_is_always_available(sm75) -> None:
    assert _resolve_gdn_prefill_backend(_config("triton"))[1] == "triton"


def test_sm90_prefers_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdn_module, "current_platform", _platform(9, 0))
    assert _resolve_gdn_prefill_backend(_config("auto"))[1] == "flashinfer"


def test_sm100_offers_cutedsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdn_module, "current_platform", _platform(10, 0))
    assert _resolve_gdn_prefill_backend(_config("cutedsl"))[1] == "cutedsl"
    assert _resolve_gdn_prefill_backend(_config("auto"))[1] == "flashinfer"


def test_a_non_cuda_platform_resolves_to_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gdn_module,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            is_cpu=lambda: True,
            is_rocm=lambda: False,
            is_xpu=lambda: False,
        ),
    )
    assert _resolve_gdn_prefill_backend(_config("flashinfer"))[1] == "triton"


# --- what the prefill adapter hands FlashInfer -------------------------------
#
# The batch maximum is a later addition to chunk_gated_delta_rule; 0.7.0 keeps
# it internal and derives the grids from the token total. The adapter has to
# read the installed signature and leave the keyword out of the builds that do
# not take it, or every architecture the base path already served raises
# TypeError. These stubs declare their parameters explicitly: a **kwargs stub
# would accept the keyword either way and prove nothing.


def _stub_without_max_seqlen(seen: dict):
    def chunk_gated_delta_rule(
        q,
        k,
        v,
        g=None,
        beta=None,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        use_qk_l2norm_in_kernel=False,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
        use_cp="auto",
        backend="auto",
    ):
        seen.update(backend=backend, has_max_seqlen=False, q_shape=tuple(q.shape))
        return torch.zeros_like(q), initial_state

    return chunk_gated_delta_rule


def _stub_with_max_seqlen(seen: dict):
    def chunk_gated_delta_rule(
        q,
        k,
        v,
        g=None,
        beta=None,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        use_qk_l2norm_in_kernel=False,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
        use_cp="auto",
        backend="auto",
        max_seqlen=None,
        cu_seqlens_cpu=None,
    ):
        seen.update(
            backend=backend,
            has_max_seqlen=True,
            max_seqlen=max_seqlen,
            cu_seqlens_cpu=cu_seqlens_cpu,
            q_shape=tuple(q.shape),
        )
        return torch.zeros_like(q), initial_state

    return chunk_gated_delta_rule


def _install_stub(monkeypatch: pytest.MonkeyPatch, fn) -> SimpleNamespace:
    """Publish `fn` as the installed GDN prefill and reset the cached probe."""
    module = SimpleNamespace(chunk_gated_delta_rule=fn)
    monkeypatch.setattr(flashinfer_utils, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        flashinfer_utils,
        "_get_submodule",
        lambda name: module if name == "flashinfer.gdn_prefill" else None,
    )
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_prefill", module)
    flashinfer_utils.flashinfer_gdn_prefill_takes.cache_clear()
    return module


@pytest.fixture(autouse=True)
def _forget_probe():
    """The probe is cached for the process, so neither side may leak."""
    flashinfer_utils.flashinfer_gdn_prefill_takes.cache_clear()
    yield
    flashinfer_utils.flashinfer_gdn_prefill_takes.cache_clear()


def _adapter_inputs(seq_lens: list[int], num_heads: int = 2, head_dim: int = 4):
    total = sum(seq_lens)
    shape = (1, total, num_heads, head_dim)
    starts = [0]
    for length in seq_lens:
        starts.append(starts[-1] + length)
    return dict(
        q=torch.zeros(shape),
        k=torch.zeros(shape),
        v=torch.zeros(shape),
        g=torch.zeros(1, total, num_heads),
        beta=torch.ones(1, total, num_heads),
        initial_state=torch.zeros(len(seq_lens), num_heads, head_dim, head_dim),
        output_final_state=True,
        cu_seqlens=torch.tensor(starts, dtype=torch.int32),
        use_qk_l2norm_in_kernel=False,
    )


@pytest.mark.parametrize(
    ("seq_lens", "batch_maximum"),
    [([1024, 1024, 1024, 1024], 1024), ([4096], 4096)],
    ids=["four-equal", "one-long"],
)
def test_adapter_passes_the_batch_maximum_when_the_build_takes_it(
    monkeypatch: pytest.MonkeyPatch, seq_lens: list[int], batch_maximum: int
) -> None:
    seen: dict = {}
    _install_stub(monkeypatch, _stub_with_max_seqlen(seen))

    inputs = _adapter_inputs(seq_lens)
    host = inputs["cu_seqlens"].clone()
    output, final_state = fi_chunk_gated_delta_rule(
        **inputs, max_seq_len=batch_maximum, cu_seqlens_cpu=host
    )

    assert seen["backend"] == "flashinfer"
    assert seen["max_seqlen"] == batch_maximum
    assert seen["cu_seqlens_cpu"] is host
    # The adapter squeezes the leading axis on the way in and restores it on
    # the way out, and returns the state the callee reports.
    assert seen["q_shape"] == (sum(seq_lens), 2, 4)
    assert tuple(output.shape) == (1, sum(seq_lens), 2, 4)
    assert final_state.shape == inputs["initial_state"].shape


def test_adapter_omits_the_batch_maximum_when_the_build_lacks_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}
    _install_stub(monkeypatch, _stub_without_max_seqlen(seen))

    output, final_state = fi_chunk_gated_delta_rule(
        **_adapter_inputs([1024, 1024, 1024, 1024]),
        max_seq_len=1024,
        cu_seqlens_cpu=torch.tensor([0, 1024, 2048, 3072, 4096]),
    )

    # No TypeError, and the backend still names the implementation to run.
    assert seen["has_max_seqlen"] is False
    assert seen["backend"] == "flashinfer"
    assert tuple(output.shape) == (1, 4096, 2, 4)
    assert final_state is not None


def test_the_probe_answers_from_the_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = flashinfer_utils.flashinfer_gdn_prefill_takes

    _install_stub(monkeypatch, _stub_without_max_seqlen({}))
    assert not probe("max_seqlen") and not probe("cu_seqlens_cpu")

    _install_stub(monkeypatch, _stub_with_max_seqlen({}))
    assert probe("max_seqlen") and probe("cu_seqlens_cpu")
