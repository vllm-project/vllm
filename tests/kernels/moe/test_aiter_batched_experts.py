# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit-level tests for the AITER ``BatchedExperts`` FP8 wrapper.

These tests guard the reshape and oracle-selection contracts without requiring
the AITER runtime:

  * the wrapper advertises ``BatchedExperts`` activation format,
  * BatchedExperts prepare/finalize can provide already-quantized activations,
  * the wrapper flattens ``(E_local, M_e, K)`` activations and dispatched
    scales before delegating to the Standard-layout AITER experts,
  * ``BATCHED_AITER`` maps to ``AiterBatchedExpertsFp8``, and
  * the FP8 oracle routes the batched ROCm AITER path to that wrapper.
"""

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
import vllm.model_executor.layers.fused_moe.oracle.fp8 as fp8_oracle
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

# ---------------------------------------------------------------------------
# Import the wrapper without requiring the AITER runtime. Importing the module
# must work even when AITER is not present — it only fails at *kernel-call*
# time. This is the same pattern as the rest of the vllm aiter modules.
# ---------------------------------------------------------------------------
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (  # noqa: E402
    AiterBatchedExpertsFp8,
    AiterExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (  # noqa: E402
    Fp8MoeBackend,
    _get_priority_backends,
    backend_to_kernel_cls,
    select_fp8_moe_backend,
)


def test_aiter_batched_experts_fp8_activation_format():
    """The wrapper must advertise ``BatchedExperts`` format."""
    assert (
        AiterBatchedExpertsFp8.activation_format()
        == mk.FusedMoEActivationFormat.BatchedExperts
    )
    # And the sibling Standard variant must still be Standard.
    assert AiterExperts.activation_format() == mk.FusedMoEActivationFormat.Standard


def test_aiter_batched_experts_does_not_expect_unquantized_inputs():
    """Critical: ``BatchedExperts`` prepare steps (DeepEP-LL, NIXL) explicitly
    reject ``defer_input_quant=True``. The wrapper must not request it."""
    # ``expects_unquantized_inputs`` is a @property on the base class, so we
    # have to query an instance, not the class. We don't construct a full
    # ``FusedMoEConfig`` (lots of plumbing) — we just check the descriptor
    # directly.
    prop = AiterBatchedExpertsFp8.__dict__["expects_unquantized_inputs"]
    assert isinstance(prop, property), (
        "expects_unquantized_inputs must be a @property to match the ABC"
    )
    # Invoke the getter with a dummy ``self`` proxy. The body of the getter
    # only returns False unconditionally.
    fget = prop.fget
    assert fget is not None
    assert fget(object.__new__(AiterBatchedExpertsFp8)) is False


def test_aiter_batched_experts_flattens_batched_layout_for_inner_aiter():
    """CPU-constructible check for the batched-to-flat wrapper contract."""

    class CapturingInner:
        def __init__(self):
            self.kwargs = None

        def apply(self, **kwargs):
            self.kwargs = kwargs
            output = kwargs["output"]
            hidden_states = kwargs["hidden_states"]
            topk_ids = kwargs["topk_ids"].to(dtype=output.dtype)
            output.copy_(hidden_states[:, : output.size(-1)] + topk_ids)

    E_local = 2
    M_e = 3
    K = 4
    hidden_states = torch.arange(E_local * M_e * K, dtype=torch.float32).reshape(
        E_local, M_e, K
    )
    output = torch.empty_like(hidden_states)
    a1q_scale = torch.arange(E_local * M_e * 2, dtype=torch.float32).reshape(
        E_local, M_e, 2
    )
    a2_scale = torch.tensor([0.5])
    expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list([2, 1], "cpu")

    inner = CapturingInner()
    wrapper = object.__new__(AiterBatchedExpertsFp8)
    wrapper._inner = inner

    wrapper.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(E_local, 1, 1),
        w2=torch.empty(E_local, 1, 1),
        topk_weights=torch.empty(1, 1),
        topk_ids=torch.empty(1, 1, dtype=torch.int64),
        activation=MoEActivation.SILU,
        global_num_experts=99,
        expert_map=torch.tensor([1, 0]),
        a1q_scale=a1q_scale,
        a2_scale=a2_scale,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=expert_tokens_meta,
        apply_router_weight_on_input=True,
    )

    assert inner.kwargs is not None
    kwargs = inner.kwargs
    assert kwargs["apply_router_weight_on_input"] is False
    expected_ids = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.int32)

    assert kwargs["hidden_states"].shape == (E_local * M_e, K)
    assert torch.equal(kwargs["hidden_states"], hidden_states.reshape(E_local * M_e, K))
    assert kwargs["output"].shape == (E_local * M_e, K)
    assert kwargs["output"].data_ptr() == output.reshape(E_local * M_e, K).data_ptr()
    assert torch.equal(kwargs["topk_ids"], expected_ids)
    assert torch.equal(kwargs["topk_weights"], torch.ones(E_local * M_e, 1))
    assert kwargs["global_num_experts"] == E_local
    assert kwargs["expert_map"] is None
    assert torch.equal(kwargs["a1q_scale"], a1q_scale.reshape(E_local * M_e, 2))
    assert kwargs["a2_scale"] is a2_scale
    assert kwargs["expert_tokens_meta"] is expert_tokens_meta

    expected_output = hidden_states.reshape(E_local * M_e, K) + expected_ids.float()
    assert torch.equal(output, expected_output.reshape(E_local, M_e, K))


def test_oracle_registers_batched_aiter_backend():
    """``BATCHED_AITER`` must exist in ``Fp8MoeBackend`` and map to the
    wrapper class."""
    # Enum membership.
    assert Fp8MoeBackend.BATCHED_AITER.value == "BATCHED_AITER"
    # Dispatch table mapping.
    classes = backend_to_kernel_cls(Fp8MoeBackend.BATCHED_AITER)
    assert classes == [AiterBatchedExpertsFp8]


def test_select_fp8_moe_backend_routes_batched_aiter_env_to_wrapper(monkeypatch):
    """ROCm AITER env selection must pick the batched wrapper for DP/EP."""

    def is_set(name):
        return name in {"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"}

    def is_supported(cls, config, weight_key, activation_key, activation_format):
        assert activation_format == mk.FusedMoEActivationFormat.BatchedExperts
        return True, None

    config = SimpleNamespace(
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=True,
            use_deepep_v2_kernels=False,
            ep_size=1,
        ),
    )
    monkeypatch.setattr(fp8_oracle.envs, "is_set", is_set)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_ROCM_USE_AITER", True)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_ROCM_USE_AITER_MOE", True)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_TEST_FORCE_FP8_MARLIN", False)
    monkeypatch.setattr(
        AiterBatchedExpertsFp8,
        "is_supported_config",
        staticmethod(is_supported),
    )

    backend, experts_cls = select_fp8_moe_backend(config, None, None)

    assert backend == Fp8MoeBackend.BATCHED_AITER
    assert experts_cls is AiterBatchedExpertsFp8


def test_oracle_priority_order_places_batched_aiter_before_fallbacks():
    """``BATCHED_AITER`` must be tried before the generic Triton/CUTLASS
    batched fallbacks on the ROCm path, so the kernel-doesn't-support error
    is never surfaced when AITER MoE is enabled."""
    # The helper can reshuffle for Hopper/XPU/CPU, but it only needs these
    # fields from the config object for the ROCm/default order we check here.
    moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            use_deepep_v2_kernels=False,
            ep_size=1,
        )
    )
    backends = _get_priority_backends(moe_config, None, None)

    assert Fp8MoeBackend.BATCHED_AITER in backends
    ba_idx = backends.index(Fp8MoeBackend.BATCHED_AITER)
    for fallback in (
        Fp8MoeBackend.BATCHED_DEEPGEMM,
        Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
        Fp8MoeBackend.BATCHED_TRITON,
    ):
        if fallback in backends:
            assert ba_idx < backends.index(fallback), (
                f"BATCHED_AITER must precede {fallback.value} in the priority list"
            )
