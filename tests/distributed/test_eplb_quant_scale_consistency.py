# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EPLB rearrangement consistency tests for quant-method derived state.

EPLB rearranges every registered Parameter of a RoutedExperts in place,
sliced along dim 0 (see RoutedExperts.get_expert_weights). Quant methods
must therefore register all derived per-expert tensors as Parameters and
alias the same storage in their FusedMoEQuantConfig, so the kernels observe
rearranged values with no extra bookkeeping. These tests verify that
contract: simulating a rearrangement on the registered Parameters must be
indistinguishable from loading a checkpoint with permuted experts.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.platforms import current_platform

EPLB_NVFP4_BACKENDS = ["flashinfer_cutedsl", "flashinfer_trtllm"]

NUM_EXPERTS = 8
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
EXPERT_PERMUTATION = [3, 0, 5, 1, 7, 2, 6, 4]

QUANT_CONFIG_TENSORS = (
    "w1_scale",
    "w2_scale",
    "g1_alphas",
    "g2_alphas",
    "a1_gscale",
    "a2_gscale",
)


class _RoutedExpertsStub(torch.nn.Module):
    """Minimal ``RoutedExperts`` stand-in for NVFP4 weight processing."""

    def __init__(self, moe_config: FusedMoEConfig) -> None:
        super().__init__()
        self.moe_config = moe_config
        self.activation = moe_config.activation

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return None


def _make_moe_config(backend: str) -> FusedMoEConfig:
    parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=True,
    )
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=2,
        hidden_dim=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        moe_backend=backend,
    )


def _make_raw_weights(
    device: torch.device, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Random raw NVFP4 checkpoint tensors, indexable per-expert on dim 0."""
    e, h, i = NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE

    def packed(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device=device, generator=generator
        )

    def block_scale(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 128, shape, dtype=torch.uint8, device=device, generator=generator
        ).view(torch.float8_e4m3fn)

    def global_scale(*shape: int) -> torch.Tensor:
        return torch.rand(shape, device=device, generator=generator) + 0.5

    return {
        "w13_weight_packed": packed(e, 2 * i, h // 2),
        "w2_weight_packed": packed(e, h, i // 2),
        "w13_weight_scale": block_scale(e, 2 * i, h // 16),
        "w2_weight_scale": block_scale(e, h, i // 16),
        "w13_weight_global_scale": global_scale(e, 2),
        "w2_weight_global_scale": global_scale(e),
        "w13_input_global_scale": global_scale(e, 2),
        "w2_input_global_scale": global_scale(e),
    }


def _build_processed_layer(
    backend: str, raw: dict[str, torch.Tensor], device: torch.device
) -> tuple[CompressedTensorsW4A4Nvfp4MoEMethod, _RoutedExpertsStub]:
    """Create the method + layer, load ``raw`` and run the real
    process_weights_after_loading (which builds the flashinfer kernel)."""
    moe_config = _make_moe_config(backend)
    method = CompressedTensorsW4A4Nvfp4MoEMethod(moe_config, "layer.0", use_a16=False)
    layer = _RoutedExpertsStub(moe_config).to(device)
    method.create_weights(
        layer,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size_per_partition=INTERMEDIATE_SIZE,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer = layer.to(device)
    with torch.no_grad():
        for name, value in raw.items():
            getattr(layer, name).copy_(value)
    method.process_weights_after_loading(layer)
    return method, layer


def _assert_tensors_equal(
    actual: torch.Tensor, expected: torch.Tensor, name: str
) -> None:
    if actual.dtype == torch.float8_e4m3fn:
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), name
    elif actual.dtype == torch.uint8:
        assert torch.equal(actual, expected), name
    else:
        torch.testing.assert_close(actual, expected, msg=lambda m: f"{name}: {m}")


def _simulate_eplb_rearrangement(layer: torch.nn.Module, perm: torch.Tensor) -> None:
    """Permute experts of every registered Parameter in place, the way
    EPLB's in-place rearrangement moves expert slices along dim 0."""
    with torch.no_grad():
        for _, param in layer.named_parameters():
            param.copy_(param[perm])


def _ensure_world1_distributed() -> None:
    """The enable_eplb config makes weight processing all-reduce activation
    scale amaxes over the EP group, so a (single-rank) EP group must exist."""
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
            local_rank=0,
        )
    ensure_model_parallel_initialized(1, 1)


@pytest.mark.parametrize("backend", EPLB_NVFP4_BACKENDS)
def test_nvfp4_eplb_rearrangement_matches_reload(backend: str) -> None:
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        pytest.skip("NVFP4 CuteDSL/TRTLLM MoE backends require Blackwell (SM100).")

    device = torch.device("cuda:0")
    torch.accelerator.set_device_index(device)
    perm = torch.tensor(EXPERT_PERMUTATION, device=device)

    with set_current_vllm_config(VllmConfig()):
        _ensure_world1_distributed()
        generator = torch.Generator(device=device).manual_seed(1234)
        raw = _make_raw_weights(device, generator)
        raw_permuted = {name: value[perm].contiguous() for name, value in raw.items()}

        try:
            method, layer = _build_processed_layer(backend, raw, device)
        except ValueError as exc:
            pytest.skip(f"{backend} NVFP4 MoE backend unavailable: {exc}")
        ref_method, ref_layer = _build_processed_layer(backend, raw_permuted, device)

        # The EPLB contract: every registered Parameter must be an
        # expert-major contiguous tensor so get_expert_weights can view it
        # as (E, -1) and rearrange expert slices in place.
        for name, param in layer.named_parameters():
            assert param.is_contiguous(), f"{name} is not contiguous"
            assert param.shape[0] == NUM_EXPERTS, (
                f"{name} is not expert-major: {tuple(param.shape)}"
            )

        # Derived per-expert scales must live in registered Parameters that
        # the quant config aliases; anything else goes stale on rearrangement.
        quant_config = method.moe_quant_config
        assert quant_config is not None
        params = dict(layer.named_parameters())
        assert quant_config.g1_alphas.data_ptr() == (
            params["w13_weight_scale_2"].data_ptr()
        )
        assert quant_config.g2_alphas.data_ptr() == (
            params["w2_weight_scale_2"].data_ptr()
        )
        assert quant_config.w1_scale.data_ptr() == (
            params["w13_weight_scale"].data_ptr()
        )
        assert quant_config.w2_scale.data_ptr() == params["w2_weight_scale"].data_ptr()

        _simulate_eplb_rearrangement(layer, perm)

        # After rearrangement, all registered state must equal what a fresh
        # load of the permuted experts produces.
        ref_params = dict(ref_layer.named_parameters())
        assert params.keys() == ref_params.keys()
        for name in params:
            _assert_tensors_equal(params[name], ref_params[name], name)

        # And so must the kernel-visible quant config tensors (the CuteDSL
        # MMA scale views alias the registered Parameters' storage).
        ref_quant_config = ref_method.moe_quant_config
        assert ref_quant_config is not None
        for name in QUANT_CONFIG_TENSORS:
            actual = getattr(quant_config, name)
            expected = getattr(ref_quant_config, name)
            assert (actual is None) == (expected is None), name
            if actual is not None:
                _assert_tensors_equal(actual, expected, f"quant_config.{name}")


def test_fp8_per_tensor_alphas_registered_and_aliased() -> None:
    """CPU-only: the fp8 per-tensor oracle must register the fused
    (w_scale * a_scale) products as layer Parameters aliased by the quant
    config, so EPLB rearrangement keeps kernels consistent."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        make_fp8_moe_quant_config,
    )

    e = NUM_EXPERTS
    generator = torch.Generator().manual_seed(1234)

    def scales() -> tuple[torch.Tensor, torch.Tensor]:
        w_scale = torch.rand((e,), generator=generator) + 0.5
        a_scale = torch.rand((), generator=generator) + 0.5
        return w_scale, a_scale.expand(e).contiguous()

    w1_scale, a1_scale = scales()
    w2_scale, a2_scale = scales()

    layer = torch.nn.Module()
    quant_config = make_fp8_moe_quant_config(
        fp8_backend=Fp8MoeBackend.FLASHINFER_CUTLASS,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        layer=layer,
    )

    params = dict(layer.named_parameters())
    assert quant_config.g1_alphas.data_ptr() == params["g1_alphas"].data_ptr()
    assert quant_config.g2_alphas.data_ptr() == params["g2_alphas"].data_ptr()

    perm = torch.tensor(EXPERT_PERMUTATION)
    ref_config = make_fp8_moe_quant_config(
        fp8_backend=Fp8MoeBackend.FLASHINFER_CUTLASS,
        w1_scale=w1_scale[perm].contiguous(),
        w2_scale=w2_scale[perm].contiguous(),
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        layer=None,
    )
    _simulate_eplb_rearrangement(layer, perm)

    torch.testing.assert_close(quant_config.g1_alphas, ref_config.g1_alphas)
    torch.testing.assert_close(quant_config.g2_alphas, ref_config.g2_alphas)
