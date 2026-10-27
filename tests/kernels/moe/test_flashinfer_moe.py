# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as fi_moe
from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
    is_valid_flashinfer_cutlass_fused_moe,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_type,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_cutlass_fused_moe,
    has_flashinfer_cutlass_fused_moe_workspace,
)
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import current_workspace_manager

if not has_flashinfer_cutlass_fused_moe() or not current_platform.has_device_capability(
    100
):
    pytest.skip(
        "Requires flashinfer_cutlass_fused_moe and nvfp4 support",
        allow_module_level=True,
    )

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


@pytest.mark.skipif(
    not has_flashinfer_cutlass_fused_moe_workspace(),
    reason="Requires caller-owned CUTLASS MoE workspace support",
)
@pytest.mark.parametrize(
    "max_num_tokens, dp_size, elastic_ep_max_dp_size, workspace_rows",
    [(32, 1, None, 8192), (32, 2, None, 8192), (4096, 2, 4, 16384)],
)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
@pytest.mark.parametrize("quant_dtype", [None, "nvfp4"])
@torch.inference_mode()
def test_flashinfer_workspace_reused_after_lock(
    max_num_tokens,
    dp_size,
    elastic_ep_max_dp_size,
    workspace_rows,
    activation,
    quant_dtype,
    workspace_init,
    monkeypatch,
):
    """Framework scratch must survive locking and graph replay without aliasing."""
    parallel = FusedMoEParallelConfig.make_no_parallel()
    parallel.dp_size = dp_size
    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=512,
        num_local_experts=8,
        num_logical_experts=8,
        activation=activation,
        device="cuda",
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=max_num_tokens,
        elastic_ep_max_dp_size=elastic_ep_max_dp_size,
    )
    max_rows = max_num_tokens * dp_size
    x = torch.randn(max_rows, 256, device="cuda", dtype=torch.bfloat16) / 10
    w1 = (
        torch.randn(
            8,
            512 * (2 if activation.is_gated else 1),
            256,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = torch.randn(8, 256, 512, device="cuda", dtype=torch.bfloat16) / 10
    quant_config = FusedMoEQuantConfig.make()
    if quant_dtype is not None:
        w1, w2, quant_config = make_test_quant_config(
            8,
            512,
            256,
            in_dtype=x.dtype,
            quant_dtype=quant_dtype,
            block_shape=None,
            per_act_token_quant=False,
            make_gate=activation.is_gated,
        )
    layers = [
        FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            FlashInferExperts(config, quant_config),
        )
        for _ in range(2)
    ]
    ids = torch.tensor([0, 1], device="cuda").expand(max_rows, 2)
    weights = torch.full((max_rows, 2), 0.5, device="cuda")
    expected_nbytes = fi_moe.flashinfer_cutlass_fused_moe_workspace_size(
        workspace_rows,
        256,
        512,
        8,
        2,
        x_dtype=torch.uint8 if quant_dtype == "nvfp4" else x.dtype,
        weight_dtype=torch.long if quant_dtype == "nvfp4" else w1.dtype,
        output_dtype=x.dtype,
        activation_type=activation_to_flashinfer_type(activation),
        device=x.device,
    )
    buffers = []
    call_args: dict = {}
    kernel = fi_moe.flashinfer_cutlass_fused_moe

    def record_workspace(**kwargs):
        call_args.clear()
        call_args.update(kwargs)
        scratch = kwargs["workspace_buffer"]
        output = kwargs["output"]
        buffers.append(scratch)
        assert scratch.dtype == torch.uint8 and scratch.ndim == 1
        assert scratch.nbytes >= expected_nbytes
        # The common workspace reserves the output before workspace2, even
        # when prepare/finalize supplies an external output allocation.
        assert scratch.storage_offset() >= output.nbytes
        assert (
            scratch.data_ptr() + scratch.nbytes <= output.data_ptr()
            or output.data_ptr() + output.nbytes <= scratch.data_ptr()
        )
        return kernel(**kwargs)

    def run(layer, rows):
        return layer.apply(
            hidden_states=x[:rows],
            w1=w1,
            w2=w2,
            topk_weights=weights[:rows],
            topk_ids=ids[:rows],
            activation=activation,
            global_num_experts=8,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    def check_reference(output):
        reference = torch.empty_like(output)
        kernel(**{**call_args, "workspace_buffer": None, "output": reference})
        torch.testing.assert_close(output, reference)

    monkeypatch.setattr(fi_moe, "flashinfer_cutlass_fused_moe", record_workspace)
    with set_current_vllm_config(VllmConfig()):
        check_reference(run(layers[0], max_rows))
        manager = current_workspace_manager()
        manager.lock()
        storage_ptr = manager.get_simultaneous(((1,), torch.uint8))[0].data_ptr()
        for layer, rows in [(layers[1], 1), (layers[0], 17), (layers[1], max_rows)]:
            check_reference(run(layer, rows))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(layers[0], 16)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run(layers[0], 16)
        for _ in range(2):
            x[:16].normal_(std=0.1)
            graph.replay()
            check_reference(output)
        assert all(b.untyped_storage().data_ptr() == storage_ptr for b in buffers)


@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUOAI_UNINTERLEAVE],
)
def test_flashinfer_swigluoai_params_are_forwarded(activation, monkeypatch):
    from flashinfer.fused_moe.core import ActivationType

    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=2,
        num_logical_experts=2,
        activation=activation,
        device="cuda",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
    )
    quant_config = FusedMoEQuantConfig.make(
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    experts = FlashInferExperts(moe_config=moe_config, quant_config=quant_config)

    call_args = {}

    def fake_flashinfer_cutlass_fused_moe(**kwargs):
        call_args.update(kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.experts."
        "flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe",
        fake_flashinfer_cutlass_fused_moe,
    )
    experts.apply(
        output=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        hidden_states=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        w1=torch.empty((2, 256, 128), device="cuda", dtype=torch.bfloat16),
        w2=torch.empty((2, 128, 128), device="cuda", dtype=torch.bfloat16),
        topk_weights=torch.ones((1, 1), device="cuda", dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), device="cuda", dtype=torch.int64),
        activation=activation,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert experts._supports_activation(activation)
    assert call_args["activation_type"] == ActivationType.Swiglu
    for name, value in (
        ("swiglu_alpha", 1.702),
        ("swiglu_beta", 1.0),
        ("swiglu_limit", 7.0),
    ):
        torch.testing.assert_close(
            call_args[name],
            torch.full((2,), value, device="cuda", dtype=torch.float32),
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
@torch.inference_mode()
def test_flashinfer_fp4_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: MoEActivation,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        quant_blocksize = 16
        is_gated_act = activation.is_gated

        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=is_gated_act,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        assert is_valid_flashinfer_cutlass_fused_moe(a, w1_q, w2_q)

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        flashinfer_experts = FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            FlashInferExperts(moe_config=moe_config, quant_config=quant_config),
        )

        flashinfer_output = flashinfer_experts.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        # Reference check:
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty(
            (e, (2 if is_gated_act else 1) * n, k), device="cuda", dtype=dtype
        )
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                quant_config.w1_scale[idx],
                (1 / quant_config.g1_alphas[idx]),
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                quant_config.w2_scale[idx],
                (1 / quant_config.g2_alphas[idx]),
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        torch_output = torch_moe(
            a_in_dtype, w1_d, w2_d, score, topk, activation=activation
        )

        torch.testing.assert_close(
            torch_output, flashinfer_output, atol=1e-1, rtol=1e-1
        )


if __name__ == "__main__":
    test_flashinfer_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)
