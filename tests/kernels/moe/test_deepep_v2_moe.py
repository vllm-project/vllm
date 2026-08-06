# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test DeepEP v2 (ElasticBuffer) dispatch-combine logic.
Compares against a pure-PyTorch reference MoE implementation.
"""

import dataclasses

import pytest
import torch.distributed
from torch.distributed import ProcessGroup

from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingGroupedExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.utils.import_utils import has_deep_ep_v2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch

if has_deep_ep_v2():
    from .parallel_utils import DeepEPV2Args, make_deepep_v2_a2a

requires_deep_ep_v2 = pytest.mark.skipif(
    not has_deep_ep_v2(),
    reason="Requires DeepEP v2 (ElasticBuffer)",
)


def _build_expert_map(
    pgi: ProcessGroupInfo,
    num_experts: int,
    num_local_experts: int,
) -> torch.Tensor:
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
    start = pgi.rank * num_local_experts
    expert_map[start : start + num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32
    )
    return expert_map.to(device=pgi.device)


def assert_fp8_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    close = torch.isclose(actual, expected, atol=2e-1, rtol=2e-1)
    close_fraction = close.float().mean().item()
    assert close_fraction > 0.99, (
        f"Only {close_fraction:.1%} of FP8 outputs are within tolerance"
    )


def dequantize_mxfp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
        dtype=torch.float32,
    )
    packed = weight.view(torch.uint8)
    indices = torch.stack((packed & 0xF, (packed >> 4) & 0xF), dim=-1).flatten(-2)
    values = lookup[indices.long()]
    scale_float = (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    return values * scale_float.repeat_interleave(32, dim=-1)


@dataclasses.dataclass
class TestConfig:
    dtype: torch.dtype
    topk: int
    m: int
    k: int
    n: int
    num_experts: int


@dataclasses.dataclass
class TestTensors:
    rank_tokens: torch.Tensor
    rank_token_scales: torch.Tensor | None
    intermediate_scales: torch.Tensor | None
    topk: torch.Tensor
    topk_weights: torch.Tensor
    config: TestConfig

    @staticmethod
    def make(config: TestConfig) -> "TestTensors":
        assert config.dtype in [torch.bfloat16, torch.float8_e4m3fn]
        token_dtype = (
            torch.bfloat16 if config.dtype == torch.float8_e4m3fn else config.dtype
        )
        rank_tokens = (
            torch.randn((config.m, config.k), device="cuda", dtype=token_dtype) / 10
        )
        if config.dtype == torch.float8_e4m3fn:
            rank_token_scales = torch.tensor(1 / 448, device="cuda")
            intermediate_scales = torch.tensor(8 / 448, device="cuda")
        else:
            rank_token_scales = None
            intermediate_scales = None

        if config.m == 0:
            topk = torch.empty(
                (0, config.topk),
                device="cuda",
                dtype=torch.int64,
            )
        else:
            topk = torch.stack(
                [
                    torch.randperm(config.num_experts, device="cuda")[: config.topk]
                    for _ in range(config.m)
                ]
            ).to(dtype=torch.int64)
        topk_weights = torch.randn(topk.shape, dtype=torch.float32, device="cuda")
        return TestTensors(
            rank_tokens=rank_tokens,
            rank_token_scales=rank_token_scales,
            intermediate_scales=intermediate_scales,
            topk=topk,
            topk_weights=topk_weights,
            config=config,
        )


def make_modular_kernel(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    dp_size: int,
    hidden_size: int,
    num_experts: int,
    num_local_experts: int,
    topk: int,
    q_dtype: torch.dtype | None,
    use_fp8_dispatch: bool,
    quant_config: FusedMoEQuantConfig,
    use_cudagraph: bool = False,
) -> FusedMoEKernel:
    v2_args = DeepEPV2Args(
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        num_topk=topk,
        hidden_size=hidden_size,
        max_tokens_per_rank=8192,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    a2a = make_deepep_v2_a2a(
        pg=pg,
        pgi=pgi,
        dp_size=dp_size,
        v2_args=v2_args,
        use_cudagraph=use_cudagraph,
    )

    moe_config = make_dummy_moe_config(
        num_experts=num_local_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
    )

    fused_experts = TritonExperts(
        moe_config=moe_config,
        quant_config=quant_config,
    )

    mk = FusedMoEKernel(
        prepare_finalize=a2a,
        fused_experts=fused_experts,
    )
    return mk


def deepep_v2_moe_impl(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    dp_size: int,
    test_tensors: TestTensors,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    num_experts: int,
    topk: int,
    use_fp8_dispatch: bool,
    per_act_token_quant: bool,
) -> torch.Tensor:
    num_local_experts = w1.size(0)

    is_quantized = w1.dtype == torch.float8_e4m3fn
    q_dtype = torch.float8_e4m3fn if is_quantized else None

    quant_config = FusedMoEQuantConfig.make(
        q_dtype,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        per_act_token_quant=per_act_token_quant,
        a1_scale=test_tensors.rank_token_scales,
        a2_scale=test_tensors.intermediate_scales,
    )

    hidden_size = test_tensors.rank_tokens.size(1)

    mk: FusedMoEKernel = make_modular_kernel(
        pg,
        pgi,
        dp_size,
        hidden_size,
        num_experts,
        num_local_experts,
        topk,
        q_dtype,
        use_fp8_dispatch,
        quant_config,
    )

    out = mk.apply(
        hidden_states=test_tensors.rank_tokens,
        w1=w1,
        w2=w2,
        topk_weights=test_tensors.topk_weights,
        topk_ids=test_tensors.topk,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=_build_expert_map(pgi, num_experts, num_local_experts),
        apply_router_weight_on_input=False,
    )

    return out


def _deep_ep_v2_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    use_fp8_dispatch: bool,
    per_act_token_quant: bool,
):
    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    is_quantized = w1.dtype == torch.float8_e4m3fn
    device_idx = torch.accelerator.current_device_index()
    w1 = w1.to(device=device_idx)
    w2 = w2.to(device=device_idx)
    if is_quantized:
        assert w1_scale is not None and w2_scale is not None
        w1_scale = w1_scale.to(device=device_idx)
        w2_scale = w2_scale.to(device=device_idx)

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    test_tensors = TestTensors.make(config)

    with set_current_vllm_config(VllmConfig()):
        # Reference
        q_dtype = torch.float8_e4m3fn if is_quantized else None
        torch_combined = torch_experts(
            test_tensors.rank_tokens,
            w1,
            w2,
            test_tensors.topk_weights,
            test_tensors.topk,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=test_tensors.rank_token_scales,
            a2_scale=test_tensors.intermediate_scales,
            quant_dtype=q_dtype,
            per_act_token_quant=per_act_token_quant,
        )

        # Splice experts for this rank
        num_local_experts = config.num_experts // pgi.world_size
        e_start = num_local_experts * pgi.rank
        e_end = e_start + num_local_experts
        w1_ep = w1[e_start:e_end]
        w2_ep = w2[e_start:e_end]

        w1_scale_ep, w2_scale_ep = None, None
        if is_quantized:
            w1_scale_ep = w1_scale[e_start:e_end]  # type: ignore
            w2_scale_ep = w2_scale[e_start:e_end]  # type: ignore

        deepep_combined = deepep_v2_moe_impl(
            pg,
            pgi,
            dp_size,
            test_tensors,
            w1_ep,
            w2_ep,
            w1_scale_ep,
            w2_scale_ep,
            config.num_experts,
            config.topk,
            use_fp8_dispatch,
            per_act_token_quant,
        )

    if is_quantized:
        assert_fp8_close(torch_combined, deepep_combined)
    else:
        torch.testing.assert_close(
            torch_combined,
            deepep_combined,
            atol=6e-2,
            rtol=6e-2,
        )


MNKs = [
    (1, 256, 256),
    (2, 256, 512),
    (3, 1024, 2048),
    (32, 256, 1024),
    (45, 512, 2048),
    (64, 1024, 1024),
    (222, 1024, 2048),
]

DTYPES = [torch.bfloat16, torch.float8_e4m3fn]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n,k", MNKs)
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@multi_gpu_test(num_gpus=2)
@requires_deep_ep_v2
def test_deep_ep_v2_moe(
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    world_dp_size: tuple[int, int],
    workspace_init,
):
    per_act_token_quant = False
    use_fp8_dispatch = False

    set_random_seed(7)
    world_size, dp_size = world_dp_size
    config = TestConfig(dtype=dtype, topk=topk, m=m, k=k, n=n, num_experts=num_experts)

    quant_dtype = dtype if dtype == torch.float8_e4m3fn else None
    (_, w1, w1_scale, _), (_, w2, w2_scale, _) = make_test_weights(
        num_experts,
        n,
        k,
        quant_dtype=quant_dtype,
        per_out_ch_quant=True,
    )

    parallel_launch(
        world_size,
        _deep_ep_v2_moe,
        dp_size,
        config,
        w1,
        w2,
        w1_scale,
        w2_scale,
        use_fp8_dispatch,
        per_act_token_quant,
    )


def _make_humming_experts(
    moe_config: FusedMoEConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: list[int],
) -> tuple[HummingGroupedExperts, FusedMoEQuantConfig, torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils import humming_utils
    from vllm.utils import humming

    layer = torch.nn.Module()
    layer.layer_name = "test_deep_ep_v2_humming"
    layer.moe_config = moe_config
    layer.params_dtype = torch.bfloat16
    layer.local_num_experts = moe_config.num_local_experts
    layer.global_num_experts = moe_config.num_experts
    layer.hidden_size = moe_config.hidden_dim
    layer.intermediate_size_per_partition = moe_config.intermediate_size_per_partition
    layer.weight_block_size = block_shape
    for name, tensor in (
        ("w13_weight", w1),
        ("w2_weight", w2),
        ("w13_weight_scale", w1_scale),
        ("w2_weight_scale", w2_scale),
    ):
        if name.endswith("_weight"):
            tensor = tensor.view(torch.int32)
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))

    weight_schema = humming.HummingWeightSchema(
        b_dtype=humming.dtypes.float8e4m3,
        bs_dtype=humming.dtypes.float32,
        weight_scale_group_size=block_shape[1],
        weight_scale_group_size_n=block_shape[0],
    )
    input_schema = humming.HummingInputSchema(
        a_dtype=humming.dtypes.float8e4m3,
        input_scale_group_size=block_shape[1],
    )
    humming_utils.convert_to_humming_moe_kernel_format(
        layer,
        weight_schema=weight_schema,
        input_schema=input_schema,
    )
    quant_config = humming_utils.get_humming_moe_quant_config(layer)
    experts = HummingGroupedExperts(
        layer=layer,
        moe_config=moe_config,
        quant_config=quant_config,
    )
    return experts, quant_config, layer.w13_weight, layer.w2_weight


def _make_mxfp4_humming_experts(
    moe_config: FusedMoEConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[HummingGroupedExperts, FusedMoEQuantConfig, torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils import humming_utils
    from vllm.utils import humming

    layer = torch.nn.Module()
    layer.layer_name = "test_deep_ep_v2_mxfp4_humming"
    layer.moe_config = moe_config
    layer.params_dtype = torch.bfloat16
    layer.local_num_experts = moe_config.num_local_experts
    layer.global_num_experts = moe_config.num_experts
    layer.hidden_size = moe_config.hidden_dim
    layer.intermediate_size_per_partition = moe_config.intermediate_size_per_partition
    for name, tensor in (
        ("w13_weight", w1),
        ("w2_weight", w2),
        ("w13_weight_scale", w1_scale),
        ("w2_weight_scale", w2_scale),
    ):
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))

    weight_schema = humming.HummingWeightSchema(
        b_dtype=humming.dtypes.float4e2m1,
        bs_dtype=humming.dtypes.float8e8m0,
        weight_scale_group_size=32,
    )
    input_schema = humming.HummingInputSchema()
    humming_utils.convert_to_humming_moe_kernel_format(
        layer,
        weight_schema=weight_schema,
        input_schema=input_schema,
    )
    quant_config = humming_utils.get_humming_moe_quant_config(
        layer,
        gemm1_clamp_limit=moe_config.swiglu_limit,
    )
    experts = HummingGroupedExperts(
        layer=layer,
        moe_config=moe_config,
        quant_config=quant_config,
    )
    return experts, quant_config, layer.w13_weight, layer.w2_weight


def _deep_ep_v2_moe_cudagraph(
    pgi: ProcessGroupInfo,
    dp_size: int,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    moe_backend: str,
    activation: MoEActivation,
    use_cudagraph: bool,
    tokens_per_rank: tuple[int, ...],
    weight_format: str,
):
    """Verify DeepEP v2 with an explicit expert backend and weight format."""
    import tempfile

    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    test_tensors = TestTensors.make(
        dataclasses.replace(config, m=tokens_per_rank[pgi.rank])
    )
    num_local_experts = config.num_experts // pgi.world_size
    hidden_size = config.k

    # All ranks must use the same global weights before taking their EP slice.
    w1_bf16 = (
        torch.randn(
            (config.num_experts, 2 * config.n, config.k),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 15
    )
    w2_bf16 = (
        torch.randn(
            (config.num_experts, config.k, config.n),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 15
    )
    torch.distributed.broadcast(w1_bf16, src=0, group=pg)
    torch.distributed.broadcast(w2_bf16, src=0, group=pg)

    from vllm.config import KernelConfig

    vllm_cfg = VllmConfig()
    vllm_cfg.parallel_config.data_parallel_size = dp_size
    vllm_cfg.parallel_config.data_parallel_rank = pgi.rank
    vllm_cfg.parallel_config.data_parallel_rank_local = pgi.local_rank
    vllm_cfg.parallel_config.enable_expert_parallel = True
    vllm_cfg.kernel_config = KernelConfig(moe_backend=moe_backend)

    with set_current_vllm_config(vllm_cfg):
        # Initialize vLLM parallel state (needed by MoERunner layer)
        temp_file = tempfile.mktemp()
        init_distributed_environment(
            world_size=pgi.world_size,
            rank=pgi.rank,
            distributed_init_method=f"file://{temp_file}",
            local_rank=pgi.local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        if weight_format == "fp8":
            from tests.kernels.moe.test_moe_layer import _quantize_fp8_halves

            block_shape = [128, 128]
            qw = _quantize_fp8_halves(
                w1_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16),
                w2_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16),
                block_shape,
            )
            assert qw.w13_weight_scale is not None
            assert qw.w2_weight_scale is not None
            w1_kernel = qw.w13_weight
            w2_kernel = qw.w2_weight
            w1_scale = qw.w13_weight_scale
            w2_scale = qw.w2_weight_scale
            w1_ref = w1_kernel
            w2_ref = w2_kernel
        else:
            assert weight_format == "mxfp4"
            assert moe_backend == "humming"
            from vllm.utils import humming

            w1_kernel, w1_scale, _, _ = humming.quantize_weight(
                w1_bf16,
                humming.dtypes.float4e2m1,
                humming.dtypes.float8e8m0,
                32,
                pack=True,
            )
            w2_kernel, w2_scale, _, _ = humming.quantize_weight(
                w2_bf16,
                humming.dtypes.float4e2m1,
                humming.dtypes.float8e8m0,
                32,
                pack=True,
            )
            assert w1_scale is not None and w2_scale is not None
            w1_ref = dequantize_mxfp4(w1_kernel, w1_scale).to(torch.bfloat16)
            w2_ref = dequantize_mxfp4(w2_kernel, w2_scale).to(torch.bfloat16)

        reference_topk_weights = test_tensors.topk_weights.to(torch.bfloat16).to(
            torch.float32
        )
        if test_tensors.rank_tokens.size(0) == 0:
            torch_combined = torch.empty_like(test_tensors.rank_tokens)
        else:
            torch_combined = torch_experts(
                test_tensors.rank_tokens,
                w1_ref,
                w2_ref,
                reference_topk_weights,
                test_tensors.topk,
                w1_scale=w1_scale if weight_format == "fp8" else None,
                w2_scale=w2_scale if weight_format == "fp8" else None,
                quant_dtype=(torch.float8_e4m3fn if weight_format == "fp8" else None),
                block_shape=block_shape if weight_format == "fp8" else None,
                activation=activation,
            )

        # EP-slice before format conversion
        e_start = num_local_experts * pgi.rank
        e_end = e_start + num_local_experts
        w1_ep = w1_kernel[e_start:e_end]
        w2_ep = w2_kernel[e_start:e_end]
        w1_scale_ep = w1_scale[e_start:e_end]
        w2_scale_ep = w2_scale[e_start:e_end]

        moe_config = make_dummy_moe_config(
            num_experts=config.num_experts,
            num_local_experts=num_local_experts,
            experts_per_token=config.topk,
            hidden_dim=hidden_size,
            intermediate_size=config.n,
            activation=activation,
        )
        if activation == MoEActivation.SITU:
            moe_config = dataclasses.replace(
                moe_config,
                activation_situ_beta=1.0,
            )
        if weight_format == "mxfp4":
            moe_config = dataclasses.replace(moe_config, swiglu_limit=10.0)
        moe_parallel_config = dataclasses.replace(
            moe_config.moe_parallel_config,
            ep_size=pgi.world_size,
            ep_rank=pgi.rank,
            dp_size=dp_size,
            dp_rank=pgi.rank,
            use_ep=True,
            all2all_backend="deepep_v2",
        )
        moe_config = dataclasses.replace(
            moe_config,
            moe_parallel_config=moe_parallel_config,
        )

        if moe_backend == "humming" and weight_format == "mxfp4":
            fused_experts, quant_config, w1_ep, w2_ep = _make_mxfp4_humming_experts(
                moe_config,
                w1_ep,
                w2_ep,
                w1_scale_ep,
                w2_scale_ep,
            )
        elif moe_backend == "humming":
            fused_experts, quant_config, w1_ep, w2_ep = _make_humming_experts(
                moe_config,
                w1_ep,
                w2_ep,
                w1_scale_ep,
                w2_scale_ep,
                block_shape,
            )
        else:
            assert moe_backend == "flashinfer_trtllm"
            assert weight_format == "fp8"
            from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
                TrtLlmFp8ExpertsModular,
            )
            from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
                Fp8MoeBackend,
                convert_to_fp8_moe_kernel_format,
            )

            class _MockLayer:
                weight_block_size = block_shape

                class moe_config:
                    is_act_and_mul = True
                    intermediate_size_per_partition = config.n

                class activation:
                    is_gated = True

            w1_ep, w2_ep, w1_scale_ep, w2_scale_ep = convert_to_fp8_moe_kernel_format(
                fp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
                layer=_MockLayer(),
                w13=w1_ep,
                w2=w2_ep,
                w13_scale=w1_scale_ep,
                w2_scale=w2_scale_ep,
                w13_input_scale=None,
                w2_input_scale=None,
            )
            quant_config = FusedMoEQuantConfig.make(
                torch.float8_e4m3fn,
                block_shape=block_shape,
                w1_scale=w1_scale_ep,
                w2_scale=w2_scale_ep,
            )
            fused_experts = TrtLlmFp8ExpertsModular(
                moe_config=moe_config,
                quant_config=quant_config,
            )

        v2_args = DeepEPV2Args(
            num_local_experts=num_local_experts,
            num_experts=config.num_experts,
            num_topk=config.topk,
            hidden_size=hidden_size,
            max_tokens_per_rank=8192,
            use_fp8_dispatch=False,
        )
        a2a = make_deepep_v2_a2a(
            pg=pg,
            pgi=pgi,
            dp_size=dp_size,
            v2_args=v2_args,
            use_cudagraph=use_cudagraph,
        )
        mk_kernel = FusedMoEKernel(
            prepare_finalize=a2a,
            fused_experts=fused_experts,
        )
        expert_map = _build_expert_map(pgi, config.num_experts, num_local_experts)

        num_tokens_across_dp = torch.tensor(
            tokens_per_rank,
            device="cpu",
            dtype=torch.int,
        )
        with set_forward_context(
            None,
            vllm_cfg,
            num_tokens=test_tensors.rank_tokens.size(0),
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            for _ in range(3):
                out = mk_kernel.apply(
                    hidden_states=test_tensors.rank_tokens,
                    w1=w1_ep,
                    w2=w2_ep,
                    topk_weights=test_tensors.topk_weights,
                    topk_ids=test_tensors.topk,
                    activation=activation,
                    global_num_experts=config.num_experts,
                    expert_map=expert_map,
                    apply_router_weight_on_input=False,
                )

        if torch_combined.numel() == 0:
            assert out.shape == torch_combined.shape
        elif moe_backend == "humming":
            assert_fp8_close(torch_combined, out)
        else:
            torch.testing.assert_close(
                torch_combined,
                out,
                atol=6e-2,
                rtol=6e-2,
            )


@pytest.mark.parametrize("m,n,k", [(32, 256, 1024)])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 2)])
@pytest.mark.parametrize(
    (
        "moe_backend",
        "activation",
        "use_cudagraph",
        "tokens_per_rank",
        "weight_format",
    ),
    [
        pytest.param(
            "flashinfer_trtllm",
            MoEActivation.SILU,
            True,
            (32, 32),
            "fp8",
            id="flashinfer_trtllm-silu-padded",
        ),
        pytest.param(
            "humming",
            MoEActivation.SILU,
            True,
            (32, 32),
            "fp8",
            id="humming-silu-padded",
        ),
        pytest.param(
            "humming",
            MoEActivation.SILU,
            False,
            (32, 32),
            "fp8",
            id="humming-silu-expanded",
        ),
        pytest.param(
            "humming",
            MoEActivation.SILU,
            True,
            (32, 0),
            "fp8",
            id="humming-silu-padded-idle-rank",
        ),
        pytest.param(
            "humming",
            MoEActivation.SILU,
            False,
            (32, 0),
            "fp8",
            id="humming-silu-expanded-idle-rank",
        ),
        pytest.param(
            "humming",
            MoEActivation.SITU,
            True,
            (32, 32),
            "fp8",
            id="humming-situ-padded",
        ),
        pytest.param(
            "humming",
            MoEActivation.SILU,
            True,
            (32, 0),
            "mxfp4",
            id="humming-mxfp4-clamped-silu-padded-idle-rank",
        ),
        pytest.param(
            "humming",
            MoEActivation.SILU,
            False,
            (32, 0),
            "mxfp4",
            id="humming-mxfp4-clamped-silu-expanded-idle-rank",
        ),
    ],
)
@multi_gpu_test(num_gpus=2)
@requires_deep_ep_v2
def test_deep_ep_v2_moe_cudagraph(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    world_dp_size: tuple[int, int],
    moe_backend: str,
    activation: MoEActivation,
    use_cudagraph: bool,
    tokens_per_rank: tuple[int, ...],
    weight_format: str,
    workspace_init,
):
    _launch_deep_ep_v2_case(
        m=m,
        n=n,
        k=k,
        num_experts=num_experts,
        topk=topk,
        world_dp_size=world_dp_size,
        moe_backend=moe_backend,
        activation=activation,
        use_cudagraph=use_cudagraph,
        tokens_per_rank=tokens_per_rank,
        weight_format=weight_format,
    )


def _launch_deep_ep_v2_case(
    *,
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    world_dp_size: tuple[int, int],
    moe_backend: str,
    activation: MoEActivation,
    use_cudagraph: bool,
    tokens_per_rank: tuple[int, ...],
    weight_format: str,
) -> None:
    set_random_seed(7)
    world_size, dp_size = world_dp_size
    config = TestConfig(
        dtype=torch.float8_e4m3fn,
        topk=topk,
        m=m,
        k=k,
        n=n,
        num_experts=num_experts,
    )

    parallel_launch(
        world_size,
        _deep_ep_v2_moe_cudagraph,
        dp_size,
        config,
        None,  # weights created inside worker
        None,
        None,
        None,
        moe_backend,
        activation,
        use_cudagraph,
        tokens_per_rank,
        weight_format,
    )


@multi_gpu_test(num_gpus=2)
@requires_deep_ep_v2
def test_deep_ep_v2_humming_dsv4_expert_topology(workspace_init):
    _launch_deep_ep_v2_case(
        m=8,
        n=2048,
        k=4096,
        num_experts=256,
        topk=6,
        world_dp_size=(2, 2),
        moe_backend="humming",
        activation=MoEActivation.SILU,
        use_cudagraph=False,
        tokens_per_rank=(8, 0),
        weight_format="mxfp4",
    )
