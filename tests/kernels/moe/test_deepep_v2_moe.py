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
from vllm.model_executor.layers.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
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

        topk = torch.stack(
            [
                torch.randperm(config.num_experts, device="cuda")[: config.topk]
                for _ in range(config.m)
            ]
        ).to(dtype=torch.int64)
        topk_weights = torch.randn(topk.shape, dtype=torch.float32, device="cuda")
        return TestTensors(
            rank_tokens=rank_tokens,
            rank_token_scales=None,
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
        inplace=False,
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

    def build_expert_map():
        expert_map = torch.full((num_experts,), fill_value=-1, dtype=torch.int32)
        s = pgi.rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)))
        device = torch.accelerator.current_device_index()
        return expert_map.to(device=device, dtype=torch.int32)

    is_quantized = w1.dtype == torch.float8_e4m3fn
    q_dtype = torch.float8_e4m3fn if is_quantized else None

    quant_config = FusedMoEQuantConfig.make(
        q_dtype,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        per_act_token_quant=per_act_token_quant,
        a1_scale=test_tensors.rank_token_scales,
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
        expert_map=build_expert_map(),
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


def _deep_ep_v2_moe_cudagraph(
    pgi: ProcessGroupInfo,
    dp_size: int,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
):
    """Worker function: verify DeepEP v2 + TrtLLM FP8 with do_expand=False."""
    import tempfile

    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    test_tensors = TestTensors.make(config)
    num_local_experts = config.num_experts // pgi.world_size
    hidden_size = config.k

    # Create FP8 weights directly, then dequantize for bf16 reference.
    w1_fp8 = torch.randn(
        (config.num_experts, 2 * config.n, config.k),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    w2_fp8 = torch.randn(
        (config.num_experts, config.k, config.n),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    w1_ref = w1_fp8.to(torch.bfloat16)
    w2_ref = w2_fp8.to(torch.bfloat16)

    from vllm.config import KernelConfig

    vllm_cfg = VllmConfig()
    vllm_cfg.kernel_config = KernelConfig(moe_backend="flashinfer_trtllm")

    with set_current_vllm_config(vllm_cfg):
        # Initialize vLLM parallel state (needed by FusedMoE layer)
        temp_file = tempfile.mktemp()
        init_distributed_environment(
            world_size=pgi.world_size,
            rank=pgi.rank,
            distributed_init_method=f"file://{temp_file}",
            local_rank=pgi.local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        # Reference MoE using dequantized bf16 weights
        torch_combined = torch_experts(
            test_tensors.rank_tokens,
            w1_ref,
            w2_ref,
            test_tensors.topk_weights,
            test_tensors.topk,
        )

        # Use the production pipeline: make_fused_moe_layer creates
        # a FusedMoE layer, quantizes weights, runs
        # process_weights_after_loading (TrtLLM W31 swap + BlockMajorK
        # shuffle), and selects the kernel.
        # Quantize weights using production helper, EP-slice, then
        # convert to TrtLLM format.
        from tests.kernels.moe.test_moe_layer import _quantize_fp8_halves
        from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
            TrtLlmFp8ExpertsModular,
        )
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            Fp8MoeBackend,
            convert_to_fp8_moe_kernel_format,
        )

        block_shape = [128, 128]
        qw = _quantize_fp8_halves(w1_ref, w2_ref, block_shape)

        # EP-slice before format conversion
        e_start = num_local_experts * pgi.rank
        e_end = e_start + num_local_experts
        w1_ep = qw.w13_weight[e_start:e_end]
        w2_ep = qw.w2_weight[e_start:e_end]
        assert qw.w13_weight_scale is not None
        assert qw.w2_weight_scale is not None
        w1_scale_ep = qw.w13_weight_scale[e_start:e_end]
        w2_scale_ep = qw.w2_weight_scale[e_start:e_end]

        # Convert to TrtLLM format (W31 swap + BlockMajorK shuffle)
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

        # Build TrtLLM expert with correct EP params
        quant_config = FusedMoEQuantConfig.make(
            torch.float8_e4m3fn,
            block_shape=block_shape,
            w1_scale=w1_scale_ep,
            w2_scale=w2_scale_ep,
        )
        moe_config = make_dummy_moe_config(
            num_experts=num_local_experts,
            experts_per_token=config.topk,
            hidden_dim=hidden_size,
            intermediate_size=config.n,
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
            use_cudagraph=True,
        )
        mk_kernel = FusedMoEKernel(
            prepare_finalize=a2a,
            fused_experts=fused_experts,
            inplace=False,
        )

        for _ in range(3):
            out = mk_kernel.apply(
                hidden_states=test_tensors.rank_tokens,
                w1=w1_ep,
                w2=w2_ep,
                topk_weights=test_tensors.topk_weights,
                topk_ids=test_tensors.topk,
                activation=MoEActivation.SILU,
                global_num_experts=config.num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        torch.testing.assert_close(
            torch_combined,
            out,
            atol=6e-2,
            rtol=6e-2,
        )


@pytest.mark.parametrize("m,n,k", [(32, 256, 1024)])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@multi_gpu_test(num_gpus=2)
@requires_deep_ep_v2
def test_deep_ep_v2_moe_cudagraph(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    world_dp_size: tuple[int, int],
    workspace_init,
):
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
    )
