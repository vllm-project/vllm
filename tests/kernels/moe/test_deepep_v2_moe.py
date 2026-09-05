# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test DeepEP v2 (ElasticBuffer) dispatch-combine logic.
Compares against a pure-PyTorch reference MoE implementation.

DeepEP v2 emits a padded contiguous layout (PaddedStandard), so it can only be
paired with padding-aware experts. All experts exercised here are TRTLLM-Gen
kernels, which skip padding rows at tile granularity.
"""

import dataclasses

import pytest
import torch.distributed

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
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

# TRTLLM-Gen experts (the only padding-aware experts usable with DeepEP v2)
# require FlashInfer on SM100.
requires_flashinfer_sm100 = pytest.mark.skipif(
    not has_flashinfer() or not current_platform.has_device_capability(100),
    reason="Requires FlashInfer TRTLLM fused MoE (SM100)",
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

# DeepEP v2 emits a padded contiguous layout (PaddedStandard). Only experts
# that skip padding rows can consume it; flashinfer_cutlass processes all rows
# (Standard) and is therefore excluded here.
EXPERTS_BACKENDS = [
    "flashinfer_trtllm",
    "trtllm_fp8",
]


def _dtype_to_backend(dtype: torch.dtype) -> str:
    return "trtllm_fp8" if dtype == torch.float8_e4m3fn else "flashinfer_trtllm"


def _make_experts(
    experts_backend: str,
    config: TestConfig,
    moe_config,
    num_local_experts: int,
    rank: int,
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    test_tensors: TestTensors,
):
    e_start = num_local_experts * rank
    e_end = e_start + num_local_experts

    if experts_backend != "trtllm_fp8":
        from vllm.model_executor.layers.fused_moe.config import (
            FUSED_MOE_UNQUANTIZED_CONFIG,
        )
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            backend_to_kernel_cls,
            convert_to_unquantized_kernel_format,
            map_unquantized_backend,
        )

        torch_combined = torch_experts(
            test_tensors.rank_tokens,
            w1_bf16,
            w2_bf16,
            test_tensors.topk_weights,
            test_tensors.topk,
        )
        backend = map_unquantized_backend(experts_backend)
        w1_ep, w2_ep = convert_to_unquantized_kernel_format(
            backend,
            moe_config,
            w1_bf16[e_start:e_end],
            w2_bf16[e_start:e_end],
        )
        experts_cls = next(
            cls
            for cls in backend_to_kernel_cls(backend)
            if issubclass(cls, mk.FusedMoEExpertsModular)
        )
        fused_experts = experts_cls(
            moe_config=moe_config,
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        )
        return fused_experts, w1_ep, w2_ep, torch_combined, 1e-1, 2e-1

    from tests.kernels.moe.test_moe_layer import _quantize_fp8_halves
    from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
        TrtLlmFp8ExpertsModular,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        convert_to_fp8_moe_kernel_format,
    )

    w1_ref = w1_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)
    w2_ref = w2_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)

    block_shape = [128, 128]
    qw = _quantize_fp8_halves(w1_ref, w2_ref, block_shape)
    assert qw.w13_weight_scale is not None
    assert qw.w2_weight_scale is not None

    reference_topk_weights = test_tensors.topk_weights.to(torch.bfloat16).to(
        torch.float32
    )
    torch_combined = torch_experts(
        test_tensors.rank_tokens,
        qw.w13_weight,
        qw.w2_weight,
        reference_topk_weights,
        test_tensors.topk,
        w1_scale=qw.w13_weight_scale,
        w2_scale=qw.w2_weight_scale,
        quant_dtype=torch.float8_e4m3fn,
        block_shape=block_shape,
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
        w13=qw.w13_weight[e_start:e_end],
        w2=qw.w2_weight[e_start:e_end],
        w13_scale=qw.w13_weight_scale[e_start:e_end],
        w2_scale=qw.w2_weight_scale[e_start:e_end],
        w13_input_scale=None,
        w2_input_scale=None,
    )

    fused_experts = TrtLlmFp8ExpertsModular(
        moe_config=moe_config,
        quant_config=FusedMoEQuantConfig.make(
            torch.float8_e4m3fn,
            block_shape=block_shape,
            w1_scale=w1_scale_ep,
            w2_scale=w2_scale_ep,
        ),
    )
    return fused_experts, w1_ep, w2_ep, torch_combined, 6e-2, 6e-2


def _deep_ep_v2_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    config: TestConfig,
    use_cudagraph: bool,
    experts_backend: str,
    poison_padding: bool = False,
):
    import tempfile

    from vllm.config import KernelConfig
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    set_random_seed(7 + pgi.rank)
    test_tensors = TestTensors.make(config)
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

    vllm_cfg = VllmConfig()
    vllm_cfg.kernel_config = KernelConfig(moe_backend="flashinfer_trtllm")

    with set_current_vllm_config(vllm_cfg):
        temp_file = tempfile.mktemp()
        init_distributed_environment(
            world_size=pgi.world_size,
            rank=pgi.rank,
            distributed_init_method=f"file://{temp_file}",
            local_rank=pgi.local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)

        moe_config = make_dummy_moe_config(
            num_experts=config.num_experts,
            num_local_experts=num_local_experts,
            experts_per_token=config.topk,
            hidden_dim=hidden_size,
            intermediate_size=config.n,
        )
        moe_config = dataclasses.replace(
            moe_config,
            moe_parallel_config=dataclasses.replace(
                moe_config.moe_parallel_config,
                ep_size=pgi.world_size,
                ep_rank=pgi.rank,
                use_ep=True,
                all2all_backend="deepep_v2",
            ),
        )

        (
            fused_experts,
            w1_ep,
            w2_ep,
            torch_combined,
            atol,
            rtol,
        ) = _make_experts(
            experts_backend,
            config,
            moe_config,
            num_local_experts,
            pgi.rank,
            w1_bf16,
            w2_bf16,
            test_tensors,
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

        # Poison the padding rows of the dispatched activation with NaN before
        # the experts run. A padding-aware expert skips those rows at tile
        # granularity, so the valid output must stay finite and bit-identical to
        # the clean run; a leak proves the padding is being read.
        poison_fired = [False]
        if poison_padding:
            _orig_apply = fused_experts.apply

            def _poison_apply(*args, **kwargs):
                hs = kwargs["hidden_states"]
                tids = kwargs["topk_ids"]
                pad = (tids == -1).all(dim=1)
                if pad.any():
                    poison_fired[0] = True
                    hs[pad] = float("nan")
                return _orig_apply(*args, **kwargs)

            fused_experts.apply = _poison_apply

        with set_forward_context(None, vllm_cfg):
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

    if poison_padding:
        assert poison_fired[0], "no padding rows were poisoned; test is vacuous"
        assert torch.isfinite(out).all(), "padding NaNs leaked into the valid output"
    torch.testing.assert_close(torch_combined, out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n,k", MNKs)
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@multi_gpu_test(num_gpus=2)
@requires_deep_ep_v2
@requires_flashinfer_sm100
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
    set_random_seed(7)
    world_size, dp_size = world_dp_size
    config = TestConfig(dtype=dtype, topk=topk, m=m, k=k, n=n, num_experts=num_experts)

    parallel_launch(
        world_size,
        _deep_ep_v2_moe,
        dp_size,
        config,
        False,
        _dtype_to_backend(dtype),
    )


@pytest.mark.parametrize("m,n,k", [(32, 256, 1024)])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@pytest.mark.parametrize("experts_backend", EXPERTS_BACKENDS)
@pytest.mark.parametrize("use_cudagraph", [True, False])
@multi_gpu_test(num_gpus=2)
@requires_deep_ep_v2
@requires_flashinfer_sm100
def test_deep_ep_v2_moe_backends(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    world_dp_size: tuple[int, int],
    experts_backend: str,
    use_cudagraph: bool,
    workspace_init,
):
    set_random_seed(7)
    world_size, dp_size = world_dp_size
    config = TestConfig(
        dtype=torch.float8_e4m3fn
        if experts_backend == "trtllm_fp8"
        else torch.bfloat16,
        topk=topk,
        m=m,
        k=k,
        n=n,
        num_experts=num_experts,
    )

    parallel_launch(
        world_size,
        _deep_ep_v2_moe,
        dp_size,
        config,
        use_cudagraph,
        experts_backend,
    )


@pytest.mark.parametrize("m,n,k", [(32, 256, 1024)])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@pytest.mark.parametrize("experts_backend", EXPERTS_BACKENDS)
@multi_gpu_test(num_gpus=2)
@requires_deep_ep_v2
@requires_flashinfer_sm100
def test_deep_ep_v2_moe_padding_robust(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    world_dp_size: tuple[int, int],
    experts_backend: str,
    workspace_init,
):
    """Padding-aware experts must not read the worst-case padding tail.

    Runs the decode dispatch (``use_cudagraph=True``), which allocates a large
    worst-case recv buffer whose tail rows are padding, then fills those rows
    with NaN before the experts run. If the kernel truly skips padding at tile
    granularity the valid output stays finite and matches the reference; a NaN
    leak means padding is being computed on.
    """
    set_random_seed(7)
    world_size, dp_size = world_dp_size
    config = TestConfig(
        dtype=torch.float8_e4m3fn
        if experts_backend == "trtllm_fp8"
        else torch.bfloat16,
        topk=topk,
        m=m,
        k=k,
        n=n,
        num_experts=num_experts,
    )

    parallel_launch(
        world_size,
        _deep_ep_v2_moe,
        dp_size,
        config,
        True,
        experts_backend,
        True,
    )
