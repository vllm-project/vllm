# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the real expert backends behind nixl_ep_v2's dispatch and combine.
Compares against the reference MoE that test_deepep_v2_moe.py builds.
"""

import dataclasses

import pytest
import torch.distributed

from tests.kernels.moe.test_deepep_v2_moe import (
    EXPERTS_BACKENDS,
    TestConfig,
    TestTensors,
    _make_experts,
)
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl_moe_nvfp4
from vllm.utils.import_utils import has_nixl_ep_v2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch


def _nixl_ep_v2_moe_backends(
    pgi: ProcessGroupInfo,
    config: TestConfig,
    use_cudagraph: bool,
    experts_backend: str,
    expand: bool,
):
    import tempfile

    from vllm.config import KernelConfig, ParallelConfig
    from vllm.distributed import (
        get_ep_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep_v2 import (
        NixlEPV2PrepareAndFinalize,
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

    # The nixl_ep_v2 buffer comes from the EP group's all2all manager,
    # which takes one DP rank per GPU.
    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=pgi.world_size,
            enable_expert_parallel=True,
            all2all_backend="nixl_ep_v2",
        )
    )
    vllm_cfg.parallel_config.data_parallel_rank = pgi.rank
    vllm_cfg.kernel_config = KernelConfig(moe_backend="flashinfer_trtllm")

    with set_current_vllm_config(vllm_cfg):
        temp_file = tempfile.mktemp()
        # init_distributed_environment multiplies these back by the DP size.
        init_distributed_environment(
            world_size=1,
            rank=0,
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
                all2all_backend="nixl_ep_v2",
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

        mgr = get_ep_group().device_communicator.all2all_manager
        quant_config = fused_experts.quant_config
        buffer = mgr.get_handle(
            dict(
                num_max_tokens_per_rank=8192,
                hidden=hidden_size,
                num_topk=config.topk,
                # As all2all_utils sizes it: block FP8 dispatches FP8.
                use_fp8_dispatch=quant_config.use_fp8_w8a8
                and quant_config.is_block_quantized,
                num_local_experts=num_local_experts,
            )
        )
        a2a = NixlEPV2PrepareAndFinalize(
            buffer,
            num_dispatchers=pgi.world_size,
            rank_expert_offset=pgi.rank * num_local_experts,
            num_sms=mgr.num_sms,
            use_cudagraph=use_cudagraph,
            expand=expand,
        )
        mk_kernel = FusedMoEKernel(
            prepare_finalize=a2a,
            fused_experts=fused_experts,
        )

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

    torch.testing.assert_close(torch_combined, out, atol=atol, rtol=rtol)
    # NVFP4 noise needs an atol that would also pass a wrong output; so compare
    # direction too.
    cosine = torch.nn.functional.cosine_similarity(
        out.flatten().float(), torch_combined.flatten().float(), dim=0
    )
    assert cosine > 0.99, f"cosine similarity {cosine:.4f} below 0.99"


@pytest.mark.parametrize("experts_backend", EXPERTS_BACKENDS)
@pytest.mark.parametrize(
    "use_cudagraph,expand",
    [
        pytest.param(True, False, id="padded"),
        pytest.param(True, True, id="expanded"),
        pytest.param(False, False, id="exact"),
    ],
)
@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not has_nixl_ep_v2(), reason="Requires NIXL EP v2 (nixl.runtime.ep)"
)
@pytest.mark.skipif(
    not has_flashinfer() or not current_platform.has_device_capability(100),
    reason="Requires FlashInfer TRTLLM fused MoE (SM100)",
)
def test_nixl_ep_v2_moe_backends(
    experts_backend: str, use_cudagraph: bool, expand: bool, workspace_init
):
    """The real experts behind nixl_ep_v2's dispatch and combine, in its
    padded, worst-case expanded (VLLM_NIXL_EP_V2_EXPAND) and exact layouts.
    Block FP8 experts take FP8 dispatch, as all2all_utils builds it."""
    if (
        experts_backend == "flashinfer_cutedsl_nvfp4"
        and not has_flashinfer_cutedsl_moe_nvfp4()
    ):
        pytest.skip("Requires FlashInfer CuTeDSL NVFP4 MoE")
    set_random_seed(7)
    # m=24 rounds up to a 32-token bucket: the padded layout gets padding rows.
    config = TestConfig(
        dtype=torch.float8_e4m3fn
        if experts_backend == "trtllm_fp8"
        else torch.bfloat16,
        topk=6,
        m=24,
        k=1024,
        n=256,
        num_experts=32,
    )

    parallel_launch(
        2,
        _nixl_ep_v2_moe_backends,
        config,
        use_cudagraph,
        experts_backend,
        expand,
    )
