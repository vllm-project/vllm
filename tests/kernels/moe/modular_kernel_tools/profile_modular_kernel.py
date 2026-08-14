# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections.abc import Callable
from itertools import product
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from .common import (
    Config,
    RankTensors,
    WeightTensors,
    _make_gscale,
    make_modular_kernel,
)
from .parallel_utils import ProcessGroupInfo, parallel_launch_with_config


def do_profile(
    fn: Callable,
    fn_kwargs: dict[Any, Any],
    pgi: ProcessGroupInfo,
    config: Config,
    num_warmups: int = 5,
):
    for _ in range(num_warmups):
        fn(**fn_kwargs)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        record_shapes=True,
    ) as tprof:
        fn(**fn_kwargs)
        device = torch.accelerator.current_device_index()
        torch.accelerator.synchronize(device)

    # TODO (varun): Add a descriptive trace file name
    tprof.export_chrome_trace(
        f"{config.torch_trace_dir_path}/m{config.M}_{pgi.rank}_trace.json"
    )


def profile_modular_kernel(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    config: Config,
    weights: WeightTensors,
    rank_tensors: RankTensors,
) -> None:
    assert isinstance(config.Ms, int)
    assert isinstance(config.topks, int)

    # weights for rank
    rank_weights = weights.slice_weights(pgi.rank, config.num_local_experts)

    if config.quant_dtype == "nvfp4":
        gscale = _make_gscale(config.num_local_experts)
    else:
        gscale = None

    quant_config = FusedMoEQuantConfig.make(
        config.quant_dtype,
        w1_scale=rank_weights.w1_scale,
        w2_scale=rank_weights.w2_scale,
        a1_scale=rank_tensors.hidden_states_scale,
        g1_alphas=(1 / rank_weights.w1_gs) if rank_weights.w1_gs is not None else None,
        g2_alphas=(1 / rank_weights.w2_gs) if rank_weights.w2_gs is not None else None,
        a1_gscale=gscale,
        a2_gscale=gscale,
        block_shape=config.quant_block_shape,
        per_act_token_quant=config.is_per_act_token_quant,
        per_out_ch_quant=config.is_per_out_ch_quant,
    )

    # make modular kernel
    mk = make_modular_kernel(config, vllm_config, quant_config)

    topk_ids = rank_tensors.topk_ids.to(
        mk.prepare_finalize.topk_indices_dtype() or rank_tensors.topk_ids.dtype
    )

    # impls might update the tensor in place
    hidden_states = rank_tensors.hidden_states.clone()

    mk_kwargs = {
        "hidden_states": hidden_states,
        "w1": rank_weights.w1,
        "w2": rank_weights.w2,
        "topk_weights": rank_tensors.topk_weights,
        "topk_ids": topk_ids,
        "activation": MoEActivation.SILU,
        "expert_map": rank_tensors.expert_map,
        "global_num_experts": config.E,
        "apply_router_weight_on_input": config.topk == 1
        and config.supports_apply_weight_on_input(),
    }

    num_tokens = hidden_states.shape[0]
    num_tokens_across_dp = torch.tensor(
        [num_tokens] * config.world_size, device="cpu", dtype=torch.int
    )

    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        do_profile(mk.apply, mk_kwargs, pgi, config)


def rank_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    config: Config,
    weights: WeightTensors,
):
    set_random_seed(pgi.rank)

    # workspace manager is normally initialized by GPUModelRunner; we initialize
    # it here for the standalone benchmark process.
    init_workspace_manager(torch.device(f"cuda:{pgi.local_rank}"))

    # get weights to this device
    weights.to_current_device()

    Ms = config.Ms
    assert isinstance(Ms, list)
    TOPKs = config.topks
    assert isinstance(TOPKs, list)

    for m, topk in product(Ms, TOPKs):
        print(f"Running m={m}, topk={topk} ...")
        # override m and topk
        cfgx = copy.deepcopy(config)
        cfgx.Ms = m
        cfgx.topks = topk

        # inputs for rank
        rank_tensors = RankTensors.make(cfgx, pgi)
        profile_modular_kernel(pgi, vllm_config, cfgx, weights, rank_tensors)


def run(config: Config):
    weights: WeightTensors = WeightTensors.make(config)
    vllm_config, env_dict = config.make_env_data()
    parallel_launch_with_config(
        config.world_size, rank_worker, vllm_config, env_dict, config, weights
    )


if __name__ == "__main__":
    from .cli_args import make_config, make_config_arg_parser

    parser = make_config_arg_parser(
        description=(
            "Run single prepare-finalize & fused-experts combination test"
            "Example : python3 -m tests.kernels.moe.modular_kernel_tools.profile_modular_kernel "  # noqa: E501
            "--pf-type DeepEPLLPrepareAndFinalize --experts-type BatchedTritonExperts"
        )
    )
    args = parser.parse_args()
    assert args.torch_trace_dir_path is not None, (
        "Please pass in a directory to store torch traces"
    )
    config = make_config(args)

    run(config)
