# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from itertools import product
from typing import Optional

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import VllmConfig, current_platform
# Fused experts imports
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
# PrepareFinalize imports
from vllm.utils import has_deep_ep, has_deep_gemm, has_pplx

from .modular_kernel_tools.common import (Config, RankTensors, WeightTensors,
                                          run_modular_kernel)
from .modular_kernel_tools.mk_objects import (
    MK_FUSED_EXPERT_TYPES, MK_MULTI_GPU_PREPARE_FINALIZE_TYPES,
    MK_QUANT_CONFIGS, MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES)
from .modular_kernel_tools.parallel_utils import (ProcessGroupInfo,
                                                  parallel_launch_with_config)

# TODO (varun): These requirements are very strict and could be relaxed.
meets_package_requirements = pytest.mark.skipif(
    not (has_deep_ep() and has_pplx() and has_deep_gemm()),
    reason="Requires deep_ep & deep_gemm & pplx packages",
)


def rank_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    config: Config,
    weights: WeightTensors,
):
    current_platform.seed_everything(pgi.rank)

    # sanity check
    from vllm import envs
    if config.fused_moe_chunk_size is not None:
        assert (config.fused_moe_chunk_size == envs.VLLM_FUSED_MOE_CHUNK_SIZE)

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
        run_modular_kernel(pgi, vllm_config, cfgx, weights, rank_tensors)


def run(config: Config):
    assert config.is_valid()
    print(f"Testing config \n{config.describe()} ...")

    weights: WeightTensors = WeightTensors.make(config)

    vllm_config, env_dict = config.make_env_data()
    parallel_launch_with_config(config.world_size, rank_worker, vllm_config,
                                env_dict, config, weights)


Ms = [64]
Ks = [7168]  # hidden sizes
Ns = [2048]
TOPKs = [4, 1]
Es = [32]
DTYPEs = [torch.bfloat16]
FUSED_MOE_CHUNK_SIZEs = [None, 64]


@pytest.mark.parametrize("k", Ks)
@pytest.mark.parametrize("n", Ns)
@pytest.mark.parametrize("e", Es)
@pytest.mark.parametrize("dtype", DTYPEs)
@pytest.mark.parametrize("quant_config", MK_QUANT_CONFIGS)
@pytest.mark.parametrize(
    "combination",
    product(MK_MULTI_GPU_PREPARE_FINALIZE_TYPES, MK_FUSED_EXPERT_TYPES))
@pytest.mark.parametrize("fused_moe_chunk_size", FUSED_MOE_CHUNK_SIZEs)
@pytest.mark.parametrize("world_size", [2])
@meets_package_requirements
def test_modular_kernel_combinations_multigpu(
        k: int, n: int, e: int, dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
        combination: tuple[mk.FusedMoEPrepareAndFinalize,
                           mk.FusedMoEPermuteExpertsUnpermute],
        fused_moe_chunk_size: Optional[int], world_size: int):

    config = Config(
        Ms=Ms,
        K=k,
        N=n,
        E=e,
        topks=TOPKs,
        dtype=dtype,
        quant_config=quant_config,
        prepare_finalize_type=combination[0],
        fused_experts_type=combination[1],
        fused_moe_chunk_size=fused_moe_chunk_size,
        world_size=world_size,
    )
    if not config.is_valid():
        pytest.skip(f"Tests config {config} is not valid. Skipping ...")
    run(config)


@pytest.mark.parametrize("k", Ks)
@pytest.mark.parametrize("n", Ns)
@pytest.mark.parametrize("e", Es)
@pytest.mark.parametrize("dtype", DTYPEs)
@pytest.mark.parametrize("quant_config", MK_QUANT_CONFIGS)
@pytest.mark.parametrize(
    "combination",
    product(MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES, MK_FUSED_EXPERT_TYPES))
@pytest.mark.parametrize("fused_moe_chunk_size", FUSED_MOE_CHUNK_SIZEs)
@pytest.mark.parametrize("world_size", [1])
@meets_package_requirements
def test_modular_kernel_combinations_singlegpu(
        k: int, n: int, e: int, dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
        combination: tuple[mk.FusedMoEPrepareAndFinalize,
                           mk.FusedMoEPermuteExpertsUnpermute],
        fused_moe_chunk_size: Optional[int], world_size: int):
    config = Config(
        Ms=Ms,
        K=k,
        N=n,
        E=e,
        topks=TOPKs,
        dtype=dtype,
        quant_config=quant_config,
        prepare_finalize_type=combination[0],
        fused_experts_type=combination[1],
        fused_moe_chunk_size=fused_moe_chunk_size,
        world_size=world_size,
    )

    if not config.is_valid():
        pytest.skip(f"Tests config {config} is not valid. Skipping ...")
    run(config)


if __name__ == '__main__':
    # Ability to test individual PrepareAndFinalize and FusedExperts combination
    from .modular_kernel_tools.cli_args import (make_config,
                                                make_config_arg_parser)
    parser = make_config_arg_parser(description=(
        "Run single prepare-finalize & fused-experts combination test"
        "Example : python3 -m tests.kernels.moe.test_modular_kernel_combinations --pf-type PplxPrepareAndFinalize --experts-type BatchedTritonExperts"  #noqa: E501
    ))
    args = parser.parse_args()
    config = make_config(args)

    run(config)
