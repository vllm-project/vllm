# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import textwrap
import traceback
from itertools import product
from typing import Optional

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils import has_deep_ep, has_deep_gemm, has_pplx
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

from ...utils import multi_gpu_test
from .modular_kernel_tools.common import (Config, RankTensors, WeightTensors,
                                          reference_moe_impl,
                                          run_modular_kernel)
from .modular_kernel_tools.mk_objects import (
    MK_FUSED_EXPERT_TYPES, MK_MULTI_GPU_PREPARE_FINALIZE_TYPES,
    MK_QUANT_CONFIGS, MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES, TestMoEQuantConfig,
    expert_info)
from .modular_kernel_tools.parallel_utils import (ProcessGroupInfo,
                                                  parallel_launch_with_config)

has_any_multi_gpu_package = (has_deep_ep() or has_deep_gemm() or has_pplx()
                             or has_flashinfer_cutlass_fused_moe())

meets_multi_gpu_requirements = pytest.mark.skipif(
    not has_any_multi_gpu_package,
    reason="Requires deep_ep or deep_gemm or pplx or flashinfer packages",
)


def format_result(verbose, msg, ex=None):
    if ex is not None:
        x = str(ex)
        newx = x.strip(" \n\t")[:16]
        if len(newx) < len(x):
            newx = newx + " ..."

        prefix = "E\t"
        print(f"{textwrap.indent(traceback.format_exc(), prefix)}")
        print(f"FAILED {msg} - {newx}\n")
    elif verbose:
        print(f"PASSED {msg}")
    else:
        print(".", end="")


def rank_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    base_config: Config,
    weights: WeightTensors,
    verbose: bool,
):
    current_platform.seed_everything(pgi.rank)

    # sanity check
    from vllm import envs
    if base_config.fused_moe_chunk_size is not None:
        assert (
            base_config.fused_moe_chunk_size == envs.VLLM_FUSED_MOE_CHUNK_SIZE)

    # get weights to this device
    weights.to_current_device()

    Ms = base_config.Ms
    assert isinstance(Ms, list)
    TOPKs = base_config.topks
    assert isinstance(TOPKs, list)

    exceptions = []
    count = 0

    for m, topk in product(Ms, TOPKs):
        # override m and topk
        config = copy.deepcopy(base_config)
        config.Ms = m
        config.topks = topk

        try:
            print(f"Running[{pgi.rank}]: m={m}, topk={topk} ...")
            count = count + 1

            # inputs for rank
            rank_tensors = RankTensors.make(config, pgi)

            # modular kernel out
            mk_out = run_modular_kernel(pgi, vllm_config, config, weights,
                                        rank_tensors)

            with set_current_vllm_config(vllm_config):
                ref_out = reference_moe_impl(config, weights, rank_tensors)

            if config.quant_dtype == "nvfp4":
                atol = 1e-1 if config.K < 4096 else 2e-1
                rtol = 1e-1 if config.K < 4096 else 2e-1
            else:
                atol = 3e-2
                rtol = 3e-2

            torch.testing.assert_close(ref_out, mk_out, atol=atol, rtol=rtol)
            format_result(verbose, config.describe())
        except Exception as ex:
            format_result(verbose, config.describe(), ex)
            exceptions.append(ex)

    if len(exceptions) > 0:
        raise RuntimeError(
            f"{len(exceptions)} of {count} tests failed in child process, "
            f"rank={pgi.rank}.")
    else:
        print(f"{count} of {count} tests passed in child process, "
              f"rank={pgi.rank}.")


def run(config: Config, verbose: bool):
    assert config.is_valid()

    weights: WeightTensors = WeightTensors.make(config)

    vllm_config, env_dict = config.make_env_data()
    parallel_launch_with_config(config.world_size, rank_worker, vllm_config,
                                env_dict, config, weights, verbose)


Ms = [32, 64]
# hidden sizes, making this too large will cause fp4 tests to fail.
# Also needs to be a multiple of 1024 for deep_gemm.
Ks = [2048]
Ns = [1024]
TOPKs = [4, 1]
Es = [32]
DTYPEs = [torch.bfloat16]
FUSED_MOE_CHUNK_SIZEs = [None, 16]


def is_nyi_config(config: Config) -> bool:
    # We know these configs to be legitimate. but still fail.
    info = expert_info(config.fused_experts_type)

    if info.needs_matching_quant:
        # The triton kernels expect both per-act-token-quant and
        # per-out-ch-quant or neither.
        unsupported_quant_config = ((config.is_per_act_token_quant +
                                     config.is_per_out_ch_quant) == 1)
        return unsupported_quant_config

    return not info.supports_expert_map


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
@multi_gpu_test(num_gpus=2)
@meets_multi_gpu_requirements
def test_modular_kernel_combinations_multigpu(
        k: int, n: int, e: int, dtype: torch.dtype,
        quant_config: Optional[TestMoEQuantConfig],
        combination: tuple[mk.FusedMoEPrepareAndFinalize,
                           mk.FusedMoEPermuteExpertsUnpermute],
        fused_moe_chunk_size: Optional[int], world_size: int, pytestconfig):

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

    if is_nyi_config(config):
        pytest.skip(f"Tests config {config} is nyi. Skipping ...")

    verbosity = pytestconfig.getoption('verbose')
    run(config, verbosity > 0)


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
def test_modular_kernel_combinations_singlegpu(
        k: int, n: int, e: int, dtype: torch.dtype,
        quant_config: Optional[TestMoEQuantConfig],
        combination: tuple[mk.FusedMoEPrepareAndFinalize,
                           mk.FusedMoEPermuteExpertsUnpermute],
        fused_moe_chunk_size: Optional[int], world_size: int, pytestconfig):
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

    if is_nyi_config(config):
        pytest.skip(f"Tests config {config} is nyi. Skipping ...")

    verbosity = pytestconfig.getoption('verbose')
    run(config, verbosity > 0)


if __name__ == '__main__':
    # Ability to test individual PrepareAndFinalize and FusedExperts combination
    from .modular_kernel_tools.cli_args import (make_config,
                                                make_config_arg_parser)
    parser = make_config_arg_parser(description=(
        "Run single prepare-finalize & fused-experts combination test"
        "Example : python3 -m tests.kernels.moe.test_modular_kernel_combinations "  #noqa: E501
        "--pf-type PplxPrepareAndFinalize --experts-type BatchedTritonExperts"
    ))
    args = parser.parse_args()
    config = make_config(args)

    run(config, True)
