# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import textwrap
import traceback
from itertools import product
from typing import Any

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.utils.import_utils import has_deep_ep, has_deep_gemm, has_pplx
from vllm.utils.torch_utils import cuda_device_count_stateless, set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from .modular_kernel_tools.common import (
    Config,
    RankTensors,
    WeightTensors,
    reference_moe_impl,
    run_modular_kernel,
)
from .modular_kernel_tools.mk_objects import (
    MK_FUSED_EXPERT_TYPES,
    MK_MULTI_GPU_PREPARE_FINALIZE_TYPES,
    MK_QUANT_CONFIGS,
    MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES,
    TestMoEQuantConfig,
    expert_info,
)
from .modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)

has_any_multi_gpu_package = (
    has_deep_ep() or has_deep_gemm() or has_pplx() or has_flashinfer_cutlass_fused_moe()
)

meets_multi_gpu_requirements = pytest.mark.skipif(
    not has_any_multi_gpu_package,
    reason="Requires deep_ep or deep_gemm or pplx or flashinfer packages",
)

if current_platform.is_fp8_fnuz():
    pytest.skip(
        "Tests in this file require float8_e4m3fn and platform does not support",
        allow_module_level=True,
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
    # Initialize workspace manager in child process
    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    set_random_seed(pgi.rank)

    # sanity check
    from vllm import envs

    if base_config.fused_moe_chunk_size is not None:
        assert base_config.fused_moe_chunk_size == envs.VLLM_FUSED_MOE_CHUNK_SIZE

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
            mk_out = run_modular_kernel(pgi, vllm_config, config, weights, rank_tensors)

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
            f"rank={pgi.rank}."
        )
    else:
        print(f"{count} of {count} tests passed in child process, rank={pgi.rank}.")


def run(config: Config, verbose: bool):
    assert config.is_valid()[0]
    assert not is_nyi_config(config)

    weights: WeightTensors = WeightTensors.make(config)

    vllm_config, env_dict = config.make_env_data()
    parallel_launch_with_config(
        config.world_size, rank_worker, vllm_config, env_dict, config, weights, verbose
    )


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
        unsupported_quant_config = (
            config.is_per_act_token_quant + config.is_per_out_ch_quant
        ) == 1
        return unsupported_quant_config

    return not info.supports_expert_map


def generate_valid_test_cases(
    world_size: int, prepare_finalize_types
) -> list[tuple[Any, ...]]:
    cases = []
    total = 0

    for k, n, e, dtype, quant_config, combination, chunk_size in product(
        Ks,
        Ns,
        Es,
        DTYPEs,
        MK_QUANT_CONFIGS,
        product(prepare_finalize_types, MK_FUSED_EXPERT_TYPES),
        FUSED_MOE_CHUNK_SIZEs,
    ):
        total = total + 1

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
            fused_moe_chunk_size=chunk_size,
            world_size=world_size,
        )

        # TODO(bnell): figure out how to get verbose flag here.
        verbose = False  # pytestconfig.getoption('verbose') > 0

        valid, reason = config.is_valid()

        if not valid:
            if verbose:
                print(f"Test config {config} is not valid: {reason}")
            continue

        if is_nyi_config(config):
            if verbose:
                print(f"Test config {config} is nyi.")
            continue

        cases.append(
            (
                k,
                n,
                e,
                dtype,
                quant_config,
                combination[0],
                combination[1],
                chunk_size,
                world_size,
            )
        )

    print(f"{len(cases)} of {total} valid configs generated.")

    return cases


@pytest.mark.parametrize(
    "k,n,e,dtype,quant_config,prepare_finalize_type,fused_experts_type,chunk_size,world_size",
    generate_valid_test_cases(
        world_size=2, prepare_finalize_types=MK_MULTI_GPU_PREPARE_FINALIZE_TYPES
    ),
)
@meets_multi_gpu_requirements
def test_modular_kernel_combinations_multigpu(
    k: int,
    n: int,
    e: int,
    dtype: torch.dtype,
    quant_config: TestMoEQuantConfig | None,
    prepare_finalize_type: mk.FusedMoEPrepareAndFinalize,
    fused_experts_type: mk.FusedMoEExperts,
    chunk_size: int | None,
    world_size: int,
    pytestconfig,
):
    if cuda_device_count_stateless() < world_size:
        pytest.skip(
            f"Not enough GPUs available to run, got "
            f"{cuda_device_count_stateless()} exepected "
            f"{world_size}."
        )

    config = Config(
        Ms=Ms,
        K=k,
        N=n,
        E=e,
        topks=TOPKs,
        dtype=dtype,
        quant_config=quant_config,
        prepare_finalize_type=prepare_finalize_type,
        fused_experts_type=fused_experts_type,
        fused_moe_chunk_size=chunk_size,
        world_size=world_size,
    )
    verbosity = pytestconfig.getoption("verbose")
    run(config, verbosity > 0)


@pytest.mark.parametrize(
    "k,n,e,dtype,quant_config,prepare_finalize_type,fused_experts_type,chunk_size,world_size",
    generate_valid_test_cases(
        world_size=1, prepare_finalize_types=MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES
    ),
)
def test_modular_kernel_combinations_singlegpu(
    k: int,
    n: int,
    e: int,
    dtype: torch.dtype,
    quant_config: TestMoEQuantConfig | None,
    prepare_finalize_type: mk.FusedMoEPrepareAndFinalize,
    fused_experts_type: mk.FusedMoEExperts,
    chunk_size: int | None,
    world_size: int,
    pytestconfig,
    workspace_init,
):
    """Note: float8_e4m3fn is not supported on CUDA architecture < 89,
    and those tests will be skipped on unsupported hardware."""
    config = Config(
        Ms=Ms,
        K=k,
        N=n,
        E=e,
        topks=TOPKs,
        dtype=dtype,
        quant_config=quant_config,
        prepare_finalize_type=prepare_finalize_type,
        fused_experts_type=fused_experts_type,
        fused_moe_chunk_size=chunk_size,
        world_size=world_size,
    )

    if (
        quant_config is not None and quant_config.quant_dtype == torch.float8_e4m3fn
    ) and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )
    verbosity = pytestconfig.getoption("verbose")
    run(config, verbosity > 0)


if __name__ == "__main__":
    # Ability to test individual PrepareAndFinalize and FusedExperts combination
    from .modular_kernel_tools.cli_args import make_config, make_config_arg_parser

    parser = make_config_arg_parser(
        description=(
            "Run single prepare-finalize & fused-experts combination test"
            "Example : python3 -m tests.kernels.moe.test_modular_kernel_combinations "
            "--pf-type PplxPrepareAndFinalize --experts-type BatchedTritonExperts"
        )
    )
    args = parser.parse_args()
    config = make_config(args)

    run(config, True)
