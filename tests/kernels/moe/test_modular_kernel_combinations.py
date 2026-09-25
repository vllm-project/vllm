# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import inspect
import textwrap
import traceback
from itertools import product
from typing import Any
from unittest import mock

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import AiterExperts
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.utils.import_utils import has_aiter, has_deep_ep, has_deep_gemm
from vllm.utils.torch_utils import set_random_seed
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
from .utils import check_accuracy

has_any_multi_gpu_package = (
    has_deep_ep() or has_deep_gemm() or has_flashinfer_cutlass_fused_moe()
)

meets_multi_gpu_requirements = pytest.mark.skipif(
    not has_any_multi_gpu_package,
    reason="Requires deep_ep or deep_gemm or flashinfer packages",
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


def assert_aiter_quant_scheme_case(config: Config) -> None:
    """Make the AITER (weight_quant_key, activation_quant_key) pair this
    config exercises explicit, instead of AiterExperts being reached only
    indirectly through the general quant-config sweep.
    See https://github.com/vllm-project/vllm/issues/54966."""
    fe_cls = config.fused_experts_type
    if fe_cls is not AiterExperts:
        return

    if config.quant_config is None:
        w_key, a_key = None, None
    else:
        w_key, a_key = config.fp8_quant_key_pair()

    assert fe_cls._supports_quant_scheme(w_key, a_key), (
        f"AITER case (weight_key={w_key}, activation_key={a_key}) reached "
        "the modular-kernel harness, but AiterExperts._supports_quant_scheme "
        "does not declare it supported."
    )
    print(f"[AITER case] weight_key={w_key}, activation_key={a_key}")


def assert_aiter_activation_case(config: Config) -> None:
    """Make the AITER activation this config exercises explicit, instead of
    AiterExperts being reached only indirectly through the general
    activation sweep. See https://github.com/vllm-project/vllm/issues/54966."""
    fe_cls = config.fused_experts_type
    if fe_cls is not AiterExperts:
        return

    assert fe_cls._supports_activation(config.activation), (
        f"AITER case activation={config.activation} reached the "
        "modular-kernel harness, but AiterExperts._supports_activation "
        "does not declare it supported."
    )
    print(f"[AITER case] activation={config.activation}")


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

            # Skip unsupported: AITER block-scaled MoE does not
            # support apply_router_weight_on_input (topk=1 path).
            # https://github.com/ROCm/aiter/issues/2418
            if (
                topk == 1
                and config.supports_apply_weight_on_input()
                and config.fused_experts_type is AiterExperts
                and config.quant_block_shape is not None
            ):
                print(
                    f"Skipping[{pgi.rank}]: m={m}, topk={topk}"
                    " (AITER block-scaled + weight-on-input,"
                    " https://github.com/ROCm/aiter/issues/2418)"
                )
                count -= 1
                continue

            # Skip unsupported: AITER x DeepEP-HT/Mori dispatch crashes
            # with an illegal memory access at world_size>1 on gfx942.
            # https://github.com/vllm-project/vllm/issues/57029
            if (
                config.world_size > 1
                and config.fused_experts_type is AiterExperts
                and getattr(config.prepare_finalize_type, "__name__", "")
                in ("DeepEPHTPrepareAndFinalize", "MoriPrepareAndFinalize")
            ):
                print(
                    f"Skipping[{pgi.rank}]: m={m}, topk={topk}"
                    " (AITER x DeepEP-HT/Mori illegal memory access,"
                    " https://github.com/vllm-project/vllm/issues/57029)"
                )
                count -= 1
                continue

            assert_aiter_quant_scheme_case(config)
            assert_aiter_activation_case(config)

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

            # On ROCm, AITER FP8 fused MoE uses hardware FP8
            # dot-product which can produce slightly larger error
            # than dequant+f32 matmul at FP8 representable-value
            # boundaries. Allow a small percentage of elements to
            # exceed the base tolerance by a bounded margin.
            # https://github.com/ROCm/aiter/issues/2421
            from vllm.platforms import current_platform as _cp

            is_aiter_fp8 = (
                _cp.is_rocm()
                and config.fused_experts_type is AiterExperts
                and config.quant_config is not None
            )
            if is_aiter_fp8:
                check_accuracy(ref_out, mk_out, atol=atol, rtol=rtol, percent=0.9)
            else:
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
MK_ACTIVATIONS = [
    MoEActivation.SILU,
    MoEActivation.GELU,
]


def is_nyi_config(config: Config) -> bool:
    # We know these configs to be legitimate. but still fail.
    info = expert_info(config.fused_experts_type)
    if info.needs_matching_quant:
        # The triton kernels expect both per-act-token-quant and
        # per-out-ch-quant or neither.
        unsupported_quant_config = (
            config.is_per_act_token_quant + config.is_per_out_ch_quant
        ) == 1
        if unsupported_quant_config:
            return True

    if config.activation != MoEActivation.SILU:
        if config.fused_experts_type is not AiterExperts:
            return True  # AITER-only for this axis, for now
        if config.quant_dtype is not None:
            return True  # unquantized-only for this axis, for now

    return False


def generate_valid_test_cases(
    world_size: int, prepare_finalize_types
) -> list[tuple[Any, ...]]:
    cases = []
    total = 0

    for k, n, e, dtype, quant_config, activation, combination in product(
        Ks,
        Ns,
        Es,
        DTYPEs,
        MK_QUANT_CONFIGS,
        MK_ACTIVATIONS,
        product(prepare_finalize_types, MK_FUSED_EXPERT_TYPES),
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
            activation=activation,
            prepare_finalize_type=combination[0],
            fused_experts_type=combination[1],
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
                activation,
                combination[0],
                combination[1],
                world_size,
            )
        )

    print(f"{len(cases)} of {total} valid configs generated.")

    return cases


@pytest.mark.parametrize(
    "k,n,e,dtype,quant_config,activation,"
    "prepare_finalize_type,fused_experts_type,world_size",
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
    activation: MoEActivation,
    prepare_finalize_type: mk.FusedMoEPrepareAndFinalize,
    fused_experts_type: mk.FusedMoEExperts,
    world_size: int,
    pytestconfig,
):
    if current_platform.device_count() < world_size:
        pytest.skip(
            f"Not enough GPUs available to run, got "
            f"{current_platform.device_count()} expected "
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
        activation=activation,
        prepare_finalize_type=prepare_finalize_type,
        fused_experts_type=fused_experts_type,
        world_size=world_size,
    )
    verbosity = pytestconfig.getoption("verbose")
    run(config, verbosity > 0)


@pytest.mark.parametrize(
    "k,n,e,dtype,quant_config,activation,"
    "prepare_finalize_type,fused_experts_type,world_size",
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
    activation: MoEActivation,
    prepare_finalize_type: mk.FusedMoEPrepareAndFinalize,
    fused_experts_type: mk.FusedMoEExperts,
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
        activation=activation,
        prepare_finalize_type=prepare_finalize_type,
        fused_experts_type=fused_experts_type,
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


# AITER sorting-backend dispatch env-var matrix (issue #54966 step 3) -------
#
# AITER_USE_CK_MOE_SORTING / AITER_USE_FLYDSL_MOE_SORTING are aiter globals
# read once at import time, so each combo needs a fresh child process (see
# parallel_launch_with_config) rather than in-process monkeypatching.
#
# AITER_MOE_SORT_BACKEND is not covered: it only matters when output_aux=True,
# which vLLM only sets on the MXFP4 path -- not yet wired into AiterExperts.


def _aiter_sort_backend_hooks_available() -> bool:
    """Whether the installed aiter build exposes the sort-backend hooks that
    AITER_USE_CK_MOE_SORTING / AITER_USE_FLYDSL_MOE_SORTING select between.
    """
    try:
        import aiter.fused_moe as aiter_fused_moe
    except ImportError:
        return False

    required = (
        "_moe_sorting_impl",
        "_flydsl_moe_sorting",
        "_USE_CK_MOE_SORTING",
        "_USE_FLYDSL_MOE_SORTING",
    )
    return all(hasattr(aiter_fused_moe, name) for name in required)


# Deliberately does NOT call _aiter_sort_backend_hooks_available() here: this
# marker is evaluated at module-import time, which also happens inside each
# spawned child (to unpickle its worker) -- before that case's env vars are
# applied, permanently freezing aiter's sort-backend globals. The hooks check
# instead runs inside the test body below (parent process only).
require_aiter_moe = pytest.mark.skipif(
    not (
        current_platform.is_rocm()
        and has_aiter()
        and rocm_aiter_ops.is_fused_moe_enabled()
    ),
    reason=(
        "AITER MoE sorting-backend dispatch needs ROCm + AITER, with "
        "VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 set before "
        "the test process starts."
    ),
)

# (AITER_USE_CK_MOE_SORTING, AITER_USE_FLYDSL_MOE_SORTING, expected_backend)
AITER_SORTING_BACKEND_ENV_CASES = [
    (1, 0, "ck"),
    (1, 1, "ck"),  # CK wins over FlyDSL even when both are requested.
    (0, 1, "flydsl"),
    (0, 0, "opus"),  # default when neither flag is set.
]


def _aiter_sorting_backend_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    config: Config,
    weights: WeightTensors,
    verbose: bool,
    ck: int,
    flydsl: int,
    expected_backend: str,
):
    # Imported lazily in the spawned child, after env vars are applied, so
    # aiter's import-time globals pick up this case's env.
    import aiter.fused_moe as aiter_fused_moe

    # Assert the env vars actually stuck in aiter's import-time globals,
    # rather than inferring it indirectly from which sorting function fires.
    assert bool(ck) == aiter_fused_moe._USE_CK_MOE_SORTING, (
        f"aiter.fused_moe._USE_CK_MOE_SORTING={aiter_fused_moe._USE_CK_MOE_SORTING} "
        f"but AITER_USE_CK_MOE_SORTING={ck} was set before this process started -- "
        "the env var never reached aiter's import-time globals."
    )
    assert bool(flydsl) == aiter_fused_moe._USE_FLYDSL_MOE_SORTING, (
        "aiter.fused_moe._USE_FLYDSL_MOE_SORTING="
        f"{aiter_fused_moe._USE_FLYDSL_MOE_SORTING} but AITER_USE_FLYDSL_MOE_SORTING="
        f"{flydsl} was set before this process started -- the env var never "
        "reached aiter's import-time globals."
    )

    # Captured before patching, since the patched attribute would resolve to
    # the mock's own (*args, **kwargs) signature instead of the real one.
    original_moe_sorting_impl = aiter_fused_moe._moe_sorting_impl

    with (
        mock.patch.object(
            aiter_fused_moe,
            "_moe_sorting_impl",
            wraps=original_moe_sorting_impl,
        ) as sorting_impl_mock,
        mock.patch.object(
            aiter_fused_moe,
            "_flydsl_moe_sorting",
            wraps=aiter_fused_moe._flydsl_moe_sorting,
        ) as flydsl_mock,
    ):
        # Reuses the real correctness/accuracy checking rank_worker already
        # does for every other AiterExperts combo, instead of duplicating it.
        rank_worker(pgi, vllm_config, cpu_group, config, weights, verbose)

        if expected_backend == "flydsl":
            assert flydsl_mock.call_count > 0, "Expected FlyDSL sorting to fire."
            assert sorting_impl_mock.call_count == 0, (
                "FlyDSL was requested and eligible, but the opus/CK sorting "
                "path fired instead -- a silent fallback."
            )
        else:
            assert flydsl_mock.call_count == 0, (
                f"Expected the '{expected_backend}' sorting path, but FlyDSL "
                "fired instead."
            )
            assert sorting_impl_mock.call_count > 0, (
                f"Expected the '{expected_backend}' sorting path to fire."
            )
            # Bind by signature so this works regardless of whether use_opus
            # is passed positionally or by keyword.
            call = sorting_impl_mock.call_args
            bound = inspect.signature(original_moe_sorting_impl).bind(
                *call.args, **call.kwargs
            )
            use_opus = bound.arguments["use_opus"]
            assert use_opus == (expected_backend == "opus"), (
                f"Expected use_opus={expected_backend == 'opus'} for the "
                f"'{expected_backend}' sorting path, got use_opus={use_opus}."
            )


@require_aiter_moe
@pytest.mark.parametrize("ck,flydsl,expected_backend", AITER_SORTING_BACKEND_ENV_CASES)
def test_aiter_moe_sorting_backend_dispatch_env_matrix(
    ck: int, flydsl: int, expected_backend: str
):
    """See https://github.com/vllm-project/vllm/issues/54966 ("Test AITER
    backend and dispatch settings")."""
    if not _aiter_sort_backend_hooks_available():
        pytest.skip(
            "Installed aiter build predates the sort-backend hooks "
            "(_moe_sorting_impl, _flydsl_moe_sorting, _USE_CK_MOE_SORTING, "
            "_USE_FLYDSL_MOE_SORTING)."
        )

    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    config = Config(
        Ms=[32],
        K=2048,
        N=1024,
        E=32,
        topks=[4],
        dtype=torch.bfloat16,
        quant_config=None,
        prepare_finalize_type=MoEPrepareAndFinalizeNoDPEPModular,
        fused_experts_type=AiterExperts,
        world_size=1,
    )
    assert config.is_valid()[0]

    weights = WeightTensors.make(config)
    vllm_config, env_dict = config.make_env_data()
    env_dict = {
        **env_dict,
        "AITER_USE_CK_MOE_SORTING": str(ck),
        "AITER_USE_FLYDSL_MOE_SORTING": str(flydsl),
    }

    parallel_launch_with_config(
        config.world_size,
        _aiter_sorting_backend_worker,
        vllm_config,
        env_dict,
        config,
        weights,
        False,
        ck,
        flydsl,
        expected_backend,
    )


def _aiter_dispatch_policy_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    config: Config,
    weights: WeightTensors,
    verbose: bool,
    dispatch_policy: int,
):
    # Drives the real AiterExperts.apply() -> rocm_aiter_fused_experts() call
    # through the modular kernel, so it also catches a regression in apply()'s
    # own forwarding (unlike test_rocm_aiter_moe.py's version, which supplies
    # the policy value directly to rocm_aiter_fused_experts()).
    with mock.patch.object(
        rocm_aiter_ops, "fused_moe", wraps=rocm_aiter_ops.fused_moe
    ) as fused_moe_mock:
        rank_worker(pgi, vllm_config, cpu_group, config, weights, verbose)

    assert fused_moe_mock.call_count > 0, (
        "Expected AiterExperts.apply() to call rocm_aiter_ops.fused_moe."
    )
    for call in fused_moe_mock.call_args_list:
        forwarded = call.kwargs["moe_sorting_dispatch_policy"]
        assert forwarded == dispatch_policy, (
            f"AiterExperts.apply() forwarded moe_sorting_dispatch_policy="
            f"{forwarded}, but VLLM_ROCM_AITER_MOE_DISPATCH_POLICY="
            f"{dispatch_policy} was set before this process started."
        )


@require_aiter_moe
@pytest.mark.parametrize("dispatch_policy", [0, 1, 2])
def test_aiter_moe_dispatch_policy_forwarded_through_apply(dispatch_policy: int):
    """See https://github.com/vllm-project/vllm/issues/54966 ("Test AITER
    backend and dispatch settings").

    Complements test_aiter_moe_dispatch_policy_forwarded_to_fused_moe in
    test_rocm_aiter_moe.py by driving the real AiterExperts.apply() call
    instead of supplying moe_sorting_dispatch_policy directly.
    """
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    config = Config(
        Ms=[32],
        K=2048,
        N=1024,
        E=32,
        topks=[4],
        dtype=torch.bfloat16,
        quant_config=None,
        prepare_finalize_type=MoEPrepareAndFinalizeNoDPEPModular,
        fused_experts_type=AiterExperts,
        world_size=1,
    )
    assert config.is_valid()[0]

    weights = WeightTensors.make(config)
    vllm_config, env_dict = config.make_env_data()
    env_dict = {
        **env_dict,
        "VLLM_ROCM_AITER_MOE_DISPATCH_POLICY": str(dispatch_policy),
    }

    parallel_launch_with_config(
        config.world_size,
        _aiter_dispatch_policy_worker,
        vllm_config,
        env_dict,
        config,
        weights,
        False,
        dispatch_policy,
    )


if __name__ == "__main__":
    # Ability to test individual PrepareAndFinalize and FusedExperts combination
    from .modular_kernel_tools.cli_args import make_config, make_config_arg_parser

    parser = make_config_arg_parser(
        description=(
            "Run single prepare-finalize & fused-experts combination test"
            "Example : python3 -m tests.kernels.moe.test_modular_kernel_combinations "
            "--pf-type DeepEPLLPrepareAndFinalize --experts-type BatchedTritonExperts"
        )
    )
    args = parser.parse_args()
    config = make_config(args)

    run(config, True)
