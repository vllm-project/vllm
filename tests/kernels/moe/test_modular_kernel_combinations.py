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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
)
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
from .utils import check_accuracy, make_test_weights

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
        config.world_size,
        rank_worker,
        vllm_config,
        env_dict,
        None,
        config,
        weights,
        verbose,
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


# --- 4b: hidden_dim_unpadded/intermediate_size_per_partition_unpadded matrix -

# K/N intentionally unaligned to AITER's 64/128 granularity (unlike the
# shared mk_objects.py defaults) so hidden_pad/intermediate_pad are forced.
_PADDING_E = 8
_PADDING_M = 16
_PADDING_TOPK = 2
_PADDING_K_UNPADDED = 896  # 7 * 128
_PADDING_K_PADDED = 1024  # +128
_PADDING_N_UNPADDED = 384  # 3 * 128
_PADDING_N_PADDED = 512  # +128
# hidden_pad = (K_padded - hidden_dim_unpadded) // 128 * 128 (rocm_aiter_moe.py)
_PADDING_HIDDEN_PAD_EXPECTED = 128
# intermediate_pad = (N_padded - intermediate_size_per_partition_unpadded)
#   // 64 * 64 * (2 if tp_size == 1 else 1) (rocm_aiter_moe.py); tp_size==1 here.
_PADDING_INTERMEDIATE_PAD_EXPECTED = 256

# {no padding, hidden-only, intermediate-only, both} -> (pad_hidden, pad_intermediate)
_PADDING_MODES: dict[str, tuple[bool, bool]] = {
    "none": (False, False),
    "hidden": (True, False),
    "intermediate": (False, True),
    "both": (True, True),
}

# unquantized + the 4 fp8 configs AiterExperts supports (excludes MXFP4,
# stubbed below pending MI350/gfx950, and MK_QUANT_CONFIGS[1], unsupported).
_PADDING_QUANT_CONFIGS = [
    MK_QUANT_CONFIGS[0],  # unquantized
    MK_QUANT_CONFIGS[2],  # fp8 channel weights / per-token activations
    MK_QUANT_CONFIGS[3],  # fp8 per-tensor weights / per-tensor activations
    MK_QUANT_CONFIGS[4],  # fp8 per-tensor weights / per-token activations
    MK_QUANT_CONFIGS[5],  # fp8 128x128-block weights / 128-block activations
]
_PADDING_QUANT_IDS = [
    "unquantized",
    "fp8_channel_token",
    "fp8_tensor_tensor",
    "fp8_tensor_token",
    "fp8_block_token",
]


def _slice_gate_up_rows(t: torch.Tensor, real_per_half: int) -> torch.Tensor:
    """Slice a (E, 2*padded_per_half, ...) tensor's dim=1 down to the real
    gate/up halves of `real_per_half` size each."""
    padded_per_half = t.shape[1] // 2
    return torch.cat(
        [
            t[:, :real_per_half],
            t[:, padded_per_half : padded_per_half + real_per_half],
        ],
        dim=1,
    )


def _slice_unpadded_weights(
    weights: WeightTensors,
    quant_config: TestMoEQuantConfig | None,
    k_unpadded: int,
    n_unpadded: int,
) -> WeightTensors:
    """Derive the unpadded sub-block of a padded WeightTensors: same values,
    restricted to the region hidden_pad/intermediate_pad keeps.

    Scale-shape choice (per_out_ch) must match _make_padding_matrix_weights()."""
    block_shape = quant_config.block_shape if quant_config is not None else None
    per_out_ch = quant_config is not None and quant_config.per_out_ch_quant

    w1 = _slice_gate_up_rows(weights.w1[:, :, :k_unpadded], n_unpadded)
    w2 = weights.w2[:, :k_unpadded, :n_unpadded]

    if weights.w1_scale is None:
        w1_scale = w2_scale = None
    elif weights.w2_scale is None:
        raise AssertionError("w1_scale and w2_scale must both be set or both None")
    elif block_shape is not None:
        block_n, block_k = block_shape
        assert n_unpadded % block_n == 0 and k_unpadded % block_k == 0
        n_scale, k_scale = n_unpadded // block_n, k_unpadded // block_k
        w1_scale = _slice_gate_up_rows(weights.w1_scale, n_scale)[..., :k_scale]
        w2_scale = weights.w2_scale[:, :k_scale, :n_scale]
    elif per_out_ch:
        w1_scale = _slice_gate_up_rows(weights.w1_scale, n_unpadded)
        w2_scale = weights.w2_scale[:, :k_unpadded, :]
    else:
        # Per-tensor scale: shape (E, 1, 1), independent of K/N -- reuse as-is.
        w1_scale = weights.w1_scale
        w2_scale = weights.w2_scale

    return WeightTensors(w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale)


def _make_padding_matrix_weights(config: Config) -> WeightTensors:
    """Like WeightTensors.make(), but scales weights by
    config.is_per_out_ch_quant instead of is_per_act_token_quant.

    WeightTensors.make() ties weight-scale shape to activation quant, so
    fp8_tensor_token would get per-channel scales despite being a per-tensor
    weight scheme. Scoped here rather than fixing WeightTensors.make(),
    which other tests depend on."""
    (_, w1, w1_scale, w1_gs), (_, w2, w2_scale, w2_gs) = make_test_weights(
        e=config.E,
        n=config.N,
        k=config.K,
        in_dtype=config.dtype,
        quant_dtype=config.quant_dtype,
        block_shape=config.quant_block_shape,
        per_out_ch_quant=config.is_per_out_ch_quant,
    )
    return WeightTensors(
        w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale, w1_gs=w1_gs, w2_gs=w2_gs
    )


def _slice_unpadded_rank_tensors(
    rank_tensors: RankTensors, k_unpadded: int
) -> RankTensors:
    return RankTensors(
        hidden_states=rank_tensors.hidden_states[:, :k_unpadded].contiguous(),
        hidden_states_scale=rank_tensors.hidden_states_scale,
        topk_weights=rank_tensors.topk_weights,
        topk_ids=rank_tensors.topk_ids,
        expert_map=rank_tensors.expert_map,
    )


def _aiter_padding_matrix_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    padded_config: Config,
    padded_weights: WeightTensors,
    verbose: bool,
    hidden_pad_expected: int,
    intermediate_pad_expected: int,
):
    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)
    set_random_seed(pgi.rank)

    weights = copy.deepcopy(padded_weights)
    weights.to_current_device()

    rank_tensors = RankTensors.make(padded_config, pgi)

    with mock.patch.object(
        rocm_aiter_ops, "fused_moe", wraps=rocm_aiter_ops.fused_moe
    ) as fused_moe_mock:
        mk_out = run_modular_kernel(
            pgi, vllm_config, padded_config, weights, rank_tensors
        )

    assert fused_moe_mock.call_count > 0, (
        "Expected AiterExperts.apply() to call rocm_aiter_ops.fused_moe."
    )
    for call in fused_moe_mock.call_args_list:
        assert call.kwargs["hidden_pad"] == hidden_pad_expected, (
            f"AiterExperts.apply() forwarded hidden_pad="
            f"{call.kwargs['hidden_pad']}, expected {hidden_pad_expected}."
        )
        assert call.kwargs["intermediate_pad"] == intermediate_pad_expected, (
            f"AiterExperts.apply() forwarded intermediate_pad="
            f"{call.kwargs['intermediate_pad']}, expected "
            f"{intermediate_pad_expected}."
        )

    # AiterExperts.apply() never slices its own output -- the buffer stays
    # raw (padded) width; only the caller-sliced unpadded prefix is checked.
    assert mk_out.shape[-1] == padded_config.K, (
        f"AiterExperts output width {mk_out.shape[-1]} != raw hidden_dim "
        f"{padded_config.K}."
    )
    mk_out = mk_out[..., :_PADDING_K_UNPADDED]
    # Only the real slice is checked: beyond it is unspecified/reused memory.
    assert torch.isfinite(mk_out).all(), (
        "AiterExperts output must not contain NaN/Inf in the real "
        "hidden_dim_unpadded-wide output slice."
    )

    unpadded_weights = _slice_unpadded_weights(
        weights,
        padded_config.quant_config,
        _PADDING_K_UNPADDED,
        _PADDING_N_UNPADDED,
    )
    unpadded_rank_tensors = _slice_unpadded_rank_tensors(
        rank_tensors, _PADDING_K_UNPADDED
    )

    with set_current_vllm_config(vllm_config):
        ref_out = reference_moe_impl(
            padded_config, unpadded_weights, unpadded_rank_tensors
        )

    # ref_out's magnitude here (~1e-3-1e-2) is below atol=3e-2, so a zeroed-out
    # mk_out would still pass check_accuracy below -- guard against that.
    ref_scale = ref_out.abs().mean()
    mk_scale = mk_out.abs().mean()
    assert mk_scale > 0.5 * ref_scale, (
        f"AiterExperts output magnitude (mean |mk_out|={mk_scale:.6f}) looks "
        f"degenerate/zeroed vs. reference (mean |ref_out|={ref_scale:.6f})."
    )

    # Lenient check for every scheme, including unquantized: a strict
    # assert_close(atol=rtol=3e-2) spuriously fails ~0.1-0.2% of elements
    # on MI300 even for unquantized AiterExperts (confirmed on hardware).
    check_accuracy(ref_out, mk_out, atol=3e-2, rtol=3e-2, percent=0.9)


@require_aiter_moe
@pytest.mark.parametrize("quant_config", _PADDING_QUANT_CONFIGS, ids=_PADDING_QUANT_IDS)
@pytest.mark.parametrize("mode", list(_PADDING_MODES))
def test_aiter_moe_padding_matrix(mode: str, quant_config: TestMoEQuantConfig | None):
    """See https://github.com/vllm-project/vllm/issues/54966 ("Test padding").

    Exercises AiterExperts's hidden_pad/intermediate_pad (rocm_aiter_moe.py)
    across {no padding, hidden-only, intermediate-only, both} x the
    unquantized + 4 fp8 quant schemes it supports. Compares against an
    unpadded reference and verifies the hidden_pad/intermediate_pad
    AiterExperts.apply() forwards to rocm_aiter_ops.fused_moe.
    """
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    pad_hidden, pad_intermediate = _PADDING_MODES[mode]

    if (
        quant_config is not None
        and quant_config.block_shape is not None
        and not pad_hidden
    ):
        pytest.skip(
            "AITER's block-quantized (128x128) CK GEMM kernel does not "
            f"support the raw (unpadded) hidden_dim={_PADDING_K_UNPADDED} "
            "shape on this hardware -- it raises 'wrong! device_gemm with "
            "the specified compilation parameters does not support this "
            "GEMM problem'. Only hidden_pad-forcing modes (raw "
            f"hidden_dim={_PADDING_K_PADDED}) are exercised for this quant "
            "scheme."
        )

    k = _PADDING_K_PADDED if pad_hidden else _PADDING_K_UNPADDED
    n = _PADDING_N_PADDED if pad_intermediate else _PADDING_N_UNPADDED

    config = Config(
        Ms=_PADDING_M,
        K=k,
        N=n,
        E=_PADDING_E,
        topks=_PADDING_TOPK,
        dtype=torch.bfloat16,
        quant_config=quant_config,
        prepare_finalize_type=MoEPrepareAndFinalizeNoDPEPModular,
        fused_experts_type=AiterExperts,
        world_size=1,
        hidden_dim_unpadded=_PADDING_K_UNPADDED if pad_hidden else None,
        intermediate_size_per_partition_unpadded=(
            _PADDING_N_UNPADDED if pad_intermediate else None
        ),
    )
    assert config.is_valid()[0]
    assert config.fe_supports_quant_scheme(), (
        f"AiterExperts does not support quant scheme {quant_config}."
    )

    weights = _make_padding_matrix_weights(config)
    vllm_config, env_dict = config.make_env_data()

    hidden_pad_expected = _PADDING_HIDDEN_PAD_EXPECTED if pad_hidden else 0
    intermediate_pad_expected = (
        _PADDING_INTERMEDIATE_PAD_EXPECTED if pad_intermediate else 0
    )

    parallel_launch_with_config(
        config.world_size,
        _aiter_padding_matrix_worker,
        vllm_config,
        env_dict,
        config,
        weights,
        False,
        hidden_pad_expected,
        intermediate_pad_expected,
    )


@pytest.mark.skip(
    reason="MXFP4 AiterExperts padding requires MI350/gfx950 hardware. "
    'See https://github.com/vllm-project/vllm/issues/54966 ("Test padding").'
)
def test_aiter_moe_padding_matrix_mxfp4():
    """Stub for the {no padding, hidden-only, intermediate-only, both} MXFP4
    padding matrix once MI350/gfx950 hardware is available in CI."""


# --- 4c: HIP-graph token padding -- inf/nan garbage rows (topk_ids=-1) ------

# AiterExperts isolates garbage rows from real rows under both AITER sorting
# backends, and reference_moe_impl's dispatch (`mask = topk_ids == i`) is
# likewise row-local, so comparing real rows against it is valid. Exception:
# AiterExperts's a1 quant scale is normally fixed before garbage injection
# (computed in RankTensors.make()), but "fp8_tensor_token" computes it live
# from the (already-injected) hidden_states -- see the skip below.

# {mode name -> {row index: fill value}}. Covers multiple garbage-row counts
# and positions, and both inf and nan.
_TOKEN_PADDING_GARBAGE_MODES: dict[str, dict[int, float]] = {
    "single_inf": {3: float("inf")},
    "single_nan": {5: float("nan")},
    "multiple_mixed": {
        1: float("inf"),
        4: float("-inf"),
        7: float("nan"),
        12: float("inf"),
    },
}


def _apply_token_padding_garbage_rows(
    rank_tensors: RankTensors, garbage_rows: dict[int, float]
) -> None:
    """Simulate the HIP-graph padding-token contract in-place: garbage rows
    get inf/nan hidden_states and their routing masked to topk_ids=-1 /
    topk_weights=0, mirroring grouped_topk_router.py's `is_padding` masking.
    """
    for row, fill in garbage_rows.items():
        rank_tensors.hidden_states[row].fill_(fill)
        rank_tensors.topk_ids[row].fill_(-1)
        rank_tensors.topk_weights[row].fill_(0.0)


def _aiter_token_padding_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    config: Config,
    weights: WeightTensors,
    verbose: bool,
    garbage_rows: dict[int, float],
):
    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)
    set_random_seed(pgi.rank)

    weights = copy.deepcopy(weights)
    weights.to_current_device()

    rank_tensors = RankTensors.make(config, pgi)
    _apply_token_padding_garbage_rows(rank_tensors, garbage_rows)

    with set_current_vllm_config(vllm_config):
        mk_out = run_modular_kernel(pgi, vllm_config, config, weights, rank_tensors)
        ref_out = reference_moe_impl(config, weights, rank_tensors)

    real_rows = [i for i in range(config.Ms) if i not in garbage_rows]

    assert torch.isfinite(mk_out[real_rows]).all(), (
        "Garbage (padding) token rows leaked NaN/Inf into real rows' output "
        "through AiterExperts."
    )

    # ref_out's magnitude here (~1e-3-1e-2) is below atol=3e-2, so a zeroed-out
    # mk_out would still pass the checks below -- guard against that.
    ref_scale = ref_out[real_rows].abs().mean()
    mk_scale = mk_out[real_rows].abs().mean()
    assert mk_scale > 0.5 * ref_scale, (
        f"AiterExperts output magnitude (mean |mk_out|={mk_scale:.6f}) looks "
        f"degenerate/zeroed vs. reference (mean |ref_out|={ref_scale:.6f})."
    )

    if config.quant_config is not None:
        check_accuracy(
            ref_out[real_rows], mk_out[real_rows], atol=3e-2, rtol=3e-2, percent=0.9
        )
    else:
        torch.testing.assert_close(
            ref_out[real_rows], mk_out[real_rows], atol=3e-2, rtol=3e-2
        )


@require_aiter_moe
@pytest.mark.parametrize("quant_config", _PADDING_QUANT_CONFIGS, ids=_PADDING_QUANT_IDS)
@pytest.mark.parametrize("garbage_mode", list(_TOKEN_PADDING_GARBAGE_MODES))
def test_aiter_moe_token_padding_garbage_rows(
    garbage_mode: str, quant_config: TestMoEQuantConfig | None
):
    """See https://github.com/vllm-project/vllm/issues/54966 ("Test padding").

    Simulates HIP-graph token padding: a subset of rows get inf/nan
    hidden_states with topk_ids=-1 / topk_weights=0, mirroring the router's
    padding-token masking contract (grouped_topk_router.py). Verifies the
    real (non-garbage) rows' output through AiterExperts remains finite and
    matches reference_moe_impl, across multiple garbage-row counts/positions
    and both inf and nan fill values, for unquantized + the 4 fp8 quant
    schemes AiterExperts supports.
    """
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    garbage_rows = _TOKEN_PADDING_GARBAGE_MODES[garbage_mode]

    # Reuses 4b's already-confirmed 128-aligned ("padded") K/N -- these are
    # known to work across all 5 quant schemes here, including AITER's
    # block-quant kernel (which 4b's _PADDING_K_UNPADDED/_PADDING_N_UNPADDED
    # sizes do not support). Token-padding safety doesn't depend on
    # hidden_pad/intermediate_pad, so there's no need for a separate shape.
    config = Config(
        Ms=_PADDING_M,
        K=_PADDING_K_PADDED,
        N=_PADDING_N_PADDED,
        E=_PADDING_E,
        topks=_PADDING_TOPK,
        dtype=torch.bfloat16,
        quant_config=quant_config,
        prepare_finalize_type=MoEPrepareAndFinalizeNoDPEPModular,
        fused_experts_type=AiterExperts,
        world_size=1,
    )
    assert config.is_valid()[0]
    assert config.fe_supports_quant_scheme(), (
        f"AiterExperts does not support quant scheme {quant_config}."
    )

    # This (weight, activation) pair is AiterExperts's real per-tensor
    # dynamic-activation-quant scheme -- its batch-wide max-abs scale is
    # poisoned by a real inf garbage row. Upstream bug:
    # https://github.com/ROCm/aiter/issues/6275.
    if (
        quant_config is not None
        and config.fp8_quant_key_pair() == (kFp8StaticTensorSym, kFp8DynamicTensorSym)
        and any(v in (float("inf"), float("-inf")) for v in garbage_rows.values())
    ):
        pytest.skip(
            "Batch-wide dynamic per-tensor quant scale is poisoned by a "
            "real inf garbage row. See "
            "https://github.com/ROCm/aiter/issues/6275."
        )

    weights = WeightTensors.make(config)
    vllm_config, env_dict = config.make_env_data()

    parallel_launch_with_config(
        config.world_size,
        _aiter_token_padding_worker,
        vllm_config,
        env_dict,
        config,
        weights,
        False,
        garbage_rows,
    )


@pytest.mark.skip(
    reason="MXFP4 AiterExperts token padding requires MI350/gfx950 hardware. "
    'See https://github.com/vllm-project/vllm/issues/54966 ("Test padding").'
)
def test_aiter_moe_token_padding_garbage_rows_mxfp4():
    pass


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
