# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.model_executor.layers.linear_attention.mamba2.decode.base import (
    MambaSSUKernel,
)
from vllm.model_executor.layers.linear_attention.mamba2.decode.cpu import CPUSSUKernel
from vllm.model_executor.layers.linear_attention.mamba2.decode.flashinfer import (
    FlashInferSSUKernel,
)
from vllm.model_executor.layers.linear_attention.mamba2.decode.replayssm_base import (
    MambaReplaySSMLayerState,
)
from vllm.model_executor.layers.linear_attention.mamba2.decode.replayssm_flashinfer import (  # noqa: E501
    FlashInferReplaySSMKernel,
)
from vllm.model_executor.layers.linear_attention.mamba2.decode.replayssm_triton import (
    TritonReplaySSMKernel,
)
from vllm.model_executor.layers.linear_attention.mamba2.decode.replayssm_triton_spec import (  # noqa: E501
    TritonReplaySSMSpecKernel,
)
from vllm.model_executor.layers.linear_attention.mamba2.decode.selector import (
    SSU_KERNELS,
)
from vllm.model_executor.layers.linear_attention.mamba2.decode.triton import (
    TritonSSUKernel,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.ops import ssu_dispatch
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    KVCacheGroup,
    KVCacheLayout,
    MambaSpec,
)
from vllm.v1.worker.replayssm_utils import (
    ReplaySSMModelContext,
    _compact_replayssm_requests_kernel,
    _postprocess_replayssm_kernel,
)
from vllm.v1.worker.utils import allocate_kv_cache

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

DEVICE = current_platform.device_type
requires_gpu = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="requires CUDA or XPU",
)


def _device_ints(values):
    return torch.tensor(values, dtype=torch.int32, device=DEVICE)


def _replayssm_mixer(kv_cache, ring_start, committed, A):
    return SimpleNamespace(
        kv_cache=kv_cache,
        ssu_state=MambaReplaySSMLayerState(
            buffer_len=16,
            num_spec=0,
            ring_start=ring_start,
            prev_num_accepted=committed,
        ),
        A=A,
        mamba_config=MambaConfig(backend=MambaBackendEnum.FLASHINFER),
        ssu_kernel=FlashInferReplaySSMKernel,
    )


@pytest.mark.parametrize("checkpointed", [False, True], ids=["replay", "checkpoint"])
@pytest.mark.parametrize(
    ("post_step", "layout", "ssm_dtype"),
    [
        pytest.param(False, KVCacheLayout.LBNHC, torch.float16, id="v1-fp16-lbnhc"),
        pytest.param(True, KVCacheLayout.BLHNC, torch.float32, id="v2-fp32-blhnc"),
    ],
)
def test_replayssm_materializes_only_accepted_boundary_with_compacted_rows(
    post_step, layout, ssm_dtype, checkpointed
):
    """Publish a boundary without modifying its live or cached-prefix sources."""
    try:
        from flashinfer.mamba.replayssm_materialize import (
            replayssm_materialize,  # noqa: F401
        )

        has_materializer = True
    except ImportError:
        has_materializer = False
    if not (torch.cuda.is_available() and has_materializer):
        reason = "requires CUDA and FlashInfer ReplaySSM materialization"
        if os.environ.get("VLLM_TEST_REQUIRE_REPLAYSSM") == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    set_random_seed(42)
    slots, heads, dim, dstate, ring_len = 19, 8, 64, 128, 20
    spec = MambaSpec(
        block_size=16,
        shapes=(
            (heads, dim),
            (heads, dim, dstate),
            (heads, ring_len, dim),
            (heads, ring_len),
            (1, ring_len, dstate),
        ),
        dtypes=(
            torch.bfloat16,
            ssm_dtype,
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
        ),
        state_alignment=16,
        block_stride_alignment=16,
    )
    spec = replace(spec, page_size_padded=spec.page_size_bytes + 128)
    vllm_config = SimpleNamespace(
        cache_config=CacheConfig(),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    vllm_config.cache_config.kv_cache_layout = layout.name
    layer_names = [
        ["mixer", "neighbor"],
        ["other_group", "other_neighbor"],
        ["last_group"],
    ]
    config = get_kv_cache_config_from_groups(
        vllm_config,
        [
            KVCacheGroup.from_layers(
                {
                    name: MambaManager.layer_config(spec, enable_prefix_caching=True)
                    for name in names
                }
            )
            for names in layer_names
        ],
        num_blocks=slots,
    )
    pages = allocate_kv_cache(config, torch.device("cuda"), layout)
    caches = {}
    for name, page in pages.items():
        layer = SimpleNamespace(
            get_state_shape=lambda: spec.shapes,
            get_state_dtype=lambda: spec.dtypes,
            state_alignment=16,
        )
        MambaBase.bind_kv_cache(layer, page)
        caches[name] = layer.kv_cache
    rings = {name: states[2:] for name, states in caches.items()}
    for name, states in caches.items():
        states[1].normal_(std=0.1)
        x, dt, B = rings[name]
        x.normal_()
        dt.fill_(0.01)
        B.normal_()
        assert (
            not x.is_contiguous() and not dt.is_contiguous() and not B.is_contiguous()
        )
    assert rings["mixer"][0].data_ptr() == rings["other_group"][0].data_ptr()
    assert rings["mixer"][0].data_ptr() == rings["last_group"][0].data_ptr()

    # Groups share backing pools, but own disjoint physical slots.
    # Each group's cached prefix remains immutable after copying to its live slot.
    for names, prefix, live in zip(layer_names, [1, 7, 13], [3, 9, 15], strict=True):
        for name in names:
            caches[name][1][live].copy_(caches[name][1][prefix])
    original = {name: states[1].clone() for name, states in caches.items()}
    ring_before = {
        name: [tensor.clone() for tensor in tensors] for name, tensors in rings.items()
    }
    A = -torch.ones(heads, device="cuda")
    null = NULL_BLOCK_ID
    block_tables = [
        [[null, 3], [null, 5], [4, 4], [null, null], [null, null]],
        [[8, 9], [null, 11], [10, 10], [null, null], [null, null]],
        [[14, 15], [null, 17], [16, 16], [null, null], [null, null]],
    ]
    histories = [
        {
            null: (11, 9),
            offset + 3: (7, 14 if checkpointed else 2),
            offset + 5: (2, 1),
            offset + 4: (6, 2),
        }
        for offset in (0, 6, 12)
    ]
    grouped = []
    for names, table, history in zip(layer_names, block_tables, histories, strict=True):
        starts, committed = [0] * slots, [0] * slots
        for slot, (start, count) in history.items():
            starts[slot], committed[slot] = start, count
        starts, committed = _device_ints(starts), _device_ints(committed)
        mixers = [
            _replayssm_mixer(caches[name], starts, committed, A) for name in names
        ]
        grouped.append((mixers, spec, _device_ints(table)))
    context = ReplaySSMModelContext.create(
        grouped, max_num_reqs=6, materialize_prefixes=True
    )
    # Row 0 crosses token 16, but only later groups have a destination.
    # Row 1 rejects all drafts and stops at 15.
    # Row 2 uses in-place replay; row 3 has null slots in every group.
    # Row 4 is a padded request, and row 5 is unused buffer capacity.
    mapping = _device_ints([2, 0, 1, 3, -1])
    # Groups use different slots but preserve equal logical replay histories.
    context.reset_new_slots(
        idx_mapping=mapping,
        src_cols=_device_ints([0, 1, 1, 1]),
        dst_cols=_device_ints([1, 0, 0, 0]),
        num_reqs=5,
    )
    reset_histories = [
        {**histories[0], 5: (0, 0)},
        {**histories[1], 8: histories[1][9], 11: (0, 0)},
        {**histories[2], 14: histories[2][15], 17: (0, 0)},
    ]
    for (mixers, _, _), expected in zip(grouped, reset_histories, strict=True):
        assert mixers[0].ssu_state.ring_start.tolist() == [
            expected.get(slot, (0, 0))[0] for slot in range(slots)
        ]
        assert mixers[0].ssu_state.prev_num_accepted.tolist() == [
            expected.get(slot, (0, 0))[1] for slot in range(slots)
        ]
    metadata = dict(
        idx_mapping=mapping,
        query_metadata=_device_ints([0, 4, 8, 12, 16, 20])
        if post_step
        else _device_ints([4] * 5),
        query_metadata_is_cumulative=post_step,
        num_computed_tokens=_device_ints([15, 17, 17, 17])
        if post_step
        else _device_ints([14] * 4),
        num_computed_is_post_step=post_step,
        num_accepted_tokens=_device_ints([1, 3, 3, 3]),
        is_prefilling=torch.zeros(5, dtype=torch.bool, device="cuda"),
        live_cols=_device_ints([1] * 4),
        num_reqs=5,
    )
    context.postprocess(**metadata)
    # After a checkpoint, the source state is canonical at the old ring end.
    # Only the two accepted tokens through the boundary remain to replay.
    plans = {0: (1, 2) if checkpointed else (7, 4), 2: (6, 4)}
    assert context.active_request_indices.tolist() == [0, 2, -1, -1, -1, -1]
    assert context.plan_ring_start.tolist() == [
        plans.get(row, (0, -1))[0] for row in range(6)
    ]
    assert context.plan_flush_count.tolist() == [
        plans.get(row, (0, -1))[1] for row in range(6)
    ]
    destinations: dict[int, set[int]] = {}
    for names, (mixers, _, _), table in zip(
        layer_names, grouped, block_tables, strict=True
    ):
        starts = mixers[0].ssu_state.ring_start
        committed = mixers[0].ssu_state.prev_num_accepted
        for row, (start, flush) in plans.items():
            dst, src = table[row]
            if dst == null:
                assert starts[src].item() == start
                assert committed[src].item() == flush + 1
                assert (starts[null].item(), committed[null].item()) == (11, 9)
                continue
            for name in names:
                destinations.setdefault(caches[name][1].data_ptr(), set()).add(dst)
                x, dt, B = rings[name]
                expected = original[name][src].float().clone()
                for offset in range(flush):
                    pos = (start + offset) % ring_len
                    delta = dt[src, :, pos]
                    expected *= torch.exp(delta * A)[:, None, None]
                    expected += (
                        delta[:, None, None]
                        * x[src, :, pos].float()[:, :, None]
                        * B[src, 0, pos].float()[None, None, :]
                    )
                torch.testing.assert_close(
                    caches[name][1][dst].float(), expected, atol=2e-3, rtol=2e-3
                )
            assert starts[dst].item() == 0
            assert committed[dst].item() == 0
            if src != dst:
                assert starts[src].item() == start
                assert committed[src].item() == flush + 1
    # Check all other slots, including immutable prefixes and each live tail.
    for name, states in caches.items():
        for slot in set(range(slots)) - destinations[states[1].data_ptr()]:
            torch.testing.assert_close(
                states[1][slot], original[name][slot], atol=0, rtol=0
            )
        for tensor, before in zip(rings[name], ring_before[name], strict=True):
            torch.testing.assert_close(tensor, before, atol=0, rtol=0)

    # Old plans beyond the new batch must not replay a second time.
    before_shrink = {name: states[1].clone() for name, states in caches.items()}
    mapping[:2].fill_(-1)
    metadata["num_reqs"] = 2
    context.postprocess(**metadata)
    assert context.active_request_indices.tolist() == [-1] * 6
    for name, states in caches.items():
        torch.testing.assert_close(states[1], before_shrink[name], atol=0, rtol=0)


@requires_gpu
@pytest.mark.parametrize("post_step", [False, True], ids=["v1", "v2"])
def test_replayssm_skips_canonical_self_copy_but_resets_prefill_trackers(post_step):
    """Skip prefill self-copies, reset trackers, and preserve in-place MTP replay."""
    mixer = _replayssm_mixer(
        kv_cache=(
            torch.empty(0, device=DEVICE),
            torch.empty(6, 1, 1, 1, device=DEVICE),
            torch.empty(6, 1, 20, 1, device=DEVICE),
            torch.empty(6, 1, 20, device=DEVICE),
            torch.empty(6, 1, 20, 1, device=DEVICE),
        ),
        ring_start=_device_ints([3] * 6),
        committed=_device_ints([2] * 6),
        A=torch.empty(1, device=DEVICE),
    )
    spec = MambaSpec(
        block_size=16,
        shapes=tuple(tensor.shape[1:] for tensor in mixer.kv_cache),
        dtypes=tuple(tensor.dtype for tensor in mixer.kv_cache),
    )
    table = _device_ints(
        [
            [0, 1, 2],
            [3, 4, 5],
            [NULL_BLOCK_ID] * 3,
            [NULL_BLOCK_ID] * 3,
        ]
    )
    context = ReplaySSMModelContext.create(
        [([mixer], spec, table)], max_num_reqs=4, materialize_prefixes=True
    )
    # Reordered rows: canonical prefill in slot 2; in-place MTP replay in slot 4;
    # padded slot; padded request.
    _postprocess_replayssm_kernel[(4, 1)](
        _device_ints([1, 2, 0, -1]),
        _device_ints([0, 48, 52, 68, 68])
        if post_step
        else _device_ints([48, 4, 16, 0]),
        _device_ints([16, 48, 32]) if post_step else _device_ints([0, 0, 28]),
        _device_ints([1, 1, 4]),
        torch.tensor([True, False, True, True], device=DEVICE),
        _device_ints([0, 2, 1]),
        context.block_table_ptrs,
        context.ring_start_ptrs,
        context.num_committed_ptrs,
        context.src_slots_ptrs,
        context.dst_slots_ptrs,
        context.plan_ring_start,
        context.plan_flush_count,
        context.block_table_strides,
        context.num_layers,
        MAX_NUM_REQS=4,
        MAMBA_BLOCK_SIZE=16,
        LOGICAL_WINDOW=16,
        RING_BUFFER_LEN=20,
        PAD_SLOT_ID=NULL_BLOCK_ID,
        QUERY_METADATA_IS_CUMULATIVE=post_step,
        NUM_COMPUTED_IS_POST_STEP=post_step,
        HAS_IDX_MAPPING=True,
        MATERIALIZE_PREFIXES=True,
        LIVE_COL_IS_ZERO=False,
    )
    _compact_replayssm_requests_kernel[(1,)](
        context.plan_flush_count,
        context.active_request_indices,
        4,
        MAX_NUM_REQS=4,
        BLOCK_SIZE=4,
    )

    assert context.plan_flush_count.tolist() == [-1, 6, -1, -1]
    assert context.active_request_indices.tolist() == [1, -1, -1, -1]
    assert context.src_slots[0, 1].item() == 4
    assert context.dst_slots[0, 1].item() == 4
    assert mixer.ssu_state.ring_start.tolist() == [3, 0, 0, 3, 0, 3]
    assert mixer.ssu_state.prev_num_accepted.tolist() == [2, 0, 0, 2, 0, 2]


SM90 = (PlatformEnum.CUDA, DeviceCapability(9, 0))
SM100 = (PlatformEnum.CUDA, DeviceCapability(10, 0))


def _cfg(
    platform,
    capability,
    state_size=128,
    state_dtype=torch.float32,
    stochastic_rounding=False,
    ssu_algorithm=None,
    use_replayssm=False,
    num_spec=0,
    buffer_len=16,
    prefix_caching=False,
    model_dtype=torch.bfloat16,
    pp_size=1,
):
    """A Nemotron-like Mamba2 layer."""
    spec = MambaSpec(
        block_size=256,
        shapes=((3, 1024), (8, 64, state_size)),
        dtypes=(torch.bfloat16, state_dtype),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
    )
    return LinearAttnSelectorConfig(
        platform,
        capability,
        spec,
        model_dtype,
        use_stochastic_rounding=stochastic_rounding,
        use_ssu_algorithm=ssu_algorithm is not None,
        use_replayssm=use_replayssm,
        replayssm_buffer_len=buffer_len,
        num_speculative_tokens=num_spec,
        use_pipeline_parallel=pp_size > 1,
        use_prefix_caching=prefix_caching,
    )


HOPPER, BLACKWELL = _cfg(*SM90), _cfg(*SM100)
CPU = _cfg(PlatformEnum.CPU, None)
FP16 = dict(state_dtype=torch.float16)


# Decode metadata for a state tensor indexed by batch row, without speculative
# decoding.
_NO_INDICES = SimpleNamespace(
    state_indices_tensor_d=None, num_accepted_tokens=None, query_start_loc_d=None
)


def _select(cfg, requested=None):
    return select_linear_attn_kernel("Mamba SSU", SSU_KERNELS, cfg, requested)


@pytest.mark.parametrize(
    ("backend", "cfg", "reason"),
    [
        (MambaBackendEnum.TRITON, CPU, "CPU"),
        (
            MambaBackendEnum.TRITON,
            _cfg(*SM90, stochastic_rounding=True, **FP16),
            "capability 10",
        ),
        (MambaBackendEnum.TRITON, _cfg(*SM100, stochastic_rounding=True), "float16"),
        (MambaBackendEnum.TRITON, _cfg(*SM100, ssu_algorithm="vertical"), "algorithm"),
        (MambaBackendEnum.CPU, BLACKWELL, "platform CUDA"),
        (MambaBackendEnum.FLASHINFER, CPU, "platform CPU"),
        (MambaBackendEnum.FLASHINFER, _cfg(*SM100, state_size=16), "state size"),
        (
            MambaBackendEnum.FLASHINFER,
            _cfg(*SM100, stochastic_rounding=True),
            "float16",
        ),
        (
            MambaBackendEnum.CPU,
            _cfg(PlatformEnum.CPU, None, ssu_algorithm="vertical"),
            "algorithm",
        ),
        (
            MambaBackendEnum.CPU,
            _cfg(PlatformEnum.CPU, None, stochastic_rounding=True, **FP16),
            "stochastic rounding",
        ),
    ],
)
def test_decode_backend_that_cannot_run_fails_at_selection(backend, cfg, reason):
    """A requested backend fails at startup with its reason, not at the first
    decode."""
    with pytest.raises(ValueError, match=reason):
        _select(cfg, backend)


def test_kernel_rejects_features_it_does_not_declare():
    """New features are off by default: a kernel must opt in to each."""

    class BareKernel(MambaSSUKernel):
        backend = MambaBackendEnum.TRITON

        @classmethod
        def supports_platform(cls, platform):
            return True

    cfg = _cfg(*SM100, ssu_algorithm="vertical", stochastic_rounding=True)
    assert BareKernel.validate_configuration(cfg) == [
        "stochastic rounding not supported",
        "--mamba-ssu-algorithm not supported",
    ]


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
@pytest.mark.parametrize("cfg", [HOPPER, BLACKWELL])
def test_flashinfer_decodes_mamba2_on_hopper_and_blackwell(cfg):
    """Unlike FlashInfer's SSD scan, its decode kernel isn't SM10x-only."""
    assert _select(cfg, MambaBackendEnum.FLASHINFER) is FlashInferSSUKernel


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [(BLACKWELL, TritonSSUKernel), (CPU, CPUSSUKernel)],
)
def test_unset_backend_selects_the_platform_default(cfg, expected):
    """CPU decode comes from selection, not a silent Triton-to-CPU swap."""
    assert _select(cfg) is expected


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
@pytest.mark.parametrize(
    "cfg",
    [
        _cfg(*SM90, stochastic_rounding=True, **FP16),
        _cfg(*SM100, ssu_algorithm="vertical"),
    ],
    ids=["stochastic-rounding-off-sm10x", "ssu-algorithm"],
)
def test_unset_backend_selects_the_kernel_that_supports_the_options(cfg):
    """Automatic selection skips kernels that can't do what's asked (Triton off
    SM10x can't round stochastically; only FlashInfer takes an algorithm)."""
    assert _select(cfg) is FlashInferSSUKernel


needs_flashinfer = pytest.mark.skipif(
    not HAS_FLASHINFER, reason="flashinfer not installed"
)


@pytest.mark.parametrize(
    ("backend", "num_spec", "expected"),
    [
        (None, 0, TritonReplaySSMKernel),
        (MambaBackendEnum.TRITON, 0, TritonReplaySSMKernel),
        (None, 3, TritonReplaySSMSpecKernel),
        (MambaBackendEnum.TRITON, 3, TritonReplaySSMSpecKernel),
        pytest.param(
            MambaBackendEnum.FLASHINFER,
            0,
            FlashInferReplaySSMKernel,
            marks=needs_flashinfer,
        ),
        pytest.param(
            MambaBackendEnum.FLASHINFER,
            3,
            FlashInferReplaySSMKernel,
            marks=needs_flashinfer,
        ),
    ],
)
def test_replayssm_kernel_follows_backend_and_speculation(backend, num_spec, expected):
    """With --use-replayssm, the backend picks the provider (unset is Triton)
    and speculative decoding picks between the two Triton kernels."""
    assert (
        _select(_cfg(*SM100, num_spec=num_spec, use_replayssm=True), backend)
        is expected
    )


def test_plain_and_replayssm_kernels_reject_each_others_caches():
    """Plain SSU would update the state without filling ReplaySSM's rings, and
    ReplaySSM kernels need them."""
    replay = _cfg(*SM100, use_replayssm=True)
    for kernel in SSU_KERNELS:
        cfg = BLACKWELL if kernel.is_replayssm() else replay
        reasons = kernel.validate_configuration(cfg)
        assert any("--use-replayssm" in reason for reason in reasons), kernel


@pytest.mark.parametrize(
    ("backend", "cfg", "reason"),
    [
        (
            MambaBackendEnum.FLASHINFER,
            _cfg(*SM100, buffer_len=17, use_replayssm=True),
            "<= 16",
        ),
        (
            MambaBackendEnum.FLASHINFER,
            _cfg(PlatformEnum.CPU, None, use_replayssm=True),
            "platform CPU",
        ),
        (
            MambaBackendEnum.FLASHINFER,
            _cfg(
                *SM100,
                prefix_caching=True,
                model_dtype=torch.float16,
                use_replayssm=True,
            ),
            "bfloat16 model inputs",
        ),
        (
            MambaBackendEnum.FLASHINFER,
            _cfg(*SM100, pp_size=2, use_replayssm=True),
            "pipeline parallelism",
        ),
        (
            MambaBackendEnum.TRITON,
            _cfg(*SM100, num_spec=3, buffer_len=3, use_replayssm=True),
            r"--replayssm-buffer-len >= 1 \+ num_speculative_tokens",
        ),
        (
            MambaBackendEnum.TRITON,
            _cfg(
                *SM100, num_spec=3, stochastic_rounding=True, **FP16, use_replayssm=True
            ),
            "stochastic rounding not supported",
        ),
        (
            MambaBackendEnum.TRITON,
            _cfg(*SM100, num_spec=3, pp_size=2, use_replayssm=True),
            "pipeline parallelism",
        ),
        (
            MambaBackendEnum.TRITON,
            _cfg(*SM90, stochastic_rounding=True, **FP16, use_replayssm=True),
            "capability 10",
        ),
        (
            MambaBackendEnum.TRITON,
            _cfg(
                PlatformEnum.ROCM,
                DeviceCapability(9, 4),
                stochastic_rounding=True,
                **FP16,
                use_replayssm=True,
            ),
            "gfx950",
        ),
        (
            MambaBackendEnum.TRITON,
            _cfg(*SM100, stochastic_rounding=True, use_replayssm=True),
            "float16",
        ),
        (
            MambaBackendEnum.FLASHINFER,
            _cfg(*SM100, stochastic_rounding=True, use_replayssm=True),
            "float16",
        ),
        (
            MambaBackendEnum.FLASHINFER,
            _cfg(*SM100, ssu_algorithm="vertical", use_replayssm=True),
            "algorithm",
        ),
        (
            MambaBackendEnum.TRITON,
            _cfg(*SM100, ssu_algorithm="vertical", use_replayssm=True),
            "algorithm",
        ),
        (
            MambaBackendEnum.TRITON,
            _cfg(*SM100, num_spec=3, ssu_algorithm="vertical", use_replayssm=True),
            "algorithm",
        ),
        (
            MambaBackendEnum.TRITON,
            _cfg(PlatformEnum.CPU, None, use_replayssm=True),
            "CPU",
        ),
    ],
)
def test_replayssm_kernel_that_cannot_run_fails_at_selection(backend, cfg, reason):
    """The ReplaySSM checks that lived in `VllmConfig` are the kernels' own."""
    with pytest.raises(ValueError, match=reason):
        _select(cfg, backend)


def test_triton_replayssm_standard_decode_keeps_pipeline_parallelism():
    cfg = _cfg(*SM100, pp_size=2, use_replayssm=True)
    assert _select(cfg, MambaBackendEnum.TRITON) is TritonReplaySSMKernel


@pytest.mark.parametrize(
    ("kernel", "num_spec", "trackers", "alignment", "ring_len"),
    [
        (TritonReplaySSMKernel, 0, False, 1, 16),
        (TritonReplaySSMSpecKernel, 3, True, 1, 20),
        (FlashInferReplaySSMKernel, 0, True, 16, 17),
        (FlashInferReplaySSMKernel, 3, True, 16, 20),
    ],
)
def test_replayssm_kernel_declares_its_cache_layout(
    kernel, num_spec, trackers, alignment, ring_len
):
    """Config-time readers (state alignment, ring shapes, the runner's trackers)
    ask the selected kernel instead of `VllmConfig`. Tracker kernels keep the
    live and current verify windows in the ring."""
    assert kernel.uses_ring_trackers is trackers
    assert kernel.state_alignment == alignment
    assert kernel.ring_len(16, num_spec) == ring_len


@pytest.mark.parametrize("stochastic_rounding", [False, True])
def test_only_flashinfer_replayssm_takes_a_shared_seed_slot(stochastic_rounding):
    mamba_config = MambaConfig(enable_stochastic_rounding=stochastic_rounding)
    assert (
        FlashInferReplaySSMKernel.uses_rand_seed_slot(mamba_config)
        is stochastic_rounding
    )
    for kernel in (TritonReplaySSMKernel, TritonReplaySSMSpecKernel):
        assert not kernel.uses_rand_seed_slot(mamba_config)


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("simple", "simple"),
        ("vertical", "vertical"),
        ("horizontal", "horizontal"),
    ],
)
def test_flashinfer_forwards_ssu_algorithm(
    algorithm: MambaSSUAlgorithm | None,
    expected: MambaSSUAlgorithm,
    monkeypatch,
):
    kernel = Mock()
    monkeypatch.setattr(FlashInferSSUKernel, "_fn", staticmethod(lambda: kernel))
    mamba_config = MambaConfig(
        backend=MambaBackendEnum.FLASHINFER, ssu_algorithm=algorithm
    )
    tensor = torch.empty(1)
    layer = SimpleNamespace(mamba_config=mamba_config)
    FlashInferSSUKernel.forward(layer, _NO_INDICES, *[tensor] * 9)
    assert kernel.call_args.kwargs["algorithm"] == expected


@requires_gpu
def test_triton_basic_call():
    set_random_seed(0)
    device = DEVICE
    batch_size = 2
    dim = 64
    dstate = 16

    state = torch.randn(batch_size, dim, dstate, device=device)
    x = torch.randn(batch_size, dim, device=device)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device=device)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device)
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)

    layer = SimpleNamespace(mamba_config=MambaConfig())
    TritonSSUKernel.forward(layer, _NO_INDICES, state, x, dt, A, B, C, D, dt_bias, out)
    assert not torch.isnan(out).any()


def test_replayssm_physical_ring_shape():
    shapes = MambaStateShapeCalculator.replayssm_ring_shapes(
        (8, 4, 16), n_groups=4, tp_world_size=2, ring_len=20
    )

    assert shapes == ((8, 20, 4), (8, 20), (2, 20, 16))


@requires_gpu
@pytest.mark.parametrize("post_step", [False, True])
def test_replayssm_postprocess_commits_staged_transition(post_step):
    """Keep group histories equal across checkpoints without touching other slots."""
    ring_start = [_device_ints([11, 3, 7, 13]), _device_ints([11, 7, 3, 13])]
    committed = [_device_ints([9, 2, 9, 17]), _device_ints([9, 9, 2, 17])]
    cache = (
        torch.empty(4, 1, device=DEVICE),
        torch.empty(4, 1, 1, 1, device=DEVICE),
        torch.empty(4, 1, 20, 1, device=DEVICE),
        torch.empty(4, 1, 20, device=DEVICE),
        torch.empty(4, 1, 20, 1, device=DEVICE),
    )
    spec = MambaSpec(
        block_size=256,
        shapes=tuple(tensor.shape[1:] for tensor in cache),
        dtypes=tuple(tensor.dtype for tensor in cache),
    )
    # Neighboring request rows provide valid canary IDs, making erroneous
    # column reads deterministically corrupt another slot rather than crash.
    block_tables = [
        torch.full((256, 1), 3, dtype=torch.int32, device=DEVICE) for _ in range(2)
    ]
    grouped = []
    for group_idx, table in enumerate(block_tables):
        table[0, 0] = group_idx + 1
        mixer = _replayssm_mixer(
            cache,
            ring_start[group_idx],
            committed[group_idx],
            torch.empty(1, device=DEVICE),
        )
        grouped.append(([mixer], spec, table))
    context = ReplaySSMModelContext.create(
        grouped, max_num_reqs=1, materialize_prefixes=False
    )
    computed = 50_000
    history_tokens = checkpoint_tokens = 0
    # Reset a later prefill chunk, then reject, partially accept, and fully accept.
    for step, accepted in enumerate([1] + [4, 1, 3, 2] * 5):
        prefilling = step == 0
        query_len = 4
        if not prefilling:
            if history_tokens - checkpoint_tokens + query_len > 16:
                checkpoint_tokens = history_tokens
            history_tokens += accepted
        computed_after = computed + (query_len if prefilling else accepted)
        context.postprocess(
            idx_mapping=_device_ints([0]) if post_step else None,
            query_metadata=_device_ints([0, query_len])
            if post_step
            else _device_ints([query_len]),
            query_metadata_is_cumulative=post_step,
            num_computed_tokens=_device_ints(
                [computed_after if post_step else computed]
            ),
            num_computed_is_post_step=post_step,
            num_accepted_tokens=_device_ints([accepted]),
            is_prefilling=torch.tensor([prefilling], device=DEVICE),
            live_cols=None,
            num_reqs=1,
        )
        expected_committed = history_tokens - checkpoint_tokens
        expected_start = checkpoint_tokens % 20
        for group_idx in range(2):
            expected_starts, expected_counts = [11, 7, 7, 13], [9, 9, 9, 17]
            expected_starts[group_idx + 1] = expected_start
            expected_counts[group_idx + 1] = expected_committed
            assert ring_start[group_idx].tolist() == expected_starts, step
            assert committed[group_idx].tolist() == expected_counts, step
        assert ring_start[0][1].item() == ring_start[1][2].item()
        assert committed[0][1].item() == committed[1][2].item()
        assert 0 <= expected_committed <= 16
        assert expected_committed + query_len <= 20
        assert context.plan_ring_start.item() == 0
        assert context.plan_flush_count.item() == -1
        computed = computed_after
    assert checkpoint_tokens >= 40  # Exercise more than one physical ring wrap.


@pytest.mark.parametrize(
    ("query_start_loc", "expected_x_shape", "expected_B_shape", "expected_max_seqlen"),
    [
        pytest.param([0, 4, 8], (2, 4, 2, 4), (2, 4, 1, 8), None, id="dense"),
        pytest.param([0, 4, 6], (1, 6, 2, 4), (1, 6, 1, 8), 4, id="packed"),
    ],
)
def test_flashinfer_replayssm_selects_mtp_layout(
    monkeypatch,
    query_start_loc,
    expected_x_shape,
    expected_B_shape,
    expected_max_seqlen,
):
    """Full verify windows go to FlashInfer as dense `(batch, 1 + num_spec)`
    rows; a step with shorter rows goes packed with `cu_seqlens`."""
    layer = SimpleNamespace(
        mamba_config=MambaConfig(backend=MambaBackendEnum.FLASHINFER),
        kv_cache=(
            torch.empty(3, 1),
            torch.empty(3, 2, 4, 8),
            torch.empty(3, 2, 20, 4),
            torch.empty(3, 2, 20),
            torch.empty(3, 1, 20, 8),
        ),
        ssu_state=MambaReplaySSMLayerState(
            buffer_len=16,
            num_spec=3,
            ring_start=torch.zeros(3, dtype=torch.int32),
            prev_num_accepted=torch.zeros(3, dtype=torch.int32),
        ),
    )
    num_decode_tokens = query_start_loc[-1]
    query_start_loc_d = torch.tensor(query_start_loc, dtype=torch.int32)
    metadata = Mamba2AttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=num_decode_tokens,
        num_reqs=2,
        has_initial_states_p=None,
        query_start_loc_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=torch.tensor([[1], [2]], dtype=torch.int32),
        query_start_loc_d=query_start_loc_d,
        num_accepted_tokens=torch.tensor([4, 2], dtype=torch.int32),
        seq_lens=torch.tensor([104, 102], dtype=torch.int32),
        replayssm_scratch=(torch.empty(0), torch.empty(0), torch.empty(0)),
        replayssm_state_indices_d=torch.tensor([1, 2], dtype=torch.int32),
    )
    kernel = Mock()
    monkeypatch.setattr(ssu_dispatch, "_checkpointing_ssu", lambda: kernel)
    x = torch.empty(num_decode_tokens, 2, 4)
    B = torch.empty(num_decode_tokens, 1, 8)

    FlashInferReplaySSMKernel.forward(
        layer,
        metadata,
        layer.kv_cache[1],
        x,
        torch.empty_like(x),
        torch.empty(2, 4, 8),
        B,
        torch.empty_like(B),
        torch.empty(2, 4),
        torch.empty(2, 4),
        torch.empty_like(x),
    )

    args = kernel.call_args.args
    assert args[6].shape == expected_x_shape
    assert args[7].shape == expected_x_shape
    assert args[9].shape == expected_B_shape
    assert args[10].shape == expected_B_shape
    assert args[11].shape == expected_x_shape
    if expected_max_seqlen is None:
        assert kernel.call_args.kwargs["cu_seqlens"] is None
    else:
        assert kernel.call_args.kwargs["cu_seqlens"] is query_start_loc_d
    assert kernel.call_args.kwargs["max_seqlen"] == expected_max_seqlen


def test_replayssm_ring_shape_extends_groups_for_head_shards():
    """B_cache retains one replicated group on each head shard."""
    shapes = MambaStateShapeCalculator.replayssm_ring_shapes(
        (2, 64, 128), n_groups=1, tp_world_size=2, ring_len=16
    )

    assert shapes == ((2, 16, 64), (2, 16), (1, 16, 128))
