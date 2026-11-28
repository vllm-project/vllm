# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sizing of the split-KV logits workspace of `TritonMLAImpl.forward_mqa`.

The reservation must cover the largest `attn_logits` that `forward_mqa`
asks the workspace manager for. A windowed layer (a DSpark draft layer) splits
at most its window, so reserving for `max_model_len` instead holds GPU memory
that the profiling run then takes from the KV cache. The decode kernel is
replaced by a recorder and the workspace manager by a recorder or a CPU
manager, so these tests need no GPU.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    MLACommonMetadataBuilder,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.v1.attention.backends.mla.prefill.cpu_sdpa import CPUSDPAMLAPrefillBackend
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    init_workspace_manager,
    lock_workspace,
    reset_workspace_manager,
    use_workspace_lane,
)

MODULE = "vllm.v1.attention.backends.mla.triton_mla"

MAX_MODEL_LEN = 1 << 20
MAX_NUM_SEQS = 64
NUM_SPECULATIVE_TOKENS = 3
QUERY_LEN = 1 + NUM_SPECULATIVE_TOKENS
NUM_HEADS = 16
KV_LORA_RANK = 512
HEAD_SIZE = KV_LORA_RANK + 64
PAGE_SIZE = 16
SM_COUNT = 304
# A DSpark draft layer's decode window: the trained window plus the draft block.
DRAFT_WINDOW = 2048 + NUM_SPECULATIVE_TOKENS
# next_power_of_2(2051 // 512) = 4; min(next_power_of_2(2**20 // 512), 2 * 304).
DRAFT_WINDOW_SPLITS = 4
MAX_MODEL_LEN_SPLITS = 608


def _logits_shape(num_kv_splits: int) -> tuple[int, ...]:
    return (MAX_NUM_SEQS * QUERY_LEN, NUM_HEADS, num_kv_splits, KV_LORA_RANK + 1)


def _vllm_config(layer_windows: dict[str, int | None]):
    layer_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(layer_config), torch.device("cpu"):
        for name, window in layer_windows.items():
            MLAAttention(
                num_heads=NUM_HEADS,
                scale=192**-0.5,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                q_lora_rank=None,
                kv_lora_rank=KV_LORA_RANK,
                kv_b_proj=ColumnParallelLinear(
                    KV_LORA_RANK, NUM_HEADS * 256, bias=False, disable_tp=True
                ),
                prefix=name,
                attn_backend=TritonMLABackend,
                sliding_window=window,
                prefill_backend_cls=CPUSDPAMLAPrefillBackend,
            )
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_SEQS),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
            parallel_drafting=False,
            max_num_new_slots_for_drafting=0,
        ),
        use_v2_model_runner=True,
        compilation_config=layer_config.compilation_config,
    )


def _make_impl(sliding_window: int | None) -> TritonMLAImpl:
    """Only the attributes reserve_workspace and forward_mqa read."""
    impl = object.__new__(TritonMLAImpl)
    impl.num_heads = NUM_HEADS
    impl.kv_lora_rank = KV_LORA_RANK
    impl.scale = HEAD_SIZE**-0.5
    impl.dcp_world_size = 1
    impl.decode_sliding_window = sliding_window
    impl._sm_count = SM_COUNT
    impl._vllm_config = _vllm_config({})
    return impl


def _stub_builder_base_init(self, kv_cache_spec, layer_names, vllm_config, device):
    """The fields TritonMLAMetadataBuilder.__init__ reads from its base."""
    self.compilation_config = vllm_config.compilation_config
    self.model_config = vllm_config.model_config
    self.num_heads = NUM_HEADS
    self.dcp_world_size = 1
    self.mla_dims = SimpleNamespace(kv_lora_rank=KV_LORA_RANK)
    self.reorder_batch_threshold = QUERY_LEN


@contextmanager
def _workspace_requests() -> Iterator[list[tuple[int, ...]]]:
    """Shapes requested from the workspace manager, in order."""
    requests: list[tuple[int, ...]] = []

    def get_simultaneous(*shapes_and_dtypes):
        requests.extend(shape for shape, _ in shapes_and_dtypes)
        return tuple(torch.empty(0) for _ in shapes_and_dtypes)

    manager = SimpleNamespace(get_simultaneous=get_simultaneous)
    with (
        patch(f"{MODULE}.is_workspace_manager_initialized", return_value=True),
        patch(f"{MODULE}.current_workspace_manager", return_value=manager),
        patch(f"{MODULE}.current_platform.num_compute_units", return_value=SM_COUNT),
    ):
        yield requests


def _run_longest_draft_decode(impl: TritonMLAImpl) -> torch.Tensor:
    """forward_mqa on a full batch of draft blocks at max_model_len context.

    Returns the `attn_logits` scratch passed to the decode kernel.
    """
    metadata = SimpleNamespace(
        max_seq_len=MAX_MODEL_LEN,
        causal=False,
        decode=SimpleNamespace(
            block_table=torch.zeros(MAX_NUM_SEQS, 1, dtype=torch.int32),
            seq_lens=torch.full((MAX_NUM_SEQS,), MAX_MODEL_LEN, dtype=torch.int32),
        ),
    )
    num_rows = MAX_NUM_SEQS * QUERY_LEN
    q = torch.zeros(num_rows, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, PAGE_SIZE, HEAD_SIZE, dtype=torch.bfloat16)
    layer = SimpleNamespace(_k_scale=torch.ones(1))
    with patch(f"{MODULE}.decode_attention_fwd") as decode_attention_fwd:
        impl.forward_mqa(q, kv_cache, metadata, layer)
    return decode_attention_fwd.call_args.args[7]


@pytest.fixture
def undersized_locked_workspace() -> Iterator[None]:
    """Both lanes hold a one-split logits reservation when the manager locks."""
    init_workspace_manager(torch.device("cpu"), num_lanes=2)
    for lane in (0, 1):
        with use_workspace_lane(lane):
            current_workspace_manager().get_simultaneous(
                (_logits_shape(1), torch.float32)
            )
    lock_workspace()
    yield
    reset_workspace_manager()


@pytest.mark.parametrize(
    "sliding_window, num_kv_splits",
    [(DRAFT_WINDOW, DRAFT_WINDOW_SPLITS), (None, MAX_MODEL_LEN_SPLITS)],
)
def test_profiling_reservation_follows_the_attended_kv(sliding_window, num_kv_splits):
    """A windowed layer reserves the splits of its window, not max_model_len's."""
    with _workspace_requests() as requests:
        _make_impl(sliding_window).reserve_workspace()

    assert requests == [_logits_shape(num_kv_splits)]


@pytest.mark.parametrize("sliding_window", [DRAFT_WINDOW, None])
def test_reservation_covers_the_largest_forward_mqa_request(sliding_window):
    """Once the workspace is locked, a larger request fails outside the draft."""
    impl = _make_impl(sliding_window)
    with _workspace_requests() as requests:
        impl.reserve_workspace()
        _run_longest_draft_decode(impl)

    reserved, requested = requests
    assert requested == reserved


@pytest.mark.usefixtures("undersized_locked_workspace")
def test_undersized_default_lane_fails_loudly():
    """Outside the DSpark draft lane, an undersized reservation is a bug."""
    with pytest.raises(AssertionError, match="Workspace is locked"):
        _run_longest_draft_decode(_make_impl(DRAFT_WINDOW))


@pytest.mark.usefixtures("undersized_locked_workspace")
def test_undersized_draft_lane_allocates_the_logits():
    """Only the draft's captures size its lane, so it may outgrow it."""
    with use_workspace_lane(1):
        attn_logits = _run_longest_draft_decode(_make_impl(DRAFT_WINDOW))

    assert attn_logits.shape == _logits_shape(DRAFT_WINDOW_SPLITS)


@pytest.fixture
def layer_parallel_state(tmp_path):
    if model_parallel_is_initialized():
        yield
        return
    created_world = not torch.distributed.is_initialized()
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{tmp_path / 'dist'}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(backend="gloo")
            yield
        finally:
            destroy_model_parallel()
            if created_world:
                destroy_distributed_environment()


@pytest.mark.parametrize(
    "layer_windows, num_kv_splits",
    [
        ({"draft.0": DRAFT_WINDOW, "draft.1": DRAFT_WINDOW}, DRAFT_WINDOW_SPLITS),
        ({"draft.0": DRAFT_WINDOW, "draft.1": 2 * DRAFT_WINDOW}, 8),
        ({"draft.0": DRAFT_WINDOW, "draft.1": None}, MAX_MODEL_LEN_SPLITS),
    ],
)
def test_builder_reserves_for_its_widest_layer(
    layer_windows, num_kv_splits, layer_parallel_state
):
    """Mixed finite windows use their maximum; full attention uses max_model_len."""
    with (
        _workspace_requests() as requests,
        patch.object(MLACommonMetadataBuilder, "__init__", _stub_builder_base_init),
    ):
        TritonMLAMetadataBuilder(
            None, list(layer_windows), _vllm_config(layer_windows), None
        )

    assert requests == [_logits_shape(num_kv_splits)]


def test_builder_rejects_non_mla_layer_before_reading_metadata():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.compilation_config.static_forward_context["not_mla"] = torch.nn.Identity()
    with pytest.raises(AssertionError, match="not_mla.*MLAAttention.*Identity"):
        TritonMLAMetadataBuilder(None, ["not_mla"], config, torch.device("cpu"))
