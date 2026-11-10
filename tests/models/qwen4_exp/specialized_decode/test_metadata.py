# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The replayed attention metadata and n-gram inputs of uniform decode steps.

`Qwen4ExpSpecializedDecodeModelState` records the stock attention metadata
build of every uniform decode shape in a CUDA graph, replays it in place of the
eager build, and writes the PLE n-gram inputs with one kernel
(`runtime/decode_metadata.py`). These tests run the state over the real
metadata builders of Qwen3.8-Flash-Next's attention groups (three GDN groups,
the PLE short convolution, the QSA main cache and its raw, compressed and
uncompressed indexer caches) and the real aligned-state-index kernel, and check
that a replayed step returns the metadata and leaves every builder, mamba
context and n-gram buffer exactly as the stock build does: at every captured
decode size of TP4 and TP1, with padding requests, stale rows in the padding,
sequence lengths across mamba block and compression group boundaries, and
replayed and stock steps interleaved across sizes.
"""

import dataclasses
import random
from collections.abc import Collection
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.models.qwen4_exp.common.qsa import Qwen4ExpQSAMetadataBuilder
from vllm.models.qwen4_exp.common.qsa_indexer import (
    QSAIndexerBackend,
    QSAIndexerMetadataBuilder,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.nvidia.specialized_decode import DecodeConfig
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import (
    decode_metadata,
    model_state,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.decode_metadata import (
    RECORDABLE_BUILDERS,
    capture_keys,
    differences,
    snapshot,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.model_state import (
    Qwen4ExpSpecializedDecodeModelState,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils import torch_utils
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadataBuilder
from vllm.v1.attention.backends import gdn_attn, short_conv_attn
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="the recorded builds need CUDA graphs and Triton",
)

NUM_SPEC = 3
QUERY_LEN = NUM_SPEC + 1
MAMBA_BLOCK = 64
ATTN_BLOCK = 64
COMPRESS_RATIO = 4
# The raw-key ring of a depth-3 speculator: round_up(4 + 3, 4).
RING_CAPACITY = 8
MAX_MODEL_LEN = 1024
# Mamba tables reach the last block of the longest sequence plus the
# speculative blocks after it.
MAMBA_TABLE_WIDTH = MAX_MODEL_LEN // MAMBA_BLOCK + NUM_SPEC
ATTN_TABLE_WIDTH = MAX_MODEL_LEN // ATTN_BLOCK
NUM_BLOCKS = 4096
NGRAM_CONTEXT = 2
EOS = 7
VOCAB = 1000
PAD_SLOT = -1
# Written over every buffer before both arms of a poisoned step, so a row that
# one arm skips cannot match the other by chance.
SENTINEL = 1 << 20
# About half a millisecond of GPU time on the current stream.
DELAY_CYCLES = 1_000_000
# The decode graph request counts of the served deployments (their logs).
CAPTURE_SIZES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
CAPTURE_SIZES += [36, 40, 44, 48, 56, 64]

GDN_SPEC = MambaSpec(
    block_size=MAMBA_BLOCK,
    shapes=((4, 64), (2, 16, 16)),
    dtypes=(torch.bfloat16, torch.float32),
    mamba_cache_mode="align",
    num_speculative_blocks=NUM_SPEC,
)
PLE_SPEC = MambaSpec(
    block_size=MAMBA_BLOCK,
    shapes=((8, 64),),
    dtypes=(torch.bfloat16,),
    mamba_cache_mode="align",
    num_speculative_blocks=NUM_SPEC,
)
QSA_SPEC = FullAttentionSpec(
    block_size=ATTN_BLOCK, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
)
RAW_SPEC = CircularBufferSpec(
    block_size=RING_CAPACITY,
    num_kv_heads=1,
    head_size=128,
    head_size_v=0,
    dtype=torch.bfloat16,
)
COMPRESSED_SPEC = MLAAttentionSpec(
    block_size=ATTN_BLOCK,
    num_kv_heads=1,
    head_size=128,
    dtype=torch.bfloat16,
    tokens_per_state=COMPRESS_RATIO,
)
UNCOMPRESSED_SPEC = MLAAttentionSpec(
    block_size=ATTN_BLOCK,
    num_kv_heads=1,
    head_size=128,
    dtype=torch.bfloat16,
    tokens_per_state=1,
)
# (name, backend, builder, spec) by KV cache group, mamba groups first. The QSA
# main cache's backend (None) is imported once the model module is: importing
# the NVIDIA QSA module first is circular.
GROUPS: list[
    tuple[
        str, type[AttentionBackend] | None, type[AttentionMetadataBuilder], KVCacheSpec
    ]
] = [
    ("gdn0", GDNAttentionBackend, GDNAttentionMetadataBuilder, GDN_SPEC),
    ("gdn1", GDNAttentionBackend, GDNAttentionMetadataBuilder, GDN_SPEC),
    ("gdn2", GDNAttentionBackend, GDNAttentionMetadataBuilder, GDN_SPEC),
    (
        "ple",
        PleShortConvAttentionBackend,
        PleShortConvAttentionMetadataBuilder,
        PLE_SPEC,
    ),
    ("qsa", None, Qwen4ExpQSAMetadataBuilder, QSA_SPEC),
    ("raw", QSAIndexerBackend, QSAIndexerMetadataBuilder, RAW_SPEC),
    ("compressed", QSAIndexerBackend, QSAIndexerMetadataBuilder, COMPRESSED_SPEC),
    ("uncompressed", QSAIndexerBackend, QSAIndexerMetadataBuilder, UNCOMPRESSED_SPEC),
]
NUM_MAMBA_GROUPS = 4


def _vllm_config(max_num_seqs: int) -> VllmConfig:
    """MTP-like speculation of three drafts, FULL decode graphs of the served
    request counts up to `max_num_seqs`, mamba prefix caching (align)."""
    config = VllmConfig()
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    compilation = config.compilation_config
    compilation.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    compilation.max_cudagraph_capture_size = max_num_seqs * QUERY_LEN
    compilation.decode_cudagraph_capture_request_sizes = CAPTURE_SIZES
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = 2048
    config.cache_config.mamba_cache_mode = "align"
    return config


@dataclass
class Scenario:
    """A model state over real builders, and the runner state it reads."""

    state: Qwen4ExpSpecializedDecodeModelState
    attn_groups: list[list[AttentionGroup]]
    kv_cache_config: KVCacheConfig
    input_buffers: InputBuffers
    block_tables: list[torch.Tensor]
    slot_mappings: torch.Tensor
    req_states: SimpleNamespace
    rng: random.Random
    # Per replayed or stock step of the state: whether its build ran on the
    # side stream.
    overlapped: list[bool] = dataclasses.field(default_factory=list)

    @property
    def max_num_reqs(self) -> int:
        return self.state.max_num_reqs

    def tables(self, num_reqs_padded: int) -> tuple[torch.Tensor, ...]:
        """The runner's block tables of a step, as it passes them."""
        return tuple(table[:num_reqs_padded] for table in self.block_tables)

    def batch(
        self, num_reqs: int, num_reqs_padded: int, short_request: bool = False
    ) -> InputBatch:
        """Write a decode step of `num_reqs` requests (padded to
        `num_reqs_padded`) into the runner's buffers, as the runner does, and
        return its batch. With `short_request` the last request schedules one
        draft fewer: not a uniform decode step."""
        rng = self.rng
        query_lens = [QUERY_LEN] * num_reqs
        if short_request:
            query_lens[-1] -= 1
        num_tokens = sum(query_lens)
        num_tokens_padded = num_reqs_padded * QUERY_LEN
        batch = InputBatch.make_dummy(num_reqs, num_tokens, self.input_buffers)
        device = self.input_buffers.device
        rows = rng.sample(range(self.max_num_reqs), num_reqs)
        # Lengths around mamba block and compression group boundaries.
        seq_lens = [
            min(
                MAX_MODEL_LEN,
                max(
                    query_len,
                    MAMBA_BLOCK * rng.randint(0, MAX_MODEL_LEN // MAMBA_BLOCK - 1)
                    + rng.randint(-2, 5),
                ),
            )
            for query_len in query_lens
        ]
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        query_lens_t = torch.tensor(query_lens, dtype=torch.int32)
        computed = self.req_states.num_computed_tokens.gpu
        computed[torch.tensor(rows, device=device)] = (seq_lens_t - query_lens_t).to(
            device
        )
        buffers = self.input_buffers
        buffers.seq_lens.zero_()
        buffers.seq_lens[:num_reqs] = seq_lens_t.to(device)
        query_start_loc = torch.full(
            (self.max_num_reqs + 1,), num_tokens, dtype=torch.int32
        )
        query_start_loc[0] = 0
        query_start_loc[1 : num_reqs + 1] = torch.cumsum(query_lens_t, 0)
        buffers.query_start_loc.copy_(query_start_loc)
        positions = torch.cat(
            [torch.arange(s - q, s) for s, q in zip(seq_lens, query_lens)]
        )
        buffers.positions.zero_()
        buffers.positions[:num_tokens] = positions.to(device)
        # Real rows of random blocks; padding rows keep stale ids.
        for table in self.block_tables:
            table.random_(1, NUM_BLOCKS)
        self.slot_mappings.fill_(PAD_SLOT)
        self.slot_mappings[:, :num_tokens].random_(0, NUM_BLOCKS * ATTN_BLOCK)
        self.state.num_accepted_tokens_gpu.random_(1, QUERY_LEN + 1)
        bound = torch.zeros(num_reqs_padded, dtype=torch.int32)
        bound[:num_reqs] = seq_lens_t
        return replace(
            batch,
            num_reqs_after_padding=num_reqs_padded,
            num_tokens_after_padding=num_tokens_padded,
            idx_mapping=torch.tensor(rows, dtype=torch.int64, device=device),
            idx_mapping_np=np.array(rows),
            num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
            num_draft_tokens=num_tokens - num_reqs,
            num_draft_tokens_per_req=np.array(query_lens, dtype=np.int32) - 1,
            query_start_loc=buffers.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_np=query_start_loc[: num_reqs_padded + 1].numpy(),
            seq_lens=buffers.seq_lens[:num_reqs_padded],
            seq_lens_cpu_upper_bound=bound,
            input_ids=buffers.input_ids[:num_tokens_padded],
            positions=buffers.positions[:num_tokens_padded],
            is_padding=buffers.is_padding[:num_tokens_padded],
        )

    def buffers(self) -> list[torch.Tensor]:
        """Every device buffer either arm writes."""
        state = self.state
        return [
            *state.metadata._buffers,
            state.ngram_context,
            state.ple_query_start_loc,
        ]


def _scenario(
    monkeypatch: pytest.MonkeyPatch,
    max_num_seqs: int,
    check_steps: int = 0,
    overlap: bool = True,
    joined: Collection[int] = (),
    stock_captured: Collection[int] = (),
) -> Scenario:
    """The model state of a deployment of `max_num_seqs` requests over real
    builders, with every uniform decode shape recorded. The forward graph
    captures that the runner makes before recording are stood in for: those
    of the padded token counts `joined` placed the waits for an overlapped
    replay, those of `stock_captured` ran the stock layers; `overlap` is
    the deployment's setting."""
    # Each scenario records into a graph pool of its own. Torch's pinned host
    # allocator keeps a private pool per graph pool id, and capturing into an
    # id whose graphs were all destroyed by an earlier scenario fails its
    # use-count assert; a server keeps its graphs alive and never does so.
    monkeypatch.setattr(
        type(current_platform), "_global_graph_pool", torch.cuda.graph_pool_handle()
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "_resolve_gdn_prefill_backend",
        lambda vllm_config: ("auto", "triton"),
    )
    device = torch.device("cuda")
    config = _vllm_config(max_num_seqs)
    attn_groups = []
    kv_cache_groups = []
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSABackend

    for index, (name, backend, builder_cls, spec) in enumerate(GROUPS):
        layers = [f"{name}.0", f"{name}.1"]
        group = AttentionGroup(backend or Qwen4ExpQSABackend, layers, spec, index)
        group.metadata_builders = [builder_cls(spec, layers, config, device)]
        attn_groups.append([group])
        kv_cache_groups.append(KVCacheGroupSpec(layers, spec))
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
    )
    block_tables = [
        torch.zeros(
            max_num_seqs,
            MAMBA_TABLE_WIDTH if index < NUM_MAMBA_GROUPS else ATTN_TABLE_WIDTH,
            dtype=torch.int32,
            device=device,
        )
        for index in range(len(GROUPS))
    ]
    max_tokens = max_num_seqs * QUERY_LEN
    slot_mappings = torch.full(
        (len(GROUPS), max_tokens), PAD_SLOT, dtype=torch.int64, device=device
    )

    # The mamba context's aligned-index kernel over the mamba groups' tables.
    ctx = object.__new__(MambaSpecDecodeGPUContext)
    ctx.block_size = MAMBA_BLOCK
    ctx.num_groups = NUM_MAMBA_GROUPS
    ctx.is_initialized = True
    ctx.block_table_stride_req = block_tables[0].stride(0)
    ctx.aligned_index_block_table_ptrs = torch.tensor(
        [table.data_ptr() for table in block_tables[:NUM_MAMBA_GROUPS]],
        dtype=torch.int64,
        device=device,
    )
    ctx.aligned_state_indices = torch.zeros(
        (NUM_MAMBA_GROUPS, max_num_seqs, 1 + NUM_SPEC), dtype=torch.int32, device=device
    )

    # Qwen4ExpModelState's fields that its attention and input paths read.
    state = object.__new__(Qwen4ExpSpecializedDecodeModelState)
    state.vllm_config = config
    state.model_config = SimpleNamespace(max_model_len=MAX_MODEL_LEN)
    state.cache_config = config.cache_config
    state.max_num_reqs = max_num_seqs
    state.device = device
    state.rope_state = None
    state.num_accepted_tokens_gpu = torch.ones(
        max_num_seqs, dtype=torch.int32, device=device
    )
    state._align_mode = True
    state._use_flashinfer_replayssm = False
    state.recoverssm = None
    state._mamba_ctx = ctx
    state._mamba_spec = GDN_SPEC
    state._mamba_group_ids = list(range(NUM_MAMBA_GROUPS))
    state.uses_ngram_embedding = True
    state.ngram_context_len = NGRAM_CONTEXT
    state.ngram_eos_token_id = EOS
    state.ngram_context = torch.full(
        (max_num_seqs, NGRAM_CONTEXT), EOS, dtype=torch.int32, device=device
    )
    state.ngram_context_offsets = torch.arange(
        -NGRAM_CONTEXT, 0, dtype=torch.int64, device=device
    )
    state.ple_query_start_loc = torch.zeros(
        max_num_seqs + 1, dtype=torch.int32, device=device
    )
    state.disk_offload_modules = []
    # No GDN records protocol here (test_gdn_records*.py test it).
    state.gdn_records = None
    state._pending_commit = None
    state.init_decode_metadata(
        DecodeConfig(
            tokens_per_request=QUERY_LEN,
            max_rows=max_tokens,
            metadata_check_steps=check_steps,
            metadata_overlap=overlap,
        )
    )
    for rows in joined:
        state.metadata.joined(rows)
    for rows in stock_captured:
        state.metadata.captured_stock(rows)
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(
            gpu=torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        ),
        all_token_ids=SimpleNamespace(
            gpu=torch.randint(
                0,
                VOCAB,
                (max_num_seqs, MAX_MODEL_LEN),
                dtype=torch.int32,
                device=device,
            )
        ),
    )
    input_buffers = InputBuffers(max_num_seqs, max_tokens, device)
    state.capture_inner_cudagraphs(
        input_buffers,
        SimpleNamespace(input_block_tables=block_tables, slot_mappings=slot_mappings),
        attn_groups,
        kv_cache_config,
    )
    return Scenario(
        state=state,
        attn_groups=attn_groups,
        kv_cache_config=kv_cache_config,
        input_buffers=input_buffers,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        req_states=req_states,
        rng=random.Random(max_num_seqs),
    )


def _aligned_indices(scenario: Scenario) -> list[tuple | None]:
    """Where the GDN builders' `mamba_aligned_state_indices` point."""
    out = []
    for groups in scenario.attn_groups:
        builder = groups[0].get_metadata_builder()
        if hasattr(builder, "mamba_aligned_state_indices"):
            indices = builder.mamba_aligned_state_indices
            out.append(
                None
                if indices is None
                else (indices.data_ptr(), tuple(indices.shape), tuple(indices.stride()))
            )
    return out


def _run(
    scenario: Scenario, batch: InputBatch, cudagraph_mode: CUDAGraphMode, stock: bool
) -> dict:
    """One arm of a step: the state's attention metadata and model inputs, or
    the stock state's; a snapshot of both and of every buffer."""
    state = scenario.state
    num_reqs_padded = (
        batch.num_reqs_after_padding
        if cudagraph_mode == CUDAGraphMode.FULL
        else batch.num_reqs
    )
    num_tokens_padded = (
        batch.num_tokens_after_padding
        if cudagraph_mode == CUDAGraphMode.FULL
        else batch.num_tokens
    )
    args = (
        batch,
        cudagraph_mode,
        scenario.tables(num_reqs_padded),
        scenario.slot_mappings[:, :num_tokens_padded],
        scenario.attn_groups,
        scenario.kv_cache_config,
    )
    if stock:
        attn_metadata = Qwen4ExpModelState.prepare_attn(state, *args)
        inputs = Qwen4ExpModelState.prepare_inputs(state, batch, scenario.req_states)
    else:
        attn_metadata = state.prepare_attn(*args)
        inputs = state.prepare_inputs(batch, scenario.req_states)
        scenario.overlapped.append(state.metadata._pending)
        # The forward graph's waits for an overlapped replay.
        state.metadata.settle()
    result = snapshot({"attn": attn_metadata, "inputs": inputs}, scenario.buffers())
    result["aligned_indices"] = _aligned_indices(scenario)
    result["replayed"] = any(
        attn_metadata is recording.attn_metadata
        for recording in state.metadata._recordings.values()
    )
    return result


def _check_step(
    scenario: Scenario,
    batch: InputBatch,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL,
    poison: bool = False,
    delay: bool = False,
) -> bool:
    """Run a step through the state and, from the same buffers, the stock
    state; assert the two equal. Returns whether the state replayed. With
    `delay` the current stream first sleeps, so a side stream that did not
    wait for it would run ahead of the step's inputs."""
    if delay:
        torch.cuda._sleep(DELAY_CYCLES)
    buffers = scenario.buffers()
    if poison:
        for buffer in buffers:
            buffer.fill_(True if buffer.dtype == torch.bool else SENTINEL)
    before = [buffer.clone() for buffer in buffers]
    fast = _run(scenario, batch, cudagraph_mode, stock=False)
    for buffer, saved in zip(buffers, before):
        buffer.copy_(saved)
    stock = _run(scenario, batch, cudagraph_mode, stock=True)
    replayed = fast.pop("replayed")
    assert not stock.pop("replayed")
    found = differences(fast, stock)
    assert not found, (
        f"{batch.num_reqs} requests padded to {batch.num_reqs_after_padding}: {found}"
    )
    return replayed


def _padded(num_reqs: int) -> int:
    """The decode graph a step of `num_reqs` requests pads to."""
    return next(size for size in CAPTURE_SIZES if size >= num_reqs)


@pytest.mark.parametrize("max_num_seqs", [16, 64], ids=["tp4", "tp1"])
def test_every_captured_size_is_recorded(monkeypatch, max_num_seqs):
    """One recording per request count up to the largest decode graph, each
    padded to the smallest graph that holds it."""
    scenario = _scenario(monkeypatch, max_num_seqs)
    want = [
        (n, _padded(n), n * QUERY_LEN, _padded(n) * QUERY_LEN)
        for n in range(1, max_num_seqs + 1)
    ]
    assert capture_keys(scenario.state.vllm_config, QUERY_LEN) == want
    assert scenario.state.metadata.keys == want


@pytest.mark.parametrize("max_num_seqs", [16, 64], ids=["tp4", "tp1"])
def test_replays_equal_the_stock_build(monkeypatch, max_num_seqs):
    """Every request count, in an order that alternates large and small
    batches of different padded sizes, then again with stock steps (a batch
    with a short request, a PIECEWISE step) between them; every other step
    starts from poisoned buffers."""
    scenario = _scenario(monkeypatch, max_num_seqs)
    counts = list(range(1, max_num_seqs + 1))
    scenario.rng.shuffle(counts)
    # Largest and smallest alternate, so a step follows one of another size.
    order = [count for pair in zip(counts, reversed(counts)) for count in pair]
    for step, num_reqs in enumerate(order[:max_num_seqs]):
        batch = scenario.batch(num_reqs, _padded(num_reqs))
        assert _check_step(scenario, batch, poison=step % 2 == 0)
    for step, num_reqs in enumerate(counts):
        padded = _padded(num_reqs)
        assert _check_step(scenario, scenario.batch(num_reqs, padded))
        if step % 3 == 0:
            short = scenario.batch(num_reqs, padded, short_request=True)
            assert not _check_step(scenario, short)
        if step % 3 == 1:
            piecewise = scenario.batch(num_reqs, num_reqs)
            assert not _check_step(scenario, piecewise, CUDAGraphMode.PIECEWISE)
        if step % 3 == 2:
            batch = scenario.batch(num_reqs, padded)
            assert _check_step(scenario, batch, poison=True)


def test_stock_build_writes_the_pinned_buffers(monkeypatch):
    """The builder buffers the recorded build writes, which the module
    docstring lists, exist with the dtypes and sizes the FULL graphs read, and
    the stock build's metadata of a uniform decode step references them; the
    stock builds stage their two host masks through the function the recording
    intercepts."""
    max_num_seqs = 16
    scenario = _scenario(monkeypatch, max_num_seqs)
    state = scenario.state
    builders = [groups[0].get_metadata_builder() for groups in scenario.attn_groups]
    assert {type(builder) for builder in builders} == set(RECORDABLE_BUILDERS)
    num_reqs, padded = 5, 6
    batch = scenario.batch(num_reqs, padded)
    attn_metadata = Qwen4ExpModelState.prepare_attn(
        state,
        batch,
        CUDAGraphMode.FULL,
        scenario.tables(padded),
        scenario.slot_mappings[:, : padded * QUERY_LEN],
        scenario.attn_groups,
        scenario.kv_cache_config,
    )

    def aliases(view: torch.Tensor, buffer: torch.Tensor) -> bool:
        start = buffer.data_ptr()
        end = start + buffer.numel() * buffer.element_size()
        return start <= view.data_ptr() < end

    max_bs = max_num_seqs * QUERY_LEN
    for index, builder in enumerate(builders[:3]):
        assert isinstance(builder, GDNAttentionMetadataBuilder)
        metadata = attn_metadata[f"gdn{index}.0"]
        pinned = {
            "spec_state_indices_tensor": (torch.int32, (max_bs, NUM_SPEC + 1)),
            "spec_sequence_masks": (torch.bool, (max_bs,)),
            "spec_token_indx": (torch.int32, (max_bs * QUERY_LEN,)),
            "spec_query_start_loc": (torch.int32, (max_bs + 1,)),
            "num_accepted_tokens": (torch.int32, (max_bs,)),
        }
        for name, (dtype, shape) in pinned.items():
            buffer = getattr(builder, name)
            assert (buffer.dtype, tuple(buffer.shape)) == (dtype, shape), name
        assert aliases(
            metadata.spec_state_indices_tensor, builder.spec_state_indices_tensor
        )
        mamba_ctx = state._mamba_ctx
        assert mamba_ctx is not None and mamba_ctx.aligned_state_indices is not None
        assert builder.mamba_aligned_state_indices is not None
        assert builder.mamba_aligned_state_indices.data_ptr() == (
            mamba_ctx.aligned_state_indices[index].data_ptr()
        )
        if index == 0:
            for name in pinned:
                assert aliases(getattr(metadata, name), getattr(builder, name)), name
    ple = builders[3]
    metadata = attn_metadata["ple.0"]
    for name, dtype, size in [
        ("spec_state_indices_tensor", torch.int32, max_num_seqs),
        ("spec_sequence_masks", torch.bool, max_num_seqs),
        ("spec_query_start_loc", torch.int32, max_num_seqs + 1),
        ("num_accepted_tokens", torch.int32, max_num_seqs),
    ]:
        buffer = getattr(ple, name)
        assert (buffer.dtype, buffer.numel()) == (dtype, size), name
        assert aliases(getattr(metadata, name), buffer), name
    for name, builder in zip(["raw", "compressed", "uncompressed"], builders[5:]):
        assert isinstance(builder, QSAIndexerMetadataBuilder)
        metadata = attn_metadata[f"{name}.0"]
        for field, buffer_name in [
            ("token_to_req", "token_to_req_buffer"),
            ("logical_positions", "logical_positions_buffer"),
            ("visible_blocks", "visible_blocks_buffer"),
            ("k_work_metadata", "k_work_metadata_buffer"),
        ]:
            buffer = getattr(builder, buffer_name)
            if buffer.numel():
                assert aliases(getattr(metadata, field), buffer), (name, field)
        slot_buffer = (
            scenario.slot_mappings
            if name == "uncompressed"
            else builder.slot_mapping_buffer
        )
        assert aliases(metadata.slot_mapping, slot_buffer), name
        assert metadata.max_seq_len == int(batch.seq_lens_cpu_upper_bound.max())
    for module in (gdn_attn, short_conv_attn):
        assert module.async_tensor_h2d is torch_utils.async_tensor_h2d
    for recording in state.metadata._recordings.values():
        assert len(recording.staged) == 2


def test_runtime_check_passes_equal_replays_and_counts(monkeypatch):
    """With the check on, the first replays of a key run the stock build too;
    later ones only replay."""
    scenario = _scenario(monkeypatch, 16, check_steps=2)
    metadata = scenario.state.metadata
    calls = []
    replay = metadata._checked_replay

    def counted(*args):
        calls.append(args[0])
        return replay(*args)

    monkeypatch.setattr(metadata, "_checked_replay", counted)
    for _ in range(3):
        _check_step(scenario, scenario.batch(3, 3))
    assert calls == [(3, 3, 12, 12)] * 2
    assert metadata._recordings[(3, 3, 12, 12)].inputs_checked == 2


def test_runtime_check_reports_a_stale_host_field(monkeypatch):
    """A replay whose `max_seq_len` is not the step's raises in the check."""
    scenario = _scenario(monkeypatch, 16, check_steps=1)
    scenario.state.metadata._recordings[(2, 2, 8, 8)].with_max_seq_len.clear()
    batch = scenario.batch(2, 2)
    with pytest.raises(RuntimeError, match="max_seq_len"):
        scenario.state.prepare_attn(
            batch,
            CUDAGraphMode.FULL,
            scenario.tables(2),
            scenario.slot_mappings[:, :8],
            scenario.attn_groups,
            scenario.kv_cache_config,
        )


def test_runtime_check_reports_wrong_ngram_inputs(monkeypatch):
    """A replayed step whose n-gram kernel skips its rows raises in the
    check."""
    scenario = _scenario(monkeypatch, 16, check_steps=1)
    monkeypatch.setattr(model_state, "fill_ngram_inputs", lambda *args: None)
    batch = scenario.batch(4, 4)
    scenario.state.ngram_context.fill_(SENTINEL)
    scenario.state.prepare_attn(
        batch,
        CUDAGraphMode.FULL,
        scenario.tables(4),
        scenario.slot_mappings[:, :16],
        scenario.attn_groups,
        scenario.kv_cache_config,
    )
    with pytest.raises(RuntimeError, match="n-gram inputs"):
        scenario.state.prepare_inputs(batch, scenario.req_states)


def test_unrecordable_builders_keep_the_stock_build(monkeypatch):
    """A deployment with a builder the module does not know records nothing
    and builds every step the stock way."""
    scenario = _scenario(monkeypatch, 16)
    monkeypatch.setattr(
        decode_metadata, "RECORDABLE_BUILDERS", frozenset({GDNAttentionMetadataBuilder})
    )
    state = scenario.state
    state.capture_inner_cudagraphs(
        scenario.input_buffers,
        SimpleNamespace(
            input_block_tables=scenario.block_tables,
            slot_mappings=scenario.slot_mappings,
        ),
        scenario.attn_groups,
        scenario.kv_cache_config,
    )
    assert state.metadata.keys == []
    assert not _check_step(scenario, scenario.batch(3, 3))


# The overlapped replay (``decode_metadata.py``, "Overlapping the replay with
# the forward graph"): the recordings of a padded token count whose forward
# graph capture placed the waits replay on the side stream; ``_run`` then
# waits for the replay as the forward graph does (``settle``).


def _all_padded_tokens(max_num_seqs: int) -> list[int]:
    """The padded token counts of every decode graph of the deployment."""
    return [size * QUERY_LEN for size in CAPTURE_SIZES if size <= max_num_seqs]


def _spy_replays(monkeypatch, scenario: Scenario) -> list[bool]:
    """The `overlap` argument of every replay of the scenario's state."""
    metadata = scenario.state.metadata
    calls: list[bool] = []
    replay = metadata._replay

    def spied(recording, input_batch, overlap):
        calls.append(overlap)
        return replay(recording, input_batch, overlap)

    monkeypatch.setattr(metadata, "_replay", spied)
    return calls


@pytest.mark.parametrize("max_num_seqs", [16, 64], ids=["tp4", "tp1"])
def test_overlapped_replays_equal_the_stock_build(monkeypatch, max_num_seqs):
    """`test_replays_equal_the_stock_build` with every padded size's
    forward graph waiting: every replay runs on the side stream, after the
    current stream's sleep and the step's inputs, and equals the stock build;
    the stock steps between them (a short request, a PIECEWISE step) build on
    the current stream, as before."""
    scenario = _scenario(
        monkeypatch, max_num_seqs, joined=_all_padded_tokens(max_num_seqs)
    )
    metadata = scenario.state.metadata
    assert metadata.overlapped_keys == metadata.keys
    counts = list(range(1, max_num_seqs + 1))
    scenario.rng.shuffle(counts)
    for step, num_reqs in enumerate(counts):
        padded = _padded(num_reqs)
        batch = scenario.batch(num_reqs, padded)
        assert _check_step(scenario, batch, poison=step % 2 == 0, delay=True)
        assert scenario.overlapped[-1]
        if step % 2 == 0:
            short = scenario.batch(num_reqs, padded, short_request=True)
            assert not _check_step(scenario, short, delay=True)
            assert not scenario.overlapped[-1]
        else:
            piecewise = scenario.batch(num_reqs, num_reqs)
            assert not _check_step(
                scenario, piecewise, CUDAGraphMode.PIECEWISE, delay=True
            )
            assert not scenario.overlapped[-1]


def test_gdn_metadata_are_final_at_gdn_ready(monkeypatch):
    """A forward graph's GDN layers wait only for `gdn_ready`: what their
    metadata hold then equals the stock build's, from poisoned buffers."""
    scenario = _scenario(monkeypatch, 16, joined=_all_padded_tokens(16))
    state = scenario.state
    gdn_layers = [f"{name}.0" for name, *_ in GROUPS[:3]]
    for num_reqs in (1, 3, 8, 16):
        padded = _padded(num_reqs)
        batch = scenario.batch(num_reqs, padded)
        buffers = scenario.buffers()
        for buffer in buffers:
            buffer.fill_(True if buffer.dtype == torch.bool else SENTINEL)
        before = [buffer.clone() for buffer in buffers]
        args = (
            batch,
            CUDAGraphMode.FULL,
            scenario.tables(padded),
            scenario.slot_mappings[:, : padded * QUERY_LEN],
            scenario.attn_groups,
            scenario.kv_cache_config,
        )
        torch.cuda._sleep(DELAY_CYCLES)
        attn_metadata = state.prepare_attn(*args)
        assert state.metadata._pending
        torch.cuda.current_stream().wait_event(state.metadata.gdn_ready)
        fast = snapshot({name: attn_metadata[name] for name in gdn_layers}, [])
        state.metadata.settle()
        for buffer, saved in zip(buffers, before):
            buffer.copy_(saved)
        expected = Qwen4ExpModelState.prepare_attn(state, *args)
        stock = snapshot({name: expected[name] for name in gdn_layers}, [])
        assert not differences(fast, stock), num_reqs


def test_a_build_that_changes_the_gdn_metadata_later_is_refused(monkeypatch):
    """Recording raises when a build after the GDN groups' writes into their
    metadata (here the PLE build bumps the last GDN builder's state indices);
    with the overlap off the GDN layers wait for the whole replay, so it
    records."""
    gdn_builders: list[GDNAttentionMetadataBuilder] = []
    gdn_build = GDNAttentionMetadataBuilder.build
    ple_build = PleShortConvAttentionMetadataBuilder.build

    def remembered(self, *args, **kwargs):
        gdn_builders.append(self)
        return gdn_build(self, *args, **kwargs)

    def late_write(self, *args, **kwargs):
        metadata = ple_build(self, *args, **kwargs)
        gdn_builders[-1].spec_state_indices_tensor.add_(1)
        return metadata

    monkeypatch.setattr(GDNAttentionMetadataBuilder, "build", remembered)
    monkeypatch.setattr(PleShortConvAttentionMetadataBuilder, "build", late_write)
    with pytest.raises(RuntimeError, match="changes the GDN metadata"):
        _scenario(monkeypatch, 16, joined=_all_padded_tokens(16))
    scenario = _scenario(monkeypatch, 16, overlap=False)
    assert scenario.state.metadata.keys


def test_a_graph_wait_binds_to_the_records_before_its_launch():
    """CUDA's rule that the overlap relies on (module docstring): with the host
    far ahead, the external event waits of a graph launched on one stream see
    the records of the launches before it on another stream, an event record
    node inside a graph and an eager record after it, not later ones (which
    follow it and would deadlock)."""
    side = torch.cuda.Stream()
    middle = torch.cuda.Event(external=True)
    end = torch.cuda.Event(external=True)
    middle.record()
    end.record()
    source, first, second, seen_first, seen_second = (
        torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(5)
    )
    iterations = 100
    out_first = torch.full((iterations,), -1, dtype=torch.int64, device="cuda")
    out_second = torch.full((iterations,), -1, dtype=torch.int64, device="cuda")
    torch.cuda._sleep(10)
    first.copy_(source)
    second.copy_(source * 10)
    torch.accelerator.synchronize()
    producer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(producer):
        torch.cuda._sleep(DELAY_CYCLES // 5)
        first.copy_(source)
        middle.record()
        torch.cuda._sleep(DELAY_CYCLES // 5)
        second.copy_(source * 10)
    consumer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(consumer):
        torch.cuda.current_stream().wait_event(middle)
        seen_first.copy_(first)
        torch.cuda.current_stream().wait_event(end)
        seen_second.copy_(second)
    torch.accelerator.synchronize()
    main = torch.cuda.current_stream()
    for step in range(iterations):
        side.wait_stream(main)
        with torch.cuda.stream(side):
            source.fill_(step)
            producer.replay()
            end.record()
        consumer.replay()
        out_first[step].copy_(seen_first[0])
        out_second[step].copy_(seen_second[0])
    torch.accelerator.synchronize()
    steps = torch.arange(iterations, device="cuda")
    assert torch.equal(out_first, steps)
    assert torch.equal(out_second, steps * 10)


# Every path that does not overlap replays or builds on the current stream and
# equals the stock build.


def test_checked_replays_of_an_overlapped_key_run_on_the_current_stream(
    monkeypatch,
):
    scenario = _scenario(monkeypatch, 16, check_steps=2, joined=_all_padded_tokens(16))
    replays = _spy_replays(monkeypatch, scenario)
    for _ in range(3):
        assert _check_step(scenario, scenario.batch(3, 3), delay=True)
    assert replays == [False, False, True]
    assert scenario.overlapped == [False, False, True]


def test_draft_pass_builds_stay_stock(monkeypatch):
    """A build of other attention groups (the MTP drafter's) does not replay,
    whatever the padded size."""
    scenario = _scenario(monkeypatch, 16, joined=_all_padded_tokens(16))
    replays = _spy_replays(monkeypatch, scenario)
    scenario.attn_groups = list(scenario.attn_groups)
    assert not _check_step(scenario, scenario.batch(3, 3), delay=True)
    assert replays == [] and scenario.overlapped == [False]


def test_a_padded_size_without_waits_replays_on_the_current_stream(monkeypatch):
    """Recordings of a padded size whose forward graph capture placed no waits
    (not joined, or another capture of it ran the stock layers) replay on the
    current stream."""
    joined = set(_all_padded_tokens(16)) - {3 * QUERY_LEN}
    scenario = _scenario(monkeypatch, 16, joined=joined, stock_captured={2 * QUERY_LEN})
    metadata = scenario.state.metadata
    overlapped = {key[3] for key in metadata.overlapped_keys}
    assert {key[3] for key in metadata.keys} - overlapped == {
        2 * QUERY_LEN,
        3 * QUERY_LEN,
    }
    replays = _spy_replays(monkeypatch, scenario)
    for num_reqs in (1, 2, 3, 4):
        assert _check_step(scenario, scenario.batch(num_reqs, num_reqs), delay=True)
    assert replays == [True, False, False, True]


def test_the_setting_off_replays_on_the_current_stream(monkeypatch):
    scenario = _scenario(monkeypatch, 16, overlap=False, joined=_all_padded_tokens(16))
    assert scenario.state.metadata.overlapped_keys == []
    replays = _spy_replays(monkeypatch, scenario)
    for num_reqs in (1, 5, 16):
        batch = scenario.batch(num_reqs, _padded(num_reqs))
        assert _check_step(scenario, batch, poison=True)
    assert replays == [False] * 3 and scenario.overlapped == [False] * 3
