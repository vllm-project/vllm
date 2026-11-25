# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SDSA layer tests: each step must match a torch reference that reads the keys
from `history`, not the caches: each row selects its K best keys, and the output
is attention over them. Indexer keys are E2M1 with a 6 per block of 16: exact."""

from types import SimpleNamespace

import pytest
import torch

import vllm.utils.gpu_sync_debug as gsd
from tests.v1.attention.utils import BatchSpec
from vllm import _custom_ops as ops
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.models.nemotron_shortcut.nvidia import sdsa
from vllm.models.nemotron_shortcut.nvidia.sdsa import backend, prefill
from vllm.models.nemotron_shortcut.nvidia.sdsa.prefill import PAGE, SdsaConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import nvfp4_split_data_scale
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.workspace import current_workspace_manager

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or bool(sdsa.flashinfer_ops.missing_requirements()),
    reason="SDSA needs SM10x and its FlashInfer and FlashAttention 4 kernels",
)
HEAD = backend.SDSA_CORE_HEAD_SIZE  # the core's head size, 256
E4M3 = torch.float8_e4m3fn
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
Q_SCALE, K_SCALE, V_SCALE = 0.03, 0.05, 0.02  # FP8 scales, distinct to catch a swap
NVFP4_KEYS = SdsaConfig(indexer_dtype="nvfp4")  # K 2048, the default kernels
SHAPES = [  # each request's tokens so far (seq_lens), and how many are new
    BatchSpec([], [], "empty_step"),
    BatchSpec([257, 129, 7], [257, 129, 7], "prefill_from_empty"),
    BatchSpec([16384, 8191, 1025], [2048, 129, 3], "prefill_over_cached_keys"),
    BatchSpec([16384], [16384], "prefill_of_two_panels"),  # a panel: 12,288 rows
    BatchSpec([4097, 8193, 7], [1, 129, 7], "decode_with_prefill"),
    BatchSpec([4097, 2051, 8193], [1, 1, 129], "two_decodes_with_prefill"),
]


@pytest.fixture(autouse=True)
def tp_group(dist_init) -> None:
    """A TP group of one rank, as a TP1 server has."""


def grid_keys(*shape: int) -> torch.Tensor:
    """Random E2M1 values, with a 6 first in each block of 16."""
    keys = torch.tensor(E2M1)[torch.randint(16, shape)]
    keys.view(-1, 16)[:, 0] = 6
    return keys


def worker_config(
    heads, max_model_len, max_tokens, kv_cache_dtype="auto", drafts=0, **compile
) -> VllmConfig:
    """A worker's config for one SDSA layer and up to 3 requests, with a stand-in
    model config."""
    config = VllmConfig(compilation_config=CompilationConfig(**compile))
    if drafts:  # MTP: a decode request verifies 1 + drafts tokens
        config.speculative_config = SimpleNamespace(
            num_speculative_tokens=drafts,
            parallel_drafting=False,
            max_num_new_slots_for_drafting=0,
        )
    config.cache_config.block_size = PAGE
    config.cache_config.cache_dtype = kv_cache_dtype
    config.scheduler_config.max_num_batched_tokens = max_tokens
    config.scheduler_config.max_num_seqs = 3
    config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(model_type="sdsa"),
        is_mm_prefix_lm=False,
        max_model_len=max_model_len,
        multimodal_config=None,
        enable_sleep_mode=False,
        sleep_mode_offload_cudagraph=False,
        get_head_size=lambda: HEAD,
        get_num_attention_heads=lambda parallel: heads // parallel.tensor_parallel_size,
        get_num_kv_heads=lambda parallel: 1,
    )
    return config


class SdsaFixture:
    """One SDSA layer set up as a worker does, with `history`: 3 requests' keys and
    values at every position, and 7 spare pages per request, as next to Mamba."""

    def __init__(self, config, sdsa_config=NVFP4_KEYS, slack=0) -> None:
        self.config, self.sdsa, self.slack = config, sdsa_config, slack
        heads = config.model_config.get_num_attention_heads(config.parallel_config)
        tokens = config.scheduler_config.max_num_batched_tokens
        self.length = length = config.model_config.max_model_len
        pages = 3 * (length // PAGE + 8)
        torch.manual_seed(0)
        with torch.device("cuda"), set_current_vllm_config(config):
            self.layer = sdsa.SdsaAttention(
                num_heads=heads,
                sdsa_config=sdsa_config,
                buffers=sdsa.allocate_sdsa_buffers(config, sdsa_config.top_k),
                cache_config=config.cache_config,
                prefix="test.sdsa",
            )
            self.table = torch.randperm(pages, dtype=torch.int32).view(3, -1)
            position = torch.arange(length)
            self.slot = self.table[:, position // PAGE].long() * PAGE + position % PAGE
            self.position_of_slot = torch.zeros(pages * PAGE, dtype=torch.long)
            self.position_of_slot[self.slot] = position
            values = torch.randn(3, length, 2 * HEAD)  # the core's keys and values
            self.history = torch.cat((grid_keys(3, length, 128), values), -1).bfloat16()
            # Buffers that a CUDA graph reads: they keep their addresses.
            self.inputs = torch.zeros(tokens, heads + 3, HEAD, dtype=torch.bfloat16)
            self.slots = torch.zeros(tokens, dtype=torch.long)
            self.lengths = torch.zeros(2, 4, dtype=torch.int32)  # query starts, lengths
            key_bytes = sum(sdsa.INDEXER_FORMATS[sdsa_config.indexer_dtype][1:])
            k_cache = torch.zeros(pages, PAGE, key_bytes, dtype=torch.uint8)
            self.layer.indexer_cache.kv_cache = k_cache
            core = self.layer.core
            if core.impl.fp8_kv:
                for name, scale in (("_q", Q_SCALE), ("_k", K_SCALE), ("_v", V_SCALE)):
                    getattr(core, f"{name}_scale").fill_(scale)
                    setattr(core, f"{name}_scale_float", scale)
            kv_dtype = core.kv_cache_torch_dtype
            self.kv_cache = torch.zeros(pages, 1, PAGE, 2 * HEAD, dtype=kv_dtype)
            core.bind_kv_cache(self.kv_cache)
        self.names = (self.layer.indexer_cache.prefix, core.layer_name)
        self.builders = []
        for cache, name in zip((self.layer.indexer_cache, core), self.names):
            builder = cache.get_attn_backend().get_builder_cls()
            spec = cache.get_kv_cache_config(config).spec
            width = {"block_table_width": self.table.shape[1]}
            kwargs = width if builder.requires_block_table_width else {}
            self.builders.append(
                builder(spec, [name], config, torch.device("cuda"), **kwargs)
            )
        with set_forward_context(None, config):  # the profiling run
            self.layer(*self.split_inputs(tokens))
        current_workspace_manager().lock()

    def split_inputs(self, rows: int) -> list[torch.Tensor]:
        """The layer's inputs: indexer query and key, core query, key, value."""
        x = self.inputs[:rows]
        indexer_query, indexer_key = x[:, 0].unflatten(1, (2, 128)).unbind(1)
        return [indexer_query, indexer_key, x[:, 1:-2], x[:, -2:-1], x[:, -1:]]

    def write_cached_keys(self, cached: torch.Tensor) -> None:
        """Write the `cached` positions' keys with vLLM's own cache writers."""
        slots = self.slot[: len(cached)][cached]
        index_key, kv = self.history[: len(cached)][cached].split((128, 2 * HEAD), -1)
        core, k_cache = self.layer.core, self.layer.indexer_cache.kv_cache
        writer_args = (core.kv_cache_dtype, core._k_scale, core._v_scale)
        cache_k, cache_v = self.kv_cache.transpose(1, 2).split(HEAD, -1)
        key, value = kv[:, None].split(HEAD, -1)
        ops.reshape_and_cache_flash(key, value, cache_k, cache_v, slots, *writer_args)
        keys, scales = prefill.indexer_pages(k_cache, self.sdsa.indexer_dtype)
        page, offset = slots // PAGE, slots % PAGE  # the page views are not flat
        if scales is None:  # BF16 keys, as they are
            keys[page, offset] = index_key
            return
        one = torch.ones(1, device="cuda")  # NVFP4 keys, through vLLM's quantizer
        values, block_scales = ops.scaled_fp4_quant(
            index_key.contiguous(), one, is_sf_swizzled_layout=False
        )
        keys[page, offset] = values
        scales[page, offset] = block_scales.view(E4M3)

    def step(self, spec: BatchSpec) -> list[torch.Tensor]:
        """Write the keys cached before the step, build the step's metadata, and
        return its inputs. The layer caches the step's new keys itself."""
        query_lens = torch.tensor(spec.query_lens, dtype=torch.int32)
        seq_lens = torch.tensor(spec.seq_lens, dtype=torch.int32)
        query_start = torch.zeros(spec.batch_size + 1, dtype=torch.int32)
        query_start[1:] = query_lens.cumsum(0)
        request = torch.arange(spec.batch_size).repeat_interleave(query_lens)
        new = torch.arange(len(request)) - query_start[request]
        self.request = request.cuda()
        self.position = ((seq_lens - query_lens)[request] + new).cuda()
        cached = (torch.arange(self.length) < (seq_lens - query_lens)[:, None]).cuda()
        if cached.any():
            self.write_cached_keys(cached)
        position = self.position.clamp(min=0)  # a padding row has none
        slots = self.slots[: len(request)]
        slots.copy_(
            self.slot[self.request, position].masked_fill(self.position < 0, -1)
        )
        self.slot_mapping = dict.fromkeys(self.names, slots)
        self.lengths[0, : len(query_start)] = query_start
        self.lengths[1, : spec.batch_size] = seq_lens
        common = CommonAttentionMetadata(
            query_start_loc=self.lengths[0, : len(query_start)],
            query_start_loc_cpu=query_start,
            seq_lens=self.lengths[1, : spec.batch_size],
            seq_lens_cpu_upper_bound=seq_lens + self.slack,
            num_reqs=spec.batch_size,
            num_actual_tokens=len(request),
            max_query_len=max(spec.query_lens, default=0),
            max_seq_len=max(spec.seq_lens, default=0),
            block_table_tensor=self.table[: spec.batch_size],
            slot_mapping=slots,
        )
        self.metadata = {}  # a host sync here fails a test that forbids them
        for name, builder in zip(self.names, self.builders):
            self.metadata[name] = gsd.with_gpu_sync_check(builder.build)(0, common)
        rows, new_keys = len(request), self.history[self.request, position]
        self.inputs[:rows, 0, :128] = grid_keys(rows, 128)  # the indexer query
        self.inputs[:rows, 0, 128:] = new_keys[:, :128]
        self.inputs[:rows, 1:-2].normal_()  # the core query
        self.inputs[:rows, -2:] = new_keys[:, 128:].unflatten(1, (2, HEAD))
        return self.split_inputs(rows)

    def run(self, steps, indexer_query=None) -> None:
        """Run each step eagerly, and check it."""
        for spec in steps:
            inputs = self.step(spec)
            if indexer_query is not None:
                inputs[0].fill_(indexer_query)
            mapping = self.slot_mapping
            with set_forward_context(self.metadata, self.config, slot_mapping=mapping):
                self.check(inputs, self.layer(*inputs))

    def check(self, inputs: list[torch.Tensor], output: torch.Tensor) -> None:
        """Each row selects distinct keys that score as its K best, and the
        output is softmax attention over them."""
        rows, top_k, core = len(self.position), self.sdsa.top_k, self.layer.core
        selected = self.selected_positions(rows)
        current_workspace_manager()._current_workspaces[0].fill_(0xA5)
        counts = (self.position + 1).clamp(min=0, max=top_k)
        valid = torch.arange(top_k, device="cuda") < counts[:, None]
        assert torch.equal(selected >= 0, valid)
        ordered = selected.sort(1).values
        assert not ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()
        keys = self.history[..., :128].flatten(0, 1).float()
        keys_values = self.history[..., 128:].flatten(0, 1).float()
        query = inputs[2][:rows].float()
        tolerance = dict(rtol=0.02, atol=0.01)
        if core.impl.fp8_kv:  # the layer rounds Q, K, V and P to E4M3
            kv_scale = torch.tensor([K_SCALE] * HEAD + [V_SCALE] * HEAD, device="cuda")
            keys_values = (keys_values / kv_scale).to(E4M3).float() * kv_scale
            query = (query / Q_SCALE).to(E4M3).float() * Q_SCALE
            tolerance = dict(rtol=0.06, atol=0.08)
        owner = torch.arange(len(keys), device="cuda") // self.length
        key_position = torch.arange(len(keys), device="cuda") % self.length
        for start in range(0, rows, 512):
            chunk = slice(start, start + 512)
            request, end = self.request[chunk, None], self.position[chunk, None] + 1
            scores = inputs[0][chunk].float() @ keys.T
            scores.masked_fill_((owner != request) | (key_position >= end), -torch.inf)
            columns = request * self.length + selected[chunk].clamp(min=0)
            chosen = scores.gather(1, columns).masked_fill(~valid[chunk], -torch.inf)
            best = scores.topk(top_k).values
            assert torch.equal(chosen.sort(1, descending=True).values, best)
            key, value = keys_values[columns].chunk(2, -1)
            logits = query[chunk] @ key.transpose(1, 2) * core.impl.scale
            logits.masked_fill_(~valid[chunk, None], -torch.inf)
            attends = counts[chunk] > 0  # a padding row attends to nothing
            expected = (logits.softmax(-1) @ value)[attends]
            actual = output[chunk][attends].float()
            torch.testing.assert_close(actual, expected, **tolerance)

    def selected_positions(self, rows: int) -> torch.Tensor:
        """Each row's selected keys as positions in its request, -1 last. A
        decode row's are in topk_indices; a prefill row's are its group's union
        of cache slots, filtered by its mask row (union core), or its own slots
        (token and FA4 cores). The buffers are then poisoned."""
        layer, indexer = self.layer, self.metadata[self.names[0]]
        selected = torch.full((rows, self.sdsa.top_k), -1, device="cuda")
        decodes = indexer.num_decodes * indexer.decode_rows
        selected[:decodes] = layer.topk_indices[:decodes]
        plan = indexer.prefill
        if plan is not None:
            tile_rows = (plan.query_rows >= 0).nonzero()[:, 0]
            query = plan.query_rows[tile_rows]
            batch = plan.group_batches[tile_rows // plan.group].long()
            columns = torch.arange(plan.unions.shape[1], device="cuda")
            chosen = columns < layer.valid_counts[batch, None]
            if plan.core == "union":
                bits = layer.mask[query][:, columns // 32] >> columns % 32
                chosen &= (bits & 1) > 0
            slots = plan.unions[batch].long().clamp(min=0)
            positions = self.position_of_slot[slots].masked_fill(~chosen, 2**40)
            positions = positions.sort(1).values[:, : self.sdsa.top_k]
            selected[decodes + query] = positions.masked_fill(positions == 2**40, -1)
        for buffer in (layer.topk_indices, layer.valid_counts, layer.mask):
            buffer.fill_(-0x5A5A5A5B)
        return selected


@pytest.mark.parametrize("top_k", [2048, 512])
@pytest.mark.parametrize("spec", SHAPES, ids=[spec.name for spec in SHAPES])
@pytest.mark.parametrize(
    ("indexer_dtype", "prefill_core"),
    [("nvfp4", "union"), ("nvfp4", "token"), ("bf16", "union")],
)
@torch.inference_mode()
def test_sdsa_matches_reference(
    spec, indexer_dtype, prefill_core, top_k, monkeypatch, workspace_init
) -> None:
    """One step of each shape matches the reference, and its metadata builds do
    not sync with the host. A row with at most K keys attends to all of them."""
    config = SdsaConfig(
        top_k=top_k, indexer_dtype=indexer_dtype, prefill_core=prefill_core
    )
    worker = worker_config(112, 16384, max(spec.compute_num_tokens(), 1))
    fixture = SdsaFixture(worker, config)
    fixture.step(spec)  # warms the pinned host pool, which syncs once
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)
    gsd._install_copy_checkers()
    fixture.run([spec])


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("route", ["dense", "fused", "split"])
@pytest.mark.parametrize(
    ("drafts", "decode_core", "prefill_core"),
    [(0, "token", "union"), (1, "fa4", "token"), (7, "token", "fa4")],
)
@torch.inference_mode()
def test_each_selector_and_core(
    route, drafts, decode_core, prefill_core, kv_cache_dtype, workspace_init
) -> None:
    """Decode of 1 + drafts tokens a request, mixed with prefill, matches the
    reference under each selector ("split": requests of 8K keys and up fused),
    core and KV cache dtype, also when the host knows only upper bounds of the
    lengths (async speculative decoding: a non-uniform decode runs as prefill)."""
    fused = {"dense": False, "fused": True, "split": (8192, 1 << 30)}[route]
    config = SdsaConfig(
        indexer_dtype="nvfp4",
        decode_core=decode_core,
        decode_fused=fused,
        prefill_core=prefill_core,
        prefill_fused=fused,
    )
    rows = 1 + drafts
    worker = worker_config(20, 20480, rows + 307, kv_cache_dtype, drafts=drafts)
    steps = [
        BatchSpec([2], [2]),
        BatchSpec([4097, 8193, 20000], [rows, rows, 129]),
        BatchSpec([16384, 7, 9000], [rows, 7, 300]),
        BatchSpec([16384, 3, 9000], [rows, rows, rows]),
        BatchSpec([9000, 5000, 20000], [rows, 1, rows]),
    ]
    SdsaFixture(worker, config, slack=drafts).run(steps)


@pytest.mark.parametrize("top_k", [2048, 512])
@pytest.mark.parametrize(
    ("pattern", "spec"),
    [
        ("short", BatchSpec([12288], [3])),
        ("ties", BatchSpec([32768], [3])),
        ("misses", BatchSpec([32768] * 3, [8] * 3)),
        ("over", BatchSpec([32768, 20000], [131, 9])),
        ("over_ties", BatchSpec([32768], [9])),
    ],
)
@torch.inference_mode()
def test_selector_repairs_rows_that_miss(
    pattern, spec, top_k, monkeypatch, workspace_init
) -> None:
    """Where the fused selector's sampled threshold misses (a row short of K, or
    too many candidates), the selection stays exact. Samples sit at offset 0 of
    each run of pages, so the pattern decides what they see."""
    monkeypatch.setattr(prefill, "PANEL_ROWS", 128)
    config = SdsaConfig(top_k=top_k, indexer_dtype="nvfp4", prefill_core="union")
    fixture = SdsaFixture(worker_config(112, 32768, spec.compute_num_tokens()), config)
    fixture.builders[0].sample_offsets.zero_()
    # Keys by page. Against a query of 6s, a "level" key scores above 0 keys.
    keys = fixture.history[..., :128].unflatten(1, (-1, PAGE))
    level = torch.tensor([6.0] + [1.0] * 15, device="cuda").repeat(8)
    keys.zero_()
    if pattern == "short":  # under RANK sampled keys score above a tied run
        keys[0, 60:92] = level
        keys[0, :60, 0] = 2 * level
    if pattern == "misses":  # only each run's first key scores above 0
        keys[:, :: prefill.PAGES, 0] = level
    if pattern in ("over", "over_ties"):  # random or tied; each page's first lowest
        keys.copy_(grid_keys(*keys.shape) if pattern == "over" else level)
        keys[:, :, 0] = -6
    fixture.run([spec], indexer_query=0 if pattern == "ties" else 6)


@pytest.mark.parametrize("top_k", [2048, 512])
@pytest.mark.parametrize(
    ("heads", "drafts", "route"),
    [(20, 0, "dense"), (112, 0, "fused"), (112, 3, "split")],
)
@pytest.mark.parametrize(
    ("indexer_dtype", "kv_cache_dtype"),
    [("nvfp4", "auto"), ("nvfp4", "fp8"), ("bf16", "auto")],
)
@torch.inference_mode()
def test_decode_cudagraph_replays_changed_metadata(
    heads, drafts, route, indexer_dtype, kv_cache_dtype, top_k, workspace_init
) -> None:
    """Decode steps of new lengths replay one full CUDA graph and match the
    reference; padding requests (length 0) select nothing. 20 heads take the
    token core, 112 heads with drafts the FA4 core."""
    context = 131072 if heads == 112 else 8192
    fused = {"dense": False, "fused": True, "split": (context // 2, context)}[route]
    config = SdsaConfig(top_k=top_k, indexer_dtype=indexer_dtype, decode_fused=fused)
    rows = 3 * (1 + drafts)
    graphs = dict(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    worker = worker_config(heads, context, rows, kv_cache_dtype, drafts, **graphs)
    worker.compilation_config.cudagraph_capture_sizes = [rows]  # VllmConfig drops
    worker.compilation_config.max_cudagraph_capture_size = rows  # them without a model
    fixture = SdsaFixture(worker, config)
    graph = CUDAGraphWrapper(fixture.layer, worker, CUDAGraphMode.FULL)
    stream = torch.cuda.Stream()  # CUDA graphs capture off the default stream
    batch = BatchDescriptor(rows, 3, uniform=True)
    q = context // 4
    steps = [[q - 1, 2 * q + 1, 3 * q - 1], [3 * q + 1, q, 7], [2 * q, 3, 0], [0, 0, 0]]
    fixture.run([BatchSpec(steps[0], [1 + drafts] * 3)])  # kernels compile eagerly
    for index, seq_lens in enumerate(steps):
        inputs = fixture.step(BatchSpec(seq_lens, [1 + drafts] * 3))
        stream.wait_stream(torch.cuda.current_stream())
        replay = dict(cudagraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=batch)
        forward = set_forward_context(
            fixture.metadata, worker, slot_mapping=fixture.slot_mapping, **replay
        )
        with torch.cuda.stream(stream), forward:
            if index == 0:  # captures the graph
                graph(*inputs)
            output = graph(*inputs).clone()
        torch.cuda.current_stream().wait_stream(stream)
        fixture.check(inputs, output)
    graph.clear_graphs()


@torch.inference_mode()
def test_union_warm_up(monkeypatch, workspace_init) -> None:
    """The warm-up builds a union kernel per FlashInfer key bitmap size: powers
    of two up to UNION_KEY_LIMIT keys. The indexer's builder registers
    them only if a step can take the union core: at 20 heads, not at 112."""
    kernel = prefill.GROUP_UNION_MASKS_KERNEL
    shapes = dict(group=4, top_k=2048, table_pages=22528, columns=8192)
    keys = kernel.get_warmup_keys(max_model_len=1441792, **shapes)
    sizes = [16384 << n for n in range(5)] + [524288]
    assert [key.num_keys for key in keys] == sizes
    groups = []
    monkeypatch.setattr(
        kernel, "register_warmup", lambda **s: groups.append(s["group"])
    )
    SdsaFixture(worker_config(20, 4096, 2))
    current_workspace_manager().unlock()  # a new worker reserves its workspace
    SdsaFixture(worker_config(112, 4096, 2))
    assert groups == [4]


def test_nvfp4_prefill_launches() -> None:
    """A prefill whose rows select at least its context converts it densely;
    one whose regions exceed the workspace splits into launches that fit."""
    launches, groups = backend._nvfp4_prefill_launches(
        (2816, 50), (200_000, 150_000), 2048, 1 << 20, 65535, 256
    )
    assert launches == [(0, 2, 0, 2866, 200_000)]
    assert groups.tolist() == [[0, 0, 2816, 0, 1], [1, 2816, 50, 200_000, 0]]
    launches, groups = backend._nvfp4_prefill_launches(
        (3000,), (900_000,), 2048, 300_000, 65535, 256
    )
    assert len(launches) == 21
    assert launches[0] == (0, 1, 0, 146, 299_008)
    assert launches[-1] == (20, 21, 2920, 3000, 163_840)
    assert groups[:, 2].sum() == 3000 and not groups[:, 4].any()


@pytest.mark.skipif(
    not sdsa.flashinfer_ops.has_nvfp4_kv_conversion(),
    reason="needs FlashInfer's NVFP4 K/V conversion by key position (#9527)",
)
@torch.inference_mode()
def test_nvfp4_writer_and_conversion_roundtrip() -> None:
    """VLLM's NVFP4 cache writer and FlashInfer's conversion by key position
    give back the BF16 inputs: a K/V scale layout both get wrong fails."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache_by_position

    torch.manual_seed(123)
    tokens = 448
    x = torch.randn(tokens, 1, HEAD, device="cuda")
    x *= torch.linspace(0.1, 6.0, tokens, device="cuda")[:, None, None]
    y = (
        torch.randn_like(x)
        * torch.linspace(3.0, 0.1, tokens, device="cuda")[:, None, None]
    )
    x, y = x.bfloat16(), y.bfloat16()
    cache = torch.empty(7, 2, PAGE, 144, device="cuda", dtype=torch.uint8)
    k_scale, v_scale = (torch.tensor(value, device="cuda") for value in (0.03, 0.07))
    key_cache, value_cache = cache.transpose(1, 2).chunk(2, dim=-2)
    ops.reshape_and_cache_flash(
        x,
        y,
        key_cache,
        value_cache,
        torch.arange(tokens, device="cuda"),
        "nvfp4",
        k_scale,
        v_scale,
    )
    k, v = cache.chunk(2, dim=1)
    kd, ks = nvfp4_split_data_scale(k)
    vd, vs = nvfp4_split_data_scale(v)
    # One row selecting positions 0 .. tokens - 1 of one request whose block
    # table is the identity, so key position p is cache slot p.
    positions = torch.full((1, 512), -1, dtype=torch.int32, device="cuda")
    positions[0, :tokens] = torch.arange(tokens, device="cuda", dtype=torch.int32)
    block_table = torch.arange(cache.shape[0], device="cuda", dtype=torch.int32)
    out = torch.empty(512, 1, 1, 2 * HEAD, dtype=E4M3, device="cuda")
    seq_lens = torch.empty(1, device="cuda", dtype=torch.int32)
    gather_nvfp4_kv_cache_by_position(
        (kd, vd),
        (ks, vs),
        positions,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        block_table[None],
        out=out,
        block_tables=torch.empty(1, 2, 512, device="cuda", dtype=torch.int32),
        seq_lens=seq_lens,
    )
    assert seq_lens.item() == tokens
    for original, converted, scale in zip(
        (x, y), out[:tokens].chunk(2, dim=-1), (k_scale, v_scale)
    ):
        # The conversion divides by 6 to fit E4M3.
        reconstructed = converted.reshape(tokens, 1, HEAD).float() * (6 * scale)
        relative_rmse = (
            (reconstructed - original.float()).square().mean()
            / original.float().square().mean()
        ).sqrt()
        assert relative_rmse.item() < 0.14
