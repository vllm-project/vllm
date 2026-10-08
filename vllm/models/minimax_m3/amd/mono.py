# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 target-layer adapter for ATOM's TP4 mono kernel library.

The native model owns this adapter, its conversion copies, peer memory and all
graph resources. Scheduler, speculative decoding and cache allocation remain
native. Initialization is collective; unsupported runtime batches use the
native model loop. Kernel failures propagate.
"""

import os
from dataclasses import fields
from typing import cast

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.indexer_aiter import MiniMaxM3IndexerAiterMetadata
from vllm.models.minimax_m3.amd.mono_weights import MonoLayerWeights, require
from vllm.models.minimax_m3.amd.sparse_attention_msa import (
    MiniMaxM3SparseAiterPADecodeMetadata,
)
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
SUPPORTED_TOKENS = (1, 4, 8, 16)


def _collective_stage(group, name, action):
    error, result = None, None
    try:
        result = action()
    except Exception as exc:
        error = f"rank {dist.get_rank(group)} {name}: {type(exc).__name__}: {exc}"
    errors: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    if any(item is not None for item in errors):
        raise RuntimeError(
            "MiniMax-M3 ATOM mono initialization failed: "
            + "; ".join(item for item in errors if item is not None)
        )
    return result


class _PeerMemory:
    def __init__(self, nbytes, group, rank, device):
        from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

        self.heap = UncachedIpcHeap
        self.opened: list[int] = []
        self.local = 0

        def allocate():
            self.local = self.heap.alloc_uncached(nbytes)

        try:
            _collective_stage(group, "peer allocation", allocate)
            handle = _collective_stage(
                group, "peer handle", lambda: self.heap.get_mem_handle_bytes(self.local)
            )
            handles = [None] * 4
            dist.all_gather_object(handles, handle, group=group)

            def open_peers():
                addresses = []
                for peer, handle in enumerate(handles):
                    address = (
                        self.local
                        if peer == rank
                        else self.heap.open_mem_handle(handle)
                    )
                    addresses.append(address)
                    if peer != rank:
                        self.opened.append(address)
                self.addresses = torch.tensor(
                    addresses, dtype=torch.int64, device=device
                )
                # This storage has no deleter. The adapter retains its HIP owner
                # until the runner drops graphs and synchronizes every rank.
                storage = torch._C._construct_storage_from_data_pointer(
                    self.local, device, nbytes
                )
                self.tensor = torch.empty(0, dtype=torch.uint8, device=device).set_(
                    storage, 0, (nbytes,), (1,)
                )

            _collective_stage(group, "peer mapping", open_peers)
            dist.barrier(group=group)
        except Exception:
            self.close()
            raise

    def close(self):
        """Release only after all users and captured graphs have been drained."""
        self.tensor = None
        self.addresses = None
        for address in self.opened:
            self.heap.close_mem_handle(address)
        self.opened.clear()
        if self.local:
            self.heap.free_device_mem(self.local)
            self.local = 0


@triton.jit
def _token_rows(
    main_source,
    index_source,
    lengths,
    main_slots,
    index_slots,
    main_out,
    index_out,
    seq_out,
    slot_out,
    index_slot_out,
    main_stride: tl.constexpr,
    index_stride: tl.constexpr,
    main_width: tl.constexpr,
    index_width: tl.constexpr,
    rows: tl.constexpr,
    query_len: tl.constexpr,
    width: tl.constexpr,
):
    token = tl.program_id(0)
    request = token // query_len
    seq = tl.load(lengths + request, request < rows, other=0)
    slot = tl.load(main_slots + token)
    index_slot = tl.load(index_slots + token)
    live = (seq > 0) & (slot >= 0)
    causal_seq = tl.where(live, seq - query_len + token % query_len + 1, 0)
    tl.store(seq_out + token, causal_seq)
    tl.store(slot_out + token, tl.where(live, slot, -1))
    tl.store(index_slot_out + token, tl.where(live, index_slot, -1))
    col = tl.arange(0, width)
    main = tl.load(
        main_source + request * main_stride + col,
        (request < rows) & (col < main_width) & live,
        other=0,
    )
    index = tl.load(
        index_source + request * index_stride + col,
        (request < rows) & (col < index_width) & live,
        other=0,
    )
    tl.store(main_out + token * width + col, main)
    tl.store(index_out + token * width + col, index)


def _mono_layer(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    residual_out: torch.Tensor,
    main_cache: torch.Tensor,
    index_cache: torch.Tensor,
    weights: list[torch.Tensor],
    resources: list[torch.Tensor],
    layer_name: str,
    token_count: int,
    query_len: int,
) -> None:
    attention = get_forward_context().no_compile_layers[layer_name]
    adapter, index = attention._atom_mono
    adapter.launch(index, residual, positions, token_count, query_len)


def _mono_layer_fake(
    hidden,
    residual,
    positions,
    output,
    residual_out,
    main_cache,
    index_cache,
    weights,
    resources,
    layer_name,
    token_count,
    query_len,
) -> None:
    return None


direct_register_custom_op(
    "minimax_m3_atom_mono",
    _mono_layer,
    mutates_args=["output", "residual_out", "main_cache", "index_cache", "resources"],
    fake_impl=_mono_layer_fake,
)


class M3Mono:
    def __init__(self, model, config, kv_cache_config):
        self.group = get_tp_group().cpu_group
        self.rank = get_tp_group().rank_in_group
        self.device = next(model.parameters()).device
        self.config = config
        self.state = "VALIDATING"
        self.fallback_counts = {}
        self.query_len = 4

        def dependencies():
            from atom.models.minimax_m3.mono.config import VLLM_CACHE_ABI_VERSION
            from atom.models.minimax_m3.mono.kernels.post_attn import (
                build_post_attn_kernel,
            )
            from atom.models.minimax_m3.mono.kernels.pre_attn import K1_ARGS
            from atom.models.minimax_m3.mono.kernels.pre_attn import (
                SCRATCH_BYTES as K1_BYTES,
            )
            from atom.models.minimax_m3.mono.layout import SCRATCH_BYTES, sym_layout

            return (
                VLLM_CACHE_ABI_VERSION,
                build_post_attn_kernel,
                K1_ARGS,
                K1_BYTES,
                SCRATCH_BYTES,
                sym_layout,
            )

        (
            abi_version,
            build_post_attn_kernel,
            K1_ARGS,
            K1_BYTES,
            SCRATCH_BYTES,
            sym_layout,
        ) = _collective_stage(self.group, "ATOM dependencies", dependencies)

        def validate():
            parallel = config.parallel_config
            require(abi_version == 1, "ATOM cache ABI version must be 1")
            require(
                os.environ.get("COMPILE_ONLY", "0") != "1",
                "COMPILE_ONLY=1 disables execution and is unsupported",
            )
            require(
                K1_ARGS
                == (
                    "ar",
                    "g_in",
                    "w_qkv",
                    "s_qkv",
                    "g_q",
                    "g_k",
                    "g_iq",
                    "g_ik",
                    "cos_sin",
                    "index_cache",
                    "iq_out",
                    "scratch",
                ),
                "ATOM ordered K1 ABI changed",
            )
            require(config.use_v2_model_runner, "V2 model runner is required")
            require(
                (
                    parallel.tensor_parallel_size,
                    parallel.pipeline_parallel_size,
                    parallel.data_parallel_size,
                )
                == (4, 1, 1),
                "requires TP4/PP1/DP1",
            )
            require(
                parallel.decode_context_parallel_size
                == parallel.prefill_context_parallel_size
                == 1
                and not parallel.enable_expert_parallel
                and not parallel.use_ubatching,
                "context/expert parallelism and microbatching are unsupported",
            )
            require(
                config.lora_config is None
                and config.kv_transfer_config is None
                and not config.model_config.enable_sleep_mode,
                "LoRA, KV transfer and sleep are unsupported",
            )
            require(
                config.cache_config.cache_dtype == "fp8"
                and config.cache_config.block_size == 128
                and not config.cache_config.enable_prefix_caching,
                "requires FP8, block size 128 and prefix caching disabled",
            )
            spec = config.speculative_config
            require(
                spec is not None
                and spec.method == "eagle3"
                and spec.num_speculative_tokens == 3,
                "requires native EAGLE3 with three speculative tokens",
            )
            require(
                config.scheduler_config.max_num_seqs <= 4
                and config.model_config.max_model_len == 16384,
                "requires at most four sequences and max_model_len=16384",
            )
            props = torch.cuda.get_device_properties(self.device)
            require(
                props.gcnArchName.split(":")[0] == "gfx950"
                and props.multi_processor_count == 256,
                "requires gfx950 with 256 CUs",
            )
            cfg = model.config
            require(
                (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads)
                == (6144, 64, 4),
                "model dimensions",
            )
            require(
                cfg.swiglu_alpha == 1.702 and cfg.swiglu_beta == 1.0, "SwiGLU constants"
            )
            sparse = cfg.sparse_attention_config
            require(
                (
                    sparse["sparse_topk_blocks"],
                    sparse["sparse_block_size"],
                    sparse.get("sparse_init_block", 0),
                    sparse.get("sparse_local_block", 0),
                )
                == (16, 128, 0, 1),
                "sparse selection configuration",
            )
            layer_ids = [
                i for i, flag in enumerate(sparse["sparse_attention_freq"]) if flag
            ]
            require(
                layer_ids == list(range(3, len(model.layers))),
                "expected contiguous sparse layers after three dense layers",
            )
            require(len(model.layers) == 60, "expected 60 decoder layers")
            self.layer_ids = list(range(3, 60))
            names_to_group = {
                name: i
                for i, group in enumerate(kv_cache_config.kv_cache_groups)
                for name in group.layer_names
            }
            main_groups = {
                names_to_group[model.layers[i].self_attn.layer_name] for i in layer_ids
            }
            index_groups = {
                names_to_group[model.layers[i].self_attn.indexer.index_cache.prefix]
                for i in layer_ids
            }
            require(
                len(main_groups) == len(index_groups) == 1,
                "sparse main layers and index layers must each share metadata",
            )

        _collective_stage(self.group, "configuration", validate)
        self.layers = _collective_stage(
            self.group,
            "weights and caches",
            lambda: [
                MonoLayerWeights.from_layer(model.layers[i]) for i in self.layer_ids
            ],
        )
        self.width = triton.next_power_of_2(
            (config.model_config.max_model_len + 127) // 128
        )

        def allocate():
            def empty(shape, dtype):
                return torch.zeros(shape, dtype=dtype, device=self.device)

            self.ars = [empty((16, 6144), torch.bfloat16) for _ in range(2)]
            self.residuals = [empty((16, 6144), torch.bfloat16) for _ in range(2)]
            self.h = empty((16, 6144), torch.bfloat16)
            self.q = empty((16, 16, 128), torch.bfloat16)
            self.iq = empty((16, 1, 128), torch.bfloat16)
            self.main_table = empty((16, self.width), torch.int32)
            self.index_table = empty((16, self.width), torch.int32)
            self.seq_lens = empty((16,), torch.int32)
            self.slots = empty((16,), torch.int64)
            self.index_slots = empty((16,), torch.int64)
            self.step = empty((1,), torch.int32)
            self.scratch1 = empty((K1_BYTES,), torch.uint8)
            self.scratch4 = empty((SCRATCH_BYTES,), torch.uint8)
            self.cache_args = torch.tensor(
                [self.index_slots.data_ptr(), self.index_table.data_ptr(), self.width],
                dtype=torch.int64,
                device=self.device,
            )
            self.k1_args: list[torch.Tensor] = []
            self.weight_dependencies: list[list[torch.Tensor]] = []
            for i, weight in enumerate(self.layers):
                pointers = {
                    name: getattr(weight, name)
                    for name in K1_ARGS
                    if name not in ("ar", "iq_out", "scratch")
                }
                pointers.update(
                    ar=self.ars[i % 2], iq_out=self.iq, scratch=self.scratch1
                )
                self.k1_args.append(
                    torch.tensor(
                        [pointers[name].data_ptr() for name in K1_ARGS],
                        dtype=torch.int64,
                        device=self.device,
                    )
                )
                self.weight_dependencies.append(
                    [
                        getattr(weight, f.name)
                        for f in fields(weight)
                        if isinstance(getattr(weight, f.name), torch.Tensor)
                        and f.name not in ("k_cache", "v_cache", "index_cache")
                    ]
                )

        _collective_stage(self.group, "scratch allocation", allocate)
        self.peers = _PeerMemory(
            sym_layout(4)["_bytes"], self.group, self.rank, self.device
        )
        self.resources = [
            *self.ars,
            *self.residuals,
            self.h,
            self.q,
            self.iq,
            self.main_table,
            self.index_table,
            self.seq_lens,
            self.slots,
            self.index_slots,
            self.step,
            self.scratch1,
            self.scratch4,
            self.cache_args,
            *self.k1_args,
            self.peers.tensor,
            self.peers.addresses,
        ]
        cfg = model.config
        try:
            self.kernels = _collective_stage(
                self.group,
                "kernel builders",
                lambda: {
                    n: build_post_attn_kernel(
                        4,
                        128**-0.5,
                        cfg.rms_norm_eps,
                        cfg.routed_scaling_factor,
                        1.0,
                        cfg.swiglu_limit,
                        0,
                        1,
                        n,
                        fuse_k1=True,
                        cache_mode="vllm",
                    )
                    for n in SUPPORTED_TOKENS
                },
            )
            self.state = "PREPARED"
            self._warmup()
        except Exception:
            self.peers.close()
            raise
        for i, weight in enumerate(self.layers):
            weight.attention._atom_mono = (self, i)
        self.state = "READY"
        logger.info(
            "ATOM mono ready: ABI=1, sparse layers=%s, buckets=%s, "
            "extra weights=%.2f MiB",
            self.layer_ids,
            SUPPORTED_TOKENS,
            sum(w.extra_weight_bytes for w in self.layers) / 2**20,
        )

    def close(self):
        """Called by the runner after destroying target graphs, before TP teardown."""
        if self.state == "CLOSED":
            return
        self.state = "CLOSED"
        torch.accelerator.synchronize(self.device)
        dist.barrier(group=self.group)
        for weight in self.layers:
            del weight.attention._atom_mono
        self.resources.clear()
        self.peers.close()
        self.layers.clear()

    def _warmup(self):
        positions = torch.zeros(16, dtype=torch.int64, device=self.device)
        residual = torch.zeros_like(self.h)
        self.slots.fill_(-1)
        self.index_slots.fill_(-1)
        self.seq_lens.fill_(1)
        for weight in self.layers:
            weight.attention.kv_cache[0].zero_()
            weight.index_cache[0].zero_()

        def compile_only():
            previous = os.environ.get("COMPILE_ONLY")
            os.environ["COMPILE_ONLY"] = "1"
            try:
                for n in SUPPORTED_TOKENS:
                    self.launch(0, residual, positions, n, 1 if n == 1 else 4)
            finally:
                if previous is None:
                    os.environ.pop("COMPILE_ONLY", None)
                else:
                    os.environ["COMPILE_ONLY"] = previous

        _collective_stage(self.group, "kernel compilation", compile_only)

        # Metadata specializes on the padded request count and query length.
        # Compile every serving specialization before CUDA graph capture.
        def warmup_metadata():
            lengths = torch.ones(4, dtype=torch.int32, device=self.device)
            for rows in (1, 2, 4):
                for query_len in (1, 4):
                    _token_rows[(16,)](
                        self.main_table,
                        self.index_table,
                        lengths,
                        self.slots,
                        self.index_slots,
                        self.main_table,
                        self.index_table,
                        self.seq_lens,
                        self.slots,
                        self.index_slots,
                        self.width,
                        self.width,
                        self.width,
                        self.width,
                        rows,
                        query_len,
                        self.width,
                    )
            torch.accelerator.synchronize()
            self.seq_lens.fill_(1)

        _collective_stage(self.group, "metadata compilation", warmup_metadata)
        for n in SUPPORTED_TOKENS:
            self.step.add_(1)
            for i in range(len(self.layers)):
                self.launch(
                    i,
                    residual if i == 0 else self.residuals[i % 2],
                    positions,
                    n,
                    1 if n == 1 else 4,
                )
            torch.accelerator.synchronize()
        _collective_stage(
            self.group,
            "kernel warmup",
            lambda: require(
                bool(torch.isfinite(self.ars[0]).all())
                and bool(torch.isfinite(self.ars[1]).all()),
                "nonfinite warmup output",
            ),
        )
        # Keep epochs monotonic: unused mailbox lanes can retain warmup tags.
        self.warmup_steps = len(SUPPORTED_TOKENS)

    def begin_forward(self, token_count):
        context = get_forward_context()
        reason = None
        if token_count not in SUPPORTED_TOKENS:
            reason = "token_count"
        elif not isinstance(context.attn_metadata, dict):
            reason = "no_metadata"
        else:
            first = self.layers[0].attention
            main = cast(
                MiniMaxM3SparseMetadata, context.attn_metadata[first.layer_name]
            )
            index = cast(
                MiniMaxM3IndexerAiterMetadata,
                context.attn_metadata[first.indexer.index_cache.prefix],
            )
            if (
                main.num_prefills
                or index.num_prefills
                or main.decode is None
                or index.decode is None
            ):
                reason = "prefill_or_mixed"
            elif (
                main.decode.decode_query_len not in (1, 4)
                or token_count % main.decode.decode_query_len
            ):
                reason = "query_shape"
        if reason is not None:
            self.fallback_counts[reason] = self.fallback_counts.get(reason, 0) + 1
            return False
        assert main.decode is not None and main.page16_slot_mapping is not None
        self.query_len = main.decode.decode_query_len
        decode = cast(MiniMaxM3SparseAiterPADecodeMetadata, main.decode)
        assert decode.page16_block_table is not None
        assert index.decode is not None
        assert context.slot_mapping is not None
        self.step.add_(1)
        _token_rows[(token_count,)](
            decode.page16_block_table,
            index.decode.block_table,
            main.decode.seq_lens,
            main.page16_slot_mapping,
            context.slot_mapping[first.indexer.index_cache.prefix],
            self.main_table,
            self.index_table,
            self.seq_lens,
            self.slots,
            self.index_slots,
            decode.page16_block_table.stride(0),
            index.decode.block_table.stride(0),
            decode.page16_block_table.shape[1],
            index.decode.block_table.shape[1],
            main.decode.seq_lens.numel(),
            self.query_len,
            self.width,
        )
        return True

    def forward_layer(self, index, hidden, residual, positions):
        n = hidden.shape[0]
        if index == 0:
            self.ars[0][:n].copy_(hidden)
        output, residual_out = (
            self.ars[(index + 1) % 2][:n],
            self.residuals[(index + 1) % 2][:n],
        )
        weight = self.layers[index]
        torch.ops.vllm.minimax_m3_atom_mono(
            hidden,
            residual,
            positions,
            output,
            residual_out,
            weight.attention.kv_cache,
            weight.index_cache,
            self.weight_dependencies[index],
            self.resources,
            weight.attention.layer_name,
            n,
            self.query_len,
        )
        return output, residual_out

    def launch(self, index, residual, positions, n, query_len):
        weight = self.layers[index]
        tensors = [
            self.h,
            self.q,
            self.main_table,
            self.seq_lens,
            weight.k_cache,
            weight.v_cache,
            weight.k_scale,
            weight.v_scale,
            weight.w_o,
            weight.s_o,
            weight.g_post,
            weight.gate,
            weight.bias,
            weight.w13,
            weight.s13,
            weight.w2,
            weight.s2,
            self.residuals[(index + 1) % 2],
            self.ars[(index + 1) % 2],
            self.scratch4,
        ]
        self.kernels[n](
            *[t.data_ptr() for t in tensors],
            self.peers.local,
            self.peers.addresses.data_ptr(),
            self.step.data_ptr(),
            self.rank,
            weight.layer_id,
            self.width,
            query_len,
            0,
            self.k1_args[index].data_ptr(),
            positions.data_ptr(),
            self.slots.data_ptr(),
            residual.data_ptr(),
            stream=torch.cuda.current_stream(self.device),
            cache_args=self.cache_args.data_ptr(),
        )


def prepare_model(model, config, kv_cache_config):
    from vllm.models.minimax_m3.amd.model import MiniMaxM3Model

    group = get_tp_group().cpu_group

    def target():
        language_model = (
            model.get_language_model()
            if hasattr(model, "get_language_model")
            else model
        )
        require(
            isinstance(language_model.model, MiniMaxM3Model), "unsupported model class"
        )
        require(
            language_model.model._mono is None,
            "cache reinitialization is unsupported while graphs reference "
            "mono resources",
        )
        return language_model.model

    target_model = _collective_stage(group, "target model", target)
    target_model._mono = M3Mono(target_model, config, kv_cache_config)


def release_model(model):
    """Release mono-owned IPC only after the runner has drained its graphs."""
    language_model = (
        model.get_language_model() if hasattr(model, "get_language_model") else model
    )
    target = getattr(language_model, "model", None)
    adapter = getattr(target, "_mono", None)
    if target is not None and adapter is not None:
        adapter.close()
        target._mono = None
