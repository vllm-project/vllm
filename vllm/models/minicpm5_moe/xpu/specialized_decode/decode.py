# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned mono plan, cache binding, and fail-stop lifecycle."""

from collections import Counter

import torch
import torch.distributed as dist

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from .config import MAX_VERIFY_ROWS, native_rotary_cache_version, native_rows, serves
from .guards import FailStop
from .peer import agree, region
from .pre_o import (
    PreOFa2SplitOne,
    attention_before_output_projection,
    rank_local_output_projection,
)
from .weights import cache_tensor, layer_weights, tiled_qkv_supported, tiled_qkv_weight

logger = init_logger(__name__)


class MonoLayer:
    def __init__(self, layer, index, config, scratch, mailboxes, ops):
        self.layer, self.index, self.config = layer, index, config
        self.weights = layer_weights(layer)
        if tiled_qkv_supported(ops, config.tp, self.weights[0].dtype):
            self.weights.append(tiled_qkv_weight(self.weights[2]))
        self.scratch, self.region, self.ops = scratch, mailboxes, ops
        self.dense = not hasattr(layer.mlp, "experts")
        self.tail_decode = None
        if getattr(config, "tail_only", False) and not self.dense:
            version = getattr(ops, "tail_only_abi_version", None)
            if version is not None and version() == 1:
                self.tail_decode = getattr(ops, "decode_tail", None)
        self.tail_rows4_decode = None
        if getattr(config, "tail_rows4", False) and not self.dense:
            version = getattr(ops, "tail_rows4_abi_version", None)
            if version is not None and version() == 1:
                self.tail_rows4_decode = getattr(ops, "decode_tail_rows4", None)
        self.pre_o_decode = None
        if getattr(config, "pre_o_split", False) and not self.dense:
            version = getattr(ops, "pre_o_abi_version", None)
            if version is not None and version() == 1:
                self.pre_o_decode = getattr(ops, "decode_pre_o", None)
        self.post_o_decode = None
        if getattr(config, "post_o_split", False) and not self.dense:
            projection = layer.self_attn.o_proj
            version = getattr(ops, "post_o_abi_version", None)
            if (
                version is not None
                and version() == 1
                and projection.input_is_parallel
                and projection.bias is None
                and projection.tp_size == 2
            ):
                self.post_o_decode = getattr(ops, "decode_post_o", None)
        self.pre_o_fa2 = None
        adaptive = getattr(config, "pre_o_fa2_adaptive_split1", False)
        if adaptive and (
            self.pre_o_decode is not None or self.post_o_decode is not None
        ):
            impl = layer.self_attn.attn.impl
            if getattr(impl, "vllm_flash_attn_version", None) == 2:
                self.pre_o_fa2 = PreOFa2SplitOne(impl.dense_attention_kernel)
                impl.dense_attention_kernel = self.pre_o_fa2
        self.prefix = layer.self_attn.attn.layer_name
        self.control = torch.zeros(
            ops.control_words(), dtype=torch.int32, device=self.weights[0].device
        )
        self.rope_cache = None
        self.verify = None
        self.key = None
        self.hits: Counter[int] = Counter()
        self.tail_hits: Counter[int] = Counter()
        self.pre_o_hits: Counter[int] = Counter()
        self.post_o_hits: Counter[int] = Counter()
        padding_tps = getattr(ops, "padding_mask_supported_tps", lambda: (2,))()
        self.padding_rows = (
            frozenset(ops.padding_mask_rows())
            if config.tp in padding_tps
            and hasattr(ops, "padding_mask_version")
            and ops.padding_mask_version() == 1
            and hasattr(ops, "padding_mask_rows")
            else frozenset()
        )

    def bind(self):
        cache = cache_tensor(self.layer)
        self.ops.validate(self.weights, cache, self.scratch, self.config.tp, self.dense)
        limit = getattr(self.config, "max_context_len", 32768)
        if limit > 32768:
            if native_rotary_cache_version(self.ops) != 1:
                raise RuntimeError(
                    "MiniCPM5 mono long context requires canonical RoPE cache"
                )
            rope = self.layer.self_attn.rotary_emb._match_cos_sin_cache_dtype(
                self.weights[0]
            )
            if (
                rope.device != self.weights[0].device
                or rope.dtype != self.weights[0].dtype
                or rope.ndim != 2
                or rope.shape[0] < limit
                or rope.shape[1] != 128
                or not rope.is_contiguous()
            ):
                raise RuntimeError(
                    "MiniCPM5 mono requires canonical RoPE cache [N,128]"
                )
            self.rope_cache = rope
        # The attention layer owns the cache. Keep only the validated identity
        # so graph profiling cleanup can release the temporary storage.
        self.key = self.cache_key(cache)
        if getattr(self.config, "speculative_tokens", 0):
            from .metadata import VerifyMetadata

            pages_version = getattr(self.ops, "shared_verify_pages_version", None)
            pages_tps = getattr(
                self.ops, "shared_verify_pages_supported_tps", lambda: (2,)
            )()
            self.verify = VerifyMetadata(
                limit,
                cache.shape[2],
                cache.device,
                shared_pages=(
                    self.config.tp in pages_tps
                    and cache.dtype == torch.float16
                    and pages_version is not None
                    and pages_version() == 1
                ),
            )

    @staticmethod
    def cache_key(cache):
        return cache.data_ptr(), tuple(cache.shape), cache.stride()

    @property
    def cache(self) -> torch.Tensor | None:
        if self.key is None:
            return None
        return cache_tensor(self.layer)

    def launch(self, metadata, positions, hidden, residual):
        cache = cache_tensor(self.layer)
        if self.key is None or self.cache_key(cache) != self.key:
            raise RuntimeError("MiniCPM5 mono cache moved without lifecycle rebind")
        if self.tail_rows4_decode is not None and self._supports_split(
            metadata, positions, hidden, cache, 4
        ):
            return self._launch_tail(
                self.tail_rows4_decode, metadata, positions, hidden, residual
            )
        if any(
            op is not None
            for op in (self.tail_decode, self.pre_o_decode, self.post_o_decode)
        ) and self._supports_split(metadata, positions, hidden, cache, 8):
            if self.tail_decode is not None:
                return self._launch_tail(
                    self.tail_decode, metadata, positions, hidden, residual
                )
            return self._launch_o_split(metadata, positions, hidden, residual)
        out, res = torch.empty_like(hidden), torch.empty_like(hidden)
        arguments: tuple[torch.Tensor | None, ...] = (
            hidden,
            residual,
            positions,
            metadata.block_table,
            metadata.seq_lens,
            metadata.slot_mapping,
            out,
            res,
        )
        for start in range(0, positions.numel(), 8):
            end = min(start + 8, positions.numel())
            rows = end - start
            chunk = arguments
            if positions.numel() > 8:
                chunk = tuple(
                    None if tensor is None else tensor[start:end]
                    for tensor in arguments
                )
            # All chunks share the layer's ordered workspace and IPC epoch.
            kwargs = {"ignore_padding": True} if rows in self.padding_rows else {}
            if self.rope_cache is not None:
                kwargs["rope_cache"] = self.rope_cache
            self.ops.decode(
                *chunk,
                self.scratch,
                self.control,
                self.weights,
                cache,
                self.region,
                self.index,
                self.config.tp,
                self.dense,
                **kwargs,
            )
            self.hits[rows] += 1
        return out, res

    def _supports_split(self, metadata, positions, hidden, cache, rows):
        table = metadata.block_table
        slots = metadata.slot_mapping
        return (
            positions.numel() == rows
            and rows in self.padding_rows
            and cache.dtype == hidden.dtype == torch.float16
            and cache.shape[2] == 64
            and getattr(metadata, "max_query_len", 0) == 1
            and not getattr(metadata, "minicpm5_verify_eligible", False)
            and table.ndim == 2
            and table.shape[0] == rows
            and table.stride(0) >= table.shape[1]
            and table.stride(1) == 1
            and slots.shape == (rows,)
            and slots.dtype == torch.int64
            and slots.is_contiguous()
        )

    def _split_input_norm(self, hidden, residual):
        norm = self.layer.input_layernorm
        if residual is None:
            return norm(hidden), hidden
        return norm(hidden, residual)

    def _launch_tail(self, decode, metadata, positions, hidden, residual):
        layer = self.layer
        hidden, residual = self._split_input_norm(hidden, residual)
        hidden = layer.self_attn(positions, hidden)
        normed, residual = layer.post_attention_layernorm(hidden, residual)
        output = torch.empty_like(normed)
        decode(
            normed,
            metadata.slot_mapping,
            output,
            self.scratch,
            self.control,
            self.weights,
            self.region,
            self.index,
        )
        rows = normed.shape[0]
        self.hits[rows] += 1
        self.tail_hits[rows] += 1
        return output, residual

    def _launch_o_split(self, metadata, positions, hidden, residual):
        hidden, residual = self._split_input_norm(hidden, residual)
        attention_layer = self.layer.self_attn
        attention = attention_before_output_projection(
            attention_layer, positions, hidden, self.pre_o_fa2
        )
        decode, hits = self.pre_o_decode, self.pre_o_hits
        if self.post_o_decode is not None:
            attention = rank_local_output_projection(attention_layer.o_proj, attention)
            decode, hits = self.post_o_decode, self.post_o_hits
        assert decode is not None
        output, new_residual = (
            hidden.new_empty(hidden.shape),
            hidden.new_empty(hidden.shape),
        )
        decode(
            attention.contiguous(),
            residual.contiguous(),
            metadata.slot_mapping,
            output,
            new_residual,
            self.scratch,
            self.control,
            self.weights,
            self.region,
            self.index,
        )
        self.hits[8] += 1
        hits[8] += 1
        return output, new_residual


class SpecializedDecode:
    def __init__(self, model, config, draft_slots=0):
        self.model, self.config = model, config
        self.tp = get_tp_group()
        reasons = []
        self.layers = {}
        try:
            from vllm_xpu_kernels.minicpm5_mono import load, workspace

            self.ops = load()
            self.rows = native_rows(self.ops)
            self.stream = torch.xpu.current_stream()
            parameter = next(model.parameters())
            self.scratch = workspace(config.tp, parameter.device, parameter.dtype)
            for i in range(model.start_layer, model.end_layer):
                self.layers[i] = MonoLayer(
                    model.layers[i], i, config, self.scratch, None, self.ops
                )
            self.host_health = torch.zeros(
                len(self.layers), dtype=torch.int32, pin_memory=True
            )
            self.health_controls = [layer.control for layer in self.layers.values()]
        except (AttributeError, ImportError, RuntimeError) as error:
            reasons.append(str(error))
        agree(reasons, self.tp.cpu_group, "loaded weights/workspace")
        draft_layout = getattr(self.ops, "draft_weight_layout_version", None)
        self.draft_slots = (
            draft_slots if draft_layout is not None and draft_layout() == 1 else 0
        )
        self.region = region(len(self.layers) + self.draft_slots, self.tp, self.ops)
        self.failstop = (
            FailStop(
                self.region.health(),
                self.tp.rank_in_group,
                config.tp,
                self.tp.cpu_group,
            )
            if hasattr(self.ops, "runtime_variant")
            and self.ops.runtime_variant()
            == "reference-shared-health-parallel-page-validation"
            else None
        )
        for layer in self.layers.values():
            layer.region = self.region
        self.bound = False
        self.steps = 0
        self.copy_event = None
        self.failed = False

    def attach(self, model):
        """Install the optional XPU forward on the loaded target instance."""
        fallback = model.forward

        def forward(
            input_ids, positions, intermediate_tensors=None, inputs_embeds=None
        ):
            metadata = self.plan(positions, intermediate_tensors)
            if metadata is None:
                return fallback(
                    input_ids, positions, intermediate_tensors, inputs_embeds
                )

            hidden_states = (
                model.embed_input_ids(input_ids)
                if inputs_embeds is None
                else inputs_embeds
            )
            residual = None
            aux_hidden_states: list[torch.Tensor] = []
            model._maybe_add_hidden_state(
                aux_hidden_states, model.start_layer, hidden_states, residual
            )
            for index, layer in self.layers.items():
                hidden_states, residual = layer.launch(
                    metadata[layer.prefix], positions, hidden_states, residual
                )
                model._maybe_add_hidden_state(
                    aux_hidden_states, index + 1, hidden_states, residual
                )
            hidden_states, _ = model.norm(hidden_states, residual)
            if aux_hidden_states:
                return hidden_states, aux_hidden_states
            return hidden_states

        model.specialized_decode = self
        model.forward = forward

    def bind_caches(self):
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError("MiniCPM5 mono cache binding inside graph capture")
        self.bound = False
        reasons = []
        for i, layer in self.layers.items():
            try:
                layer.bind()
            except (RuntimeError, ValueError) as error:
                reasons.append(f"layer{i}: {error}")
        agree(reasons, self.tp.cpu_group, "cache/weight layout")
        self.bound = True
        logger.info(
            "MiniCPM5 mono: bound %d layers for TP%d", len(self.layers), self.config.tp
        )

    def plan(self, positions, intermediate_tensors):
        rows = positions.numel()
        chunked_verify = (
            8 < rows <= MAX_VERIFY_ROWS
            and 8 in self.rows
            and (rows % 8 == 0 or rows % 8 in self.rows)
            and getattr(getattr(self, "config", None), "speculative_tokens", 0) > 0
        )
        if (
            not self.bound
            or (rows not in self.rows and not chunked_verify)
            or intermediate_tensors is not None
            or not is_forward_context_available()
        ):
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        self.use_current_stream()
        metadata = context.attn_metadata
        if not isinstance(metadata, dict):
            return None
        seen = set()
        verify = {}
        for layer in self.layers.values():
            item = metadata.get(layer.prefix)
            key = id(item)
            if key not in seen:
                if not serves(
                    item, rows, getattr(self.config, "max_context_len", 32768)
                ):
                    return None
                if getattr(item, "max_query_len", 1) > 1:
                    if layer.verify is None or not layer.verify.supports(item):
                        return None
                    verify[key] = (layer.verify, item)
                seen.add(key)
        if verify:
            expanded = {
                key: buffers.expand(item, positions)
                for key, (buffers, item) in verify.items()
            }
            return {
                name: expanded.get(id(item), item) for name, item in metadata.items()
            }
        return metadata

    def use_current_stream(self):
        current = torch.xpu.current_stream()
        if current != self.stream:
            if not self.config.graph or torch.xpu.is_current_stream_capturing():
                raise RuntimeError("MiniCPM5 mono executor used on a different stream")
            # MRV2 warms each capture stream before capture. Its serialized
            # execution can transfer this shared workspace with a stream wait.
            current.wait_stream(self.stream)
            self.region.rebind()
            self.stream = current

    def after_step(self):
        if self.failstop is not None:
            self.failstop.after_step()
            return
        if torch.xpu.is_current_stream_capturing():
            return
        if self.copy_event is None or self.copy_event.query():
            if self.copy_event is not None:
                self.failed |= bool(self.host_health.any())
            offset = self.ops.health_offset()
            for i, control in enumerate(self.health_controls):
                self.host_health[i : i + 1].copy_(
                    control[offset : offset + 1], non_blocking=True
                )
            self.copy_event = torch.xpu.Event()
            self.copy_event.record()
        self.steps += 1
        if self.steps % 16:
            return
        failed = torch.tensor([int(self.failed)], dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.tp.cpu_group)
        if failed.item():
            raise RuntimeError(
                "MiniCPM5 mono detected a failed device or peer wait; stopping engine"
            )

    def add_health_control(self, control):
        if torch.xpu.is_current_stream_capturing() or self.copy_event is not None:
            raise RuntimeError("MiniCPM health state must bind before execution")
        self.health_controls.append(control)
        self.host_health = torch.zeros(
            len(self.health_controls), dtype=torch.int32, pin_memory=True
        )
