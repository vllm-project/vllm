# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash frozen n-gram embeddings with TP head sharding and V2 lookback."""

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import NgramHashState
from vllm.models.qwen4_exp.common.ngram_embedding import (
    _lookup_ple_embedding_from_pinned_kernel,
)
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class FlashEngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids or ())
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_head_per_ngram
        self.head_dim = config.engram_n_embed_per_ngram // self.n_heads
        self.pad_token_id = config.engram_pad_id
        self.vocab_size = config.vocab_size
        self.seed = config.engram_seed
        self.lookup_table = None
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("engram_pad_id is outside the vocabulary")
        groups = config.engram_layer_head_vocab_sizes
        self.primes = tuple(
            groups.get(str(layer), groups.get(layer)) for layer in self.layer_ids
        )
        flat = [[p for order in group for p in order] for group in self.primes]
        self.offsets = np.array(
            [np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64
        )
        if any(sum(sizes) > np.iinfo(np.int32).max for sizes in flat):
            raise ValueError("Engram flattened bucket offsets must fit int32")

    def load_lookup(self, value):
        if (
            value.dtype not in (torch.int32, torch.int64)
            or value.ndim != 1
            or not 0 < value.numel() <= self.vocab_size
        ):
            raise ValueError("Invalid frozen Engram token map")
        lookup = value.detach().cpu().numpy()
        if np.any(lookup < 0) or np.any(lookup > np.iinfo(np.int32).max):
            raise ValueError("Invalid frozen Engram token map")
        if self.lookup_table is not None:
            if not np.array_equal(self.lookup_table, lookup):
                raise ValueError("Engram layers must share the same token map")
            return False
        self.compressed_vocab_size = int(lookup.max()) + 1
        self.lookup_table = lookup.copy()
        self.token_map = lookup.tolist()
        self.token_map.extend(range(len(self.token_map), self.vocab_size))
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        self.multipliers = np.stack(
            [
                np.random.default_rng(self.seed + 10007 * layer).integers(
                    0, bound, size=self.max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
                for layer in self.layer_ids
            ]
        )

        return True


def head_range(sizes, tp, rank):
    if not 1 <= tp <= len(sizes) or not 0 <= rank < tp:
        raise ValueError("Engram requires at least one complete hash head per TP rank")
    prefix = np.cumsum([0, *sizes]).tolist()
    boundaries = [0]
    for r in range(1, tp):
        target = prefix[-1] * r / tp
        candidates = range(boundaries[-1] + 1, len(sizes) - (tp - r) + 1)
        boundaries.append(min(candidates, key=lambda i: abs(prefix[i] - target)))
    boundaries.append(len(sizes))
    return boundaries[rank], boundaries[rank + 1]


class FlashNgramHashState(NgramHashState):
    def __init__(self, config, *, dummy=False):
        nn.Module.__init__(self)
        layout = FlashEngramLayout(config)
        self.layout: FlashEngramLayout = layout  # type: ignore[assignment]
        self.lookback_depth = self.layout.max_ngram_size - 1
        self.block_size = 1
        self.use_slot_cache = False
        self._cache = None
        self.pad_id = 0
        self.visual_token_ids: tuple[int, ...] = ()
        self.register_buffer(
            "token_map",
            torch.zeros(config.vocab_size, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            torch.zeros(
                len(self.layout.layer_ids),
                self.layout.max_ngram_size,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        for name, values in (
            ("primes", self.layout.primes),
            ("offsets", self.layout.offsets),
        ):
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.int64), persistent=False
            )
        if dummy:
            self.load_lookup(torch.arange(config.vocab_size, device="cpu"))

    def load_lookup(self, value):
        if self.layout.load_lookup(value):
            self.token_map.copy_(
                torch.as_tensor(
                    self.layout.token_map,
                    dtype=torch.int32,
                    device=self.token_map.device,
                )
            )
            self.multipliers.copy_(
                torch.as_tensor(self.layout.multipliers, device=self.multipliers.device)
            )
            self.pad_id = self.layout.token_map[self.layout.pad_token_id]

    @eager_break_during_capture
    def prepare_hashes_with_output(self, input_ids, lookback, layer_name, output):
        metadata = get_forward_context().attn_metadata
        output.copy_(
            self.prepare_hashes(
                input_ids,
                lookback,
                metadata[layer_name] if isinstance(metadata, dict) else None,
            )
        )

    def prepare_hashes(self, input_ids, lookback, metadata):
        tokens = input_ids.numel()
        if (
            metadata is None
            or tokens == 0
            or metadata.num_actual_tokens == 0
            or metadata.seq_lens.numel() == 0
        ):
            return input_ids.new_full(
                (len(self.layout.layer_ids), tokens, self.offsets.shape[1]),
                -1,
                dtype=torch.int32,
            )
        if self.layout.lookup_table is None:
            raise ValueError("Missing checkpoint Engram token map")
        if lookback is None or lookback.shape[1] != self.lookback_depth:
            raise ValueError("Flash Engram requires V2 lookback_token_ids")
        starts = metadata.query_start_loc
        rows = torch.arange(tokens, device=input_ids.device)
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = (
            metadata.seq_lens[requests]
            - (starts[requests + 1] - starts[requests])
            + rows
            - starts[requests]
        ).long()
        dead = (input_ids < 0) | (input_ids >= self.token_map.numel())
        safe_ids = input_ids.clamp(0, self.token_map.numel() - 1)
        lookback_dead = lookback >= self.token_map.numel()
        safe_lookback = lookback.clamp(-1, self.token_map.numel() - 1)
        hashes = super().forward(
            safe_ids,
            logical,
            starts,
            dead,
            safe_lookback,
            lookback_dead,
        )
        offsets = self.offsets.unsqueeze(0)
        primes = self.primes.flatten(1).unsqueeze(0)
        hashes = torch.where(hashes < offsets, hashes + primes, hashes).int()
        no_hash = dead | (rows >= metadata.num_actual_tokens) | (rows >= starts[-1])
        no_hash |= input_ids == self.layout.pad_token_id
        for token_id in self.visual_token_ids:
            no_hash |= input_ids == token_id
        hashes.masked_fill_(no_hash[:, None, None], -1)
        return hashes.permute(1, 0, 2).contiguous()


class FlashEmbedding(nn.Module):
    def __init__(self, sizes, dim, dtype, cpu_offload):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_start, self.head_end = head_range(
            sizes, self.tp_size, get_tensor_model_parallel_rank()
        )
        self.vocab_start = sum(sizes[: self.head_start])
        self.vocab_end = sum(sizes[: self.head_end])
        self.num_embeddings = sum(sizes)
        self.n_hash_cols = len(sizes)
        self.dim = dim
        self.cpu_offload = cpu_offload
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Flash Engram CPU offload requires UVA")
        rows = self.vocab_end - self.vocab_start
        storage = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        self.embedding = nn.Module()
        self.embedding.weight = nn.Parameter(
            torch.empty(
                (0,) if dtype is None else (rows, dim),
                dtype=dtype or torch.bfloat16,
                **storage,
            ),
            requires_grad=False,
        )
        self.fp8 = dtype == torch.float8_e4m3fn
        self.embedding.register_parameter("weight_scale_inv", None)
        if self.fp8:
            self.embedding.weight_scale_inv = nn.Parameter(
                torch.empty(rows, dtype=torch.bfloat16, **storage), requires_grad=False
            )
        for parameter in self.embedding.parameters():
            set_weight_attrs(parameter, {"weight_loader": self._load_shard})
        self._uva_weight = None
        self._uva_scale = None
        logger.info(
            "Flash Engram TP table: %d rows x %d, dtype=%s, cpu_offload=%s",
            rows,
            dim,
            dtype,
            cpu_offload,
        )

    def load_weight(self, leaf, value):
        if leaf == "weight_scale_inv" and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                self.embedding.weight.new_empty(0, dtype=torch.bfloat16),
                requires_grad=False,
            )
            set_weight_attrs(
                self.embedding.weight_scale_inv, {"weight_loader": self._load_shard}
            )
        self._load_shard(getattr(self.embedding, leaf), value)

    def _load_shard(self, parameter, value):
        is_weight = parameter is self.embedding.weight
        expected = (
            (self.num_embeddings, self.dim) if is_weight else (self.num_embeddings,)
        )
        dtypes = (
            (torch.bfloat16, torch.float8_e4m3fn) if is_weight else (torch.bfloat16,)
        )
        if tuple(value.shape) != expected or value.dtype not in dtypes:
            raise ValueError(
                f"Invalid Engram table: {tuple(value.shape)}/{value.dtype}"
            )
        if parameter.numel() and parameter.dtype != value.dtype:
            raise ValueError("Engram table dtype changed during weight reload")
        if is_weight:
            if (
                value.dtype != torch.float8_e4m3fn
                and self.embedding.weight_scale_inv is not None
            ):
                raise ValueError("BF16 Engram table must not have FP8 row scales")
        elif self.embedding.weight.numel() and not self.fp8:
            raise ValueError("BF16 Engram table must not have FP8 row scales")
        if parameter.numel() == 0:
            parameter.data = torch.empty(
                (self.vocab_end - self.vocab_start, *expected[1:]),
                dtype=value.dtype,
                device=parameter.device,
                pin_memory=self.cpu_offload,
            )
        default_weight_loader(parameter, value[self.vocab_start : self.vocab_end])
        if is_weight:
            self.fp8 = value.dtype == torch.float8_e4m3fn
        if is_weight and self.fp8 and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                parameter.new_empty(0, dtype=torch.bfloat16), requires_grad=False
            )
            set_weight_attrs(
                self.embedding.weight_scale_inv, {"weight_loader": self._load_shard}
            )
        self._uva_weight = None
        self._uva_scale = None

    def _lookup(self, parameter, indices, out, *, scale=False):
        view_name = "_uva_scale" if scale else "_uva_weight"
        view = getattr(self, view_name)
        if view is None:
            view = (
                get_accelerator_view_from_cpu_tensor(parameter)
                if self.cpu_offload
                else parameter
            )
            setattr(self, view_name, view)
        row_bytes = parameter.element_size() * (1 if scale else self.dim)
        _lookup_ple_embedding_from_pinned_kernel[(indices.numel(),)](
            view.view(torch.uint8),
            indices,
            out.view(torch.uint8),
            row_bytes,
            self.vocab_start,
            self.vocab_end,
            BLOCK_D=triton.next_power_of_2(row_bytes),
        )

    def forward(self, hashes):
        indices = hashes[:, self.head_start : self.head_end].long().contiguous()
        shape = (*indices.shape, self.dim)
        rows = torch.empty(
            shape, dtype=self.embedding.weight.dtype, device=hashes.device
        )
        if indices.numel() == 0:
            return rows.float()
        self._lookup(self.embedding.weight, indices, rows)
        if self.fp8:
            scales = torch.empty(
                indices.shape, dtype=torch.bfloat16, device=hashes.device
            )
            self._lookup(self.embedding.weight_scale_inv, indices, scales, scale=True)
            return rows.float() * scales.float().unsqueeze(-1)
        return rows.float()


class FlashEngram(nn.Module):
    def __init__(self, vllm_config, layout, layer_id, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        index = layout.layer_ids.index(layer_id)
        sizes = [p for order in layout.primes[index] for p in order]
        storage = vllm_config.engram_config
        if (
            storage.disk_offload
            or storage.embedding_across_dp
            or storage.dp_shared_memory
        ):
            raise ValueError(
                "Flash Engram requires private DP replicas with TP-local tables; "
                "DP sharing, cross-DP sharding and disk offload are unsupported"
            )
        dtype = None
        if vllm_config.load_config.load_format == "dummy":
            dtype = (
                torch.float8_e4m3fn if config.engram_fp8_embedding else torch.bfloat16
            )
        self.multi_head_embedding = FlashEmbedding(
            sizes, layout.head_dim, dtype, storage.cpu_offload
        )
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        local = (
            self.multi_head_embedding.head_end - self.multi_head_embedding.head_start
        )
        self.lookup_per_head_norms = nn.ModuleList(
            [nn.RMSNorm(layout.head_dim) for _ in range(local)]
        )
        self.value_proj = nn.Linear(local * layout.head_dim, self.hidden_size)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(local * layout.head_dim, self.hidden_size)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm_key = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.norm_query = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_size)) for _ in range(self.hc_mult)]
        )
        self.total_width = len(sizes) * layout.head_dim
        self.start = self.multi_head_embedding.head_start * layout.head_dim
        self.stop = self.multi_head_embedding.head_end * layout.head_dim
        self.rank = get_tensor_model_parallel_rank()
        for projection in [self.value_proj, *self.key_projs]:
            set_weight_attrs(
                projection.weight, {"weight_loader": self._load_projection}
            )
            set_weight_attrs(projection.bias, {"weight_loader": self._load_bias})

    def _load_projection(self, parameter, value):
        if value.shape != (self.hidden_size, self.total_width):
            raise ValueError("Flash Engram projection weight shape mismatch")
        default_weight_loader(parameter, value[:, self.start : self.stop])

    def _load_bias(self, parameter, value):
        default_weight_loader(
            parameter, value if self.rank == 0 else torch.zeros_like(value)
        )

    def forward(self, hidden, hashes):
        rows = self.multi_head_embedding(hashes)
        eps = torch.finfo(torch.float32).eps
        gamma = torch.stack(
            [norm.weight for norm in self.lookup_per_head_norms]
        ).float()
        local = self.multi_head_embedding
        local_hashes = hashes[:, local.head_start : local.head_end]
        x = rows.masked_fill(local_hashes.unsqueeze(-1) < 0, 0)
        x = (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma).to(
            hidden.dtype
        )
        x = x.flatten(1)
        gates = []
        for channel in range(self.hc_mult):
            key = self.key_projs[channel](x)
            if local.tp_size > 1:
                key = tensor_model_parallel_all_reduce(key)
            gate = (
                self.norm_key[channel](key)
                * self.norm_query[channel](hidden[:, channel])
            ).sum(-1)
            gate = gate * self.hidden_size**-0.5
            gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
        gate = torch.stack(gates, dim=1)
        value = self.value_proj(x)
        if local.tp_size > 1:
            value = tensor_model_parallel_all_reduce(value)
        value = value.float()
        signal = (
            self.unified_pre_gamma.float()
            * value
            * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
        )
        scales = torch.stack(list(self.unified_per_hc_scales)).float()
        result = ((gate.float().unsqueeze(-1) + scales) * signal.unsqueeze(1)).to(
            hidden.dtype
        )
        return hidden + result.masked_fill((hashes < 0).all(-1)[:, None, None], 0)
