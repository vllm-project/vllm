# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash frozen n-gram embeddings with TP head sharding and V2 lookback."""

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_engram_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import DPSharedEngramStorage
from vllm.models.hy_v4_flash.common.engram_ops import (
    gated_update,
    lookup_norm,
    ngram_hashes,
)
from vllm.platforms import current_platform
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
        pad = self.token_map[self.pad_token_id]
        self.token_map.extend([pad] * (self.vocab_size - len(self.token_map)))
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


class FlashNgramHashState(nn.Module):
    """Global Engram buckets of the n-grams ending at each token."""

    def __init__(self, config, *, dummy=False):
        super().__init__()
        self.layout = FlashEngramLayout(config)
        self.lookback_depth = self.layout.max_ngram_size - 1
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
        self.register_buffer(
            "primes",
            torch.tensor(self.layout.primes, dtype=torch.int64).flatten(1),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            torch.tensor(self.layout.offsets, dtype=torch.int64),
            persistent=False,
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
        self.prepare_hashes(
            input_ids,
            lookback,
            metadata[layer_name] if isinstance(metadata, dict) else None,
            output,
        )

    def prepare_hashes(self, input_ids, lookback, metadata, output):
        """Fill `output` (`[layers, T, columns]`) with each token's buckets."""
        if (
            metadata is None
            or input_ids.numel() == 0
            or metadata.num_actual_tokens == 0
            or metadata.seq_lens.numel() == 0
        ):
            output.fill_(-1)
            return
        if self.layout.lookup_table is None:
            raise ValueError("Missing checkpoint Engram token map")
        if lookback is None or lookback.shape[1] != self.lookback_depth:
            raise ValueError("Flash Engram requires V2 lookback_token_ids")
        ngram_hashes(
            input_ids,
            lookback,
            metadata.query_start_loc,
            metadata.seq_lens,
            metadata.num_actual_tokens,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
            self.pad_id,
            (self.layout.pad_token_id, *self.visual_token_ids),
            output,
        )


class FlashEmbedding(nn.Module):
    def __init__(
        self, sizes, dim, dtype, cpu_offload, share_group=None, *, prefix=""
    ):
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
        self.share_group = share_group
        self.prefix = prefix
        if share_group is not None and (not cpu_offload or not prefix):
            raise ValueError("Shared Flash Engram requires CPU offload and a prefix")
        self._shared_ptrs: set[int] = set()
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Flash Engram CPU offload requires UVA")
        rows = self.vocab_end - self.vocab_start
        self.embedding = nn.Module()
        if dtype is None:
            # The table dtype is unknown until the checkpoint tensor arrives.
            self.embedding.weight = nn.Parameter(
                torch.empty(
                    (0,),
                    dtype=torch.bfloat16,
                    **({"device": "cpu", "pin_memory": True} if cpu_offload else {}),
                ),
                requires_grad=False,
            )
        else:
            self.embedding.weight = nn.Parameter(
                self._allocate_storage((rows, dim), dtype), requires_grad=False
            )
        self.fp8 = dtype == torch.float8_e4m3fn
        self.embedding.register_parameter("weight_scale_inv", None)
        if self.fp8:
            self.embedding.weight_scale_inv = nn.Parameter(
                self._allocate_storage((rows,), torch.bfloat16), requires_grad=False
            )
        for parameter in self.embedding.parameters():
            set_weight_attrs(parameter, {"weight_loader": self._load_shard})
        self._uva_weight = None
        self._uva_scale = None
        logger.info(
            "Flash Engram TP table: %d rows x %d, dtype=%s, "
            "cpu_offload=%s, sharing_requested=%s",
            rows,
            dim,
            dtype,
            cpu_offload,
            share_group is not None,
        )

    def _allocate_storage(self, shape, dtype, device=None):
        """Device-local, private pinned, or DP-shared host storage."""
        if self.share_group is not None:
            numel = math.prod(shape)
            try:
                table, _ = DPSharedEngramStorage.share_shm_table(
                    numel * dtype.itemsize, self.share_group
                )
            except RuntimeError as exc:
                # share_shm_table raises on every rank of the group, so all
                # of them fall back together.
                logger.warning_once(
                    "Keeping one Flash Engram table per DP replica: %s", exc
                )
            else:
                tensor = table.view(dtype).view(shape)
                self._shared_ptrs.add(tensor.data_ptr())
                return tensor
        if self.cpu_offload:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        return torch.empty(shape, dtype=dtype, device=device)

    def _check_shared_load_order(self, parameter, value, is_weight):
        # Lazy allocation is itself collective, not just the subsequent write
        # fence. Match the logical tensor before mapping anything, including
        # after a previous allocation fell back to private storage.
        group = self.share_group
        signature = (
            self.prefix,
            "weight" if is_weight else "weight_scale_inv",
            tuple(value.shape),
            str(value.dtype),
            tuple(parameter.shape),
            str(parameter.dtype),
            parameter.data_ptr() in self._shared_ptrs,
        )
        signatures: list[object] = [None] * group.world_size
        torch.distributed.all_gather_object(
            signatures, signature, group=group.cpu_group
        )
        if any(other != signature for other in signatures):
            raise RuntimeError(
                "Flash Engram DP sharing requires identical tensor load order "
                f"across replicas; got {signatures}. Disable dp_shared_memory "
                "or use a loader with consistent ordering."
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, value in weights:
            if name not in ("embedding.weight", "embedding.weight_scale_inv"):
                raise ValueError(f"Unknown Flash Engram embedding weight: {name}")
            leaf = name.removeprefix("embedding.")
            if leaf == "weight_scale_inv" and self.embedding.weight_scale_inv is None:
                self.embedding.weight_scale_inv = nn.Parameter(
                    self.embedding.weight.new_empty(0, dtype=torch.bfloat16),
                    requires_grad=False,
                )
                set_weight_attrs(
                    self.embedding.weight_scale_inv, {"weight_loader": self._load_shard}
                )
            self._load_shard(getattr(self.embedding, leaf), value)
            loaded.add(name)
        return loaded

    def _load_shard(self, parameter, value):
        is_weight = parameter is self.embedding.weight
        if self.share_group is not None:
            self._check_shared_load_order(parameter, value, is_weight)
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
            parameter.data = self._allocate_storage(
                (self.vocab_end - self.vocab_start, *expected[1:]),
                value.dtype,
                device=parameter.device,
            )
        shared = parameter.data_ptr() in self._shared_ptrs
        if not shared or self.share_group.rank_in_group == 0:
            default_weight_loader(parameter, value[self.vocab_start : self.vocab_end])
        if shared:
            # The identity check above pairs the same tensor on every rank.
            # Only the group leader writes; fence before peers use the table.
            torch.distributed.barrier(group=self.share_group.cpu_group)
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

    def tables(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """CUDA views of the local table rows and their FP8 row scales."""
        if self._uva_weight is None:
            view = (
                get_accelerator_view_from_cpu_tensor
                if self.cpu_offload
                else lambda tensor: tensor
            )
            self._uva_weight = view(self.embedding.weight)
            if self.fp8:
                self._uva_scale = view(self.embedding.weight_scale_inv)
        return self._uva_weight, self._uva_scale


class FlashEngram(nn.Module):
    # Checkpoint (family, leaf) -> stacked parameter holding it, one row per
    # hyper-connection stream (keys first, then the value) or hash head.
    _PARAMETERS = {
        ("key_projs", "weight"): "proj_weight",
        ("value_proj", "weight"): "proj_weight",
        ("key_projs", "bias"): "proj_bias",
        ("value_proj", "bias"): "proj_bias",
        ("norm_key", "weight"): "key_norm_weight",
        ("norm_query", "weight"): "query_norm_weight",
        ("unified_per_hc_scales", ""): "hc_scale_weight",
        ("lookup_per_head_norms", "weight"): "lookup_norm_weight",
        ("unified_pre_gamma", ""): "unified_pre_gamma",
    }

    def __init__(self, vllm_config, layout, layer_id, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        index = layout.layer_ids.index(layer_id)
        sizes = [p for order in layout.primes[index] for p in order]
        storage = vllm_config.engram_config
        if storage.disk_offload or storage.embedding_across_dp:
            raise ValueError(
                "Flash Engram requires TP-local tables; cross-DP sharding "
                "and disk offload are unsupported"
            )
        share_group = None
        if storage.dp_shared_memory:
            if not current_platform.is_cuda():
                logger.warning_once(
                    "Flash Engram DP sharing is currently supported on CUDA only; "
                    "keeping private offloaded tables."
                )
            else:
                share_group = get_engram_dp_group()
                if share_group is None:
                    logger.warning_once(
                        "Flash Engram DP replicas are not co-located on one node; "
                        "keeping private offloaded tables."
                    )
        dtype = None
        if vllm_config.load_config.load_format == "dummy":
            dtype = (
                torch.float8_e4m3fn if config.engram_fp8_embedding else torch.bfloat16
            )
        self.multi_head_embedding = FlashEmbedding(
            sizes,
            layout.head_dim,
            dtype,
            storage.cpu_offload,
            share_group,
            prefix=prefix,
        )
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        local = self.multi_head_embedding
        local_heads = local.head_end - local.head_start
        self.total_width = len(sizes) * layout.head_dim
        self.start = local.head_start * layout.head_dim
        self.stop = local.head_end * layout.head_dim
        self.rank = get_tensor_model_parallel_rank()
        shapes = {
            "proj_weight": (self.hc_mult + 1, self.hidden_size, self.stop - self.start),
            "proj_bias": (self.hc_mult + 1, self.hidden_size),
            "lookup_norm_weight": (local_heads, layout.head_dim),
            "key_norm_weight": (self.hc_mult, self.hidden_size),
            "query_norm_weight": (self.hc_mult, self.hidden_size),
            "hc_scale_weight": (self.hc_mult, self.hidden_size),
            "unified_pre_gamma": (self.hidden_size,),
        }
        for name, shape in shapes.items():
            setattr(self, name, nn.Parameter(torch.empty(shape), requires_grad=False))

    def load_weight(
        self, name: str, value: torch.Tensor
    ) -> tuple[str, int | None] | None:
        """Load one checkpoint tensor.

        Returns:
            The filled parameter and its row, or None for a hash head that
            another TP rank owns.

        """
        embedding = self.multi_head_embedding
        if name.startswith("multi_head_embedding."):
            embedding.load_weights(
                [(name.removeprefix("multi_head_embedding."), value)]
            )
            return name, None
        family, _, leaf = name.partition(".")
        index = None
        if family == "value_proj":
            index = self.hc_mult
        elif family != "unified_pre_gamma":
            number, _, leaf = leaf.partition(".")
            if not number.isdigit():
                raise ValueError(f"Invalid Flash Engram checkpoint weight: {name}")
            index = int(number)
        target = self._PARAMETERS.get((family, leaf))
        if target is None:
            raise ValueError(f"Unknown Flash Engram checkpoint weight: {name}")
        if index is None:
            pass
        elif family == "lookup_per_head_norms":
            if index >= embedding.n_hash_cols:
                raise ValueError(f"Invalid Engram hash head weight: {name}")
            if not embedding.head_start <= index < embedding.head_end:
                return None
            index -= embedding.head_start
        elif family == "key_projs" and index >= self.hc_mult:
            raise ValueError(f"Invalid Flash Engram stream weight: {name}")
        if family in ("key_projs", "value_proj"):
            if leaf == "weight":
                if value.shape != (self.hidden_size, self.total_width):
                    raise ValueError("Flash Engram projection weight shape mismatch")
                value = value[:, self.start : self.stop]
            elif self.rank != 0:
                # The partial projections are summed across ranks.
                value = torch.zeros_like(value)
        parameter = getattr(self, target).data
        if index is not None:
            if index >= parameter.shape[0]:
                raise ValueError(f"Invalid Flash Engram stream weight: {name}")
            parameter = parameter[index]
        if parameter.shape != value.shape:
            raise ValueError(f"Flash Engram weight shape mismatch: {name}")
        parameter.copy_(value)
        return target, index

    def forward(self, hidden, hashes):
        local = self.multi_head_embedding
        table, scales = local.tables()
        x = lookup_norm(
            table,
            scales,
            hashes[:, local.head_start : local.head_end],
            local.vocab_start,
            local.vocab_end,
            self.lookup_norm_weight,
            hidden.dtype,
        )
        proj = F.linear(x, self.proj_weight.flatten(0, 1), self.proj_bias.flatten())
        if local.tp_size > 1:
            proj = tensor_model_parallel_all_reduce(proj)
        return gated_update(
            hidden,
            proj,
            hashes,
            self.key_norm_weight,
            self.query_norm_weight,
            self.unified_pre_gamma,
            self.hc_scale_weight,
        )
