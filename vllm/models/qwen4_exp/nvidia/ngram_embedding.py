# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp n-gram embeddings with device, pinned-host and checkpoint-mapped storage."""

from collections.abc import Iterable

import regex as re
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)

from ..common.ngram_embedding import (
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from .ops.ple import ple_ngram_ids
from .ple_pageable import (
    MappedTable,
    discover_table_layout,
    require_pageable_access,
    verify_shard_bytes,
)

logger = init_logger(__name__)

__all__ = [
    "Qwen4ExpPLEDeviceEmbedding",
    "Qwen4ExpPLEEmbedding",
    "Qwen4ExpPLEEmbeddingMethod",
    "Qwen4ExpPLEFp8EmbeddingMethod",
    "Qwen4ExpPLEPageableHostEmbedding",
    "Qwen4ExpPLEPinnedHostEmbedding",
    "Qwen4ExpPLEUnquantizedEmbeddingMethod",
    "Qwen4ExpNGramEmbedding",
]


class Qwen4ExpPLEPageableHostEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    """PLE table read in place from the checkpoint's safetensors files.

    For GPUs that dereference pageable host memory through the host page tables
    (e.g. unified-memory GB10). There is no table-sized device or pinned
    allocation and no resident duplicate of the table: its rows stay in the page
    cache as clean, file-backed pages that the kernel can drop and re-read, shared
    between processes. The prefetch buffer, ETP reduction and finalize path are
    inherited from the pinned-host backend; the lookup itself runs on the current
    stream (not the pinned backend's side stream) and gathers from the mapping.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        device = torch.device("cuda", torch.accelerator.current_device_index())
        require_pageable_access(device.index)
        if not torch.cuda.get_device_properties(device).is_integrated:
            logger.warning_once(
                "Engram checkpoint_mapped has been validated only on integrated "
                "unified-memory GPUs (DGX Spark / GB10). This device also reads "
                "pageable host memory through the host page tables (e.g. Grace "
                "Hopper / Grace Blackwell over NVLink-C2C), but performance and "
                "correctness there are untested."
            )
        # Skip the pinned-host constructor: there is no host tensor to view.
        Qwen4ExpPLEEmbedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        layer_match = re.search(r"layers\.(\d+)\.", prefix)
        if layer_match is None:
            raise ValueError(f"Cannot derive the decoder layer index from {prefix}")
        self.layer_index = int(layer_match.group(1))
        vllm_config = get_current_vllm_config()
        self._model_config = vllm_config.model_config
        self._load_config = vllm_config.load_config
        self._load_format = str(vllm_config.load_config.load_format)
        self.table: MappedTable | None = None
        self._rebind_pending = False
        self._pending_table: MappedTable | None = None
        self._reload_error: Exception | None = None
        self._prefetch_stream = torch.cuda.Stream(device=device)
        self._prefetch_buffer = torch.empty(
            max_total_tokens * self.etp_data_parallel_size,
            num_ngram_heads,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=device,
        )
        self._output_dim = num_ngram_heads * self.embedding_dim

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """No storage: rows are read from the mapped checkpoint files."""
        del num_embeddings
        return torch.empty(0, embedding_dim, dtype=dtype, device="cpu")

    def current_table(self) -> MappedTable | None:
        """The mapping in use now (rebuilt when a reload remaps the files)."""
        return self.table

    def record_incoming_shard(
        self, checkpoint_start: int, loaded_weight: torch.Tensor
    ) -> None:
        """Account for a shard delivered by a (re)load.

        On the first load the loader streams the very files that get mapped, so
        nothing is compared. On a reload (a table is already bound), the files
        the loader now points at are mapped and every incoming shard is compared
        with them in full: a reload from disk (``weights_path``) streams those
        same files and matches, while PLE weights delivered from memory (e.g.
        weight sync) cannot be served from a file mapping and are rejected.
        """
        if self._reload_error is not None:
            # A rejected load raised out of load_weights; a shard arriving now
            # starts a new load attempt, verified from scratch.
            self._reload_error = None
            self._pending_table = None
        self._rebind_pending = True
        if self.table is None:
            return
        try:
            if self._pending_table is None:
                self._pending_table = self._map_checkpoint_files()
            verify_shard_bytes(self._pending_table, checkpoint_start, loaded_weight)
        except Exception as e:
            self._reload_error = e
            raise

    def bind_storage_after_loading(self) -> None:
        """(Re)map this layer's shards once the checkpoint is on local disk."""
        if self.table is not None and not self._rebind_pending:
            return
        self._bind()

    def _map_checkpoint_files(self) -> MappedTable:
        layout = discover_table_layout(
            resolve_checkpoint_files(self._model_config, self._load_config),
            self.layer_index,
            self.org_vocab_size,
            self.embedding_dim,
            self.weight.dtype,
            int(getattr(self._model_config.hf_text_config, "split_ngram_parts", 512)),
        )
        return MappedTable(layout, self._prefetch_buffer.device)

    def _bind(self) -> None:
        if self._reload_error is not None:
            # A rejected reload stays rejected until a load succeeds; never fall
            # back to the previous mapping.
            raise RuntimeError(
                "The last weight load was rejected by checkpoint_mapped PLE storage"
            ) from self._reload_error
        if self._load_format == "dummy":
            table = MappedTable.zeros(
                self.org_vocab_size,
                self.embedding_dim * self.weight.dtype.itemsize,
                self._prefetch_buffer.device,
            )
        else:
            table = self._pending_table or self._map_checkpoint_files()
        # Commit only after everything above succeeded.
        self.table = table
        self._pending_table = None
        self._rebind_pending = False
        if table.layout is not None:
            logger.info(
                "Mapped PLE table of layer %d in place: %d rows x %d B from %d "
                "files; no table-sized device or pinned allocation",
                self.layer_index,
                table.layout.num_rows,
                table.layout.row_bytes,
                len({shard.path for shard in table.layout.shards}),
            )

    def _lookup_on_current_stream(self, ngram_ids: torch.Tensor) -> None:
        if self._rebind_pending or self._reload_error is not None:
            # A reload delivered shards but its processing did not rebind (a
            # layer without loadable elements is only restored): rebind now,
            # or raise if that reload was rejected.
            self._bind()
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        self._lookup(
            gathered_ids, output=self._prefetch_buffer[: gathered_ids.shape[0]]
        )

    @eager_break_during_capture
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Look up on the current stream, not the pinned backend's side stream.

        With the side-stream lookup, greedy outputs on GB10 were not reproducible
        within one server start (identical prompts matched in 2 of 8, logprobs
        differed by up to 1.4); on the current stream they matched in 8 of 8 with
        zero logprob difference. The CPU page prefetch already runs ahead of the
        step, so the side stream buys no overlap worth keeping here.
        """
        self._lookup_on_current_stream(ngram_ids)

    def _join_prefetch_stream(self) -> None:
        """Nothing to join: start_prefetch ran the lookup on the current stream.

        Joining the side stream anyway fails a FULL cudagraph capture, because
        that stream is never part of the capture
        (cudaErrorStreamCaptureIsolation).
        """

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather this rank's rows from the mapped files; zeros elsewhere."""
        if self.table is None:
            raise RuntimeError("PLE checkpoint mapping was not bound after loading")
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape, dtype=self.weight.dtype, device=input_ids.device
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != self.weight.dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, weight dtype, "
                "and input device"
            )
        self.table.gather_into(
            input_ids.reshape(-1).long(),
            output.view(torch.uint8).reshape(-1, self.table.row_bytes),
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
        )
        return output


def resolve_checkpoint_files(model_config, load_config) -> list[str]:
    """The safetensors files the loader read, resolved exactly as it resolves them.

    Reuses the default loader's preparation so ``--download-dir``, subfolders
    and the ``model.safetensors.index.json`` filter all apply. The weights are
    already local by the time the storage is bound, so nothing is downloaded.
    """
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    _, files, use_safetensors, _ = DefaultModelLoader(load_config)._prepare_weights(
        model_config.model_weights or model_config.model,
        None,
        model_config.revision,
        fall_back_to_pt=False,
        allow_patterns_overrides=None,
    )
    if not use_safetensors:
        raise ValueError("Engram checkpoint_mapped requires a safetensors checkpoint")
    return files


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        *,
        data_parallel_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        embedding_prefix = f"{prefix}.ngram_embedding"
        embedding_quant_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            quant_config,
            embedding_prefix,
            getattr(config, "ple_embedding_dtype", None),
        )
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        engram_config = get_current_vllm_config().engram_config
        embedding_cls: type[Qwen4ExpPLEEmbedding]
        if engram_config is not None and engram_config.checkpoint_mapped:
            embedding_cls = Qwen4ExpPLEPageableHostEmbedding
        elif engram_config is not None and engram_config.cpu_offload:
            embedding_cls = Qwen4ExpPLEPinnedHostEmbedding
        else:
            embedding_cls = Qwen4ExpPLEDeviceEmbedding
        self.ngram_embedding = embedding_cls(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=embedding_prefix,
            embedding_method=embedding_quant_method,
            num_ngram_heads=self.ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        if self.ngram_embedding.supports_prefetch:
            # The side-stream lookup outlives eager-break args, whose
            # graph-pool storage later segments may reuse.
            self._prefetch_ids = torch.empty(
                max_total_tokens, self.ngram_heads, dtype=torch.long
            )
        weight = self.ngram_embedding.weight
        logger.info(
            "Initialized PLE embedding %s: quantization_method=%s, "
            "weight_dtype=%s, weight_device=%s, pinned=%s",
            embedding_prefix,
            type(embedding_quant_method).__name__,
            weight.dtype,
            weight.device,
            weight.is_pinned(),
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute n-gram embedding indices for the current request layout."""
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        if input_ids.is_cuda:
            return ple_ngram_ids(
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                layer_multipliers=self.layer_multipliers,
                ngram_heads_vocab_sizes=self.ngram_heads_vocab_sizes,
                ngram_heads_offsets=self.ngram_heads_offsets,
                eos_token_id=self.eos_token_id,
                heads_per_ngram=self.heads_per_ngram,
                output=output,
            )
        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()
        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, num_tokens),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.ngram_embedding
        if embedding.supports_prefetch:
            return embedding(hidden_states)
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def cpu_ngram_ids_fn(self):
        """A CPU-only ``compute_ngram_ids`` for host-side page prefetching."""
        cpu = copy_module_for_cpu_ids(self)
        return lambda ids, qsl, ctx: Qwen4ExpNGramEmbedding.compute_ngram_ids(
            cpu, ids, qsl, ctx
        )

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Start the pinned lookup while the preceding decoder layer runs."""
        embedding = self.ngram_embedding
        if not embedding.supports_prefetch:
            return
        ngram_ids = self.compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
            output=self._prefetch_ids[: input_ids.numel()],
        )
        embedding.start_prefetch(hidden_states, ngram_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""
        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                if isinstance(embedding, Qwen4ExpPLEPageableHostEmbedding):
                    # Rows are read in place from the checkpoint files; sample
                    # this shard so the mapping can be checked against it.
                    embedding.record_incoming_shard(checkpoint_start, loaded_weight)
                    loaded.add("ngram_embedding.weight")
                    continue
                embedding.weight.weight_loader(
                    embedding.weight,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


def copy_module_for_cpu_ids(module: "Qwen4ExpNGramEmbedding"):
    """An object with CPU copies of the hash buffers ``compute_ngram_ids`` reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        layer_multipliers=module.layer_multipliers.cpu(),
        ngram_heads_vocab_sizes=module.ngram_heads_vocab_sizes.cpu(),
        ngram_heads_offsets=module.ngram_heads_offsets.cpu(),
        eos_token_id=module.eos_token_id,
        ngram_size=module.ngram_size,
        heads_per_ngram=module.heads_per_ngram,
        _shift_precompute=Qwen4ExpNGramEmbedding._shift_precompute,
        _shift_apply=Qwen4ExpNGramEmbedding._shift_apply,
    )


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEPageableHostEmbedding",
]
