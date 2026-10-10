# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp n-gram embeddings with device, pinned-host and host-file-gather storage."""

import mmap
import os
from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from ..common.ngram_embedding import (
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLENvFp4EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from ..common.ops.ple import ple_ngram_ids

logger = init_logger(__name__)

__all__ = [
    "Qwen4ExpPLEDeviceEmbedding",
    "Qwen4ExpPLEEmbedding",
    "Qwen4ExpPLEEmbeddingMethod",
    "Qwen4ExpPLEFp8EmbeddingMethod",
    "Qwen4ExpPLENvFp4EmbeddingMethod",
    "Qwen4ExpPLEPinnedHostEmbedding",
    "Qwen4ExpPLEUnquantizedEmbeddingMethod",
    "Qwen4ExpNGramEmbedding",
]


def _maps_entry(ptr: int) -> tuple[str, int]:
    """Return the path and file offset of ``ptr`` from ``/proc/self/maps``."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            low, high = (int(bound, 16) for bound in fields[0].split("-"))
            if low <= ptr < high:
                path = fields[5].rstrip("\n") if len(fields) > 5 else ""
                return path, ptr - low + int(fields[2], 16)
    return "", 0


class Qwen4ExpPLEFileGatherEmbedding(Qwen4ExpPLEEmbedding):
    """PLE rows read on the host from the checkpoint shard files."""

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
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            data_parallel_rank=data_parallel_rank,
        )
        self._dummy = get_current_vllm_config().load_config.load_format == "dummy"
        # NVFP4 rows are packed E2M1 bytes plus a separate plane of block scales.
        self._nvfp4 = isinstance(embedding_method, Qwen4ExpPLENvFp4EmbeddingMethod)
        self._planes = ("weight", "weight_scale") if self._nvfp4 else ("weight",)
        device, pin = self.weight.device, self.weight.device.type != "cpu"
        self._host_ids = torch.empty(
            (max_total_tokens, num_ngram_heads),
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin,
        )
        self._plane_buffers: dict[str, tuple[torch.Tensor, ...]] = {}
        for name in self._planes:
            parameter = getattr(self, name)
            shape = (max_total_tokens, num_ngram_heads, parameter.shape[1])
            staging = torch.zeros(shape, dtype=parameter.dtype, device=device)
            # Step N+1 writes these rows only after its stream sync, so after
            # step N's H2D.
            host = torch.empty(
                shape, dtype=parameter.dtype, device="cpu", pin_memory=pin
            )
            rows = torch.empty_like(host).view(torch.uint8).flatten(0, 1)
            self._plane_buffers[name] = (staging, host, rows)
        self._staging, self._host_rows, _ = self._plane_buffers["weight"]
        if self._nvfp4:
            self._output = torch.zeros(
                (max_total_tokens, num_ngram_heads, embedding_dim),
                dtype=params_dtype,
                device=device,
            )
            self._row_ids = torch.arange(
                max_total_tokens * num_ngram_heads, device=device
            )
        self._shards: dict[str, dict[int, torch.Tensor]] | None = {
            name: {} for name in self._planes
        }
        self._shard_size = 0
        self._bound = False
        self._fds: dict[str, int] = {}
        self._sources: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate no rows; they stay in the checkpoint files."""
        return torch.empty(0, embedding_dim, dtype=dtype)

    def accept_checkpoint_shard(
        self,
        shard_index: int,
        loaded_weight: torch.Tensor,
        shard_size: int,
        name: str = "weight",
    ) -> None:
        """Keep a checkpoint shard as a zero-copy byte view of its storage."""
        if self._bound or self._shards is None:
            self._shards = None
            raise RuntimeError("PLE host file gather does not support weight reload")
        dtype = getattr(self, name).dtype
        if loaded_weight.dtype != dtype:
            raise ValueError(
                f"PLE shard {name} dtype {loaded_weight.dtype} must match the "
                f"embedding dtype {dtype}; host file gather serves rows as stored"
            )
        if shard_index in self._shards[name]:
            raise ValueError(f"Duplicate PLE embedding shard {shard_index} {name}")
        self._shards[name][shard_index] = loaded_weight.view(torch.uint8)
        self._shard_size = shard_size

    def bind_file_shards(self) -> None:
        """Record each shard's file and offset and release the loader's views."""
        if self._shards is None:
            raise RuntimeError("PLE host file gather does not support weight reload")
        if not self._dummy and not self._bound:
            for name, shards in self._shards.items():
                rows = sum(shard.shape[0] for shard in shards.values())
                if rows != self.org_vocab_size:
                    plane = "" if name == "weight" else f"{name} "
                    raise ValueError(
                        f"PLE {plane}shards cover {rows} of {self.org_vocab_size} rows"
                    )
                fds_t, bases = torch.zeros(2, len(shards)).long()
                for index, shard in shards.items():
                    path, offset = _maps_entry(shard.data_ptr())
                    if not path or not os.path.isfile(path):
                        raise ValueError(
                            f"PLE embedding shard {index} {name} is not a "
                            f"file-backed view ({path or 'anonymous'})"
                        )
                    if path not in self._fds:
                        self._fds[path] = os.open(path, os.O_RDONLY)
                    fds_t[index], bases[index] = self._fds[path], offset
                self._sources[name] = (fds_t, bases)
        self._shards = {name: {} for name in self._planes}
        self._bound = True

    def stage_rows(self, num_tokens: int) -> None:
        """Read the rows for ``_host_ids[:num_tokens]`` and copy them to staging."""
        if self._shards is None or not self._bound:
            raise RuntimeError("PLE host file gather is unbound or was reloaded")
        ids = self._host_ids[:num_tokens].flatten()
        if not self._fds:
            for _, host, _ in self._plane_buffers.values():
                host[:num_tokens].view(torch.uint8).zero_()
        else:
            ids, inverse = ids.unique(return_inverse=True)
            if ids.numel() and not 0 <= ids[0] <= ids[-1] < self.org_vocab_size:
                raise IndexError(f"PLE id out of range for {self.org_vocab_size} rows")
            shard = ids // self._shard_size
            local = ids - shard * self._shard_size
            for name, (_, host, rows) in self._plane_buffers.items():
                out = host[:num_tokens].flatten(0, 1).view(torch.uint8)
                self._read_rows(name, shard, local, rows[: ids.numel()])
                torch.index_select(rows[: ids.numel()], 0, inverse, out=out)
        for staging, host, _ in self._plane_buffers.values():
            staging[:num_tokens].copy_(host[:num_tokens], non_blocking=True)

    def _read_rows(
        self,
        name: str,
        shard: torch.Tensor,
        local: torch.Tensor,
        rows: torch.Tensor,
    ) -> None:
        """Read one plane's rows for sorted distinct ids into `rows`."""
        fds_t, bases = self._sources[name]
        page, row_bytes = mmap.PAGESIZE, rows.shape[1]
        start = bases[shard] + local * row_bytes
        first, last = start // page, (start + row_bytes - 1) // page
        run = torch.ones_like(first, dtype=torch.bool)
        run[1:] = (first[1:] > last[:-1] + 1) | (shard[1:] != shard[:-1])
        fds, lows = fds_t[shard[run]].tolist(), first[run].tolist()
        # Queue every page read before the serial reads.
        for fd, lo, hi in zip(fds, lows, last[run.roll(-1)].tolist()):
            os.posix_fadvise(
                fd, lo * page, (hi - lo + 1) * page, os.POSIX_FADV_WILLNEED
            )
        offsets = start.tolist()
        for row, fd, offset in zip(rows.numpy(), fds_t[shard].tolist(), offsets):
            if os.preadv(fd, [row], offset) != row_bytes:
                raise ValueError(
                    f"PLE shard file {os.readlink(f'/proc/self/fd/{fd}')} is "
                    f"truncated at offset {offset}"
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the rows staged for this step, decoded if they are NVFP4."""
        num_tokens = hidden_states.shape[0]
        method = self.embedding_method
        if not isinstance(method, Qwen4ExpPLENvFp4EmbeddingMethod):
            return self._staging[:num_tokens].flatten(-2)
        weight, scale = (self._plane_buffers[n][0] for n in self._planes)
        output = self._output[:num_tokens]
        # Staged rows are compact, so the i-th staged row has id i.
        method.lookup_rows(
            weight,
            scale,
            self.weight_scale_2,
            self._row_ids[: num_tokens * weight.shape[1]],
            output,
            0,
            num_tokens * weight.shape[1],
        )
        return output.flatten(-2)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Rows are staged in ``prepare_inputs``, so prefetch is a no-op."""
        return None


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
        embedding_cls = (
            Qwen4ExpPLEFileGatherEmbedding
            if engram_config is not None and engram_config.host_file_gather
            else Qwen4ExpPLEPinnedHostEmbedding
            if engram_config is not None and engram_config.cpu_offload
            else Qwen4ExpPLEDeviceEmbedding
        )
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
        if embedding.supports_prefetch or isinstance(
            embedding, Qwen4ExpPLEFileGatherEmbedding
        ):
            return embedding(hidden_states)
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        return self.ngram_embedding(ngram_ids).flatten(-2)

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
        embedding = self.ngram_embedding
        method = getattr(embedding, "embedding_method", None)
        nvfp4_method = (
            method if isinstance(method, Qwen4ExpPLENvFp4EmbeddingMethod) else None
        )
        # NVFP4 tables split their block scales into shards like the rows.
        shard_parameters = ("weight", "weight_scale") if nvfp4_method else ("weight",)
        shard_size = (
            embedding.org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts

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
            if name.startswith(shard_prefix):
                shard_text, _, suffix = name[len(shard_prefix) :].partition(".")
                if not shard_text.isdigit() or suffix not in shard_parameters:
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                parameter = getattr(embedding, suffix)
                expected_shape = (expected_rows, parameter.shape[1])
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index} "
                        f"{suffix}: expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                if nvfp4_method is not None:
                    # A cast would silently reinterpret packed E2M1 bytes.
                    if loaded_weight.dtype != parameter.dtype:
                        raise ValueError(
                            f"NVFP4 PLE shard {shard_index} {suffix} requires "
                            f"{parameter.dtype}, got {loaded_weight.dtype}"
                        )
                    nvfp4_method.record_loaded_rows(
                        embedding, suffix, checkpoint_start, expected_rows
                    )
                if isinstance(embedding, Qwen4ExpPLEFileGatherEmbedding):
                    embedding.accept_checkpoint_shard(
                        shard_index, loaded_weight, shard_size, suffix
                    )
                    loaded.add(f"ngram_embedding.{suffix}")
                    continue
                parameter.weight_loader(
                    parameter,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add(f"ngram_embedding.{suffix}")
                continue
            if nvfp4_method is not None and name in (
                "ngram_embedding.weight",
                "ngram_embedding.weight_scale",
            ):
                raise ValueError(
                    f"NVFP4 PLE tables must be stored as {shard_prefix}<i>.weight "
                    f"and {shard_prefix}<i>.weight_scale shards, got unsharded {name}"
                )
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


def stage_checkpoint_rows(
    modules: list[Qwen4ExpNGramEmbedding],
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
) -> None:
    """Stage host-file-gather PLE rows for one step behind a single stream sync."""
    num_tokens = input_ids.shape[0]
    for module in modules:
        ids = module.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        module.ngram_embedding._host_ids[:num_tokens].copy_(ids, non_blocking=True)
    if input_ids.device.type != "cpu":
        torch.accelerator.current_stream().synchronize()
    for module in modules:
        module.ngram_embedding.stage_rows(num_tokens)


__all__ = [
    "Qwen4ExpNGramEmbedding",
]
