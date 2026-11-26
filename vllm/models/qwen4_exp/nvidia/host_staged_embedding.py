# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side PLE row staging for NVIDIA Qwen4Exp.

When unified virtual addressing (UVA) is unavailable, CPU offload gathers the
requested rows from host RAM and copies them to the GPU before each forward.
Disk offload uses the same staging buffers for rows read from checkpoint files.
"""

from abc import abstractmethod

import torch

from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from ..common.ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLENvFp4EmbeddingMethod,
)


class Qwen4ExpPLEHostStagedEmbedding(Qwen4ExpPLEEmbedding):
    """Stage host rows outside the forward into fixed CUDA graph inputs."""

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
        reduce_results: bool = True,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        # NVFP4 rows are packed E2M1 bytes plus a separate plane of block scales.
        self._nvfp4 = isinstance(embedding_method, Qwen4ExpPLENvFp4EmbeddingMethod)
        device = torch.get_default_device()
        self._pin_memory = (
            device.type == "cuda" and current_platform.is_pin_memory_available()
        )
        rows = (max_total_tokens, num_ngram_heads)
        self._host_ids = torch.empty(
            rows, dtype=torch.long, device="cpu", pin_memory=self._pin_memory
        )
        # The step sync completes the previous H2D before we reuse these buffers.
        self._planes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name in ("weight", "weight_scale") if self._nvfp4 else ("weight",):
            plane = getattr(self, name)
            shape, dtype = (*rows, plane.shape[1]), plane.dtype
            host = torch.empty(
                shape, dtype=dtype, device="cpu", pin_memory=self._pin_memory
            )
            self._planes[name] = (
                torch.zeros(shape, dtype=dtype, device=device),
                host,
            )
        if self._nvfp4:
            self._output = torch.zeros(
                (*rows, embedding_dim), dtype=params_dtype, device=device
            )
            self._row_ids = torch.arange(rows[0] * rows[1], device=device)

    def _copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if self._pin_memory:
            dst.copy_(src, non_blocking=True)
        else:
            # Pageable host buffers require a synchronous transfer.
            with gpu_sync_allowed():
                dst.copy_(src)

    def copy_ids_to_host(self, ids: torch.Tensor) -> None:
        self._copy(self._host_ids[: ids.shape[0]], ids)

    def stage_rows(self, num_tokens: int) -> None:
        self._gather_host_rows(self._host_ids[:num_tokens])
        for staging, host in self._planes.values():
            self._copy(staging[:num_tokens], host[:num_tokens])

    @abstractmethod
    def _gather_host_rows(self, ids: torch.Tensor) -> None:
        """Fill the host buffers in the token and head order of `ids`."""
        raise NotImplementedError

    def staged_rows(self, num_tokens: int) -> torch.Tensor:
        """Return this step's staged rows, decoded if they are NVFP4."""
        weight, *scale = (staging[:num_tokens] for staging, _ in self._planes.values())
        if not self._nvfp4:
            return weight.flatten(-2)
        num_rows = weight.shape[0] * weight.shape[1]
        output = self._output[:num_tokens]
        # Staged rows are compact, so the i-th staged row has id i.
        Qwen4ExpPLENvFp4EmbeddingMethod.decode_rows(
            weight,
            scale[0],
            self.weight_scale_2,
            self._row_ids[:num_rows],
            output,
            0,
            num_rows,
        )
        return output.flatten(-2)

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        return None

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        raise RuntimeError("Host-staged PLE embedding does not support prefetch")


class Qwen4ExpPLEHostMemoryEmbedding(Qwen4ExpPLEHostStagedEmbedding):
    """Gather this rank's CPU-resident PLE rows when UVA is unavailable."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.etp_data_parallel_size != 1:
            raise NotImplementedError(
                "Host-memory PLE embedding does not support ETP spanning DP ranks"
            )

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the local ETP rows in ordinary CPU memory."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu")

    def _gather_host_rows(self, ids: torch.Tensor) -> None:
        num_tokens = ids.shape[0]
        ids = ids.flatten()
        unowned = None
        if self.tp_size > 1:
            if ((ids < 0) | (ids >= self.org_vocab_size)).any():
                raise IndexError(f"PLE id out of range for {self.org_vocab_size} rows")
            start = self.shard_indices.org_vocab_start_index
            end = self.shard_indices.org_vocab_end_index
            unowned = (ids < start) | (ids >= end)
            ids = (ids - start).masked_fill_(unowned, 0)

        for name, (_, host) in self._planes.items():
            table = getattr(self, name).view(torch.uint8)
            out = host[:num_tokens].flatten(0, 1).view(torch.uint8)
            torch.index_select(table, 0, ids, out=out)
            if unowned is not None:
                out[unowned] = 0

    def staged_rows(self, num_tokens: int) -> torch.Tensor:
        """Return the staged rows combined across ETP ranks."""
        embeddings = super().staged_rows(num_tokens)
        if self.reduce_results:
            return self._reduce_etp_embeddings(embeddings)
        return embeddings
