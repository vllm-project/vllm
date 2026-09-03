# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

from . import ple_mmap
from .ple_layer import Qwen4ExpNGramEmbedding

logger = init_logger(__name__)


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        self._mmap_ple_modules: tuple[Qwen4ExpNGramEmbedding, ...] = ()
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

        if ple_mmap.enabled():
            self._mmap_ple_modules = self._discover_mmap_ple_modules(vllm_config, model)
            self._initialize_mmap_staging(self._mmap_ple_modules)

    def _discover_mmap_ple_modules(
        self, vllm_config: VllmConfig, model: nn.Module
    ) -> tuple[Qwen4ExpNGramEmbedding, ...]:
        """Closed-world inventory of this rank's local mmap PLE modules.

        Cross-checks the compiled graph's view (``static_forward_context``,
        which forward actually reads through ``no_compile_layers``) against
        the live module tree (``model.modules()``): a mismatch means some
        PLE layer would be captured without this model state ever having
        (correctly) initialized its staging buffer.

        Raises:
            RuntimeError: the two walks' ``id()`` sets disagree.
        """
        from_static_context: list[Qwen4ExpNGramEmbedding] = []
        for layer in vllm_config.compilation_config.static_forward_context.values():
            ple_embedding_module = getattr(layer, "ple_embedding", None)
            if ple_embedding_module is None:
                continue
            if isinstance(
                getattr(ple_embedding_module, "ngram_embedding", None),
                ple_mmap.MmapNgramEmbedding,
            ):
                from_static_context.append(ple_embedding_module)

        from_model_modules = [
            module
            for module in model.modules()
            if isinstance(module, Qwen4ExpNGramEmbedding)
            and isinstance(module.ngram_embedding, ple_mmap.MmapNgramEmbedding)
        ]

        static_ids = {id(m) for m in from_static_context}
        module_ids = {id(m) for m in from_model_modules}
        if static_ids != module_ids:
            raise RuntimeError(
                "PLE mmap: static_forward_context and model.modules() PLE "
                f"mmap module inventories disagree (static_forward_context "
                f"has {len(static_ids)}, model.modules() has "
                f"{len(module_ids)}); the closed-world discovery invariant "
                "that authorizes FULL cudagraph capture does not hold."
            )
        return tuple(from_static_context)

    def _initialize_mmap_staging(
        self, modules: tuple[Qwen4ExpNGramEmbedding, ...]
    ) -> None:
        """Aggregate allocation preflight, then allocate every layer's buffer.

        Computes and logs the aggregate staging allocation BEFORE allocating
        any individual layer's buffer, and fails closed if it does not fit
        ``MemorySnapshot(device=...).free_memory`` -- UMA-aware (falls back
        to ``psutil.virtual_memory().available`` on integrated devices,
        where reclaimable page cache is real device headroom).

        Raises:
            RuntimeError: the aggregate allocation would not fit in the
                currently free device memory.
        """
        if not modules:
            return
        per_module_bytes = [m.mmap_staging_nbytes(self.max_num_tokens) for m in modules]
        total_bytes = sum(per_module_bytes)
        snapshot = MemorySnapshot(device=self.device)
        if total_bytes > snapshot.free_memory:
            raise RuntimeError(
                f"PLE mmap: aggregate staging allocation for {len(modules)} "
                f"layer(s) at max_num_tokens={self.max_num_tokens} requires "
                f"{format_gib(total_bytes)} GiB but only "
                f"{format_gib(snapshot.free_memory)} GiB is free"
            )
        logger.info(
            "PLE mmap: staging aggregate allocation %s GiB across %d "
            "layer(s) (max_num_tokens=%d, free=%s GiB)",
            format_gib(total_bytes),
            len(modules),
            self.max_num_tokens,
            format_gib(snapshot.free_memory),
        )
        for module in modules:
            module.initialize_mmap_staging(self.max_num_tokens, self.device)

    def _dummy_query_start_loc_and_context(
        self, num_reqs: int, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded cumsum ``query_start_loc`` plus all-EOS context.

        Shared by ``prepare_dummy_inputs`` (capture, raw int args) and
        ``prepare_runtime_dummy_inputs`` (runtime profile/dummy run, derived
        from an ``InputBatch``) — neither may call the other, since one
        belongs to capture setup and the other must reuse
        ``ModelState.prepare_inputs``'s non-PLE-field behavior.
        """
        query_start_loc = self.ple_query_start_loc[: num_reqs + 1]
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])

        ngram_context = self.ngram_context[:num_reqs]
        ngram_context.fill_(self.ngram_eos_token_id)
        return query_start_loc, ngram_context

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_reqs_padded = input_batch.num_reqs_after_padding
        context = self.ngram_context[:num_reqs_padded]
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc[: num_reqs_padded + 1]
        query_start_loc.copy_(input_batch.query_start_loc[: num_reqs_padded + 1])
        ngram_context = self._prepare_ngram_context(input_batch, req_states)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )

        if self._mmap_ple_modules:
            # Actual (unpadded) extents live only in these local variables —
            # never in model_inputs, which keeps query_start_loc/ngram_context
            # at their padded extents so traced argument shapes stay stable.
            actual_tokens = input_batch.num_tokens
            padded_tokens = input_batch.num_tokens_after_padding
            num_reqs = input_batch.num_reqs
            actual_input_ids = input_batch.input_ids[:actual_tokens]
            actual_query_start_loc = query_start_loc[: num_reqs + 1]
            actual_ngram_context = ngram_context[:num_reqs]
            for module in self._mmap_ple_modules:
                module.prepare_mmap_rows(
                    actual_input_ids,
                    actual_query_start_loc,
                    actual_ngram_context,
                    actual_tokens,
                    padded_tokens,
                )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc, ngram_context = self._dummy_query_start_loc_and_context(
            num_reqs, num_tokens
        )
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        for module in self._mmap_ple_modules:
            module.prepare_dummy_mmap_rows(num_tokens)
        return model_inputs

    def prepare_runtime_dummy_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        # super().prepare_inputs, deliberately NOT self.prepare_inputs: skip
        # this class's own PLE hashing/gather entirely. A runtime dummy or
        # profile batch has no real request state to hash against, and mmap
        # staging must see only zeros here -- never a table read.
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        num_tokens_padded = input_batch.num_tokens_after_padding
        query_start_loc, ngram_context = self._dummy_query_start_loc_and_context(
            num_reqs_padded, num_tokens_padded
        )
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        for module in self._mmap_ple_modules:
            module.prepare_dummy_mmap_rows(num_tokens_padded)
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
