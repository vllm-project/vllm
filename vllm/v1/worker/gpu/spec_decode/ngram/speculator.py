# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.spec_decode.ngram.numba_utils import (
    batch_propose_numba,
)


class NgramSpeculator(BaseSpeculator):
    """CPU-based N-gram speculative decoding speculator for ModelRunnerV2.

    Uses numba-accelerated KMP-based string matching on the CPU to find
    the longest matching n-gram suffix in each request's token history,
    then proposes the k tokens that follow the match as draft tokens.

    This speculator does NOT run any GPU model — it operates purely on
    CPU with numpy arrays and therefore inherits directly from
    BaseSpeculator (not DraftModelSpeculator).

    The full token history is received via the ``all_token_ids`` parameter
    in :meth:`propose`, which must be a GPU tensor of shape
    ``[num_reqs, max_model_len]`` containing all token IDs per request.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.device = device

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        assert spec_config.prompt_lookup_min is not None
        assert spec_config.prompt_lookup_max is not None

        self.min_n = spec_config.prompt_lookup_min
        self.max_n = spec_config.prompt_lookup_max
        self.k = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs

        # Pre-allocate numpy buffers for numba batch propose.
        # These match the buffer layout from the old NgramProposer.
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros(max_num_seqs, dtype=np.int32)

        # Warm up numba JIT compilation using actual max_num_seqs,
        # NOT a hardcoded value, to avoid out-of-bounds writes that
        # cause segfaults in Numba's @njit mode (no bounds checking).
        self._max_num_seqs = max_num_seqs

        # Ngram speculator does not support multimodal inputs.
        self.supports_mm_inputs = False

        # Ngram speculator does not produce draft logits (CPU-based
        # n-gram matching has no model). The rejection sampler accepts
        # draft_logits=None for greedy verification.
        self.draft_logits: torch.Tensor | None = None

        # Threshold of total number of tokens in the batch to enable
        # multi-threading in numba batch propose.
        self.num_tokens_threshold = 8192
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        if cpu_count:
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            self.num_numba_thread_available //= tp_size
        else:
            self.num_numba_thread_available = 1

        self._warmup()

    # ── BaseSpeculator abstract methods ────────────────────────────

    def load_model(self, target_model: torch.nn.Module) -> None:
        """No model to load for N-gram proposer."""
        pass

    def set_attn(self, *args, **kwargs) -> None:
        """No attention needed for N-gram proposer."""
        pass

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """No CUDA graph needed for CPU-based N-gram proposer."""
        pass

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        """No CUDA graph capture needed for CPU-based N-gram proposer."""
        pass

    # ── Core propose logic ─────────────────────────────────────────

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        all_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import sys

        num_reqs = input_batch.num_reqs

        if dummy_run:
            # During dummy/profile runs, there are no real token IDs to match.
            # Return empty draft tokens.
            with open("/home/log.txt", "a") as f:
                f.write(
                    f"[NgramSpeculator.propose] dummy_run=True, "
                    f"num_reqs={num_reqs}, k={self.k}, returning zeros\n"
                )
            return torch.zeros(
                num_reqs, self.k, dtype=torch.int64, device=self.device
            )

        assert all_token_ids is not None, (
            "all_token_ids is required for NgramSpeculator.propose()"
        )

        # Use idx_mapping to map batch indices → req_state_idx (slot index).
        # all_token_ids is [max_num_reqs, max_model_len] indexed by slot index,
        # NOT by batch index. Without this mapping we would read the wrong
        # request's token history (e.g. warmup dummy data instead of real tokens).
        idx_mapping_np = input_batch.idx_mapping_np[:num_reqs]
        gpu_indices = torch.from_numpy(idx_mapping_np).to(self.device)
        token_ids_cpu = all_token_ids[gpu_indices].cpu().numpy()

        # Compute num_tokens_no_spec by scanning token_ids_cpu directly.
        # UVA buffer layout: real tokens (> 0 or valid IDs), rejected spec
        # tokens (-1), and padding (0).  Scan each row backwards to find
        # the last real token (non-zero and non-(-1)).
        #
        # This avoids relying on model_runner-synced num_computed / total_len,
        # keeping all ngram-specific logic self-contained.
        num_sampled_np = num_sampled.cpu().numpy()
        num_tokens_no_spec = np.zeros(num_reqs, dtype=np.int32)
        for i in range(num_reqs):
            row = token_ids_cpu[i]
            n = row.shape[0]
            while n > 0 and row[n - 1] <= 0:
                n -= 1
            num_tokens_no_spec[i] = n

        # Log input state
        with open("/home/log.txt", "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[NgramSpeculator.propose] num_reqs={num_reqs}\n")
            f.write(f"[NgramSpeculator.propose] min_n={self.min_n}, max_n={self.max_n}, k={self.k}\n")
            f.write(f"[NgramSpeculator.propose] max_model_len={self.max_model_len}\n")
            f.write(f"[NgramSpeculator.propose] idx_mapping_np={idx_mapping_np.tolist()}\n")
            f.write(f"[NgramSpeculator.propose] num_sampled_np={num_sampled_np.tolist()}\n")
            f.write(f"[NgramSpeculator.propose] prefill_len_np={input_batch.prefill_len_np[:num_reqs].tolist()}\n")
            for i in range(num_reqs):
                f.write(
                    f"[NgramSpeculator.propose] req[{i}]: "
                    f"num_computed_tokens_np={input_batch.num_computed_tokens_np[i]}, "
                    f"num_tokens_no_spec={num_tokens_no_spec[i]}, "
                    f"num_sampled_np={num_sampled_np[i]}\n"
                )
                # Show token_ids for this request
                n_toks = int(num_tokens_no_spec[i])
                tok_slice = token_ids_cpu[i, :n_toks].tolist()
                f.write(
                    f"[NgramSpeculator.propose] req[{i}] token_ids[:{n_toks}]="
                    f"{tok_slice}\n"
                )

        # Determine which requests need ngram proposals.
        valid_indices = []
        for i in range(num_reqs):
            if num_sampled_np[i] == 0:
                continue
            if num_tokens_no_spec[i] >= self.max_model_len:
                continue
            valid_indices.append(i)

        with open("/home/log.txt", "a") as f:
            f.write(f"[NgramSpeculator.propose] valid_indices={valid_indices}\n")

        # Run numba batch propose.
        if valid_indices:
            from numba import get_num_threads, set_num_threads

            original_threads = get_num_threads()
            total_tokens = num_tokens_no_spec.sum()
            if total_tokens >= self.num_tokens_threshold:
                n_threads = max(
                    1, min(self.num_numba_thread_available, len(valid_indices))
                )
                set_num_threads(n_threads)
            else:
                set_num_threads(1)

            with open("/home/log.txt", "a") as f:
                f.write(
                    f"[NgramSpeculator.propose] calling batch_propose_numba: "
                    f"valid_indices={valid_indices}, "
                    f"num_tokens_no_spec={num_tokens_no_spec[valid_indices].tolist()}, "
                    f"n_threads={get_num_threads()}\n"
                )

            # Zero out num_drafts for valid_indices BEFORE calling
            # batch_propose_numba, so that indices without a match
            # stay at 0 instead of retaining stale values from
            # previous steps.
            for i in valid_indices:
                self.valid_ngram_num_drafts[i] = 0
                self.valid_ngram_draft[i, :] = 0

            batch_propose_numba(
                valid_indices,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                self.k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )
            set_num_threads(original_threads)

            # Log results after numba
            with open("/home/log.txt", "a") as f:
                for i in range(num_reqs):
                    f.write(
                        f"[NgramSpeculator.propose] after numba req[{i}]: "
                        f"num_drafts={self.valid_ngram_num_drafts[i]}, "
                        f"draft_tokens={self.valid_ngram_draft[i, :self.k].tolist()}\n"
                    )

        # Zero out num_drafts for batch indices that were NOT processed.
        # batch_propose_numba only overwrites indices in valid_indices;
        # others retain stale values from previous steps when requests
        # move between batch indices (V2 scheduler reorders batches).
        valid_set = set(valid_indices)
        for i in range(num_reqs):
            if i not in valid_set:
                self.valid_ngram_num_drafts[i] = 0

        # Build draft_tokens tensor in the format MRV2 expects:
        # [num_reqs, k] int64, zeros where no draft tokens exist.
        # Zero is safe as model input (token ID 0 = <unk>, never proposed
        # by ngram matching on real text).
        draft_tokens = torch.zeros(
            num_reqs, self.k, dtype=torch.int64, device=self.device
        )
        for i in range(num_reqs):
            n = self.valid_ngram_num_drafts[i]
            if n > 0:
                draft_tokens[i, :n] = torch.from_numpy(
                    self.valid_ngram_draft[i, :n]
                ).to(self.device)

        return draft_tokens

    # ── Private helpers ────────────────────────────────────────────

    def _warmup(self) -> None:
        """Trigger Numba JIT compilation for the N-gram proposer.

        Uses self._max_num_seqs instead of a hardcoded value to avoid
        out-of-bounds writes into self.valid_ngram_draft (which is
        allocated with max_num_seqs rows). Numba @njit does not
        bounds-check, so writing past the array end causes a segfault.
        """
        n = self._max_num_seqs
        dummy_num_tokens = np.zeros(n, dtype=np.int32)
        dummy_token_ids = np.zeros((n, self.max_model_len), dtype=np.int32)
        valid_indices = list(range(n))
        batch_propose_numba(
            valid_indices,
            dummy_num_tokens,
            dummy_token_ids,
            self.min_n,
            self.max_n,
            self.max_model_len,
            self.k,
            self.valid_ngram_draft[:n],
            self.valid_ngram_num_drafts[:n],
        )
