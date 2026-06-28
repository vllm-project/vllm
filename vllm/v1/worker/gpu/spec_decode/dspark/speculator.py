# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark speculator.

DSpark is a block speculative decoder stored under the DeepSeek V4 checkpoint's
``mtp.*`` namespace, but it is not the serial DeepSeek MTP architecture. The
draft consumes target hidden states from configured target layers and predicts a
noise-token block through DSpark attention, Markov logits and a confidence head.

This file intentionally refuses to fall back to serial MTP. A wrong fallback
would load cleanly but silently measure the wrong algorithm.
"""

import contextlib
from collections.abc import Callable

import torch
import torch.nn as nn

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.utils.flashinfer import autotune as flashinfer_autotune
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    load_deepseek_v4_dspark_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


class DSparkSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        self.supports_mm_inputs = False
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "DSpark currently requires pipeline parallel size 1."
            )
        if parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError(
                "DSpark currently requires prefill context parallel size 1."
            )
        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "DSpark currently requires decode context parallel size 1."
            )
        hf_config = self.draft_model_config.hf_config
        self.block_size = int(getattr(hf_config, "dspark_block_size", 0) or 0)
        self.noise_token_id = int(getattr(hf_config, "dspark_noise_token_id", -1))
        self.context_window_size = int(getattr(hf_config, "sliding_window", 0) or 0)
        self.target_layer_ids = tuple(
            int(i) for i in getattr(hf_config, "dspark_target_layer_ids", ())
        )
        self.main_hidden_size = hf_config.hidden_size * len(self.target_layer_ids)
        if self.block_size <= 0:
            raise ValueError("DSpark requires dspark_block_size in the model config.")
        if self.noise_token_id < 0:
            raise ValueError(
                "DSpark requires dspark_noise_token_id in the model config."
            )
        if self.context_window_size <= 0:
            raise ValueError("DSpark requires sliding_window in the model config.")
        if not self.target_layer_ids:
            raise ValueError(
                "DSpark requires dspark_target_layer_ids in the model config."
            )
        if self.num_speculative_steps > self.block_size:
            raise ValueError(
                "DSpark num_speculative_tokens must be <= dspark_block_size "
                f"({self.num_speculative_steps} > {self.block_size})."
            )
        self.last_token_indices = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            device=device,
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.active_num_reqs = torch.tensor(0, dtype=torch.int32, device=device)
        self.draft_step_cols = torch.arange(
            self.block_size,
            dtype=torch.int64,
            device=device,
        )
        self.graph_input_ids = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.graph_main_hidden = torch.zeros(
            self.max_num_reqs,
            self.main_hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self.graph_positions = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            device=device,
        )
        self.forward_cudagraph_manager: CudaGraphManager | None = None
        self.enable_prefix_caching = cache_config.enable_prefix_caching
        self._cache_req_ids: list[str | None] = [None] * self.max_num_reqs
        self._cache_trusted_start = [0] * self.max_num_reqs
        self._cache_trusted_end = [0] * self.max_num_reqs
        self._cache_ready = [False] * self.max_num_reqs
        if self.enable_prefix_caching:
            logger.info(
                "DSpark prefix cache support enabled; draft proposals are "
                "deferred after a prefix-cache hit until the private rolling "
                "KV window is rebuilt from fresh target rows."
            )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        del target_attn_layer_names
        return load_deepseek_v4_dspark_model(target_model, self.vllm_config)

    def capture(self, attn_states: dict | None = None) -> None:
        del attn_states

        input_ids = torch.full(
            (1,),
            self.noise_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        context_hidden = torch.zeros(
            (1, self.main_hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        context_positions = torch.zeros(
            (1,),
            dtype=torch.long,
            device=self.device,
        )
        context_query_start_loc = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device=self.device,
        )
        context_num_rejected = torch.zeros(
            (1,),
            dtype=torch.int32,
            device=self.device,
        )
        decode_positions = torch.ones(
            (1,),
            dtype=torch.long,
            device=self.device,
        )

        self.idx_mapping[:1].zero_()
        self.temperature[:1].fill_(1.0)
        self.seeds[:1].zero_()
        self.active_num_reqs.fill_(1)

        batch_descriptor = BatchDescriptor(num_tokens=self.block_size)
        tune_ctx = contextlib.nullcontext()
        if self.vllm_config.kernel_config.enable_flashinfer_autotune:
            # DSpark draft MoE runs with M=block_size (5 for the published
            # checkpoint). FlashInfer's default mapper rounds this to bucket 8,
            # so tune that bucket to make the later non-tuning forward hit cache.
            tune_ctx = flashinfer_autotune(
                True,
                tuning_buckets=(8,),
                round_up=True,
            )
        try:
            with (
                tune_ctx,
                set_forward_context(
                    None,
                    self.vllm_config,
                    num_tokens=self.block_size,
                    batch_descriptor=batch_descriptor,
                    slot_mapping=None,
                ),
            ):
                self.model.precompute_context_kv_flat(
                    context_hidden,
                    context_positions,
                    context_query_start_loc,
                    context_num_rejected,
                    self.idx_mapping[:1],
                    1,
                )
                self.model.forward_spec(
                    input_ids,
                    context_hidden,
                    decode_positions,
                    idx_mapping=self.idx_mapping[:1],
                    temperature=self.temperature,
                    seeds=self.seeds,
                    draft_logits=self.draft_logits,
                    draft_step_cols=self.draft_step_cols,
                    active_num_reqs=self.active_num_reqs,
                    use_fp64_gumbel=self.use_fp64_gumbel,
                )
            torch.cuda.synchronize(self.device)
        finally:
            self.model.reset_dspark_kv_cache()
            self.draft_tokens.zero_()
            if self.draft_logits is not None:
                self.draft_logits.zero_()
        logger.info_once("DSpark eager warmup completed.")

        if self.forward_cudagraph_manager is None:
            return
        if not self.forward_cudagraph_manager.needs_capture():
            return
        logger.info("Capturing DSpark block-forward CUDA graphs...")
        try:
            self.forward_cudagraph_manager.capture(
                self._create_forward_graph_fn,
                progress_bar_desc="Capturing DSpark CUDA graphs",
            )
        except Exception:
            logger.exception(
                "DSpark CUDA graph capture failed; falling back to eager draft forward."
            )
            self.forward_cudagraph_manager = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE
        self.forward_cudagraph_manager = CudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.block_size,
        )

    def _prepare_forward_graph_inputs(self, num_reqs: int) -> None:
        self.graph_input_ids[:num_reqs].fill_(self.noise_token_id)
        self.graph_main_hidden[:num_reqs].zero_()
        self.graph_positions[:num_reqs].fill_(1)
        self.idx_mapping[:num_reqs].copy_(
            torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        )
        self.temperature.fill_(1.0)
        self.seeds.zero_()
        self.active_num_reqs.fill_(num_reqs)
        self.draft_tokens.zero_()
        if self.draft_logits is not None:
            self.draft_logits.zero_()

    def _generate_draft_graph(
        self,
        num_reqs: int,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
    ) -> None:
        num_tokens = num_reqs * self.block_size
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=None,
            batch_descriptor=batch_descriptor,
        ):
            output_ids, _, _ = self.model.forward_spec(
                self.graph_input_ids[:num_reqs],
                self.graph_main_hidden[:num_reqs],
                self.graph_positions[:num_reqs],
                idx_mapping=self.idx_mapping[:num_reqs],
                temperature=self.temperature,
                seeds=self.seeds,
                draft_logits=self.draft_logits,
                draft_step_cols=self.draft_step_cols,
                active_num_reqs=self.active_num_reqs,
                use_fp64_gumbel=self.use_fp64_gumbel,
            )
        steps = self.num_speculative_steps
        self.draft_tokens[:num_reqs, :steps].copy_(output_ids[:, 1 : 1 + steps])

    def _create_forward_graph_fn(
        self,
        desc: BatchExecutionDescriptor,
        warmup: bool,
    ) -> tuple[Callable[[CUDAGraphMode], None], AttentionState]:
        del warmup
        num_reqs = desc.num_reqs or max(1, desc.num_tokens // self.block_size)
        num_tokens_across_dp = (
            torch.full(
                (self.dp_size,),
                desc.num_tokens,
                dtype=torch.int32,
                device="cpu",
            )
            if self.dp_size > 1
            else None
        )
        self._prepare_forward_graph_inputs(num_reqs)
        fwd = lambda cg_mode: self._generate_draft_graph(
            num_reqs,
            num_tokens_across_dp,
            cg_mode,
        )
        return fwd, AttentionState(None, {})

    def _update_context_cache_state(self, input_batch: InputBatch) -> bool:
        """Track whether DSpark's private rolling-KV rows are complete.

        vLLM prefix cache can start a request with target KV already available
        while DSpark's private target-KV window is empty for those cached
        tokens.  We index the DSpark cache by persistent request-state index and
        only allow proposals once fresh target forwards have repopulated the
        whole rolling window needed by the current anchor position.
        """
        all_ready = True
        for req_idx in range(input_batch.num_reqs):
            state_idx = int(input_batch.idx_mapping_np[req_idx])
            req_id = input_batch.req_ids[req_idx]
            computed_start = int(input_batch.num_computed_tokens_np[req_idx])
            scheduled = int(input_batch.num_scheduled_tokens[req_idx])
            computed_end = computed_start + scheduled

            if self._cache_req_ids[state_idx] != req_id:
                self._cache_req_ids[state_idx] = req_id
                self._cache_trusted_start[state_idx] = computed_start
                self._cache_trusted_end[state_idx] = computed_start
                self._cache_ready[state_idx] = computed_start == 0

            if computed_start > self._cache_trusted_end[state_idx]:
                # A discontinuity means target KV came from vLLM's cache or a
                # resumed request, not from DSpark's private cache population.
                self._cache_trusted_start[state_idx] = computed_start
                self._cache_trusted_end[state_idx] = computed_start
                self._cache_ready[state_idx] = False
            elif computed_start < self._cache_trusted_start[state_idx]:
                # Request state was rewound. Treat the current row as a fresh
                # contiguous region.
                self._cache_trusted_start[state_idx] = computed_start
                self._cache_trusted_end[state_idx] = computed_start
                self._cache_ready[state_idx] = computed_start == 0

            self._cache_trusted_end[state_idx] = max(
                self._cache_trusted_end[state_idx],
                computed_end,
            )
            if scheduled <= 0:
                all_ready = False
                continue

            anchor_pos = computed_end - 1
            needed_start = max(0, anchor_pos - self.context_window_size + 1)
            if (
                self._cache_trusted_start[state_idx] <= needed_start
                and self._cache_trusted_end[state_idx] > anchor_pos
            ):
                self._cache_ready[state_idx] = True

            if not self._cache_ready[state_idx]:
                all_ready = False

        return all_ready

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict,
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            attn_metadata,
            last_hidden_states,
            mm_inputs,
        )
        num_reqs = input_batch.num_reqs
        if any(
            req_id.startswith(("_warmup_", "_v2_mixed_warmup", "_sparse_mla_v2_warmup"))
            for req_id in input_batch.req_ids[:num_reqs]
        ):
            self.draft_tokens[:num_reqs].zero_()
            return self.draft_tokens[:num_reqs]
        if dummy_run and skip_attn_for_dummy_run:
            self.draft_tokens[:num_reqs].zero_()
            return self.draft_tokens[:num_reqs]
        if not aux_hidden_states:
            raise RuntimeError("DSpark requires target auxiliary hidden states.")
        if len(aux_hidden_states) != len(self.target_layer_ids):
            raise RuntimeError(
                "DSpark auxiliary hidden-state count mismatch: "
                f"expected {len(self.target_layer_ids)}, got {len(aux_hidden_states)}."
            )

        with record_function_or_nullcontext("vllm:v2/speculator/dspark/prepare"):
            prepare_prefill_inputs(
                self.last_token_indices,
                self.current_draft_step,
                self.input_buffers,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                self.max_num_reqs,
            )
            self._copy_request_inputs(
                num_reqs,
                input_batch.idx_mapping,
                temperature,
                seeds,
            )
            self.active_num_reqs.fill_(num_reqs)
            last_token_indices = self.last_token_indices[:num_reqs]
            # prepare_prefill_inputs writes the fresh target token at each
            # request's last_token_index. DSpark must use that same token and
            # position together with the corresponding aux hidden state.
            input_ids = self.input_buffers.input_ids[last_token_indices]
            positions = self.input_buffers.positions[last_token_indices]
            if dummy_run:
                self.draft_tokens[:num_reqs].zero_()
                return self.draft_tokens[:num_reqs]
            main_hidden_all = torch.cat(aux_hidden_states, dim=-1)
            self.model.precompute_context_kv_flat(
                main_hidden_all[: input_batch.num_tokens],
                input_batch.positions[: input_batch.num_tokens],
                input_batch.query_start_loc,
                num_rejected[:num_reqs],
                input_batch.idx_mapping,
                num_reqs,
            )
            context_cache_ready = self._update_context_cache_state(input_batch)
            # The initial target prefill only initializes DSpark's rolling
            # context KV.  The public DSpark algorithm starts proposing once a
            # real decode token exists, so position 0 must not run the draft
            # block model.
            anchor_lengths = (
                input_batch.num_computed_tokens_np[:num_reqs]
                + input_batch.num_scheduled_tokens[:num_reqs]
            )
            if int(anchor_lengths.min()) <= 1:
                self.draft_tokens[:num_reqs].zero_()
                return self.draft_tokens[:num_reqs]
            if not context_cache_ready:
                return self.draft_tokens[:num_reqs, :0]
            main_hidden = main_hidden_all[last_token_indices]

        num_query_tokens = num_reqs * self.block_size
        batch_desc = None
        if self.forward_cudagraph_manager is not None:
            batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
                self.forward_cudagraph_manager,
                num_reqs,
                num_query_tokens,
                uniform_token_count=self.block_size,
                dp_size=self.dp_size,
                dp_rank=self.dp_rank,
                need_eager=is_profile,
            )
        if batch_desc is not None and batch_desc.cg_mode == CUDAGraphMode.FULL:
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            self.graph_input_ids[:num_reqs].copy_(input_ids)
            self.graph_main_hidden[:num_reqs].copy_(main_hidden)
            self.graph_positions[:num_reqs].copy_(positions)
            if num_reqs_padded > num_reqs:
                self.graph_input_ids[num_reqs:num_reqs_padded].fill_(
                    self.noise_token_id
                )
                self.graph_main_hidden[num_reqs:num_reqs_padded].zero_()
                self.graph_positions[num_reqs:num_reqs_padded].fill_(1)
                self.idx_mapping[num_reqs:num_reqs_padded].zero_()
            self.active_num_reqs.fill_(num_reqs)
            self.forward_cudagraph_manager.run_fullgraph(batch_desc)
            return self.draft_tokens[:num_reqs]

        with record_function_or_nullcontext("vllm:v2/speculator/dspark/forward"):
            # DSpark uses its own TileLang attention path, but the inherited
            # DeepSeek MoE layers still read vLLM's forward context to resolve
            # static layer state.
            batch_descriptor = BatchDescriptor(num_tokens=num_query_tokens)
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_query_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
                batch_descriptor=batch_descriptor,
            ):
                output_ids, logits, _confidence = self.model.forward_spec(
                    input_ids,
                    main_hidden,
                    positions,
                    idx_mapping=self.idx_mapping[:num_reqs],
                    temperature=self.temperature,
                    seeds=self.seeds,
                    draft_logits=self.draft_logits,
                    draft_step_cols=self.draft_step_cols,
                    active_num_reqs=self.active_num_reqs,
                    use_fp64_gumbel=self.use_fp64_gumbel,
                )
        steps = self.num_speculative_steps
        self.draft_tokens[:num_reqs, :steps].copy_(output_ids[:, 1 : 1 + steps])
        return self.draft_tokens[:num_reqs]
