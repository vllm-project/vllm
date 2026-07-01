# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, graph_capture
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.dspark_sampling import (
    sample_dspark_markov_block,
    sample_dspark_markov_block_fused,
)

logger = init_logger(__name__)


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def _spec_bool(spec_config: Any, attr: str, env_name: str) -> bool:
    return bool(getattr(spec_config, attr, False)) or _env_bool(env_name)


class _DSparkForwardCUDAGraph:
    """Opt-in fixed-shape graph wrapper for the DSpark draft forward.

    The normal vLLM drafter cudagraph dispatcher is not initialized for this
    DSpark path. This wrapper is deliberately narrower: it captures only the
    draft model forward after DSpark context KV has been prepared, and only when
    the proposer supplied stable buffers for the dynamic per-proposal inputs.
    """

    def __init__(self, proposer: "DSparkProposer", model: Any) -> None:
        self.proposer = proposer
        self.model = model
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: torch.Tensor | None = None
        self.capture_args: tuple[Any, ...] | None = None
        self.capture_kwargs: dict[str, Any] | None = None
        self.input_ptrs: tuple[int, ...] | None = None
        self.input_key: tuple[Any, ...] | None = None
        self.disabled = False
        self._num_warmups = 3

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def unwrap(self) -> Any:
        return self.model

    def _make_key(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        main_positions: torch.Tensor | None,
        main_x: torch.Tensor | None,
    ) -> tuple[Any, ...] | None:
        if input_ids is None or hidden_states is None or main_positions is None:
            return None
        if main_x is not None:
            return None
        tensors = (input_ids, positions, hidden_states, main_positions)
        return tuple(
            (tuple(t.shape), t.dtype, t.device.type, t.device.index) for t in tensors
        )

    def _make_ptrs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        main_positions: torch.Tensor | None,
    ) -> tuple[int, ...] | None:
        if input_ids is None or hidden_states is None or main_positions is None:
            return None
        return (
            input_ids.data_ptr(),
            positions.data_ptr(),
            hidden_states.data_ptr(),
            main_positions.data_ptr(),
        )

    def __call__(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        main_positions: torch.Tensor | None = None,
        main_x: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        proposer = self.proposer
        if (
            self.disabled
            or not proposer._dspark_forward_graph_call_ready
            or inputs_embeds is not None
            or kwargs
        ):
            proposer._dspark_forward_graph_call_ready = False
            return self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                main_positions=main_positions,
                main_x=main_x,
                **kwargs,
            )

        proposer._dspark_forward_graph_call_ready = False
        try:
            if torch.cuda.is_current_stream_capturing():
                return self.model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                    main_positions=main_positions,
                    main_x=main_x,
                )
        except RuntimeError:
            return self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                main_positions=main_positions,
                main_x=main_x,
            )

        input_key = self._make_key(
            input_ids,
            positions,
            hidden_states,
            main_positions,
            main_x,
        )
        input_ptrs = self._make_ptrs(input_ids, positions, hidden_states, main_positions)
        if input_key is None or input_ptrs is None:
            return self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                main_positions=main_positions,
                main_x=main_x,
            )

        if self.graph is not None:
            if input_key != self.input_key or input_ptrs != self.input_ptrs:
                return self.model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                    main_positions=main_positions,
                    main_x=main_x,
                )
            self.graph.replay()
            assert self.output is not None
            return self.output

        graph = torch.cuda.CUDAGraph()
        capture_kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            inputs_embeds=None,
            main_positions=main_positions,
            main_x=None,
        )
        try:
            # Bare torch.cuda.graph() is not enough here: the draft forward
            # contains a TP all-reduce (o_proj), and vLLM's low-latency
            # all-reduce backends (custom-allreduce/FlashInfer) require the
            # rank-synchronized capture()/register_graph_buffers() handshake
            # that get_tp_group().graph_capture() performs (see
            # custom_all_reduce.py's capture()) before any of their buffer
            # addresses are safe to replay. Warm up at this exact shape first
            # so first-touch JIT/algo-selection/communicator setup happens
            # eagerly, not while capturing, mirroring
            # GPUModelRunner._warmup_and_capture.
            with graph_capture(device=self.proposer.device):
                for _ in range(self._num_warmups):
                    self.model(**capture_kwargs)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    output = self.model(**capture_kwargs)
            self.graph = graph
            self.output = output
            self.capture_args = ()
            self.capture_kwargs = capture_kwargs
            self.input_key = input_key
            self.input_ptrs = input_ptrs
            logger.info(
                "Captured DSpark draft forward CUDA graph for input shape %s "
                "via tp_group.graph_capture (%d warmups)",
                tuple(input_ids.shape),
                self._num_warmups,
            )
            return output
        except Exception:
            self.disabled = True
            logger.warning(
                "Disabling DSpark draft forward CUDA graph after capture failure",
                exc_info=True,
            )
            return self.model(**capture_kwargs)


class DSparkProposer(DFlashProposer):
    """Fixed-block DSpark proposer.

    This MVP reuses DFlash's parallel-drafting input layout, then replaces the
    independent block sampler with DSpark's sequential Markov-head correction.
    Confidence-scheduled variable-prefix verification is deliberately out of
    scope for this first pass.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        super().__init__(vllm_config, device, runner)
        spec_config = vllm_config.speculative_config
        hf_config = self.draft_model_config.hf_config
        aux_hidden_size = int(hf_config.hidden_size) * len(
            getattr(hf_config, "dspark_target_layer_ids", ())
        )
        if aux_hidden_size and aux_hidden_size != self.hidden_size:
            self.hidden_size = aux_hidden_size
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=device,
            )
            if self.parallel_drafting_hidden_state_tensor is not None:
                self.parallel_drafting_hidden_state_tensor = torch.empty(
                    self.hidden_size, dtype=self.dtype, device=device
                )
        self._dspark_first_prev_token_ids: torch.Tensor | None = None
        self._dspark_main_hidden: torch.Tensor | None = None
        self._dspark_main_positions: torch.Tensor | None = None
        self._dspark_context_hidden_states: torch.Tensor | None = None
        self._dspark_context_positions: torch.Tensor | None = None
        self._dspark_query_start_loc: torch.Tensor | None = None
        self._dspark_main_indices: torch.Tensor | None = None
        self._dspark_batch_size: int = 0
        self._dspark_num_rejected_tokens: torch.Tensor | None = None
        self._dspark_tokens_out = torch.empty(
            (self.max_batch_size, self.num_speculative_tokens),
            dtype=torch.int64,
            device=device,
        )
        self._dspark_draft_probs_out: torch.Tensor | None = None
        self.use_fused_markov_sampler = _spec_bool(
            spec_config,
            "dspark_fused_markov_sampler",
            "VLLM_DSPARK_FUSED_MARKOV_SAMPLER",
        )
        self.use_forward_cudagraph = _spec_bool(
            spec_config,
            "dspark_forward_cudagraph",
            "VLLM_DSPARK_FORWARD_CUDAGRAPH",
        )
        self.allow_tp_forward_cudagraph = _spec_bool(
            spec_config,
            "dspark_forward_cudagraph_allow_tp",
            "VLLM_DSPARK_FORWARD_CUDAGRAPH_ALLOW_TP",
        )
        self._dspark_forward_graph_call_ready = False
        self._dspark_forward_graph_hidden: torch.Tensor | None = None
        self._dspark_forward_graph_main_positions: torch.Tensor | None = None
        self._dspark_dummy_query_start_loc = torch.empty(
            (2,), dtype=torch.int32, device=device
        )

    @override
    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        if self.use_forward_cudagraph:
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if tp_size > 1 and not self.allow_tp_forward_cudagraph:
                self.use_forward_cudagraph = False
                logger.warning(
                    "DSpark draft forward CUDA graph disabled for tensor "
                    "parallel size %s. Set "
                    "VLLM_DSPARK_FORWARD_CUDAGRAPH_ALLOW_TP=1 to opt in to "
                    "the TP graph path.",
                    tp_size,
                )
                return
            self.model = _DSparkForwardCUDAGraph(self, self.model)
            logger.info("DSpark draft forward CUDA graph is enabled.")

    @override
    def _maybe_share_embeddings(self, target_language_model: Any) -> None:
        if get_pp_group().world_size != 1:
            logger.info(
                "DSpark draft model keeps separate embeddings under pipeline "
                "parallelism."
            )
            return

        target_inner_model = getattr(target_language_model, "model", None)
        target_embed_tokens = getattr(target_inner_model, "embed_tokens", None)
        if target_embed_tokens is None or not hasattr(self.model, "embed_tokens"):
            logger.warning(
                "DSpark could not share target embeddings; keeping draft "
                "embed_tokens."
            )
            return

        del self.model.embed_tokens
        self.model.embed_tokens = target_embed_tokens
        logger.info("Shared target model embeddings with DSpark draft model.")

    @override
    def _maybe_share_lm_head(self, target_language_model: Any) -> None:
        if get_pp_group().world_size != 1:
            logger.info(
                "DSpark draft model keeps separate lm_head under pipeline "
                "parallelism."
            )
            return

        if not hasattr(target_language_model, "lm_head") or not hasattr(
            self.model, "head"
        ):
            logger.warning(
                "DSpark could not share target lm_head; keeping draft head."
            )
            return

        del self.model.head
        self.model.head = target_language_model.lm_head
        logger.info("Shared target model lm_head with DSpark draft model.")

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        del kv_cache_config, kernel_block_sizes
        self.draft_attn_groups = []
        self.kv_cache_gid = 0
        self.block_size = 1

    @override
    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del num_tokens, slot_mapping
        return {}

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        del target_token_ids
        # The sampled target token is the first Markov predecessor for each
        # speculative block. The DSpark projection consumes the target hidden
        # states that produced those sampled tokens.
        batch_size = cad.batch_size()
        self._dspark_first_prev_token_ids = next_token_ids
        valid_context_end = cad.query_start_loc[1:]
        if num_rejected_tokens_gpu is not None:
            valid_context_end = valid_context_end - num_rejected_tokens_gpu
        sample_indices = valid_context_end - 1
        self._dspark_main_indices = sample_indices
        self._dspark_main_hidden = target_hidden_states[sample_indices.long()]
        self._dspark_main_positions = target_positions[sample_indices.long()]
        self._dspark_context_hidden_states = target_hidden_states
        self._dspark_context_positions = target_positions
        self._dspark_query_start_loc = cad.query_start_loc
        self._dspark_batch_size = batch_size
        self._dspark_num_rejected_tokens = num_rejected_tokens_gpu

        num_tokens = batch_size * self.num_speculative_tokens
        draft_input_ids = self.input_ids[:num_tokens].view(
            batch_size, self.num_speculative_tokens
        )
        draft_input_ids.fill_(self.parallel_drafting_token_id)
        draft_input_ids[:, 0] = next_token_ids[:batch_size]
        draft_offsets = torch.arange(
            1,
            self.num_speculative_tokens + 1,
            device=self.device,
            dtype=self.positions.dtype,
        )
        self.positions[:num_tokens].view(
            batch_size, self.num_speculative_tokens
        ).copy_(
            self._dspark_main_positions.to(self.positions.dtype).unsqueeze(1)
            + draft_offsets.unsqueeze(0)
        )
        token_indices_to_sample = torch.arange(
            num_tokens, device=self.device, dtype=torch.int32
        )
        return num_tokens, token_indices_to_sample, cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        del is_graph_capturing, slot_mappings
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )
        self.input_ids[:num_input_tokens].fill_(self.parallel_drafting_token_id)
        with torch.inference_mode():
            self._dspark_dummy_query_start_loc[0] = 0
            self._dspark_dummy_query_start_loc[1] = num_input_tokens
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_input_tokens],
                self._get_positions(num_input_tokens),
                query_start_loc=self._dspark_dummy_query_start_loc,
                batch_size=1,
            )
            with self._forward_context(
                num_input_tokens,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            ):
                self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self._get_positions(num_input_tokens),
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=None,
                )

    def _forward_context(
        self,
        num_input_tokens: int,
        num_tokens_across_dp: int,
        cudagraph_runtime_mode,
    ):
        from vllm.forward_context import set_forward_context

        return set_forward_context(
            {},
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping={},
        )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        del mm_embed_inputs
        if self._dspark_main_hidden is None:
            raise RuntimeError("DSpark target hidden states were not initialized")
        context_main_x = None
        if (
            self._dspark_context_hidden_states is not None
            and self._dspark_context_positions is not None
            and self._dspark_query_start_loc is not None
        ):
            context_main_x = self.model.precompute_and_store_context_kv(
                self._dspark_context_hidden_states,
                self._dspark_context_positions,
                query_start_loc=self._dspark_query_start_loc,
                batch_size=self._dspark_batch_size,
                num_rejected_tokens=self._dspark_num_rejected_tokens,
            )
        model_kwargs = dict(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self._get_positions(num_input_tokens),
            hidden_states=self._dspark_main_hidden,
            inputs_embeds=None,
            main_positions=self._dspark_main_positions,
        )
        del context_main_x
        if self.use_forward_cudagraph:
            model_kwargs = self._maybe_prepare_forward_cudagraph_inputs(
                model_kwargs,
                num_input_tokens,
            )
        return (model_kwargs, num_input_tokens)

    def _maybe_prepare_forward_cudagraph_inputs(
        self,
        model_kwargs: dict[str, Any],
        num_input_tokens: int,
    ) -> dict[str, Any]:
        self._dspark_forward_graph_call_ready = False
        if not torch.cuda.is_available():
            return model_kwargs
        if (
            self._dspark_batch_size != 1
            or num_input_tokens < self.num_speculative_tokens
            or model_kwargs.get("main_x") is not None
            or model_kwargs.get("inputs_embeds") is not None
        ):
            return model_kwargs

        hidden_states = model_kwargs.get("hidden_states")
        main_positions = model_kwargs.get("main_positions")
        if not isinstance(hidden_states, torch.Tensor) or not isinstance(
            main_positions, torch.Tensor
        ):
            return model_kwargs
        if hidden_states.shape[0] != 1 or main_positions.shape[0] != 1:
            return model_kwargs

        if (
            self._dspark_forward_graph_hidden is None
            or self._dspark_forward_graph_hidden.shape != hidden_states.shape
            or self._dspark_forward_graph_hidden.dtype != hidden_states.dtype
            or self._dspark_forward_graph_hidden.device != hidden_states.device
        ):
            self._dspark_forward_graph_hidden = torch.empty_like(hidden_states)
        if (
            self._dspark_forward_graph_main_positions is None
            or self._dspark_forward_graph_main_positions.shape != main_positions.shape
            or self._dspark_forward_graph_main_positions.dtype != main_positions.dtype
            or self._dspark_forward_graph_main_positions.device != main_positions.device
        ):
            self._dspark_forward_graph_main_positions = torch.empty_like(main_positions)

        self._dspark_forward_graph_hidden.copy_(hidden_states)
        self._dspark_forward_graph_main_positions.copy_(main_positions)

        graph_kwargs = dict(model_kwargs)
        graph_kwargs["hidden_states"] = self._dspark_forward_graph_hidden
        graph_kwargs["main_positions"] = self._dspark_forward_graph_main_positions
        self._dspark_forward_graph_call_ready = True
        return graph_kwargs

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del spec_step_idx
        if self._dspark_first_prev_token_ids is None:
            raise RuntimeError("DSpark first predecessor tokens were not initialized")

        greedy_draft = sampling_metadata.all_greedy
        logits = self._compute_logits(hidden_states)
        if logits.shape[0] % self.num_speculative_tokens != 0:
            raise ValueError(
                "DSpark logits rows must be request-major blocks: "
                f"rows={logits.shape[0]}, block={self.num_speculative_tokens}"
            )
        batch_size = logits.shape[0] // self.num_speculative_tokens
        base_logits = logits.view(batch_size, self.num_speculative_tokens, -1)

        model_apply = getattr(self.model, "apply_dspark_markov_bias", None)
        if model_apply is None:
            raise NotImplementedError(
                "DSpark draft model must implement apply_dspark_markov_bias("
                "base_logits, prev_token_ids, step_idx)"
            )

        return_probs = self._enable_probabilistic_draft_probs and not greedy_draft
        tokens_out = self._dspark_tokens_out[:batch_size, : self.num_speculative_tokens]
        draft_probs_out = None
        if return_probs:
            vocab_size = base_logits.shape[-1]
            if (
                self._dspark_draft_probs_out is None
                or self._dspark_draft_probs_out.shape[0] < batch_size
                or self._dspark_draft_probs_out.shape[1] < self.num_speculative_tokens
                or self._dspark_draft_probs_out.shape[2] != vocab_size
                or self._dspark_draft_probs_out.device != base_logits.device
            ):
                self._dspark_draft_probs_out = torch.empty(
                    (batch_size, self.num_speculative_tokens, vocab_size),
                    dtype=torch.float32,
                    device=base_logits.device,
                )
            draft_probs_out = self._dspark_draft_probs_out

        if self.use_fused_markov_sampler and return_probs:
            return sample_dspark_markov_block_fused(
                base_logits,
                self._dspark_first_prev_token_ids[:batch_size],
                model_apply,
                sampling_metadata,
                use_fp64_gumbel=self.use_fp64_gumbel,
                tokens_out=tokens_out,
                draft_probs_out=draft_probs_out,
            )

        return sample_dspark_markov_block(
            base_logits,
            self._dspark_first_prev_token_ids[:batch_size],
            model_apply,
            sampling_metadata,
            return_probs=return_probs,
            use_fp64_gumbel=self.use_fp64_gumbel,
            tokens_out=tokens_out,
            draft_probs_out=draft_probs_out,
        )
