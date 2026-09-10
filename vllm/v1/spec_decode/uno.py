# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uno parallel drafting with a shared target model and draft-only LoRA.

Each request queries one base seed followed by K-1 noise tokens. All K rows
predict candidates; ordinary verification recomputes the seed and candidates.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.uno_noise import fill_uno_noise
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

UNO_LORA_INT_ID = 1_000_003


def uno_query_layout(
    batch_size: int,
    k: int,
    uno_lora_id: int,
    device: torch.device,
    num_input_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Return noise mask, sampled rows, and noise-only adapter mapping."""
    if batch_size < 0 or k < 1:
        raise ValueError("Uno requires a nonnegative batch size and K >= 1")
    num_queries = batch_size * k
    if num_input_tokens is None:
        num_input_tokens = num_queries
    if num_input_tokens < num_queries:
        raise ValueError("Uno input capacity is smaller than its query count")
    rows = torch.arange(num_input_tokens, device=device)
    is_noise = (rows < num_queries) & (rows % k != 0)
    sample_indices = torch.arange(num_queries, dtype=torch.int32, device=device)
    lora_mapping = tuple(
        uno_lora_id if i < num_queries and i % k else 0 for i in range(num_input_tokens)
    )
    return is_noise, sample_indices, lora_mapping


class UnoProposer(SpecDecodeBaseProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.use_uno()
        if not current_platform.is_cuda() or device.type != "cuda":
            raise ValueError("Uno currently requires an NVIDIA CUDA device")
        self.runner = runner
        super().__init__(
            vllm_config, device, pass_hidden_states_to_model=False, runner=runner
        )
        self.uno_lora_id = UNO_LORA_INT_ID
        self.lora_request = LoRARequest(
            lora_name="uno",
            lora_int_id=self.uno_lora_id,
            lora_path=self.speculative_config.uno_lora_path,
        )
        self._lora_hook: Callable[[tuple[int, ...] | None], None] | None = None
        self._pending_lora_map: tuple[int, ...] = ()
        self._step = 0
        self.max_query_tokens = self.max_batch_size * self.num_speculative_tokens
        self.input_ids = torch.zeros(
            self.max_query_tokens, dtype=torch.int32, device=device
        )
        self.positions = torch.zeros(
            self.max_query_tokens, dtype=torch.int64, device=device
        )
        self._slot_mapping_buffer = torch.full(
            (self.max_query_tokens,), PADDING_SLOT_ID, dtype=torch.int64, device=device
        )

    @override
    def _init_parallel_drafting_params(self) -> None:
        # Noise is generated from the configured vocabulary range, without masks.
        self.parallel_drafting_token_id = 0

    @override
    def _raise_if_padded_drafter_batch_disabled(self) -> None:
        # The Uno query layout does not expand or reuse the target input buffer.
        pass

    @override
    def _warn_if_multimodal(self) -> None:
        if self.supports_mm_inputs:
            raise ValueError("Uno currently supports text-only models")

    def set_lora_hook(self, fn: Callable[[tuple[int, ...] | None], None]) -> None:
        """Set the runner callback; None restores the base-model mapping."""
        self._lora_hook = fn

    @contextmanager
    def _draft_lora(self, mapping: tuple[int, ...]) -> Iterator[None]:
        if self._lora_hook is None:
            raise RuntimeError("Uno draft adapter callback has not been installed")
        try:
            self._lora_hook(mapping)
            yield
        finally:
            self._lora_hook(None)

    @override
    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model
        all_attn = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = {
            name
            for name, layer in all_attn.items()
            if layer.get_kv_cache_spec(self.vllm_config) is not None
        }
        if not self._draft_attn_layer_names:
            raise ValueError("Uno requires target attention layers with a KV cache")
        for name in self._draft_attn_layer_names:
            if all_attn[name].get_attn_backend().get_name() != "FLASH_ATTN":
                raise ValueError("Uno currently requires the FLASH_ATTN backend")

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError("Uno requires one homogeneous full-attention KV group")
        spec = groups[0].kv_cache_spec
        if (
            type(spec) is not FullAttentionSpec
            or spec.sliding_window is not None
            or spec.attention_chunk_size is not None
        ):
            raise ValueError("Uno requires homogeneous full attention")
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    @override
    def model_returns_tuple(self) -> bool:
        return False

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
        batch_size = cad.batch_size()
        k = self.num_speculative_tokens
        num_queries = batch_size * k
        is_noise, sample_indices, self._pending_lora_map = uno_query_layout(
            batch_size, k, self.uno_lora_id, self.device
        )
        valid_ends = cad.query_start_loc[1 : batch_size + 1]
        seq_lens = cad.seq_lens
        if num_rejected_tokens_gpu is not None:
            valid_ends = valid_ends - num_rejected_tokens_gpu
            seq_lens = seq_lens - num_rejected_tokens_gpu
        first_positions = target_positions[valid_ends.long() - 1] + 1
        offsets = torch.arange(k, device=self.device)
        positions = first_positions[:, None] + offsets
        self.positions[:num_queries].copy_(positions.reshape(-1))
        self.input_ids[:num_queries].copy_(next_token_ids.repeat_interleave(k))
        req_seeds = (
            torch.arange(batch_size, device=self.device, dtype=torch.int64)
            + (self.speculative_config.uno_noise_seed & ((1 << 62) - 1))
        ).repeat_interleave(k)
        self._step += 1
        noise_high = self.speculative_config.uno_mask_token_id
        assert noise_high is not None
        fill_uno_noise(
            self.input_ids[:num_queries],
            is_noise,
            req_seeds,
            self._step,
            1,
            noise_high,
        )

        # Noise KV occupies the real suffix, so later noise queries see it.
        # Verification overwrites this suffix with true-token KV before commitment.
        block_numbers = positions // self.block_size
        block_table = cad.block_table_tensor[:batch_size]
        safe_blocks = block_numbers.clamp(max=block_table.shape[1] - 1)
        block_ids = block_table.gather(1, safe_blocks.long()).to(torch.int64)
        slots = block_ids * self.block_size + positions % self.block_size
        valid_slots = (
            (positions < self.max_model_len)
            & (block_numbers < block_table.shape[1])
            & (block_ids != 0)
        )
        slots.masked_fill_(~valid_slots, PADDING_SLOT_ID)
        self._slot_mapping_buffer[:num_queries].copy_(slots.reshape(-1))
        query_start_loc_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * k
        upper = cad.seq_lens_cpu_upper_bound
        new_cad = CommonAttentionMetadata(
            query_start_loc=query_start_loc_cpu.to(self.device),
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens + k,
            seq_lens_cpu_upper_bound=None if upper is None else upper + k,
            num_reqs=batch_size,
            num_actual_tokens=num_queries,
            max_query_len=k,
            max_seq_len=cad.max_seq_len + k,
            block_table_tensor=block_table,
            slot_mapping=self._slot_mapping_buffer[:num_queries],
            causal=True,
        )
        return num_queries, sample_indices, new_cad

    @override
    @torch.inference_mode()
    def propose(
        self,
        num_speculative_tokens: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        self._last_draft_probs = None
        self._pending_lora_map = ()
        k = num_speculative_tokens
        if not 0 <= k <= self.speculative_config.num_speculative_tokens:
            raise ValueError("Uno draft length exceeds the configured capacity")
        self.num_speculative_tokens = k
        if k == 0:
            return torch.empty(
                common_attn_metadata.batch_size(),
                0,
                device=self.device,
                dtype=torch.int64,
            )
        try:
            num_tokens, sample_indices, cad = self.set_inputs_first_pass(
                target_token_ids,
                next_token_ids,
                target_positions,
                target_hidden_states,
                token_indices_to_sample,
                common_attn_metadata,
                num_rejected_tokens_gpu,
            )
            _, per_layer_metadata = self.build_per_group_and_layer_attn_metadata(cad)
            with self._draft_lora(self._pending_lora_map):
                with set_forward_context(
                    per_layer_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=self._get_slot_mapping(num_tokens),
                ):
                    hidden_states = self.model(
                        input_ids=self.input_ids[:num_tokens],
                        positions=self.positions[:num_tokens],
                        inputs_embeds=None,
                    )
                draft_ids, draft_probs = self._sample_draft_tokens(
                    hidden_states[sample_indices], sampling_metadata
                )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, k, draft_probs.shape[-1]
                ).contiguous()
            return draft_ids.view(-1, k)
        except Exception:
            self._last_draft_probs = None
            raise
        finally:
            self._pending_lora_map = ()

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # The target owns graph capture; draft execution is always eager.
        if is_graph_capturing or num_tokens == 0:
            return
        k = self.speculative_config.num_speculative_tokens
        batch_size = min(self.max_batch_size, (num_tokens + k - 1) // k)
        num_queries = batch_size * k
        _, _, mapping = uno_query_layout(batch_size, k, self.uno_lora_id, self.device)
        self.input_ids[:num_queries].zero_()
        self.positions[:num_queries].zero_()
        self._slot_mapping_buffer[:num_queries].fill_(PADDING_SLOT_ID)
        with self._draft_lora(mapping):
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_queries,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                # Profiling runs before the shared target KV cache is allocated.
                slot_mapping=(
                    self._get_slot_mapping(num_queries) if slot_mappings else {}
                ),
            ):
                hidden_states = self.model(
                    input_ids=self.input_ids[:num_queries],
                    positions=self.positions[:num_queries],
                    inputs_embeds=None,
                )
            logits = self.model.compute_logits(hidden_states)
            if self._enable_probabilistic_draft_probs:
                assert self.runner is not None
                metadata = replace(
                    self.runner.input_batch.sampling_metadata,
                    temperature=torch.ones(num_queries, device=self.device),
                    all_greedy=False,
                    all_random=True,
                )
                self._sample_from_logits(logits, metadata)
