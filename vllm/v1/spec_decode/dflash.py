# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer


class DFlashProposer(EagleProposer):
    """Dedicated proposer for method='dflash' with DFlash-specific config."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        self.runtime_mode = self._get_dflash_runtime_mode_from_config()
        self.mask_token_id = self._resolve_mask_token_id_from_config()

    def _get_dflash_runtime_mode_from_config(self) -> str:
        dflash_config = getattr(self.draft_model_config.hf_config, "dflash_config", {})
        if not isinstance(dflash_config, dict):
            dflash_config = {}

        runtime_mode = dflash_config.get("runtime_mode", "shared_eagle")
        if runtime_mode not in ("shared_eagle", "block_drafting"):
            raise ValueError(
                "Invalid dflash_config.runtime_mode. "
                "Expected one of ['shared_eagle', 'block_drafting'], "
                f"got {runtime_mode!r}."
            )
        return runtime_mode

    def _resolve_mask_token_id_from_config(self) -> int | None:
        hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(hf_config, "dflash_config", {})
        if not isinstance(dflash_config, dict):
            dflash_config = {}

        candidates = (
            dflash_config.get("mask_token_id"),
            getattr(hf_config, "mask_token_id", None),
            getattr(hf_config, "pard_token", None),
            getattr(hf_config, "ptd_token_id", None),
            getattr(hf_config, "pad_token_id", None),
        )
        for candidate in candidates:
            if not isinstance(candidate, int):
                continue
            if candidate < 0:
                continue
            vocab_size = getattr(hf_config, "vocab_size", None)
            if isinstance(vocab_size, int) and candidate >= vocab_size:
                raise ValueError(
                    "Resolved DFlash mask token id is out of vocab bounds: "
                    f"{candidate} >= vocab_size ({vocab_size})."
                )
            return candidate
        return None

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        DFlash config precedence:
        1) dflash_config.use_aux_hidden_state
        2) eagle_config.use_aux_hidden_state
        3) default True
        """
        use_aux_hidden_state = True

        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if isinstance(eagle_config, dict):
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)

        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if isinstance(dflash_config, dict):
            use_aux_hidden_state = dflash_config.get(
                "use_aux_hidden_state", use_aux_hidden_state
            )

        return use_aux_hidden_state

    def _maybe_combine_target_hidden_states(
        self, target_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self.model, "combine_hidden_states"):
            return target_hidden_states
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        if target_hidden_states.shape[-1] != self.hidden_size:
            raise RuntimeError(
                "DFlash combined hidden size mismatch: "
                f"expected {self.hidden_size}, got {target_hidden_states.shape[-1]}."
            )
        return target_hidden_states

    def _propose_shared_eagle(
        self,
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
        return super().propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            mm_embed_inputs=mm_embed_inputs,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            slot_mappings=slot_mappings,
        )

    def _propose_block_drafting(
        self,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> torch.Tensor:
        if self.mask_token_id is None:
            raise ValueError(
                "DFlash block_drafting requires a resolved mask token id. "
                "Set dflash_config.mask_token_id (or compatible fallback fields)."
            )

        batch_size = common_attn_metadata.batch_size()
        if (
            self.uses_mrope
            or self.uses_xdrope_dim > 0
            or self.draft_uses_xdrope_dim > 0
        ):
            raise NotImplementedError(
                "DFlash block_drafting does not support M-RoPE/XD-RoPE in this release."
            )
        if target_positions.dim() != 1:
            raise NotImplementedError(
                "DFlash block_drafting currently expects 1D position ids."
            )
        if self.indexer_layer_names:
            raise NotImplementedError(
                "DFlash block_drafting does not support indexer layers in this release."
            )

        num_query_tokens = 1 + self.num_speculative_tokens
        if next_token_ids.shape[0] != batch_size:
            raise ValueError(
                "DFlash block_drafting expects one next token per sequence. "
                f"got next_token_ids.shape[0]={next_token_ids.shape[0]}, "
                f"batch_size={batch_size}."
            )
        last_positions = common_attn_metadata.seq_lens.to(torch.long) - 1
        max_last_position = int(last_positions.max().item())
        if max_last_position + num_query_tokens >= self.max_model_len:
            raise RuntimeError(
                "DFlash block_drafting query positions exceed max_model_len. "
                f"max_last_position={max_last_position}, "
                f"num_query_tokens={num_query_tokens}, "
                f"max_model_len={self.max_model_len}."
            )

        assert self.runner is not None
        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        num_context_tokens = target_hidden_states.shape[0]
        num_query_tokens_total = batch_size * num_query_tokens
        num_kv_tokens = num_context_tokens + num_query_tokens_total

        query_positions = (
            torch.arange(
                num_query_tokens,
                device=target_positions.device,
                dtype=target_positions.dtype,
            ).view(1, -1)
            + last_positions.view(-1, 1)
            + 1
        )
        query_positions_flat = query_positions.reshape(-1)
        position_ids = torch.cat([target_positions, query_positions_flat], dim=0)

        block_size = attn_metadata_builder.kv_cache_spec.block_size
        block_table_tensor = getattr(common_attn_metadata, "block_table_tensor", None)
        if block_table_tensor is None:
            raise RuntimeError(
                "DFlash block_drafting requires block_table_tensor in attention "
                "metadata."
            )
        max_block_number = int((query_positions_flat.max() // block_size).item())
        if max_block_number >= block_table_tensor.shape[1]:
            raise RuntimeError(
                "DFlash block_drafting needs more block_table entries than "
                f"available ({max_block_number + 1} > {block_table_tensor.shape[1]})."
            )
        block_numbers = query_positions // block_size
        block_ids = block_table_tensor.gather(dim=1, index=block_numbers)
        slot_mapping = (
            block_ids * block_size + (query_positions % block_size)
        ).reshape(-1)

        metadata_snapshot = self._snapshot_common_attn_metadata(common_attn_metadata)
        try:
            # Build non-causal metadata for query tokens only.
            common_attn_metadata.slot_mapping = slot_mapping
            common_attn_metadata.num_actual_tokens = num_query_tokens_total
            common_attn_metadata.max_query_len = num_query_tokens
            common_attn_metadata.query_start_loc = (
                self.arange[: batch_size + 1] * num_query_tokens
            )
            common_attn_metadata.query_start_loc_cpu = (
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_tokens
            )
            common_attn_metadata.max_seq_len += num_query_tokens
            common_attn_metadata.seq_lens += num_query_tokens
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None
            common_attn_metadata.causal = False

            attn_metadata = attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=0
            )
            if hasattr(attn_metadata, "causal") and attn_metadata.causal:
                raise NotImplementedError(
                    "DFlash block_drafting requires non-causal attention metadata."
                )
            per_layer_attn_metadata = {
                layer_name: attn_metadata for layer_name in self.attn_layer_names
            }

            # Query input is [next_token, mask, mask, ...].
            self.input_ids[:num_query_tokens_total].fill_(self.mask_token_id)
            self.input_ids[:num_query_tokens_total:num_query_tokens] = next_token_ids
            self._set_positions(num_kv_tokens, position_ids)
            self.hidden_states[:num_context_tokens] = target_hidden_states

            # NOTE: block_drafting currently runs in eager mode to avoid
            # cudagraph/padding complexity while we stabilize correctness.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_query_tokens_total,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=self._get_slot_mapping(
                    num_query_tokens_total, common_attn_metadata.slot_mapping
                ),
            ):
                ret_hidden_states = self.model(
                    input_ids=self.input_ids[:num_query_tokens_total],
                    positions=self._get_positions(num_kv_tokens),
                    hidden_states=self.hidden_states[:num_context_tokens],
                    inputs_embeds=None,
                )
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                else:
                    last_hidden_states, _ = ret_hidden_states
        finally:
            self._restore_common_attn_metadata(common_attn_metadata, metadata_snapshot)

        # Skip the first query token (the sampled next token), and use mask slots.
        sample_hidden_states = last_hidden_states.view(
            batch_size, num_query_tokens, -1
        )[:, 1:, :].reshape(
            batch_size * self.num_speculative_tokens,
            -1,
        )
        logits = self.model.compute_logits(sample_hidden_states)
        return logits.argmax(dim=-1).view(batch_size, self.num_speculative_tokens)

    @staticmethod
    def _snapshot_common_attn_metadata(
        common_attn_metadata: CommonAttentionMetadata,
    ) -> dict[str, object]:
        return {
            "slot_mapping": common_attn_metadata.slot_mapping,
            "num_actual_tokens": common_attn_metadata.num_actual_tokens,
            "max_query_len": common_attn_metadata.max_query_len,
            "query_start_loc": common_attn_metadata.query_start_loc,
            "query_start_loc_cpu": common_attn_metadata.query_start_loc_cpu,
            "max_seq_len": common_attn_metadata.max_seq_len,
            "seq_lens": common_attn_metadata.seq_lens.clone(),
            "seq_lens_cpu": common_attn_metadata._seq_lens_cpu,
            "num_computed_tokens_cpu": common_attn_metadata._num_computed_tokens_cpu,
            "causal": common_attn_metadata.causal,
        }

    @staticmethod
    def _restore_common_attn_metadata(
        common_attn_metadata: CommonAttentionMetadata,
        metadata_snapshot: dict[str, object],
    ) -> None:
        common_attn_metadata.slot_mapping = metadata_snapshot["slot_mapping"]  # type: ignore[assignment]
        common_attn_metadata.num_actual_tokens = metadata_snapshot["num_actual_tokens"]  # type: ignore[assignment]
        common_attn_metadata.max_query_len = metadata_snapshot["max_query_len"]  # type: ignore[assignment]
        common_attn_metadata.query_start_loc = metadata_snapshot["query_start_loc"]  # type: ignore[assignment]
        common_attn_metadata.query_start_loc_cpu = metadata_snapshot[
            "query_start_loc_cpu"
        ]  # type: ignore[assignment]
        common_attn_metadata.max_seq_len = metadata_snapshot["max_seq_len"]  # type: ignore[assignment]
        common_attn_metadata.seq_lens.copy_(metadata_snapshot["seq_lens"])  # type: ignore[arg-type]
        common_attn_metadata._seq_lens_cpu = metadata_snapshot["seq_lens_cpu"]  # type: ignore[assignment]
        common_attn_metadata._num_computed_tokens_cpu = metadata_snapshot[
            "num_computed_tokens_cpu"
        ]  # type: ignore[assignment]
        common_attn_metadata.causal = metadata_snapshot["causal"]  # type: ignore[assignment]

    def propose(
        self,
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
        target_hidden_states = self._maybe_combine_target_hidden_states(
            target_hidden_states
        )

        if self.runtime_mode == "shared_eagle":
            draft_token_ids = self._propose_shared_eagle(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )
        else:
            draft_token_ids = self._propose_block_drafting(
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                common_attn_metadata=common_attn_metadata,
            )

        expected_shape = (
            common_attn_metadata.batch_size(),
            self.num_speculative_tokens,
        )
        if tuple(draft_token_ids.shape) != expected_shape:
            raise RuntimeError(
                "DFlash proposer returned an unexpected draft token shape. "
                f"Expected {expected_shape}, got {tuple(draft_token_ids.shape)}."
            )
        return draft_token_ids
