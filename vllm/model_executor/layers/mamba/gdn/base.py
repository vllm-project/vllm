# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

import torch
from transformers import PreTrainedConfig

from vllm.config import (
    VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.abstract import (
    MambaBase,
    get_mamba_prefill_checkpoint_alignment,
    get_mamba_prefill_checkpoint_grid,
)
from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointExporter
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.gdn_attn import (
    FLASHINFER_GDN_CHECKPOINT_ALIGNMENT,
    GDNAttentionMetadata,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.worker.workspace import current_workspace_manager


class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    """Base class for GatedDeltaNet attention layer."""

    def __init__(
        self,
        config: PreTrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.vllm_config = vllm_config
        self._checkpoint_alignment: int | None = None
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def _init_prefill_checkpointing(self) -> None:
        # The FlashInfer and Triton/FLA prefills emit mid-prefill states, so an
        # align-mode checkpoint needs no split prefill.
        self._checkpoint_alignment = None
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            if self.gdn_prefill_backend == "flashinfer":
                self._checkpoint_alignment = FLASHINFER_GDN_CHECKPOINT_ALIGNMENT
            elif self.gdn_prefill_backend == "triton" and current_platform.is_cuda():
                self._checkpoint_alignment = FLA_CHUNK_SIZE
        if self._checkpoint_alignment is not None:
            self._max_num_batched_tokens = (
                self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self._checkpoint_exporter = MambaPrefillCheckpointExporter(
                state_len=self.conv_kernel_size - 1
            )

    def get_prefill_checkpoint_alignment(self) -> int | None:
        return self._checkpoint_alignment

    def get_kv_cache_checkpoint_alignment(self, vllm_config: VllmConfig) -> int | None:
        if not self._exports_checkpoints:
            return None
        # FlashInfer emits states at one stride for the whole batch, the gcd of
        # the offsets; keeping every offset on the prefix-cache grid keeps that
        # stride a multiple of the grid, which bounds the states emitted.
        return get_mamba_prefill_checkpoint_grid(vllm_config)

    @functools.cached_property
    def _exports_checkpoints(self) -> bool:
        # Known only once every layer is built: a Mamba layer that cannot
        # checkpoint (e.g. Qwen4Exp's PLE short conv) keeps the scheduler
        # splitting the prefill instead.
        return (
            self._checkpoint_alignment is not None
            and get_mamba_prefill_checkpoint_alignment(self.vllm_config) is not None
        )

    @functools.cached_property
    def _checkpoint_buffer_spec(self) -> tuple[tuple[int, ...], torch.dtype]:
        state_shape = (
            self.num_v_heads // self.tp_size,
            self.head_v_dim,
            self.head_k_dim,
        )
        if self.gdn_prefill_backend == "triton":
            # One state per sequence, in the cache dtype.
            return (
                (self.vllm_config.scheduler_config.max_num_seqs, *state_shape),
                self.get_state_dtype()[1],
            )
        # Offsets on the grid emit at most one state per grid step of a batch.
        # Read lazily: the block size is final only after the model is built.
        return (
            (
                cdiv(
                    self._max_num_batched_tokens,
                    get_mamba_prefill_checkpoint_grid(self.vllm_config),
                ),
                *state_shape,
            ),
            torch.float32,
        )

    def _forward_triton_prefill_with_checkpoints(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
        *,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
        conv_input: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Triton/FLA chunked prefill that exports the batch's internal
        prefill checkpoints.

        The kernel stores each checkpoint's recurrent state in a per-sequence
        buffer, which the exporter moves into the checkpoint block together
        with the conv window. Returns the output and the rows' final states.
        """
        assert self.gdn_prefill_backend == "triton"
        checkpoint = attn_metadata.checkpoint
        assert checkpoint is not None
        assert attn_metadata.prefill_state_indices is not None
        assert attn_metadata.prefill_has_initial_state is not None
        assert attn_metadata.non_spec_query_start_loc is not None
        assert conv_input is not None
        initial_state = gather_initial_states(
            ssm_state,
            attn_metadata.prefill_state_indices,
            attn_metadata.prefill_has_initial_state,
        )
        num_rows = checkpoint.checkpoint_offsets.shape[0]
        (checkpoint_state,) = current_workspace_manager().get_simultaneous(
            self._checkpoint_buffer_spec
        )
        checkpoint_state = checkpoint_state[:num_rows]
        # The prefill kernel runs the trailing checkpoint rows.
        first = num_rows - initial_state.shape[0]
        core_out, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=attn_metadata.prefill_query_start_loc,
            chunk_indices=attn_metadata.chunk_indices,
            chunk_offsets=attn_metadata.chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            checkpoint_state=checkpoint_state[first:],
            checkpoint_offsets=checkpoint.checkpoint_offsets[first:],
        )
        self._checkpoint_exporter.export(
            checkpoint,
            raw_qkv=conv_input,
            conv_state=conv_state,
            recurrent_checkpoint=checkpoint_state,
            recurrent_state=ssm_state,
            cu_seqlens=attn_metadata.non_spec_query_start_loc,
        )
        return core_out, final_state
