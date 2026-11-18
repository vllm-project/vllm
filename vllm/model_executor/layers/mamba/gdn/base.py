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
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
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
        # align-mode checkpoint needs no split prefill. AITER FlyDSL cannot, so
        # its steps that carry a checkpoint run the Triton/FLA prefill instead.
        self._checkpoint_alignment = None
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            if self.gdn_prefill_backend == "flashinfer":
                self._checkpoint_alignment = FLASHINFER_GDN_CHECKPOINT_ALIGNMENT
            elif (
                self.gdn_prefill_backend in ("triton", "aiter_flydsl")
                and current_platform.is_cuda_alike()
            ):
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
        if self.gdn_prefill_backend != "flashinfer":
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
        assert self.gdn_prefill_backend in ("triton", "aiter_flydsl")
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

    def _forward_core_impl(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
        *,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> None:
        """Shared convolution, batch splitting, recurrence and state I/O."""
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]
        a, b = self._prepare_gating(a, b)

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                a_spec = a
                b_spec = b
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            assert num_accepted_tokens is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        conv_input_non_spec = mixed_qkv_non_spec
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        # Split mixed non-spec-decode+prefill to process independently
        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            if split_non_spec:
                conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                a_prefill = a_non_spec[num_decode_tokens:]
                b_prefill = b_non_spec[num_decode_tokens:]
            else:
                conv_output_prefill = mixed_qkv_non_spec
                a_prefill = a_non_spec
                b_prefill = b_non_spec

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = self._prepare_prefill_inputs(conv_output_prefill, a_prefill, b_prefill)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = self._recurrent_update(
                a=a_spec,
                b=b_spec,
                q=query_spec,
                k=key_spec,
                v=value_spec,
                initial_state=ssm_state,
                cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_spec_decodes
                    + 1  # type: ignore[attr-defined]
                ],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process non-spec-decode part
        if split_non_spec:
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec[:num_decode_tokens]  # type: ignore[index]
            )
            core_attn_out_decode, _ = self._recurrent_update(
                a=a[:num_decode_tokens],
                b=b[:num_decode_tokens],
                q=query_decode,
                k=key_decode,
                v=value_decode,
                initial_state=ssm_state,
                cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        else:
            core_attn_out_decode = None

        # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)
        prefill_in_place = False
        if attn_metadata.num_prefills > 0:
            # State indices, initial-state mask and cu_seqlens for the chunk
            # kernel are precomputed by the metadata builder (the prefill tail
            # when decodes are peeled off, else the full non-spec batch), so they
            # don't need to be re-derived per layer.
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            if self.gdn_prefill_backend == "flashinfer":
                # Without spec decodes the prefill rows follow the peeled
                # decodes contiguously, so FlashInfer writes them straight
                # into core_attn_out. Spec batches scatter by token index.
                prefill_out = None
                if spec_sequence_masks is None:
                    prefill_out = core_attn_out[num_decode_tokens:num_actual_tokens]
                core_attn_out_non_spec, last_recurrent_state = (
                    self._forward_flashinfer_prefill(
                        query_non_spec,
                        key_non_spec,
                        value_non_spec,
                        g_non_spec,
                        beta_non_spec,
                        attn_metadata,
                        ssm_state=ssm_state,
                        conv_state=conv_state,
                        conv_input=conv_input_non_spec,
                        output=prefill_out,
                    )
                )
                prefill_in_place = prefill_out is not None
            elif attn_metadata.checkpoint is not None:
                core_attn_out_non_spec, last_recurrent_state = (
                    self._forward_triton_prefill_with_checkpoints(
                        query_non_spec,
                        key_non_spec,
                        value_non_spec,
                        g_non_spec,
                        beta_non_spec,
                        attn_metadata,
                        ssm_state=ssm_state,
                        conv_state=conv_state,
                        conv_input=conv_input_non_spec,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                )
            else:
                initial_state = gather_initial_states(
                    ssm_state, prefill_state_indices, prefill_has_initial_state
                )
                core_attn_out_non_spec, last_recurrent_state = (
                    self._forward_chunk_prefill(
                        query_non_spec,
                        key_non_spec,
                        value_non_spec,
                        g_non_spec,
                        beta_non_spec,
                        initial_state,
                        attn_metadata,
                    )
                )
            # Init cache
            scatter_states(ssm_state, last_recurrent_state, prefill_state_indices)

            if split_non_spec and prefill_in_place:
                core_attn_out[:num_decode_tokens] = core_attn_out_decode.squeeze(0)
            elif split_non_spec:
                # Stitch the peeled decode outputs in front of the prefill
                # outputs (decode-first order).
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec], dim=1
                )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = self._recurrent_update(
                a=a,
                b=b,
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                initial_state=ssm_state,
                cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_decodes
                    + 1  # type: ignore[attr-defined]
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            core_attn_out.index_copy_(0, spec_token_indx, core_attn_out_spec.squeeze(0))
            core_attn_out.index_copy_(
                0, non_spec_token_indx, core_attn_out_non_spec.squeeze(0)
            )
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        elif not prefill_in_place:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _prepare_gating(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Qwen keeps raw projections for its fused prefill/decode kernels.
        return a, b

    def _prepare_prefill_inputs(
        self,
        conv_output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _recurrent_update(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def _forward_chunk_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return chunk_gated_delta_rule(
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
            use_qk_l2norm_in_kernel=True,
        )
