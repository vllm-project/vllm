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
    MambaStateShapeCalculator,
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
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.v1.attention.backends.gdn_attn import (
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

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def _get_conv_state(self) -> torch.Tensor:
        state = self.kv_cache[0]
        return state if is_conv_state_dim_first() else state.transpose(-1, -2)

    def _init_prefill_checkpointing(self, alignment: int | None) -> None:
        self._checkpoint_alignment = (
            alignment if self.cache_config.enable_prefix_caching else None
        )
        if self._checkpoint_alignment is not None:
            self._checkpoint_exporter = MambaPrefillCheckpointExporter(
                state_len=self.conv_kernel_size - 1
            )

    def get_prefill_checkpoint_alignment(self) -> int | None:
        return self._checkpoint_alignment

    def get_kv_cache_checkpoint_alignment(self, vllm_config: VllmConfig) -> int | None:
        if not self._exports_checkpoints:
            return None
        return get_mamba_prefill_checkpoint_grid(vllm_config)

    @functools.cached_property
    def _exports_checkpoints(self) -> bool:
        # The scheduler still splits prefills if any Mamba layer cannot checkpoint.
        return (
            self._checkpoint_alignment is not None
            and get_mamba_prefill_checkpoint_alignment(self.vllm_config) is not None
        )

    @functools.cached_property
    def _checkpoint_buffer_spec(self) -> tuple[tuple[int, ...], torch.dtype]:
        return (
            (
                self.vllm_config.scheduler_config.max_num_seqs,
                *self.get_state_shape()[1],
            ),
            self.get_state_dtype()[1],
        )

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
        conv_state = self._get_conv_state()
        ssm_state = self.kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

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

        num_decode_tokens = (
            attn_metadata.num_decode_tokens
            if spec_sequence_masks is None and attn_metadata.num_prefills > 0
            else 0
        )

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

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = self._prepare_prefill_inputs(
                mixed_qkv_non_spec[num_decode_tokens:],
                a_non_spec[num_decode_tokens:],
                b_non_spec[num_decode_tokens:],
            )
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
        if num_decode_tokens:
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
        if attn_metadata.num_prefills > 0:
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            initial_state = gather_initial_states(
                ssm_state, prefill_state_indices, prefill_has_initial_state
            )
            core_attn_out_non_spec, last_recurrent_state = self._forward_prefill(
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
                initial_state,
                attn_metadata,
                conv_input=conv_input_non_spec,
                output=core_attn_out,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            # Init cache
            scatter_states(ssm_state, last_recurrent_state, prefill_state_indices)

            if num_decode_tokens and core_attn_out_non_spec is None:
                core_attn_out[:num_decode_tokens] = core_attn_out_decode.squeeze(0)
            elif num_decode_tokens:
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
        elif core_attn_out_non_spec is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
        *,
        conv_input: torch.Tensor | None,
        output: torch.Tensor,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return prefill output, or `None` after writing it into `output`."""
        checkpoint = attn_metadata.checkpoint
        checkpoint_state = checkpoint_offsets = None
        if checkpoint is not None:
            num_rows = checkpoint.checkpoint_offsets.shape[0]
            (buffer,) = current_workspace_manager().get_simultaneous(
                self._checkpoint_buffer_spec
            )
            buffer = buffer[:num_rows]
            first = num_rows - initial_state.shape[0]
            checkpoint_state = buffer[first:]
            checkpoint_offsets = checkpoint.checkpoint_offsets[first:]

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
            checkpoint_state=checkpoint_state,
            checkpoint_offsets=checkpoint_offsets,
        )
        if checkpoint is not None:
            assert conv_input is not None
            assert attn_metadata.non_spec_query_start_loc is not None
            self._checkpoint_exporter.export(
                checkpoint,
                raw_qkv=conv_input,
                conv_state=self._get_conv_state(),
                recurrent_checkpoint=buffer,
                recurrent_state=self.kv_cache[1],
                cu_seqlens=attn_metadata.non_spec_query_start_loc,
            )
        return core_out, final_state
