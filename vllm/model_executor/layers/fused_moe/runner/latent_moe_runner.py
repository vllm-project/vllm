# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import aux_stream, current_stream

from .moe_runner import MoERunner, _unpack

logger = init_logger(__name__)


class LatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    Fused path (tp>1, un-reduced combine output, shared expert, no SP):
    concatenates the un-reduced latent partial (dim d) and the un-reduced
    shared partial (dim D) into one contiguous buffer, all-reduces once, then
    splits. The latent half is normed and up-projected locally (replicated
    up-proj -> full hidden), and the shared add folds into the GEMM epilogue
    (``torch.addmm``). One collective total, no post-reduction communication.

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size
    (using two collectives instead of the fused path's one).
    """

    def __init__(
        self,
        *args,
        enable_k3_latent_moe_tail_fusion: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_k3_latent_moe_tail_fusion = enable_k3_latent_moe_tail_fusion
        use_fused_path = self._use_fused_path()
        if (
            self.enable_k3_latent_moe_tail_fusion
            and use_fused_path
            and self.moe_config.tp_size not in (8, 16)
        ):
            logger.warning_once(
                "K3 latent-MoE tail fusion currently supports TP=8 and TP=16, "
                "but TP=%d is configured. Falling back to the default path.",
                self.moe_config.tp_size,
            )
            self.enable_k3_latent_moe_tail_fusion = False

        if self.enable_k3_latent_moe_tail_fusion and use_fused_path:
            vllm_config = get_current_vllm_config()
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError(
                    "K3 latent-MoE tail fusion does not support DBO or ubatching."
                )
            if vllm_config.model_config.enable_sleep_mode:
                raise ValueError(
                    "K3 latent-MoE tail fusion does not support sleep mode."
                )
            transform = self.routed_output_transform
            assert transform is not None
            norm = transform.norm
            assert norm is not None
            from vllm.models.kimi_k3.nvidia.ops.latent_moe_tail import (
                KimiK3LatentMoETailOp,
            )

            op = KimiK3LatentMoETailOp.initialize(
                hidden_size=transform.up_proj.weight.shape[0],
                latent_size=norm.weight.shape[0],
                dtype=norm.weight.dtype,
                device=norm.weight.device,
                rms_eps=norm.variance_epsilon,
            )
            self._k3_latent_moe_tail_op = op

    def _get_zero_residual(
        self,
        hidden_states: torch.Tensor,
        max_token_num: int,
    ) -> torch.Tensor:
        """Read-only zero ``residual_in`` for the fused AR+RMSNorm kernel.

        flashinfer requires a residual buffer even when there is no residual to
        add.
        """
        buf = getattr(self, "_zero_residual", None)
        if buf is None:
            buf = torch.zeros(
                max_token_num * hidden_states.shape[-1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self._zero_residual = buf

        assert buf.dtype == hidden_states.dtype
        assert buf.device == hidden_states.device
        assert hidden_states.numel() <= buf.numel()

        return buf[: hidden_states.numel()].view_as(hidden_states)

    def _use_fused_path(self) -> bool:
        # The fused path merges the latent and shared reductions into one
        # all-reduce, so it needs actual TP parallelism, a shared expert (to
        # concat), an un-reduced combine output, and no sequence parallelism.
        return (
            self.moe_config.tp_size > 1
            and self._shared_experts is not None
            and not self._fused_output_is_reduced
            and not self.moe_config.is_sequence_parallel
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._use_fused_path():
            return self._fused_forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # When the caller pre-applies the routed input transform outside the
        # runner (e.g. to overlap it on a separate stream), it passes the
        # already-transformed routed input as ``hidden_states`` and the original
        # hidden states as ``shared_experts_input``; skip the transform then.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )
        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = _unpack(result)
        assert shared_output is not None

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        transform = self.routed_output_transform
        assert transform is not None

        if self.enable_k3_latent_moe_tail_fusion:
            op = self._k3_latent_moe_tail_op
            if 0 < fused_output.shape[0] <= op.contract.max_num_tokens:
                norm = transform.norm
                assert norm is not None
                result = op(
                    fused_output,
                    shared_output,
                    norm.weight,
                    transform.up_proj.weight,
                )
                result = self._maybe_reduce_final_output(
                    result, og_hidden_dim_post_xform, output_is_reduced=True
                )
                return self._maybe_add_zero_expert_output(result)

        fused_latent = None
        if transform.norm is not None:
            fused_latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
        else:
            fused_latent = tensor_model_parallel_all_reduce(fused_output)

        shared_expert_stream = (
            aux_stream()
            if shared_output.size(0) <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            else None
        )
        if shared_expert_stream is not None:
            # overlap shared expert allreduce with latent up_proj
            main = current_stream()
            shared_output.record_stream(shared_expert_stream)
            shared_expert_stream.wait_stream(main)
            with torch.cuda.stream(shared_expert_stream):
                shared_output = tensor_model_parallel_all_reduce(shared_output)
            result = torch.mm(fused_latent, transform.up_proj.weight.t())
            main.wait_stream(shared_expert_stream)
        else:
            shared_output = tensor_model_parallel_all_reduce(shared_output)
            result = torch.mm(fused_latent, transform.up_proj.weight.t())
        result.add_(shared_output)

        # Output is already fully reduced; this only strips padding.
        result = self._maybe_reduce_final_output(
            result, og_hidden_dim_post_xform, output_is_reduced=True
        )

        return self._maybe_add_zero_expert_output(result)

    def allreduce_norm_latent_out(
        self,
        hidden_states: torch.Tensor,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce + add residual + (standard) RMSNorm, fused via flashinfer."""
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
            _AR_RESIDUAL_RMS_NORM,
            _can_use_flashinfer,
            flashinfer_trtllm_fused_allreduce_norm,
        )

        if self.moe_config.tp_size == 1:
            return norm(hidden_states)

        if flashinfer_trtllm_fused_allreduce_norm is not None:
            ok, max_token_num = _can_use_flashinfer(
                hidden_states, self.moe_config.tp_size
            )
            if ok:
                norm_out = torch.empty_like(hidden_states)
                # With norm_out provided, the kernel writes the new residual
                # (all_reduce(hidden_states) + residual) into the hidden_states
                # buffer and the normalized result into norm_out.
                flashinfer_trtllm_fused_allreduce_norm(
                    allreduce_in=hidden_states,
                    residual=self._get_zero_residual(hidden_states, max_token_num),
                    rms_gamma=norm.weight,
                    rms_eps=norm.variance_epsilon,
                    world_size=self.moe_config.tp_size,
                    weight_bias=0.0,
                    launch_with_pdl=True,
                    fp32_acc=True,
                    max_token_num=max_token_num,
                    pattern_code=_AR_RESIDUAL_RMS_NORM,
                    norm_out=norm_out,
                )
                return norm_out

        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return norm(reduced)
