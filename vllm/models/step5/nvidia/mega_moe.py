# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 NVFP4/FP8/BF16 DeepGEMM MegaMoE with shared symmetric storage."""

import weakref
from importlib import import_module

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.deep_gemm import _import_deep_gemm


class _BufferPool:
    # Modules retain their pool; removing the model releases the allocation.
    pools: weakref.WeakValueDictionary[
        tuple[int, torch.device, int, int, int, int, int, str], "_BufferPool"
    ] = weakref.WeakValueDictionary()

    def __init__(self, dg, group, experts, tokens, topk, hidden, intermediate, act):
        symm_buffer = import_module(f"{dg.__name__}.mega").SymmBuffer
        alignment = dg._C.get_token_alignment_for_mega_moe()
        tokens = (tokens + alignment - 1) // alignment * alignment
        sizes = {
            dtype: dg._C.get_symm_buffer_size_for_mega_moe(
                group.size(),
                experts,
                tokens,
                topk,
                hidden,
                intermediate,
                dtype,
                act,
                0,
            )[0]
            for dtype in ("fp8xfp8", "bf16xbf16")
        }
        sizes["nvfp4"] = dg._C.get_symm_buffer_size_for_nvfp4_mega_moe(
            group.size(), experts, tokens, topk, hidden, intermediate, 0, False
        )[0]
        owner_type = max(sizes, key=lambda dtype: sizes[dtype])
        kwargs = dict(
            group=group,
            num_experts=experts,
            num_max_tokens_per_rank=tokens,
            num_topk=topk,
            hidden=hidden,
            intermediate_hidden=intermediate,
            num_shared_experts=0,
        )

        def make_buffer(dtype, base=None):
            if dtype == "nvfp4":
                return dg.NVFP4SymmBuffer(**kwargs, base=base)
            return symm_buffer(**kwargs, mma_type=dtype, activation=act, base=base)

        self.owner = make_buffer(owner_type)
        self.views = {
            dtype: self.owner if dtype == owner_type else make_buffer(dtype, self.owner)
            for dtype in sizes
        }

    @classmethod
    def get(cls, dg, group, experts, tokens, topk, hidden, intermediate, act, device):
        key = (
            id(group),
            device,
            experts,
            tokens,
            topk,
            hidden,
            intermediate,
            act,
        )
        pool = cls.pools.get(key)
        if pool is None:
            # Keep first-launch ordering consistent across EP ranks.
            torch.accelerator.synchronize()
            torch.distributed.barrier(group=get_ep_group().cpu_group)
            pool = cls(dg, group, experts, tokens, topk, hidden, intermediate, act)
            cls.pools[key] = pool
        return pool


class _MegaMoEMethod(QuantizeMethodBase):
    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError("MegaMoE allocates weights at construction")

    def apply(self, layer, *args, **kwargs):
        return layer(*args, **kwargs)

    def process_weights_after_loading(self, layer):
        layer.prepare_weights()


class Step5MegaMoEExperts(nn.Module):
    def __init__(
        self,
        vllm_config,
        *,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        quant_config,
        activation,
        activation_clamp,
        prefix,
        max_num_tokens=None,
    ):
        super().__init__()
        parallel = vllm_config.parallel_config
        if not parallel.enable_expert_parallel:
            raise ValueError("DeepGEMM MegaMoE requires expert parallelism")
        if parallel.tensor_parallel_size > 1 and parallel.pipeline_parallel_size > 1:
            raise ValueError(
                "Step5 MegaMoE with TP requires sequence parallelism and PP=1"
            )
        if parallel.enable_eplb or parallel.eplb_config.num_redundant_experts:
            raise ValueError("Step5 MegaMoE does not yet support EPLB")
        if parallel.enable_dbo or getattr(parallel, "ubatch_size", 1) > 1:
            raise ValueError("Shared MegaMoE buffers require serial microbatches")
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError("MegaMoE dimensions must be multiples of 128")
        if activation not in ("silu", "swiglustep"):
            raise ValueError(f"Unsupported MegaMoE activation: {activation}")
        self.group = get_ep_group().device_group
        if num_experts % self.group.size():
            raise ValueError("MegaMoE experts must divide evenly across EP ranks")
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.group.size()
        self.expert_start = self.group.rank() * self.num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
            if max_num_tokens is None
            else max_num_tokens
        )
        self.activation = "swiglu" if activation == "silu" else activation
        self.activation_clamp = activation_clamp
        self.prefix = prefix
        self.fp8 = False
        self.nvfp4 = False
        if isinstance(quant_config, ModelOptNvFp4Config):
            self.nvfp4 = not quant_config.is_layer_excluded(
                prefix.removesuffix(".routed_experts")
            )
            if self.nvfp4 and (
                not quant_config.is_checkpoint_nvfp4_serialized
                or quant_config.group_size != 16
                or quant_config.quant_method != "NVFP4"
            ):
                raise ValueError(
                    "MegaMoE NVFP4 requires a group-size-16 W4A4 checkpoint"
                )
        elif quant_config is not None:
            if not isinstance(quant_config, Fp8Config):
                raise ValueError(
                    "MegaMoE supports BF16, block-FP8, or NVFP4 checkpoints"
                )
            self.fp8 = not is_layer_skipped(
                prefix=prefix.removesuffix(".routed_experts"),
                ignored_layers=quant_config.ignored_layers,
                fused_mapping=quant_config.packed_modules_mapping,
                match_mode=quant_config.ignored_layers_match_mode,
            )
            if self.fp8 and (
                not quant_config.is_checkpoint_fp8_serialized
                or quant_config.weight_block_size != [128, 128]
                or quant_config.store_dtype == "mxfp4"
            ):
                raise ValueError("MegaMoE FP8 requires a 128x128 FP8 checkpoint")
        self.mma_type = (
            "nvfp4" if self.nvfp4 else "fp8xfp8" if self.fp8 else "bf16xbf16"
        )
        dtype = (
            torch.uint8
            if self.nvfp4
            else torch.float8_e4m3fn
            if self.fp8
            else torch.bfloat16
        )
        for name, n, k in (
            ("w13", 2 * intermediate_size, hidden_size),
            ("w2", hidden_size, intermediate_size),
        ):
            param = nn.Parameter(
                torch.empty(
                    self.num_local_experts, n, k // 2 if self.nvfp4 else k, dtype=dtype
                ),
                requires_grad=False,
            )
            set_weight_attrs(param, {"weight_loader": self.weight_loader})
            self.register_parameter(f"{name}_weight", param)
            if self.fp8:
                scale = nn.Parameter(
                    torch.empty(
                        self.num_local_experts, n // 128, k // 128, dtype=torch.float32
                    ),
                    requires_grad=False,
                )
                set_weight_attrs(scale, {"weight_loader": self.weight_loader})
                self.register_parameter(f"{name}_weight_scale_inv", scale)
            if self.nvfp4:
                for suffix, shape, scale_dtype in (
                    (
                        "weight_scale",
                        (self.num_local_experts, n, k // 16),
                        torch.float8_e4m3fn,
                    ),
                    (
                        "weight_scale_2",
                        (self.num_local_experts, 2)
                        if name == "w13"
                        else (self.num_local_experts,),
                        torch.float32,
                    ),
                    (
                        "input_scale",
                        (self.num_local_experts, 2)
                        if name == "w13"
                        else (self.num_local_experts,),
                        torch.float32,
                    ),
                ):
                    scale = nn.Parameter(
                        torch.empty(shape, dtype=scale_dtype), requires_grad=False
                    )
                    set_weight_attrs(scale, {"weight_loader": self.weight_loader})
                    self.register_parameter(f"{name}_{suffix}", scale)
        self.quant_method = _MegaMoEMethod()
        self.ready = False
        self.pool = None

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name=None,
        shard_id=None,
        expert_id=None,
        return_success=False,
    ):
        if self.ready:
            raise RuntimeError("Reloading transformed MegaMoE weights is unsupported")
        target = param.data
        if expert_id is None:
            if loaded_weight.shape[0] == self.num_experts:
                loaded_weight = loaded_weight.narrow(
                    0, self.expert_start, self.num_local_experts
                )
        else:
            local_id = expert_id - self.expert_start
            if not 0 <= local_id < self.num_local_experts:
                return False if return_success else None
            target = target[local_id]
            if shard_id in ("w1", "w3"):
                if param.ndim == 2:
                    target = target[0 if shard_id == "w1" else 1]
                else:
                    size = target.shape[0] // 2
                    target = target.narrow(0, 0 if shard_id == "w1" else size, size)
            elif shard_id != "w2":
                raise ValueError(f"Invalid MegaMoE weight shard: {shard_id}")
        if target.shape != loaded_weight.shape:
            raise ValueError(
                f"MegaMoE weight shape mismatch: {target.shape} "
                f"vs {loaded_weight.shape}"
            )
        target.copy_(loaded_weight)
        return True if return_success else None

    def _convert_fp8(self, name):
        cast = import_module(f"{self.dg.__name__}.utils").per_token_cast_to_fp8
        weight = getattr(self, f"{name}_weight")
        scale = getattr(self, f"{name}_weight_scale_inv")
        e, n, k = weight.shape
        converted = torch.empty_like(weight)
        scales = torch.empty((e, n, k // 32), device=weight.device, dtype=torch.float32)
        # Convert one expert at a time to bound temporary memory while loading.
        for i in range(e):
            dense = (
                weight[i].float().view(n // 128, 128, k // 128, 128)
                * scale[i][:, None, :, None]
            ).reshape(n, k)
            converted[i], scales[i] = cast(dense, use_ue8m0=True, gran_k=32)
        scales = self.dg.transform_sf_into_required_layout(scales, n, k, (1, 32), e)
        return converted, scales

    def _prepare_nvfp4(self):
        if not torch.equal(
            self.w13_weight_scale_2[:, 0], self.w13_weight_scale_2[:, 1]
        ):
            raise ValueError("MegaMoE requires matching NVFP4 gate/up global scales")
        for scales in (self.w13_weight_scale_2, self.w2_weight_scale_2):
            if not torch.all(torch.isfinite(scales) & (scales > 0)):
                raise ValueError(
                    "NVFP4 weight global scales must be finite and positive"
                )
        for scales in (self.w13_input_scale, self.w2_input_scale):
            if not torch.all(torch.isfinite(scales) & (scales > 0)) or not torch.all(
                scales == scales.flatten()[0]
            ):
                raise ValueError(
                    "MegaMoE requires a shared positive input scale per projection"
                )
        input_scale = self.w13_input_scale.flatten()[0]
        self.register_buffer("input_global_scale", input_scale.reciprocal().reshape(1))
        self.register_buffer(
            "l1_alpha", (self.w13_weight_scale_2[:, 0] * input_scale).contiguous()
        )
        self.l2_activation_scale = self.w2_input_scale[0].item()
        self.register_buffer(
            "l2_alpha", self.w2_weight_scale_2 * self.l2_activation_scale
        )
        weights = []
        for name in ("w13", "w2"):
            weight = getattr(self, f"{name}_weight").data.view(torch.int8)
            scale = getattr(self, f"{name}_weight_scale").data.view(torch.int32)
            scale = scale.transpose(-1, -2).contiguous().transpose(-1, -2)
            weights.append((weight, scale))
        return weights

    def prepare_weights(self):
        if self.ready:
            return
        if torch.cuda.get_device_capability(self.w13_weight.device)[0] != 10:
            raise ValueError("DeepGEMM MegaMoE requires SM100 GPUs")
        self.dg = _import_deep_gemm()
        if self.dg is None or not hasattr(self.dg, "bf16_mega_moe"):
            raise RuntimeError("Install the pinned vLLM DeepGEMM build")
        if self.nvfp4:
            w1, w2 = self._prepare_nvfp4()
        else:
            w1 = self._convert_fp8("w13") if self.fp8 else self.w13_weight.data
            w2 = self._convert_fp8("w2") if self.fp8 else self.w2_weight.data
        w1, w2 = self.dg.transform_weights_for_mega_moe(
            w1, w2, activation=self.activation
        )
        self.w13_weight.data = w1[0] if self.fp8 or self.nvfp4 else w1
        self.w2_weight.data = w2[0] if self.fp8 or self.nvfp4 else w2
        if self.fp8:
            self.w13_weight_scale_inv.data = w1[1]
            self.w2_weight_scale_inv.data = w2[1]
        if self.nvfp4:
            self.w13_weight_scale.data = w1[1]
            self.w2_weight_scale.data = w2[1]
        self.ready = True

    def get_symm_buffer(self):
        # Defer rank synchronization until runtime; weights may be prepared
        # in a separate weight-cache process.
        if self.pool is not None:
            return self.pool.views[self.mma_type]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up MegaMoE before capturing a CUDA graph")
        self.pool = _BufferPool.get(
            self.dg,
            self.group,
            self.num_experts,
            self.max_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.activation,
            self.w13_weight.device,
        )
        return self.pool.views[self.mma_type]

    def _forward_impl(self, hidden_states, topk_weights, topk_ids, padding=None):
        from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
            prepare_megamoe_inputs,
        )

        if not self.ready:
            raise RuntimeError("MegaMoE weights must be prepared before inference")
        n = hidden_states.shape[0]
        if n > self.max_tokens:
            raise ValueError("MegaMoE input exceeds its allocated token capacity")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("MegaMoE expects BF16 hidden states")
        buffer = self.get_symm_buffer()
        if self.fp8:
            prepare_megamoe_inputs(
                hidden_states,
                topk_weights,
                topk_ids,
                buffer.x,
                buffer.x_sf,
                buffer.topk_idx,
                buffer.topk_weights,
                is_padding=padding,
            )
            w1 = (self.w13_weight, self.w13_weight_scale_inv)
            w2 = (self.w2_weight, self.w2_weight_scale_inv)
            kernel = self.dg.fp8_fp4_mega_moe
        else:
            if self.nvfp4:
                if n:
                    torch.ops._C.scaled_fp4_quant.out(
                        hidden_states,
                        self.input_global_scale,
                        False,
                        output=buffer.x[:n].view(torch.uint8),
                        output_scale=buffer.x_sf[:n].view(torch.uint8),
                    )
            else:
                buffer.x[:n].copy_(hidden_states)
            buffer.topk_idx[:n].copy_(topk_ids)
            buffer.topk_weights[:n].copy_(topk_weights)
            if padding is not None:
                buffer.topk_idx[:n].masked_fill_(padding[:n, None], -1)
                buffer.topk_weights[:n].masked_fill_(padding[:n, None], 0)
            w1, w2 = self.w13_weight, self.w2_weight
            kernel = self.dg.bf16_mega_moe
        kernel_kwargs = {}
        if self.nvfp4:
            w1 = (self.w13_weight, self.w13_weight_scale)
            w2 = (self.w2_weight, self.w2_weight_scale)
            kernel = self.dg.nvfp4_mega_moe
            kernel_kwargs = dict(
                l1_alpha=self.l1_alpha,
                l2_alpha=self.l2_alpha,
                l2_activation_scale=self.l2_activation_scale,
            )
        output = torch.empty_like(hidden_states)
        kernel(
            y=output,
            l1_weights=w1,
            l2_weights=w2,
            sym_buffer=buffer,
            activation=self.activation,
            activation_clamp=self.activation_clamp,
            **kernel_kwargs,
        )
        return output

    def forward(self, hidden_states, topk_weights, topk_ids, padding=None):
        return self._forward_impl(hidden_states, topk_weights, topk_ids, padding)


Step5MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


class Step5MegaMoE(nn.Module):
    """Router plus EP routed experts."""

    def __init__(
        self,
        vllm_config,
        *,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        quant_config,
        activation,
        activation_clamp,
        prefix,
        renormalize,
        scoring_func,
        e_score_correction_bias,
        routed_scaling_factor,
        sequence_parallel=False,
    ):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.layer_id = extract_layer_index(prefix)
        self.router = create_fused_moe_router(
            top_k=top_k,
            global_num_experts=num_experts,
            renormalize=renormalize,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if sequence_parallel:
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            max_tokens = (max_tokens + tp_size - 1) // tp_size
        self.routed_experts = Step5MegaMoEExperts(
            vllm_config,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            activation=activation,
            activation_clamp=activation_clamp,
            prefix=f"{prefix}.routed_experts",
            max_num_tokens=max_tokens,
        )

    def forward(self, hidden_states, router_logits, padding=None):
        weights, ids = self.router.select_experts(
            hidden_states, router_logits, topk_indices_dtype=torch.int64
        )
        num_tokens = hidden_states.shape[0]
        if (
            padding is None
            and not self.sequence_parallel
            and envs.VLLM_MOE_SKIP_PADDING
            and is_forward_context_available()
        ):
            mask = get_forward_context().is_padding
            padding = mask[:num_tokens] if mask is not None else None
        return self.routed_experts(hidden_states, weights, ids, padding)
