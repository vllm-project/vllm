# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt safetensors block codecs with BF16 activations."""

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.utils.b12x import B12X_BLOCK_CODECS, B12xWarmupUnit, get_b12x_blockscaled


@torch.library.custom_op(
    "vllm::modelopt_block_embedding_chunked", mutates_args=("out",)
)
def _modelopt_block_embedding_chunked(
    weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor, plan_handle: int
) -> None:
    from b12x.preparation.types import plan_from_handle
    from b12x.sequence import embedding

    plan = plan_from_handle(plan_handle)
    capacity = plan.query.max_rows
    for start in range(0, ids.numel(), capacity):
        embedding.run(
            weight,
            ids[start : start + capacity],
            out=out[start : start + capacity],
            plan=plan,
        )


@_modelopt_block_embedding_chunked.register_fake
def _modelopt_block_embedding_chunked_fake(
    weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor, plan_handle: int
) -> None:
    return None


class ModelOptBlockQuantMoEMethod(FusedMoEMethodBase):
    """Load native blocks and retain compact b12x prepared expert storage."""

    def __init__(self, moe_config: FusedMoEConfig, codec: str = "iq2_xs"):
        super().__init__(moe_config)
        self.codec = codec.lower()
        self.block_size, self.block_bytes = B12X_BLOCK_CODECS[self.codec]
        if moe_config.moe_backend not in ("auto", "b12x"):
            raise ValueError(
                "Block-quantized routed experts require the b12x MoE backend"
            )
        if not B12xExperts._supports_current_device():
            raise ValueError("Block-quantized routed experts require b12x on SM12x")
        if not B12xExperts._supports_parallel_config(moe_config.moe_parallel_config):
            raise ValueError(
                "Block-quantized supports tensor parallelism without EP or EPLB"
            )
        if moe_config.in_dtype != torch.bfloat16:
            raise ValueError("Block-quantized routed experts require BF16 activations")
        if (
            moe_config.activation
            not in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)
            or moe_config.has_bias
        ):
            raise ValueError(
                "Block-quantized routed experts require bias-free SiLU or ReLU2"
            )
        self.gated = moe_config.activation == MoEActivation.SILU
        if any(
            value is not None
            for value in (
                moe_config.swiglu_limit,
                moe_config.swiglu_alpha,
                moe_config.swiglu_beta,
            )
        ):
            raise ValueError(
                "Block-quantized routed experts require standard SiLU parameters"
            )

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        alignment = max(128, self.block_size)
        if hidden_size % alignment or intermediate_size_per_partition % alignment:
            raise ValueError(
                f"{self.codec.upper()} hidden and per-rank intermediate sizes "
                f"must align to {alignment}"
            )
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        self.maybe_roundup_sizes(
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            self.moe.moe_parallel_config,
        )
        for name, shape in (
            (
                "w13_weight",
                (
                    num_experts,
                    (2 if self.gated else 1) * intermediate_size_per_partition,
                    hidden_size // self.block_size,
                    self.block_bytes,
                ),
            ),
            (
                "w2_weight",
                (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.block_size,
                    self.block_bytes,
                ),
            ),
        ):
            layer.register_parameter(
                name,
                ModelWeightParameter(
                    data=torch.empty(shape, dtype=torch.uint8),
                    input_dim=2,
                    output_dim=1,
                    weight_loader=self.weight_loader,
                ),
            )

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"invalid Block-quantized expert projection: {shard_id}")
        if not self.gated and shard_id == "w3":
            raise ValueError("non-gated Block-quantized experts have no w3 projection")
        local_i = self.moe.intermediate_size_per_partition
        global_i = local_i * self.moe.moe_parallel_config.tp_size
        hidden = self.moe.hidden_dim
        expected = (
            (hidden, global_i // self.block_size, self.block_bytes)
            if shard_id == "w2"
            else (global_i, hidden // self.block_size, self.block_bytes)
        )
        if loaded_weight.dtype != torch.uint8 or tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"invalid packed tensor {weight_name}: expected uint8{expected}, "
                f"got {loaded_weight.dtype}{tuple(loaded_weight.shape)}"
            )
        if not 0 <= expert_id < param.shape[0]:
            raise ValueError(f"invalid Block-quantized expert index: {expert_id}")
        destination = param.data[expert_id]
        if shard_id == "w2":
            source = loaded_weight.narrow(
                1,
                self.moe.tp_rank * (local_i // self.block_size),
                local_i // self.block_size,
            )
        else:
            source = loaded_weight.narrow(0, self.moe.tp_rank * local_i, local_i)
            destination = destination.narrow(
                0, 0 if shard_id == "w1" else local_i, local_i
            )
        destination.copy_(source)
        return True if return_success else None

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(dtype=self.codec),
            _w2=FusedMoEQuantDesc(dtype=self.codec),
        )

    def process_weights_after_loading(self, layer):
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True,
        )
        assert prepare_finalize is not None
        experts = B12xExperts(self.moe, self.moe_quant_config)
        self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)
        experts.process_weights_after_loading(layer)

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
    ):
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


ModelOptBlockQuantMoEMethod.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


@register_weight_loader_v2_supported_method
class ModelOptBlockQuantLinearMethod(LinearMethodBase):
    """Load native block payloads and execute the prepared b12x dense API."""

    def __init__(self, codec: str = "iq2_xs"):
        self.codec = codec.lower()
        self.block_size, self.block_bytes = B12X_BLOCK_CODECS[self.codec]
        self.is_embedding = False
        from vllm.model_executor.kernels.linear import _get_linear_backend
        from vllm.utils.b12x import get_b12x_blockscaled

        if _get_linear_backend(quantization=self.codec) not in ("auto", "b12x"):
            raise ValueError(
                "Block-quantized dense weights require the b12x linear backend"
            )
        api = get_b12x_blockscaled()
        if (
            api is None
            or not hasattr(api, "BlockQuantLinearWeight")
            or not api.is_supported()
        ):
            raise ValueError("Block-quantized dense weights require b12x on SM12x")

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        n = sum(output_partition_sizes)
        self.is_embedding = isinstance(
            layer, VocabParallelEmbedding
        ) and not isinstance(layer, ParallelLMHead)
        if self.is_embedding and self.codec != "q8_0":
            raise ValueError("packed embedding currently requires Q8_0")
        if (
            params_dtype != torch.bfloat16
            or input_size_per_partition % self.block_size
            or n % 8
        ):
            raise ValueError(
                f"{self.codec.upper()} dense weights require BF16, "
                f"K{self.block_size} and N8"
            )
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    (n, input_size_per_partition // self.block_size, self.block_bytes),
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=extra_weight_attrs["weight_loader"],
            ),
        )

    def process_weights_after_loading(self, layer):
        from b12x.sequence import embedding

        from vllm.model_executor.utils import replace_parameter

        api = get_b12x_blockscaled()
        assert api is not None
        config = get_current_vllm_config()
        scheduler = config.scheduler_config
        capture_sizes = config.compilation_config.cudagraph_capture_sizes or []
        capacity = max(
            scheduler.max_num_batched_tokens,
            scheduler.max_num_scheduled_tokens or 0,
            *capture_sizes,
        )
        weight = layer.weight.data
        if self.is_embedding:
            bases = weight[..., :2].contiguous().view(torch.float16)
            if not torch.isfinite(bases).all().item():
                raise ValueError("Q8_0 embedding scales must be finite")
            layer.b12x_embedding_plans = {}
            width = weight.shape[1] * self.block_size
            if config.model_config is not None:
                capacity = max(capacity, config.model_config.max_model_len)

            for dtype in (torch.int32, torch.int64):
                plan = embedding.plan(
                    embedding.EmbeddingQuery(
                        max_rows=capacity,
                        table_rows=weight.shape[0],
                        width=width,
                        row_stride=weight.stride(0),
                        weight_dtype="uint8",
                        id_dtype=str(dtype).removeprefix("torch."),
                    ),
                    device=weight.device,
                )
                layer.b12x_embedding_plans[dtype] = plan
        else:
            packed = api.pack_weight(weight, recipe=self.codec)
            query = api.BlockscaledQuery(
                recipe=self.codec,
                num_tokens=capacity,
                in_features=packed.in_features,
                padded_in_features=packed.in_features,
                out_features=packed.out_features,
                activation_mode="a16",
            )
            plan = api.plan_regimes(
                query,
                exact_m=tuple(sorted({m for m in capture_sizes if 0 < m < capacity})),
            )
            layer.b12x_block_weight = packed
            layer.b12x_block_plan = plan
        if not self.is_embedding:
            replace_parameter(
                layer, "weight", torch.empty(0, dtype=torch.uint8, device=weight.device)
            )

        layer.b12x_warmup_provider = self

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        if self.is_embedding:
            plans = tuple(layer.b12x_embedding_plans.values())
        else:
            plans = (layer.b12x_block_plan,)

        def compile() -> None:
            if self.is_embedding:
                for dtype, plan in layer.b12x_embedding_plans.items():
                    ids = torch.zeros(
                        plan.query.max_rows, dtype=dtype, device=layer.weight.device
                    )
                    self.embedding(layer, ids)
            else:
                packed = layer.b12x_block_weight
                for tokens in sorted(set(token_counts) | set(plans[0].token_counts)):
                    source = torch.zeros(
                        (tokens, packed.in_features),
                        dtype=output_dtype,
                        device=packed.values.device,
                    )
                    self.apply(layer, source)

        return B12xWarmupUnit(
            name=(
                self.codec.upper() + (" embedding" if self.is_embedding else " linear")
            ),
            # Every layer owns a plan that must be prepared before capture.
            key=(type(self), *(plan.handle for plan in plans)),
            compile=compile,
        )

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.bfloat16:
            raise ValueError(
                "Block-quantized dense execution requires BF16 activations"
            )
        api = get_b12x_blockscaled()
        assert api is not None
        output = api.mm(
            x.reshape(-1, x.shape[-1]).contiguous(),
            layer.b12x_block_weight,
            plan=layer.b12x_block_plan,
            bias=bias,
        )
        return output.view(*x.shape[:-1], layer.b12x_block_weight.out_features)

    def embedding(self, layer, input_):
        plan = layer.b12x_embedding_plans[input_.dtype]
        width = layer.weight.shape[1] * self.block_size
        out = torch.empty(
            (*input_.shape, width), dtype=torch.bfloat16, device=layer.weight.device
        )
        ids, rows = input_.reshape(-1), out.view(-1, width)
        _modelopt_block_embedding_chunked(layer.weight, ids, rows, plan.handle)
        return out
