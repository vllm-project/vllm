# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_fbgemm_gpu_gen_ai

from .dispatch_combine import MetaShufflingDispatchAndCombine, RouteInfo
from .routed_experts import MetaShufflingMoERoutedExperts

if current_platform.is_cuda_alike() and has_fbgemm_gpu_gen_ai():
    from fbgemm_gpu.experimental.gen_ai.moe import index_shuffling
from vllm.logger import init_logger

logger = init_logger(__name__)


# We only need the weight loader from unquantized fused moe method.
class MetaShufflingMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(
        self,
        moe: FusedMoEConfig,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    # Override to no ops.
    def init_prepare_finalize(self, layer: torch.nn.Module):
        assert self.moe is not None


class MetaShufflingMoE(FusedMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        shared_experts: torch.nn.Module | None = None,
        scoring_func: str = "softmax",
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        is_sequence_parallel: bool = False,
        **kwargs,
    ):
        CustomOp.__init__(self)

        logger.info_once("Initialized with MetaShufflingMoE")

        assert current_platform.is_cuda_alike(), (
            "MetaShufflingMoE only supports CUDA and AMD for now."
        )
        assert has_fbgemm_gpu_gen_ai(), (
            "MetaShufflingMoE requires fbgemm_gpu_gen_ai. \
            Run pip install fbgemm-gpu-genai"
        )

        params_dtype = kwargs.get("params_dtype", torch.get_default_dtype())
        tp_size_ = kwargs.get("tp_size", get_tensor_model_parallel_world_size())
        dp_size_ = kwargs.get("dp_size", get_dp_group().world_size)
        assert not is_sequence_parallel, "Sequence parallel is not supported yet."
        # Parallelism
        vllm_config = get_current_vllm_config()
        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            dp_size_=dp_size_,
            vllm_parallel_config=vllm_config.parallel_config,
        )
        etp_size_ = 1 if self.use_ep else tp_size_
        assert not self.use_ep, "Ep is not supported yet."
        self.tp2ep_size = tp_size_ // etp_size_
        self.dp2ep = self.ep_size // self.tp2ep_size
        assert self.dp2ep == dp_size_, "Doesn't support dp > dp2ep yet"

        # Determine expert maps
        assert num_experts % self.ep_size == 0, (
            "Does not support duplicate experts for now."
        )
        self.global_num_experts = num_experts
        self.local_num_experts = self.global_num_experts
        self.group_expert_start = 0
        self.group_expert_end = self.global_num_experts
        self.experts_mask = torch.arange(
            self.group_expert_start, self.group_expert_end, device="cuda"
        ).view(-1, 1, 1)
        self.local_num_experts, self.expert_map, self.expert_mask = (
            self.global_num_experts,
            None,
            None,
        )

        # Layer setup
        # TODO: Most of the weights loading logic is
        # similar to base fused_moe. We should probably refactor
        # the code so that common shared logic can be shared.
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.scoring_func = scoring_func
        self.apply_router_weight_on_input = apply_router_weight_on_input
        assert self.apply_router_weight_on_input, (
            "Only support apply_router_weight_on_input=True for now."
        )
        self.activation = activation
        self.top_k = top_k

        if vllm_config.model_config is not None:
            model_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            model_dtype = params_dtype

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=model_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=False,
        )
        self.moe_config = moe

        self.is_routed_fp8_rowwise: bool = False
        assert quant_config is None, "Quantization is not supported yet."
        self.quant_config = quant_config

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        self.quant_method = MetaShufflingMoEMethod(moe, quant_config=quant_config)

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self._shared_experts = shared_experts
        self.dispatch_and_combine = MetaShufflingDispatchAndCombine()
        self.routed_experts = MetaShufflingMoERoutedExperts(
            quant_config=self.quant_config
        )

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return self._shared_experts

    def route(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, RouteInfo]:
        assert self.scoring_func == "sigmoid", (
            "only support sigmoid scoring function for now "
        )
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits.to(torch.float32))
        top_k = self.moe_config.experts_per_token
        if top_k in {1, 2, 4} and self.global_num_experts in {16, 128}:
            token_counts, expert_indices, token_indices = index_shuffling(
                scores,  # num_tokens
                self.group_expert_start,
                self.group_expert_end,
                top_k=top_k,
            )
            num_routed_tokens = token_counts[-1]
            token_counts = token_counts[self.group_expert_start : self.group_expert_end]
        else:
            # Slow route using torch topk.
            _, global_selected_indices = torch.topk(scores, top_k, dim=1)
            expert_indices, token_indices = torch.sort(
                global_selected_indices.flatten(), dim=0, stable=True
            )
            token_indices = token_indices // top_k
            mask = self.experts_mask == expert_indices
            token_counts = (mask).sum(dim=2, dtype=torch.int32).flatten()
            num_routed_tokens = token_counts.sum().view(
                -1,
            )
        return scores, RouteInfo(
            expert_indices=expert_indices,
            token_indices=token_indices,
            token_counts=token_counts,
            num_routed_tokens=num_routed_tokens,
        )

    def forward_impl(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        scores, route_info = self.route(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        shuffled_recv_tokens, recv_token_counts = self.dispatch_and_combine.dispatch(
            tokens=hidden_states,
            scores=scores,
            route_info=route_info,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        shared_out = None
        # TODO: add using separate streams for shared experts.
        if self._shared_experts is not None:
            shared_out = self._shared_experts(hidden_states)

        routed_out = self.routed_experts.run(
            x=shuffled_recv_tokens,
            token_counts=recv_token_counts,
            w1=self.w13_weight.data,
            w2=self.w2_weight.data,
            activation=self.activation,
            scores=scores,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            num_valid_tokens=route_info.num_recv_tokens,
            shared_out=shared_out if not self.use_ep else None,
            token_indices=route_info.token_indices if not self.use_ep else None,
        )

        output = self.dispatch_and_combine.combine(
            routed_out=routed_out,
            shared_out=shared_out,
            route_info=route_info,
            scores=scores,
        )
        output = output.view(hidden_states.shape)
        if shared_out is None:
            return output
        else:
            # create a fake shared_output as moe_forward_shared expect to return a tuple
            return torch.empty_like(output), output
