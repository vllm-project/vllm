# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to Nvidia and the vLLM project
from collections import defaultdict
from collections.abc import Iterable
from copy import copy

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import gpt_oss
from vllm.utils.math_utils import cdiv

from .utils import (
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class GptOssPuzzleTransformerBlock(gpt_oss.TransformerBlock):
    def __init__(
        self,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        """
        Build a vanilla gpt-oss hf config that matches this layer's architecture,
        then use it to instantiate a standard gpt-oss TransformerBlock.
        """
        layer_idx = extract_layer_index(prefix)
        puzzle_hf_config = vllm_config.model_config.hf_config
        vanilla_hf_config = puzzle_hf_config.get_gpt_oss_config_for_layer(layer_idx)

        # shallow copy to avoid OOM since vllm_config contains weights
        vllm_config = copy(vllm_config)
        vllm_config.model_config = copy(vllm_config.model_config)
        vllm_config.model_config.hf_config = vanilla_hf_config

        super().__init__(
            vllm_config=vllm_config, quant_config=quant_config, prefix=prefix
        )


@support_torch_compile
class GptOssPuzzleModel(gpt_oss.GptOssModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.parallel_config = vllm_config.parallel_config
        self.config.hidden_size = self.config.hidden_size
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: GptOssPuzzleTransformerBlock(
                vllm_config,
                prefix=prefix,
                quant_config=self.quant_config,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        We override load_weights to accommodate per-layer weight sizes:
        num experts, expert intermediate size.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]

        quant_method = (
            self.config.quantization_config["quant_method"]
            if hasattr(self.config, "quantization_config")
            else None
        )
        if quant_method == "mxfp4":
            return self._load_weights_mxfp4(weights, stacked_params_mapping)
        else:
            return self._load_weights_other(weights, stacked_params_mapping)

    def _load_weights_mxfp4(
        self,
        weights,
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        qkv_scale_counts = defaultdict[str, int](int)  # just for logging
        qkv_scale_ignore_counts = defaultdict[str, int](int)

        mxfp4_block = 32
        use_ep = self.parallel_config.enable_expert_parallel

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank

        intermediate_size = self.config.intermediate_size

        for name, weight in weights:
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            # FIXME(woosuk): Remove this after testing.
            weight = weight.cuda()

            try:
                layer_idx = extract_layer_index(name)
            except AssertionError:  # no layer index e.g. embeddings layer
                layer_idx = None

            if layer_idx is not None:
                block_config = self.config.block_configs[layer_idx]

                num_experts = block_config.num_local_experts
                experts_per_rank = num_experts // ep_size
                ep_rank_start = ep_rank * experts_per_rank
                ep_rank_end = (ep_rank + 1) * experts_per_rank

                intermediate_size_block = intermediate_size // mxfp4_block
                per_rank_intermediate_size_block = cdiv(
                    intermediate_size_block, tp_size
                )
                per_rank_intermediate_size = (
                    per_rank_intermediate_size_block * mxfp4_block
                )

                # Calculate common slicing bounds for current rank
                tp_rank_start = tp_rank * per_rank_intermediate_size
                tp_rank_end = min(
                    (tp_rank + 1) * per_rank_intermediate_size, intermediate_size
                )

            if ".w13_weight_scale" in name:
                # Handle MLP gate and up projection weights scale
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w2_weight_scale" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[
                        ..., tp_rank_start // mxfp4_block : tp_rank_end // mxfp4_block
                    ]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # flat weight from (E, 2 * N, block_size, entry_per_block)
                # to (E, 2 * N, -1), shouldn't trigger copy for contiguous
                weight = weight.view(
                    num_experts, 2 * intermediate_size, -1
                ).contiguous()

                # Extract gate and up projection parts
                # since the weight is shuffled, we can slice directly
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                # same flatten here, but since 2 mx4 value are packed in 1
                # uint8, divide by 2
                weight = weight.view(
                    num_experts, -1, intermediate_size // 2
                ).contiguous()
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[..., tp_rank_start // 2 : tp_rank_end // 2]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                weight_loader(
                    param, weight, weight_name=name, shard_id=None, expert_id=None
                )
                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif name.endswith((".q_scale", ".k_scale", ".v_scale")):
                module_name, scale_name = name.rsplit(".", 1)
                # support both layers.#.attn.x_scale and layers.#.attn.x_proj.x_scale
                if module_name.endswith("_proj"):
                    module_name = module_name.rsplit(".", 1)[0]
                module = self.get_submodule(module_name).attn
                if scale_name == "q_scale":
                    should_ignore = module.query_quant is None
                else:
                    should_ignore = module.kv_cache_torch_dtype not in {
                        torch.float8_e4m3fn,
                        torch.uint8,
                        torch.int8,
                    }
                if should_ignore:
                    qkv_scale_ignore_counts[scale_name] += 1
                else:
                    qkv_scale_counts[scale_name] += 1
                    assert hasattr(module, f"_{scale_name}"), (
                        f"Module {module_name} does not have {scale_name}"
                    )
                    getattr(module, f"_{scale_name}").copy_(weight)
                    assert hasattr(module, f"_{scale_name}_float"), (
                        f"Module {module_name} does not have {scale_name}_float"
                    )
                    setattr(module, f"_{scale_name}_float", weight.item())
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    logger.warning(
                        "Weight %s not found in params_dict"
                        " and was not handled by any other"
                        " case, therefore it will be ignored",
                        name,
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)

        if sum(qkv_scale_counts.values()) > 0:
            logger.info(
                "Loaded qkv scales: %s",
                ", ".join(
                    f"{count} x {k}"
                    for k, count in qkv_scale_counts.items()
                    if count != 0
                ),
            )
        if sum(qkv_scale_ignore_counts.values()) > 0:
            logger.warning(
                "Skipped loading qkv scales (maybe try"
                " passing --kv-cache-dtype fp8): %s",
                ", ".join(
                    f"{count} x {k}"
                    for k, count in qkv_scale_ignore_counts.items()
                    if count != 0
                ),
            )

        return loaded_params

    def _load_weights_other(
        self,
        weights,
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        qkv_scale_counts = defaultdict[str, int](int)  # just for logging
        qkv_scale_ignore_counts = defaultdict[str, int](int)

        use_ep = self.parallel_config.enable_expert_parallel

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank

        intermediate_size = self.config.intermediate_size

        for name, weight in weights:
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            try:
                layer_idx = extract_layer_index(name)
            except AssertionError:  # no layer index e.g. embeddings layer
                layer_idx = None

            if layer_idx is not None:
                block_config = self.config.block_configs[layer_idx]

                num_experts = block_config.num_local_experts
                experts_per_rank = num_experts // ep_size
                ep_rank_start = ep_rank * experts_per_rank
                ep_rank_end = (ep_rank + 1) * experts_per_rank

                per_rank_intermediate_size = cdiv(intermediate_size, tp_size)
                tp_rank_start = tp_rank * per_rank_intermediate_size
                tp_rank_end = min(
                    (tp_rank + 1) * per_rank_intermediate_size, intermediate_size
                )

            if ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # Extract gate and up projection parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, :, 2 * tp_rank_start : 2 * tp_rank_end]

                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, tp_rank_start:tp_rank_end, :]
                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]

                param = params_dict[name]
                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                param = params_dict[name]
                param.copy_(weight)
                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif name.endswith((".q_scale", ".k_scale", ".v_scale")):
                module_name, scale_name = name.rsplit(".", 1)
                # support both layers.#.attn.x_scale and layers.#.attn.x_proj.x_scale
                if module_name.endswith("_proj"):
                    module_name = module_name.rsplit(".", 1)[0]
                module = self.get_submodule(module_name).attn
                if scale_name == "q_scale":
                    should_ignore = module.query_quant is None
                else:
                    should_ignore = module.kv_cache_torch_dtype not in {
                        torch.float8_e4m3fn,
                        torch.uint8,
                        torch.int8,
                    }
                if should_ignore:
                    qkv_scale_ignore_counts[scale_name] += 1
                else:
                    qkv_scale_counts[scale_name] += 1
                    assert hasattr(module, f"_{scale_name}"), (
                        f"Module {module_name} does not have {scale_name}"
                    )
                    getattr(module, f"_{scale_name}").copy_(weight)
                    assert hasattr(module, f"_{scale_name}_float"), (
                        f"Module {module_name} does not have {scale_name}_float"
                    )
                    setattr(module, f"_{scale_name}_float", weight.item())
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)

        if sum(qkv_scale_counts.values()) > 0:
            logger.info(
                "Loaded qkv scales: %s",
                ", ".join(
                    f"{count} x {k}"
                    for k, count in qkv_scale_counts.items()
                    if count != 0
                ),
            )
        if sum(qkv_scale_ignore_counts.values()) > 0:
            logger.warning(
                "Skipped loading qkv scales (maybe try"
                " passing --kv-cache-dtype fp8): %s",
                ", ".join(
                    f"{count} x {k}"
                    for k, count in qkv_scale_ignore_counts.items()
                    if count != 0
                ),
            )

        return loaded_params


class GptOssPuzzleForCausalLM(gpt_oss.GptOssForCausalLM):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        self.model = GptOssPuzzleModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
