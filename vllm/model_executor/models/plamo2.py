# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only PLaMo2 model."""

from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba2.plamo2_mixer2 import Plamo2MambaMixer
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op

# Only used for type hinting.
if TYPE_CHECKING:

    class Plamo2Config(PretrainedConfig):  # type: ignore
        model_type: str = "plamo2"

        hidden_size: int
        num_hidden_layers: int
        rms_norm_eps: float
        # Attention
        num_attention_heads: int
        hidden_size_per_head: int
        num_key_value_heads: int
        # Mamba
        mamba_d_state: int
        mamba_d_conv: int
        mamba_num_heads: int
        mamba_step: int
        # MLP
        intermediate_size: int
        # Tokenizer
        vocab_size: int


def is_mamba(config: "Plamo2Config", i: int) -> bool:
    assert config.mamba_step > 1

    if config.num_hidden_layers <= (config.mamba_step // 2):
        # use attention in last layer
        return i != config.num_hidden_layers - 1
    return (i % config.mamba_step) != (config.mamba_step // 2)


def plamo2_mamba_mixer(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_impl(hidden_states=hidden_states, output=output)


def plamo2_mamba_mixer_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="plamo2_mamba_mixer",
    op_func=plamo2_mamba_mixer,
    mutates_args=["output"],
    fake_impl=plamo2_mamba_mixer_fake,
)


class DenseMLP(nn.Module):
    def __init__(
        self,
        config: "Plamo2Config",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            quant_config=quant_config,
            return_bias=False,
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
            quant_config=quant_config,
            return_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.gate_up_proj(hidden_states)
        h = self.act(h)
        return self.down_proj(h)


class Plamo2AttentionMixer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size_per_head
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        max_position = config.max_position_embeddings
        if hasattr(vllm_config.model_config, "max_model_len") and isinstance(
            vllm_config.model_config.max_model_len, int
        ):
            max_position = min(max_position, vllm_config.model_config.max_model_len)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = RMSNorm(config.hidden_size_per_head, eps=config.rms_norm_eps)
        self.q_norm.weight = torch.nn.Parameter(
            torch.ones((self.num_heads, config.hidden_size_per_head))
        )
        set_weight_attrs(
            self.q_norm.weight, {"weight_loader": sharded_weight_loader(0)}
        )
        self.k_norm = RMSNorm(config.hidden_size_per_head, eps=config.rms_norm_eps)
        self.k_norm.weight = torch.nn.Parameter(
            torch.ones((self.num_kv_heads, config.hidden_size_per_head))
        )
        # Tensor-parallelism shards the K norm weights to the tp ranks
        # in a head-wise manner. This approach does not work if there is only
        # a single KV head, as is the case for PLaMo 2-1B.
        if self.total_num_kv_heads != 1:
            set_weight_attrs(
                self.k_norm.weight, {"weight_loader": sharded_weight_loader(0)}
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_shape = q.shape
        q = q.reshape(q_shape[:-1] + self.q_norm.weight.shape)
        q = self.q_norm.forward_native(q).reshape(q_shape)
        k_shape = k.shape
        k = k.reshape(k_shape[:-1] + self.k_norm.weight.shape)
        k = self.k_norm.forward_native(k).reshape(k_shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Plamo2DecoderLayer(nn.Module):
    def __init__(
        self, vllm_config: VllmConfig, layer_idx: int, prefix: str = "", **kwargs
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.is_mamba = is_mamba(config, layer_idx)
        if self.is_mamba:
            self.mixer = Plamo2MambaMixer(
                vllm_config=vllm_config, prefix=f"{prefix}.mixer"
            )
        else:
            self.mixer = Plamo2AttentionMixer(
                vllm_config=vllm_config, prefix=f"{prefix}.mixer"
            )

        self.mlp = DenseMLP(
            config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )
        self.pre_mixer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mixer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_mixer_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_mixer_norm(hidden_states, residual)

        if self.is_mamba:
            # Plamo2MambaMixer writes output to this tensor
            output = torch.empty_like(hidden_states)
            mixer_kwargs = {
                "output": output,
            }
        else:
            mixer_kwargs = {
                "positions": positions,
            }
        hidden_states = self.mixer(
            hidden_states=hidden_states,
            **mixer_kwargs,
        )
        if self.is_mamba:
            hidden_states = output
        hidden_states = self.post_mixer_norm(hidden_states)
        # Fully Connected
        hidden_states, residual = self.pre_mlp_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_norm(hidden_states)
        return hidden_states, residual


class Plamo2Decoder(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        extra_kwargs = {"is_lora_enabled": bool(vllm_config.lora_config)}

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Plamo2DecoderLayer(
                vllm_config=vllm_config,
                layer_idx=layer_idx,
                prefix=prefix,
                **extra_kwargs,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        return hidden_states, residual


@support_torch_compile
class Plamo2Model(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.layers = Plamo2Decoder(vllm_config=vllm_config, prefix=f"{prefix}.layers")
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        hidden_states, residual = self.layers(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # Update the weight names to be compatible with the vllm version
            # of the model.
            # Do not change the order of the replacements.
            replacements = {
                # Rename incompatible weight names.
                ".A_log": ".A",
                ".B_norm_weight": ".B_norm.weight",
                ".C_norm_weight": ".C_norm.weight",
                ".dt_norm_weight": ".dt_norm.weight",
                ".q_weight": ".q_norm.weight",
                ".k_weight": ".k_norm.weight",
            }
            # Apply replacements based on the defined mappings
            for old, new in replacements.items():
                if old in name:
                    name = name.replace(old, new)

            # Reshape the in_proj weights to match the shape expected
            # by MergedColumnParallelLinear.
            # This works both for unquantized weights and
            # for quantized weights.
            # In the quantized case, the weights are already transposed.
            # Also, in addition to the quantized weights,
            # the zero points and scales have to be reshaped as well.
            # Packing should not be affected by this.
            if (
                ".mixer.in_proj.weight" in name
                or "mixer.in_proj.qweight" in name
                or "mixer.in_proj.scales" in name
                or "mixer.in_proj.qzeros" in name
            ):
                if "mixer.in_proj.weight" in name:
                    loaded_weight = loaded_weight.transpose(0, 1)
                # for weight:
                # loaded_weight.shape[0] == self.config.hidden_size
                # for qweight:
                # loaded_weight.shape[0] == self.config.hidden_size // param.pack_factor  # noqa
                # for scales and qzeros:
                # loaded_weight.shape[0] == self.config.hidden_size // self.vllm_config.quant_config.group_size  # noqa
                loaded_weight = loaded_weight.reshape(
                    loaded_weight.shape[0], self.config.mamba_num_heads, -1
                )
                gate_weight, hidden_states_weight = loaded_weight.chunk(2, dim=-1)
                gate_weight = gate_weight.reshape(loaded_weight.shape[0], -1)
                hidden_states_weight = hidden_states_weight.reshape(
                    loaded_weight.shape[0], -1
                )
                loaded_weight = torch.cat([gate_weight, hidden_states_weight], dim=-1)
                if "mixer.in_proj.weight" in name:
                    loaded_weight = loaded_weight.transpose(0, 1)

            # Offset parameter with vllm's RMSNorm haven't been supported yet.
            if ".pre_mixer_norm" in name:
                loaded_weight += 1.0
            elif ".post_mixer_norm" in name:
                loaded_weight += 1.0 / 5
            elif ".pre_mlp_norm" in name:
                loaded_weight += 1.0
            elif ".post_mlp_norm" in name:
                loaded_weight += 1.0 / (5**1.5)
            elif name == "norm.weight":
                loaded_weight += 1.0

            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Plamo2ForCausalLM(
    torch.nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid
):
    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_up_proj"],
        "in_proj": ["in_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        scheduler_config = vllm_config.scheduler_config

        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config

        # ModelConfig.get_head_size assumes head_dim is set or calculated as
        # hidden_size // num_attention_heads. However, this is not always
        # the case for PLaMo2, as indicated by the FIXME comment.
        self.config.head_dim = self.config.hidden_size_per_head

        self.model = Plamo2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.vocab_size = self.config.vocab_size
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.lm_head",
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            config.vocab_size, self.config.vocab_size
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.
        Args:
            vllm_config: vLLM config
        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        intermediate_size = hf_config.mamba_num_heads * hf_config.hidden_size_per_head

        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=0,
            num_heads=hf_config.mamba_num_heads,
            head_dim=hf_config.hidden_size_per_head,
            state_size=hf_config.mamba_d_state,
            conv_kernel=hf_config.mamba_d_conv,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba2_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
