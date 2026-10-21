# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm, RMSNormGated
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal.inputs import NestedTensors

from .qwen2 import Qwen2MLP
from .qwen3 import Qwen3ForCausalLM
from .qwen3_dflash import DFlashQwen3Attention
from .qwen3_dspark import DSparkMarkovHead
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)


class MambaMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        num_groups: int,
        conv_kernel: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.num_groups = num_groups
        self.inner_size = num_heads * head_dim
        self.conv_dim = self.inner_size + 2 * num_groups * state_size
        self.in_proj = nn.Linear(
            hidden_size, self.inner_size + self.conv_dim + num_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
        )
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.norm = RMSNormGated(self.inner_size, eps=1e-5, norm_before_gate=False)
        self.out_proj = nn.Linear(self.inner_size, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, initial_state: torch.Tensor
    ) -> torch.Tensor:
        num_blocks, block_len, _ = hidden_states.shape
        gate, xbc, dt = torch.split(
            self.in_proj(hidden_states),
            [self.inner_size, self.conv_dim, self.num_heads],
            dim=-1,
        )
        assert self.conv1d.bias is not None
        kernel = self.conv1d.weight.squeeze(1).float()
        padded = F.pad(xbc.float(), (0, 0, kernel.shape[1] - 1, 0))
        conv = self.conv1d.bias.float() + sum(
            kernel[:, k] * padded[:, k : k + block_len] for k in range(kernel.shape[1])
        )
        x, B, C = torch.split(
            F.silu(conv.to(xbc.dtype)),
            [
                self.inner_size,
                self.num_groups * self.state_size,
                self.num_groups * self.state_size,
            ],
            dim=-1,
        )

        heads_per_group = self.num_heads // self.num_groups
        x = x.reshape(num_blocks, block_len, self.num_heads, self.head_dim).float()
        B = B.reshape(num_blocks, block_len, self.num_groups, self.state_size).float()
        C = C.reshape(num_blocks, block_len, self.num_groups, self.state_size).float()
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        state = initial_state
        outputs = []
        for t in range(block_len):
            decay = torch.exp(dt[:, t] * A)
            update = dt[:, t][..., None, None] * (
                x[:, t][..., None] * B[:, t][:, :, None, :]
            )
            state = decay[..., None, None] * state + update
            outputs.append(
                (state * C[:, t][:, :, None, :]).sum(-1) + D[None, :, None] * x[:, t]
            )
        y = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        y = y.reshape(num_blocks, block_len, self.inner_size)
        return self.out_proj(self.norm(y, gate))


class MambaLayer(nn.Module):
    def __init__(self, config, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.ssm_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ssm = MambaMixer(
            config.hidden_size,
            config.mamba_num_heads,
            config.mamba_head_dim,
            config.mamba_d_state,
            config.mamba_n_groups,
            config.mamba_conv_kernel,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        blocks = self.ssm_norm(hidden_states).reshape(
            -1, self.block_size, hidden_states.shape[-1]
        )
        output = self.ssm(blocks, initial_state)
        return hidden_states + output.reshape(hidden_states.shape)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        config,
        sliding_window: int | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DFlashQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            sliding_window=sliding_window,
            causal=sliding_window is not None,
            cache_config=get_current_vllm_config().cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        attn = self.self_attn.attn
        shared_layer_name = attn.kv_sharing_target_layer_name
        attn.kv_sharing_target_layer_name = None
        output = self.self_attn(positions, self.input_layernorm(hidden_states))
        attn.kv_sharing_target_layer_name = shared_layer_name
        return hidden_states + output


class MLPLayer(nn.Module):
    def __init__(
        self, config, quant_config: QuantizationConfig | None, prefix: str
    ) -> None:
        super().__init__()
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


@support_torch_compile
class MambaAttnHybridModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"seed_projs.0.": "seed_proj."},
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ) -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        config = speculative_config.draft_model_config.hf_config
        self.config = config
        self.block_size = speculative_config.num_speculative_tokens + 1
        quant_config = get_draft_quant_config(vllm_config)

        layer_types = set(getattr(config, "layer_types", None) or ["full_attention"])
        if len(layer_types) > 1:
            raise ValueError(
                "mamba_attn_hybrid needs every attention layer to have the same "
                f"layer type, got {sorted(layer_types)}."
            )
        sliding_window = (
            config.sliding_window if "sliding_attention" in layer_types else None
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        layers: list[nn.Module] = []
        for i, layer_type in enumerate(config.block_pattern):
            layer_prefix = maybe_prefix(prefix, f"layers.{i + start_layer_id}")
            if layer_type == "mamba":
                layers.append(MambaLayer(config, self.block_size))
            elif layer_type == "attention":
                layers.append(
                    AttentionLayer(config, sliding_window, quant_config, layer_prefix)
                )
            else:
                layers.append(MLPLayer(config, quant_config, layer_prefix))
        self.layers = nn.ModuleList(layers)

        num_target_layers = len(config.eagle_aux_hidden_state_layer_ids)
        self.fc = nn.Linear(
            num_target_layers * config.hidden_size, config.hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.seed_proj = nn.Linear(
            config.hidden_size,
            config.mamba_num_heads * config.mamba_head_dim * config.mamba_d_state,
            bias=False,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        latent_seed: torch.Tensor,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        num_reqs = hidden_states.shape[0] // self.block_size
        initial_state = (
            self.seed_proj(self.hidden_norm(self.fc(latent_seed[:num_reqs])))
            .view(
                num_reqs,
                self.config.mamba_num_heads,
                self.config.mamba_head_dim,
                self.config.mamba_d_state,
            )
            .float()
        )
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, initial_state)
        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class MambaAttnHybridForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.config = speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = self.config.vocab_size
        self.model = MambaAttnHybridModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=vllm_config.model_config.get_total_num_hidden_layers(),
        )
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=getattr(self.config, "logit_scale", 1.0)
        )
        self.draft_id_to_target_id: nn.Parameter | None = None
        if self.config.draft_vocab_size < vllm_config.model_config.get_vocab_size():
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        latent_seed: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds, latent_seed)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        pass

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        for name, loaded_weight in weights:
            if "t2d" in name or "confidence_head" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
            elif "lm_head" not in name:
                name = "model." + name
            process_eagle_weight(self, name)
            model_weights[name] = loaded_weight
        AutoWeightsLoader(self).load_weights(model_weights.items())
