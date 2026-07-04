# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import regex as re
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model, DFlashQwen3DecoderLayer
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)

class DominoPrefixGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))

    def forward(
        self, input_embed: torch.Tensor, prev_hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = input_embed.shape[0]
        
        gi = F.linear(input_embed, self.weight_ih_l0)
        gi_fused = gi.view(batch_size, 3, self.hidden_size)

        if prev_hidden is None:
            update_gate = torch.sigmoid(gi_fused[:, 1, :])
            new_gate = torch.tanh(gi_fused[:, 2, :])
            return (1.0 - update_gate) * new_gate

        gh = F.linear(prev_hidden, self.weight_hh_l0)
        gh_fused = gh.view(batch_size, 3, self.hidden_size)

        reset_gate = torch.sigmoid(gi_fused[:, 0, :] + gh_fused[:, 0, :])
        update_gate = torch.sigmoid(gi_fused[:, 1, :] + gh_fused[:, 1, :])
        new_gate = torch.tanh(gi_fused[:, 2, :] + reset_gate * gh_fused[:, 2, :])

        return (1.0 - update_gate) * new_gate + update_gate * prev_hidden


@support_torch_compile
class DominoQwen3Model(DFlashQwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix)
        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        assert drafter_config.get("projector_type", None) == "domino"
        self.gru_hidden_dim = drafter_config["gru_hidden_dim"]
        self.emb_dim = drafter_config["emb_dim"]

        # TODO change to ParallelLinear
        self.prefix_gru = DominoPrefixGRUCell(
            input_size=self.config.hidden_size,
            hidden_size=self.gru_hidden_dim,
        )
        in_dim = self.config.hidden_size + self.gru_hidden_dim
        embed_proj_dims = drafter_config.get("embed_proj_dims", None)
        emb_dim = drafter_config.get("emb_dim", None)
        
        assert emb_dim is not None 
        "emb_dim must be provided in domino"
        
        if embed_proj_dims is None:
            embed_proj_dims = [in_dim, emb_dim, self.config.vocab_size]

        self.embed_proj = nn.ModuleList(
            [
                ReplicatedLinear(
                    input_size=embed_proj_dims[i],
                    output_size=embed_proj_dims[i + 1],
                    bias=False,
                    params_dtype=vllm_config.model_config.dtype,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"embed_proj.{2 * i}"),
                    return_bias=False,
                )
                for i in range(len(embed_proj_dims) - 1)
            ]
        )
        
        self.embed_proj_act = nn.SiLU()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            # replace embed_proj.{0, 2, 4, ...}.weight to
            # embed_proj.{0, 1, 2, ...}.weight
            if "embed_proj" in name:
                name = re.sub(r"\d+", lambda x: str(int(x.group()) // 2), name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DominoQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DominoQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def gru_forward(self, hidden_states: torch.Tensor, prev_input: torch.Tensor | None):
        return self.model.prefix_gru(hidden_states, prev_input)

    def domino_mlp_forward(
        self,
        hidden_states: torch.Tensor,
    ):
        x = self.model.embed_proj[0](hidden_states)
        return self.model.embed_proj[1](self.model.embed_proj_act(x))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
