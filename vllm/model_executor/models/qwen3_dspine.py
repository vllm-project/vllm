# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    get_masked_input_and_mask,
)
from vllm.v1.worker.gpu.spec_decode.dspine.config import DSpineSpeculatorConfig

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import WeightsMapper, maybe_prefix

_TENCENT_METADATA = (
    "draft_block_causal_version",
    "last_write_refine_version",
    "selected_prefix_bridge.independent_write_version",
    "selected_prefix_bridge.terminal_disabled_version",
    "selected_prefix_bridge.out.weight",
)


class AdjacentCausalInjection(nn.Module):
    """Tencent's BF16 adjacent injection and FP32 last-write refinement."""

    def __init__(
        self,
        *,
        hidden_size: int,
        transfer_dim: int,
        message_dim: int,
        num_layers: int,
        block_size: int,
        params_dtype: torch.dtype,
        quant_config: QuantizationConfig | None,
        eps: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.transfer_dim = transfer_dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.eps = eps

        def linear(input_size, output_size, name):
            return ReplicatedLinear(
                input_size,
                output_size,
                bias=False,
                params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, name),
                return_bias=False,
            )

        self.readouts = nn.ModuleList(
            linear(hidden_size, transfer_dim, f"readouts.{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self.writes = nn.ModuleList(
            linear(message_dim, hidden_size, f"writes.{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self.message = linear(transfer_dim, message_dim, "message")
        self.receiver_gate = linear(transfer_dim, message_dim, "receiver_gate")
        self.predecessor_gate = linear(transfer_dim, message_dim, "predecessor_gate")
        self.gate_bias = nn.Parameter(
            torch.empty(message_dim, dtype=params_dtype), requires_grad=False
        )

    @staticmethod
    def rms(states: torch.Tensor) -> torch.Tensor:
        return (
            states.float()
            .square()
            .mean(-1, keepdim=True)
            .clamp_min(torch.finfo(torch.float32).tiny)
            .sqrt()
        )

    def project(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float() / (self.rms(hidden_states) + self.eps)
        projected = self.readouts[layer_idx](normalized.to(hidden_states.dtype)).float()
        features = (
            math.sqrt(self.transfer_dim)
            * projected
            / (projected.norm(dim=-1, keepdim=True) + self.eps)
        )
        return features.to(hidden_states.dtype)

    def token_features(
        self, codebook: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return (codebook[token_ids.long()].float() * math.sqrt(self.transfer_dim)).to(
            codebook.dtype
        )

    def write(
        self,
        layer_idx,
        receiver_states,
        receiver_features,
        predecessor_features,
        *,
        refine: bool = False,
    ):
        def linear(layer, features):
            if refine:
                return F.linear(features, layer.weight.float())
            return layer(features).float()

        gate = torch.sigmoid(
            self.receiver_gate(receiver_features).float()
            + linear(self.predecessor_gate, predecessor_features)
            + self.gate_bias.float()
        )
        message = linear(self.message, predecessor_features)
        gated_message = gate * message
        if not refine:
            gated_message = gated_message.to(receiver_states.dtype)
        update = linear(self.writes[layer_idx], gated_message)
        result = receiver_states.float() + self.rms(receiver_states) * update
        return result if refine else result.to(receiver_states.dtype)

    def forward(
        self,
        layer_idx: int,
        pre_injection_states: torch.Tensor,
        *,
        anchor_features: torch.Tensor,
    ) -> torch.Tensor:
        blocks = pre_injection_states.reshape(-1, self.block_size, self.hidden_size)
        features = self.project(layer_idx, blocks)
        predecessor = torch.cat((anchor_features[:, None, :], features[:, 1:-1]), dim=1)
        successors = self.write(layer_idx, blocks[:, 1:], features[:, 1:], predecessor)
        return torch.cat((blocks[:, :1], successors), dim=1).reshape_as(
            pre_injection_states
        )


def _score_transitions(
    injection: AdjacentCausalInjection,
    norm: nn.Module,
    lm_head: ParallelLMHead,
    transfer_codebook: torch.Tensor,
    candidate_ids: torch.Tensor,
    final_pre_injection: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    final_layer_idx: int,
) -> torch.Tensor:
    top_k = candidate_ids.shape[-1]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    receiver_features = injection.project(final_layer_idx, final_pre_injection)
    receiver_states = final_pre_injection[:, :, None, :]
    local_ids = candidate_ids.long()
    if lm_head.tp_size > 1:
        shard = lm_head.shard_indices
        local_ids, nonlocal_mask = get_masked_input_and_mask(
            local_ids,
            shard.org_vocab_start_index,
            shard.org_vocab_end_index,
            shard.num_org_vocab_padding,
            shard.added_vocab_start_index,
            shard.added_vocab_end_index,
        )
    successor_weights = lm_head.weight[local_ids].float()
    exact_features = transfer_codebook[predecessor_ids].float() * math.sqrt(
        injection.transfer_dim
    )
    refined = injection.write(
        final_layer_idx,
        receiver_states,
        receiver_features[:, :, None, :],
        exact_features,
        refine=True,
    )
    normalized = (
        refined
        / (refined.square().mean(-1, keepdim=True) + norm.variance_epsilon).sqrt()
        * norm.weight.float()
    )
    scores = torch.einsum("bnkh,bnjh->bnkj", normalized, successor_weights)
    if lm_head.tp_size > 1:
        scores.masked_fill_(nonlocal_mask.unsqueeze(-2), 0)
        scores = tensor_model_parallel_all_reduce(scores)
    return scores


class DSpineQwen3Model(DFlashQwen3Model):
    hf_to_vllm_mapper = DFlashQwen3Model.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_substr={"transfer_embeddings": "transfer_codebook"},
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        block_size = 1 + speculative_config.num_speculative_tokens
        configured_block_size = int(draft_config.get("block_size", block_size))
        if configured_block_size != block_size:
            raise ValueError(
                "DSpine block_size must equal 1 + num_speculative_tokens: "
                f"{configured_block_size} != {block_size}."
            )
        transfer_dim = int(draft_config["transfer_dim"])
        if transfer_dim > self.config.hidden_size:
            raise ValueError("DSpine transfer_dim cannot exceed hidden_size.")
        self.transition_top_k = int(draft_config["transition_top_k"])
        self.injection = AdjacentCausalInjection(
            hidden_size=self.config.hidden_size,
            transfer_dim=transfer_dim,
            message_dim=int(draft_config["message_dim"]),
            num_layers=len(self.layers),
            block_size=block_size,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            eps=float(draft_config.get("transfer_space_eps", 1e-6)),
            prefix=maybe_prefix(prefix, "injection"),
        )
        self.transfer_codebook = nn.Parameter(
            torch.empty(
                self.config.vocab_size,
                transfer_dim,
                dtype=vllm_config.model_config.dtype,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        hidden_states = input_embeds
        residual = None
        anchor_features = self.injection.token_features(
            self.transfer_codebook, input_ids[:: self.injection.block_size]
        )
        final_pre_injection = hidden_states
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            pre_injection = (
                hidden_states if residual is None else hidden_states + residual
            )
            if layer_idx == len(self.layers) - 1:
                final_pre_injection = pre_injection
            hidden_states = self.injection(
                layer_idx, pre_injection, anchor_features=anchor_features
            )
            residual = None
        hidden_states = self.norm(hidden_states)
        return hidden_states, final_pre_injection

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        if not any(name.startswith("selected_prefix_bridge.") for name, _ in weights):
            return super().load_weights(weights)
        self._validate_tencent_metadata(dict(weights))
        mapped = list(self._tencent_weights(weights))
        loaded = super().load_weights(mapped)
        required = {"transfer_codebook"} | {
            f"injection.{name}" for name, _ in self.injection.named_parameters()
        }
        if missing := required - loaded:
            raise ValueError(f"Missing DSpine bridge weights: {sorted(missing)}")
        return loaded

    def _validate_tencent_metadata(self, weights: dict[str, torch.Tensor]) -> None:
        if missing := set(_TENCENT_METADATA[2:]) - weights.keys():
            raise ValueError(f"Missing DSpine bridge metadata: {sorted(missing)}")
        for name in _TENCENT_METADATA[:-1]:
            if name in weights:
                weight = weights[name]
                if weight.numel() != 1 or weight.item() != 1:
                    raise ValueError(f"Unsupported DSpine checkpoint version: {name}")
        terminal = weights["selected_prefix_bridge.out.weight"]
        if (
            tuple(terminal.shape)
            != (self.injection.transfer_dim, self.injection.message.output_size)
            or terminal.count_nonzero().item()
        ):
            raise ValueError("DSpine requires a zero terminal projection")

    def _tencent_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        aliases = {
            "codebook": "transfer_codebook",
            "read_weight": f"injection.readouts.{self.injection.num_layers - 1}.weight",
            "gate.weight": "injection.receiver_gate.weight",
            "gate.bias": "injection.gate_bias",
            "predecessor_gate.weight": "injection.predecessor_gate.weight",
            "prev.weight": "injection.message.weight",
        }
        for name, weight in weights:
            if name in _TENCENT_METADATA:
                continue
            key = name.removeprefix("selected_prefix_bridge.")
            if key in ("early_read_weights", "feature_write_weights"):
                read = key == "early_read_weights"
                count = self.injection.num_layers - int(read)
                if weight.ndim != 3 or weight.shape[0] != count:
                    raise ValueError(f"Invalid DSpine layer stack: {name}")
                for index, tensor in enumerate(weight):
                    module = "readouts" if read else "writes"
                    yield f"injection.{module}.{index}.weight", tensor
            else:
                yield aliases.get(key, name), weight


class DSpineQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    speculator_config_class = DSpineSpeculatorConfig

    model_cls = DSpineQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise ValueError("DSpine requires the full target vocabulary.")

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.transition_top_k
        )

    def compute_transition_scores(
        self,
        candidate_ids: torch.Tensor,
        final_pre_injection: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _score_transitions(
            self.model.injection,
            self.model.norm,
            self.lm_head,
            self.model.transfer_codebook,
            candidate_ids,
            final_pre_injection,
            anchor_token_ids,
            len(self.model.layers) - 1,
        )


EntryClass = DSpineQwen3ForCausalLM
