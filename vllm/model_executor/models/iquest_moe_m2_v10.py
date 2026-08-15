# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only IQuest MoE M2 (v1.0) model compatible with HF weights.

Reference implementation: the ``modeling_m2.py`` / ``configuration_m2.py``
shipped inside the M2 checkpoint (``M2ForCausalLM``).

M2 shares most of its skeleton with v1.3 (``iquest_moe_v13.py``): reordered
norm, per-branch output scaling, attention sink, over-encoding, and the
global/sliding-window layer interleave.  The M2-specific differences handled
here are:

1. **Latent-bottleneck MoE.**  Routed experts do not live in ``hidden_size``
   space.  ``fc1_latent_proj`` maps ``hidden_size -> moe_latent_size`` before
   routing and ``fc2_latent_proj`` maps back afterwards; both projections are
   shared by all experts.  Implemented with ``SharedFusedMoE``'s
   ``routed_input_transform`` hook (same pattern as ``nemotron_h.py``).

2. **Sink expert joined into the router softmax.**  The shared expert's gate
   logit takes part in the *same* softmax as the top-k routed logits, so the
   shared expert competes with the routed experts for probability mass.  See
   ``M2MoE._routing_split`` for the algebraic identity that lets us keep using
   the stock fused-MoE routing.

3. **Cross-layer shortcut.**  ``shortcut_pairs`` mixes the output of one layer
   back into a much later layer.  Carried across pipeline stages through
   ``IntermediateTensors`` when PP is enabled.

4. **Zero-centred gamma on the q/k norms** (``1 + w`` instead of ``w``).

5. **Fused over-encoding table.**  The M2 checkpoint stores a single
   ``over_encoding.embedding.weight`` covering all 16 hash heads, whereas v1.3
   stores one tensor per head.

Deployment notes
----------------
* Register the architecture in ``vllm/model_executor/models/registry.py``::

      "M2ForCausalLM": ("iquest_moe_m2_v10", "IquestMoeM2ForCausalLM"),

* No edits to the checkpoint's ``config.json`` are needed.  The over-encoding
  input ids and the sink-token metadata are prepared by ``GPUModelRunner``
  *before* the model is constructed, so this file cannot patch the config in
  time.  Instead ``ModelConfig`` keys off ``model_type == "m2"`` and translates
  the switches whose spelling differs from v1.3 -- see ``_M2_CONFIG_ALIASES``
  in ``vllm/config/model.py``:

  - ``use_over_encoding`` is spelled ``oe_enable``
  - ``n_head_per_ngram`` is spelled ``oe_n_head_per_ngram``
  - ``n_embed_per_ngram`` is spelled ``oe_n_embed_per_ngram``
  - ``max_ngram_size`` is spelled ``oe_max_ngram_size``
  - ``oe_padding_token_id`` is spelled ``oe_pad_token_id``
  - ``enable_sink_attention`` is spelled ``enable_sink_token``

  ``oe_vocab_size`` and ``num_sink_tokens`` are spelled the same either way.
  ``embed_scale`` / ``over_embed_scale`` are absent from M2 and default to 1.0,
  which is exactly the reference's ``(base + oe) / sqrt(2)``.

  A new M2 variant therefore has to add its ``model_type`` to
  ``_M2_MODEL_TYPES``.  If it does not, the over-encoding table and the
  attention sink are silently skipped -- the server still starts and still
  emits fluent text -- so that case logs a warning rather than failing.

* ``mtp_num_layers`` weights (``model.mtp.*``) are skipped here.  Speculative
  decoding needs a separate draft-model module, following the layout of
  ``iquest_moe_v13_mtp.py``.

Config keys that are inert in the reference implementation
----------------------------------------------------------
``modeling_m2.py`` never reads these, so the value in ``config.json`` does not
describe the model's behaviour.  Do not "fix" this file to honour them without
first changing the reference:

* ``moe_router_pre_softmax`` / ``moe_router_score_function`` -- routing is
  unconditionally top-k-then-softmax, which is what we reproduce.
* ``attention_dropout``, ``initializer_range``, ``use_cache`` -- training or
  HF-plumbing only.
* ``qk_layernorm_clip_value`` -- declared in ``M2Config`` (default 10.0) but
  never applied.  If training clips the q/k norm gamma, this is a silent
  train/inference mismatch in the *reference*, not in this port.
* ``max_position_embeddings`` -- the reference RoPE builds frequencies on the
  fly, so 8192 is not a real ceiling (see ``M2Attention``).
* ``moe_permute_fusion``, ``moe_use_sonicmoe``, ``sink_use_ubiq_kernel`` --
  Megatron kernel selection; vLLM has its own kernels.

Known deviation from the reference implementation
-------------------------------------------------
``config.fp32_residual_connection`` is **not** honoured: the residual stream is
kept in the model dtype (bf16) as in ``iquest_moe_v13.py``, rather than fp32.
Each RMSNorm still accumulates internally in fp32, so the difference is one
bf16 rounding of the residual per add.  This matches the v1.3 path that was
validated for M1, but it has *not* been verified for M2 -- treat it as an open
item, not as a settled equivalence.
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, SharedFusedMoE
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.oe_embedding import OEEmbedding
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# Weights under this prefix belong to the multi-token-prediction head, which is
# loaded by a separate draft-model module.
_MTP_PREFIX = "mtp."


def _layer_uses_rope(config, layer_idx: int) -> bool:
    """``no_rope_pattern[i] == 1`` means layer ``i`` does *not* apply RoPE.

    Note the inverted polarity relative to ``window_attention_pattern``.
    """
    pattern = getattr(config, "no_rope_pattern", None)
    if not pattern:
        return True
    return not pattern[layer_idx]


def _layer_sliding_window(config, layer_idx: int) -> int | None:
    """Left-context window for layer ``layer_idx``, or ``None`` for full attn.

    ``window_size`` is the reference implementation's ``(left, right)`` pair;
    only the left half is meaningful for causal decoding.
    ``window_attention_pattern[i] == 0`` means layer ``i`` is a full-attention
    layer even though a window size is configured.
    """
    window_size = getattr(config, "window_size", None)
    if not window_size:
        return None
    pattern = getattr(config, "window_attention_pattern", None)
    if pattern is not None and not pattern[layer_idx]:
        return None
    left = window_size[0] if isinstance(window_size, (list, tuple)) else window_size
    if left is None or left < 0:
        return None
    # vLLM counts the query token itself as part of the window, the reference
    # mask is `i - j <= left`, i.e. `left` history tokens plus self.
    return int(left) + 1


def _layer_is_moe(config, layer_idx: int) -> bool:
    pattern = getattr(config, "moe_layer_pattern", None)
    if not pattern:
        return True
    return bool(pattern[layer_idx])


class M2RMSNorm(nn.Module):
    """RMSNorm with optional zero-centred gamma.

    With ``zero_centered=True`` the stored weight is an *offset*: the effective
    scale is ``1 + w``.  M2 uses this for the q/k norms
    (``qk_layernorm_zero_centered_gamma``), so dropping the ``+ 1`` would scale
    q/k by ~0.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, zero_centered: bool = False
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(hidden_size) if zero_centered else torch.ones(hidden_size)
        )
        self.variance_epsilon = eps
        self.zero_centered = zero_centered

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = self.weight
        if self.zero_centered:
            weight = weight + 1.0
        return (weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return (
            f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, "
            f"zero_centered={self.zero_centered}"
        )


class M2MLP(nn.Module):
    """SwiGLU MLP used by the dense layer and by the shared (sink) expert."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        return out


class M2SinkExperts(nn.Module):
    """Shared ("sink") experts plus the gate whose logit joins the router.

    The module tree mirrors the checkpoint exactly::

        mlp.shared_experts.sink_gate
        mlp.shared_experts.shared_experts.{i}.{gate,up,down}_proj.weight

    ``forward`` returns the *unweighted* sum of the shared experts; the mixing
    coefficient is applied by :class:`M2MoE`, which owns the joint softmax.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        router_dtype: torch.dtype = torch.float32,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_shared = config.moe_sink_expert_share_num
        self.join_softmax = config.moe_sink_expert_join_softmax
        if self.join_softmax:
            self.sink_gate = nn.Parameter(
                torch.empty(self.num_shared, config.hidden_size, dtype=router_dtype)
            )
            set_weight_attrs(self.sink_gate, {"weight_loader": default_weight_loader})
        else:
            self.register_parameter("sink_gate", None)

        self.shared_experts = nn.ModuleList(
            [
                M2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_shared_expert_intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    # SharedFusedMoE performs the reduction together with the
                    # routed output.
                    reduce_results=False,
                    prefix=f"{prefix}.shared_experts.{i}",
                )
                for i in range(self.num_shared)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.shared_experts[0](hidden_states)
        for expert in self.shared_experts[1:]:
            out = out + expert(hidden_states)
        return out


class M2MoE(nn.Module):
    """Latent-bottleneck MoE with a sink expert sharing the router softmax.

    Reference (``modeling_m2.py``)::

        top_logits, top_idx = topk(gate(x), k)  # raw logits
        sink_logit = x @ sink_gate.T  # [T, 1]
        joint = softmax(cat([top_logits, sink_logit]))  # (k + 1)-way
        probs = joint[:, :k]  # routed experts
        p_sink = joint[:, k:]  # shared expert

    Materialising that joint softmax would mean writing a custom routing
    function.  It is unnecessary: with ``L = logsumexp(top_logits)`` and ``s``
    the sink logit,

        probs  = sigmoid(L - s) * softmax(top_logits)
        p_sink = 1 - sigmoid(L - s)

    ``softmax(top_logits)`` is exactly what the stock fused-MoE router produces
    for ``scoring_func="softmax", renormalize=True`` (it takes the top-k of the
    logits and softmaxes *those*).  So we only have to scale the two branch
    outputs by a per-token scalar, which is both cheaper and numerically
    better-behaved than building the (k+1)-way softmax.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.latent_size = config.moe_latent_size or config.hidden_size
        self.use_latent = self.latent_size != self.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()

        # The reference router runs in fp32 (``moe_router_dtype``); the top-k
        # boundary among 512 experts is sensitive enough that this matters.
        router_dtype = torch.float32
        if getattr(config, "moe_router_dtype", None) == "fp64":
            # fp64 gates are not supported by the fused kernels; fp32 is the
            # closest we can do, and is what the checkpoint actually uses.
            logger.warning_once(
                "moe_router_dtype=fp64 is not supported, falling back to fp32."
            )
        self.router_dtype = router_dtype

        self.topk_scaling_factor = getattr(
            config, "moe_router_topk_scaling_factor", None
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=router_dtype,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.enable_sink_expert = getattr(config, "moe_sink_expert_enable", False)
        if self.enable_sink_expert:
            # Registered before ``self.experts`` so that ``named_parameters``
            # reports the canonical ``mlp.shared_experts.*`` names rather than
            # the alias inside SharedFusedMoE.
            self.shared_experts = M2SinkExperts(
                config,
                quant_config=quant_config,
                router_dtype=router_dtype,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        if self.use_latent:
            self.fc1_latent_proj = ReplicatedLinear(
                config.hidden_size,
                self.latent_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1_latent_proj",
            )
            self.fc2_latent_proj = ReplicatedLinear(
                self.latent_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2_latent_proj",
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.num_local_experts,
            top_k=self.top_k,
            hidden_size=self.latent_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            # topk-then-softmax, matching moe_router_pre_softmax=False.
            renormalize=True,
            scoring_func="softmax",
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            router_logits_dtype=router_dtype,
            # Latent bottleneck: applied to the routed branch only, so the
            # shared expert keeps seeing full-width hidden states.
            routed_input_transform=self.fc1_latent_proj,
        )

    def _routing_split(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | None:
        """Return ``alpha``: the share of probability mass left to the routed
        experts once the sink expert has taken its cut.  ``None`` when the sink
        expert does not join the router softmax.
        """
        if not self.enable_sink_expert or self.shared_experts is None:
            return None
        sink_gate = self.shared_experts.sink_gate
        if sink_gate is None:
            # Shared expert present but weighted implicitly by 1.0.
            return None

        # [T, num_shared]; the reference only ever configures num_shared == 1.
        sink_logits = torch.nn.functional.linear(
            hidden_states.to(self.router_dtype), sink_gate
        )
        # logsumexp over the *selected* logits only -- this is the normaliser of
        # softmax(top_logits), which is what the fused router applies.
        top_logits = torch.topk(router_logits, self.top_k, dim=-1).values
        log_z = torch.logsumexp(top_logits, dim=-1, keepdim=True)
        return torch.sigmoid(log_z - sink_logits)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits, _ = self.gate(hidden_states.to(self.router_dtype))
        alpha = self._routing_split(hidden_states, router_logits)

        # SharedFusedMoE runs the shared experts on the full-width input and
        # applies ``routed_input_transform`` (fc1_latent_proj) to the routed
        # branch.
        shared_out, routed_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.use_latent:
            routed_out, _ = self.fc2_latent_proj(routed_out)

        routed_scale = alpha
        if self.topk_scaling_factor is not None:
            routed_scale = (
                self.topk_scaling_factor
                if alpha is None
                else alpha * self.topk_scaling_factor
            )
        if routed_scale is not None:
            routed_out = routed_out * (
                routed_scale.to(routed_out.dtype)
                if isinstance(routed_scale, torch.Tensor)
                else routed_scale
            )

        if shared_out is not None:
            if alpha is not None:
                shared_out = shared_out * (1.0 - alpha).to(shared_out.dtype)
            routed_out = routed_out + shared_out

        # Both branches were built with reduce_results=False; reduce once here.
        if self.tp_size > 1:
            routed_out = self.experts.maybe_all_reduce_tensor_model_parallel(routed_out)

        return routed_out.view(orig_shape)


class M2Attention(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.cache_config = vllm_config.cache_config

        self.hidden_size = config.hidden_size
        # The reference M2RotaryEmbedding builds its frequencies on the fly and
        # never consults max_position_embeddings, so the checkpoint value (8192)
        # is not a real ceiling.  vLLM precomputes a cos/sin cache instead, so
        # size it to whatever context is actually being served.
        max_position_embeddings = max(
            getattr(config, "max_position_embeddings", 8192) or 8192,
            vllm_config.model_config.max_model_len,
        )

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        zero_centered = getattr(config, "qk_layernorm_zero_centered_gamma", False)
        if getattr(config, "qk_layernorm", True):
            self.q_norm = M2RMSNorm(
                self.head_dim, eps=config.rms_norm_eps, zero_centered=zero_centered
            )
            self.k_norm = M2RMSNorm(
                self.head_dim, eps=config.rms_norm_eps, zero_centered=zero_centered
            )
        else:
            self.q_norm = None
            self.k_norm = None

        layer_idx = 0 if is_mtp_layer else extract_layer_index(prefix)

        # Partial RoPE: rotary_percent of head_dim, rounded down to even.
        rotary_dim = int(self.head_dim * getattr(config, "rotary_percent", 1.0))
        rotary_dim -= rotary_dim % 2
        rope_parameters = dict(getattr(config, "rope_parameters", None) or {})
        rope_parameters["rope_theta"] = getattr(config, "rope_theta", 10000.0)
        if rotary_dim != self.head_dim:
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim

        # Full-attention layers use NoPE, sliding-window layers use RoPE.
        uses_rope = _layer_uses_rope(config, layer_idx) and rotary_dim > 0
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=max_position_embeddings,
                rope_parameters=rope_parameters,
                # GPT-NeoX half-split rotation, matching `_rotate_half`.
                is_neox_style=True,
            )
            if uses_rope
            else None
        )

        sliding_window = (
            None if is_mtp_layer else _layer_sliding_window(config, layer_idx)
        )

        self.enable_sink_attention = getattr(config, "enable_sink_token", False) or (
            getattr(config, "enable_sink_attention", False)
        )
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=self.cache_config,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            enable_sinks_kv=self.enable_sink_attention,
        )

        if self.enable_sink_attention:
            num_sink_tokens = getattr(config, "num_sink_tokens", 1)
            if num_sink_tokens != 1:
                raise ValueError(
                    f"Only num_sink_tokens == 1 is supported, got {num_sink_tokens}."
                )
            # A learnable *key* per KV head (not a scalar bias): it enters the
            # softmax denominator and contributes no value.
            self.sink_token = nn.Parameter(
                torch.zeros(
                    (1, self.num_kv_heads, self.head_dim),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                ),
                requires_grad=False,
            )
            set_weight_attrs(
                self.sink_token, {"weight_loader": self.sink_token_weight_loader}
            )
            self.sinks_k = torch.zeros(
                (self.max_num_seqs, 1, self.num_kv_heads, self.head_dim),
                device=self.sink_token.device,
                dtype=config.torch_dtype,
            )
            self.sinks_v = torch.zeros_like(self.sinks_k)
            self.attn.populate_sinks_kv(sinks_k=self.sinks_k, sinks_v=self.sinks_v)

    def sink_token_weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ):
        # Checkpoint shape is [num_sink_tokens, num_kv_heads, head_dim].
        assert loaded_weight.dim() == 3, (
            f"expected a 3-dim sink_token, got {tuple(loaded_weight.shape)}"
        )
        # Sink keys are sharded along the KV-head axis, which is why TP is
        # capped at num_key_value_heads for this model family.
        shard_start = self.tp_rank * self.num_kv_heads
        shard_end = (self.tp_rank + 1) * self.num_kv_heads
        param.copy_(loaded_weight[:, shard_start:shard_end])
        self.sinks_k.copy_(
            param.unsqueeze(0).expand(self.max_num_seqs, -1, -1, -1).contiguous()
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.q_norm is not None:
            q_by_head = q.view(*q.shape[:-1], -1, self.head_dim)
            q = self.q_norm(q_by_head).view(q.shape)
            k_by_head = k.view(*k.shape[:-1], -1, self.head_dim)
            k = self.k_norm(k_by_head).view(k.shape)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class M2DecoderLayer(nn.Module):
    """Reordered-norm decoder layer.

    For every layer but the first, the residual is taken *after*
    ``input_layernorm``, i.e. the norm applies to the residual stream itself
    rather than only to the branch input::

        x = input_layernorm(h)
        h = x + post_self_attention_layernorm(attn(x)) * attn_out_scale
        h = h + mlp(pre_mlp_layernorm(h)) * ffn_out_scale

    The first layer keeps the raw residual and adds a post-MLP norm (sandwich
    norm), and uses scale 1.0 on both branches.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.layer_idx = extract_layer_index(prefix)
        self.reordered_norm = getattr(config, "reordered_norm", True)
        self.is_first_layer = self.layer_idx == 0
        # Sandwich norm == the first-layer variant of reordered norm.
        self.use_sandwich_norm = self.reordered_norm and self.is_first_layer

        self.self_attn = M2Attention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )

        if _layer_is_moe(config, self.layer_idx):
            self.mlp = M2MoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = M2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = M2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attention_layernorm = M2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_mlp_layernorm = M2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_sandwich_norm:
            self.post_mlp_layernorm = M2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        if self.is_first_layer:
            self.attn_out_scale = getattr(config, "first_layer_attn_out_scale", 1.0)
            self.ffn_out_scale = getattr(config, "first_layer_ffn_out_scale", 1.0)
        else:
            self.attn_out_scale = getattr(config, "attn_out_scale", 1.0)
            self.ffn_out_scale = getattr(config, "ffn_out_scale", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(positions=positions, hidden_states=normalized)
        attn_out = self.post_self_attention_layernorm(attn_out)

        if self.use_sandwich_norm:
            # First layer: residual is the *un*-normalized stream.
            h = hidden_states + attn_out * self.attn_out_scale
            mlp_out = self.mlp(self.pre_mlp_layernorm(h))
            return h + self.post_mlp_layernorm(mlp_out) * self.ffn_out_scale

        # reordered norm: the normalized stream becomes the residual.
        residual = normalized if self.reordered_norm else hidden_states
        h = residual + attn_out * self.attn_out_scale
        mlp_out = self.mlp(self.pre_mlp_layernorm(h))
        return h + mlp_out * self.ffn_out_scale


@support_torch_compile
class M2Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = M2DecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = M2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # Cross-layer shortcut. ``shortcut_pairs`` is 1-based in the reference
        # implementation (`enumerate(self.layers, start=1)`), so convert to
        # 0-based layer indices here -- an off-by-one is silent and expensive.
        shortcut_pairs = getattr(config, "shortcut_pairs", None)
        self.shortcut_alpha = float(getattr(config, "shortcut_alpha", 0.0) or 0.0)
        if shortcut_pairs and self.shortcut_alpha:
            self.shortcut_source = int(shortcut_pairs[0]) - 1
            self.shortcut_target = int(shortcut_pairs[1]) - 1
            if self.shortcut_source >= self.shortcut_target:
                raise ValueError(
                    "shortcut_pairs must be (source, target) with source < "
                    f"target, got {tuple(shortcut_pairs)}"
                )
            self.shortcut_target_scale = (1.0 - self.shortcut_alpha**2) ** 0.5
        else:
            self.shortcut_source = None
            self.shortcut_target = None
            self.shortcut_target_scale = 1.0

        # The shortcut may span a pipeline boundary, in which case it has to
        # travel with the hidden states.  One extra hidden-size tensor per PP
        # hop; only paid when both PP and the shortcut are enabled.
        self.shortcut_crosses_pp = (
            self.shortcut_source is not None and get_pp_group().world_size > 1
        )
        intermediate_keys = ["hidden_states"]
        if self.shortcut_crosses_pp:
            intermediate_keys.append("shortcut")
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            intermediate_keys, config.hidden_size
        )

        self.use_oe_embedding = vllm_config.model_config.get_enable_oe_embedding()
        if self.use_oe_embedding:
            if get_pp_group().is_first_rank:
                self.over_encoding = OEEmbedding(vllm_config.model_config)
            else:
                self.over_encoding = PPMissingLayer()
        self.enable_sink_attention = getattr(config, "enable_sink_token", False)

    def embed_input_ids(
        self, input_ids: torch.Tensor, oe_input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.use_oe_embedding and oe_input_ids is not None:
            return self.over_encoding(embeds, oe_input_ids)
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        oe_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        shortcut: torch.Tensor | None = None
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(
                    input_ids, oe_input_ids=oe_input_ids
                )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            if self.shortcut_crosses_pp:
                shortcut = intermediate_tensors["shortcut"]

        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer), self.start_layer
        ):
            hidden_states = layer(positions, hidden_states)
            if self.shortcut_source is None:
                continue
            if layer_idx == self.shortcut_target and shortcut is not None:
                hidden_states = (
                    self.shortcut_target_scale * hidden_states
                    + self.shortcut_alpha * shortcut
                )
            if layer_idx == self.shortcut_source:
                shortcut = hidden_states

        if not get_pp_group().is_last_rank:
            tensors = {"hidden_states": hidden_states}
            if self.shortcut_crosses_pp:
                tensors["shortcut"] = (
                    shortcut
                    if shortcut is not None
                    else torch.zeros_like(hidden_states)
                )
            return IntermediateTensors(tensors)

        return self.norm(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_local_experts,
        )

    def _load_over_encoding(
        self,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> str | None:
        """Load the fused n-gram hash table.

        The M2 checkpoint stores one tensor covering all ``oe_total_heads``
        hash tables (~5.12e8 rows), while v1.3 stores one tensor per head.
        Slice out this TP rank's row range.
        """
        target = "over_encoding.oe_embeder.weight"
        if target not in params_dict:
            return None
        param = params_dict[target]
        shard = self.over_encoding.oe_embeder.shard_indices
        start, end = shard.org_vocab_start_index, shard.org_vocab_end_index
        param.data[: end - start].copy_(loaded_weight[start:end])
        return target

    def load_weights(  # noqa: C901
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            if name.startswith(_MTP_PREFIX):
                # Belongs to the speculative-decoding draft module.
                continue
            if is_pp_missing_parameter(name, self):
                continue

            if "over_encoding" in name:
                if not self.use_oe_embedding:
                    logger.warning_once("over encoding feature is disabled")
                    continue
                if name.endswith("over_encoding.embedding.weight"):
                    target = self._load_over_encoding(loaded_weight, params_dict)
                    if target is not None:
                        loaded_params.add(target)
                    continue
                # over_encoding.proj.weight maps straight through.

            if not self.enable_sink_attention and "sink_token" in name:
                logger.warning_once("sink attention feature is disabled")
                continue

            # The shared (sink) expert lives at
            #   mlp.shared_experts.shared_experts.{i}.*
            # whose substring "experts.{i}.down_proj." also matches the routed
            # expert mapping. Flag it so only the stacked mapping can claim it.
            is_shared_expert = "shared_experts" in name

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts are handled by expert_params_mapping below;
                # rewriting the name here would corrupt that lookup.
                if "mlp.experts" in name and not is_shared_expert:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name.endswith(".bias") and new_name not in params_dict:
                    continue
                if is_pp_missing_parameter(new_name, self):
                    continue
                if new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                param.weight_loader(param, loaded_weight, shard_id)
                name, handled = new_name, True
                break
            if handled:
                loaded_params.add(name)
                continue

            if not is_shared_expert:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    new_name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(new_name, self):
                        continue
                    param = params_dict[new_name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        new_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    name, handled = new_name, True
                    break
                if handled:
                    loaded_params.add(name)
                    continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name.endswith("kv_scale"):
                remapped = name.replace(".kv_scale", ".attn.kv_scale")
                if remapped not in params_dict:
                    logger.warning_once(
                        "Found kv scale in the checkpoint (e.g. %s), but not "
                        "the expected name in the model (e.g. %s). kv-scale is "
                        "not loaded.",
                        name,
                        remapped,
                    )
                    continue
                name = remapped
            if name not in params_dict:
                logger.warning_once("Skipping unexpected weight %s", name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class IquestMoeM2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = M2DecoderLayer,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = M2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
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
        oe_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            oe_input_ids=oe_input_ids,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["model.mtp."])
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
