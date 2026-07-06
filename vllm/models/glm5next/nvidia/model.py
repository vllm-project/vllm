# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    VllmConfig,
)
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import (
    MHCPostOp,
    MHCPreOp,
    hc_contract,
    hc_expand,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP as Glm5NextMLP
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE as Glm5NextMoE
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from .attention import Glm5NextLinearAttention, Glm5NextMLAAttention

logger = init_logger(__name__)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        self.layer_kind = "kda" if config.is_kda_layer(layer_idx) else "mla"

        if config.is_kda_layer(layer_idx):
            self.self_attn = Glm5NextLinearAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # MLA layers require the latent head dims, which are guaranteed set
            # on MLA configs; narrow away the `int | None`.
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=getattr(config, "mla_nope", False),
            )

        if (
            self.is_moe
            and self.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc and not is_mtp_layer:
            # mhc config
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_no_norm_weight = config.mhc_no_norm_weight
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            # attn hc
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            # ffn hc
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # 70B or MTP layers: KDA + MoE without HC
        if not self.mhc or self.is_mtp_layer:
            residual = x
            x = self.input_layernorm(x)

            attn_output = torch.empty_like(x)
            self.self_attn(
                hidden_states=x,
                positions=positions,
                output=attn_output,
            )
            x = residual + attn_output

            residual = x
            x = self.post_attention_layernorm(x)
            x = self.mlp(x)
            x = residual + x
            return x

        # mHC start
        if self.layer_idx == 0:
            x = hc_expand(x, self.n)

        # Self Attention
        residual = x
        post, comb, x = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)

        attn_output = torch.empty_like(x)
        self.self_attn(
            hidden_states=x,
            positions=positions,
            output=attn_output,
        )
        x = attn_output

        x = self.hc_post(x, residual, post, comb)

        residual = x
        post, comb, x = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )

        # Fully Connected
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)

        x = self.hc_post(x, residual, post, comb)

        # mHC end
        if self.layer_idx == self.num_hidden_layers - 1:
            x = hc_contract(x, self.n)

        return x

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        post_mix, res_mix, layer_input = self.mhc_pre_op(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
        )
        return post_mix, res_mix, layer_input

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post_op(x, residual, post, comb)


@support_torch_compile
class Glm5NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        """
        if config.index_topk is not None:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
        """
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            # kpool widens the topk buffer: selecting topk_tokens//kpool pools and
            # expanding them yields topk_tokens token indices, plus an always-
            # selected tail of up to kpool-1 incomplete-pool tokens. The attention
            # backend reads the width dynamically via topk_indices.shape[1].
            kpool = getattr(config, "index_kpool", 1) or 1
            buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
            # The sparse MLA attention kernel
            # (triton_convert_req_index_to_global_index) tiles the topk
            # dimension in BLOCK_N=128 columns and requires the buffer width
            # to be a multiple of it; otherwise it raises
            # "NUM_TOPK_TOKENS must be divisible by BLOCK_N". Round up: the
            # extra slots stay -1 (the indexer op initializes the buffer to
            # -1) and are masked out by the attention kernel, so they do not
            # affect the softmax over the selected tokens.
            sparse_topk_block_n = 128
            buffer_width = (
                (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
            ) * sparse_topk_block_n
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            hidden_states = layer(
                positions=positions,
                x=hidden_states,
            )

        if not get_pp_group().is_last_rank:
            # PP: intermediate tensor may be 3D [T, n, H] (after hc_expand)
            # or 2D [T, H] (before hc_expand). Layers handle both correctly
            # since hc_expand only runs at layer 0 (first PP rank) and
            # hc_contract at the last layer (last PP rank).
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA: fuse q_a_proj and kv_a_proj_with_mqa
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            # Indexer: fuse wk and weights_proj
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # GLM5-Next NoPE: checkpoint's kv_a_proj_with_mqa has only kv_lora_rank
        # rows, but the model expects kv_lora_rank + qk_rope_head_dim rows.
        # Pad the missing rope portion with zeros.
        kv_a_pad_size = 0
        if self.config.mla_nope and self.config.qk_rope_head_dim > 0:
            kv_a_pad_size = self.config.qk_rope_head_dim

        _pending_wk_fp8: dict = {}

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs: dict = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # Handle FP8 indexer WK: dequantize to BF16 for fusion with
            # weights_proj into wk_weights_proj.
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # FP8 checkpoint: dequantize BF16-kept MLA projections
            # (q_a_proj / kv_a_proj_with_mqa / o_proj) to BF16.
            if _try_load_fp8_attn_proj(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            ):
                continue

            # Pad kv_a_proj_with_mqa for NoPE models
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                pad = torch.zeros(
                    kv_a_pad_size,
                    *loaded_weight.shape[1:],
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                loaded_weight = torch.cat([loaded_weight, pad], dim=0)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # QKV fusion: skip if fused module doesn't exist in model
                if param_name == ".fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in enumerate(expert_params_mapping):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params


class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
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
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[
        MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc
    ]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

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


def get_spec_layer_idx_from_weight_name(
    config: Glm5NextConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"layers.{layer_idx + i}."):
                return layer_idx + i
    return None


def _try_load_fp8_indexer_wk(name, tensor, buf, params_dict, loaded_params):
    if "indexer.wk." not in name or "wk_weights" in name:
        return False
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False
    layer_prefix = name.rsplit(".wk.", 1)[0]
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    del buf[layer_prefix]
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale_inv,
        group_shape=GroupShape(block_size, block_size),
        out_dtype=torch.bfloat16,
    )

    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


def _dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a block-FP8 (e4m3) weight with per-block scale to BF16.

    Unlike ``scaled_dequantize`` this tolerates a non-divisible (partial last
    block) shape by zero-padding to a multiple of ``block_size`` before the
    scale broadcast and trimming back afterwards (e.g. kv_a_proj_with_mqa is
    576 rows = 4*128 + 64).
    """
    out_dim, in_dim = weight_fp8.shape
    pad_out = (-out_dim) % block_size
    pad_in = (-in_dim) % block_size
    w = weight_fp8
    if pad_out or pad_in:
        w = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    # scale_inv is (ceil(out/block), ceil(in/block)); broadcast to (out, in).
    s = scale_inv.to(torch.float32)
    s_full = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    out = (w.to(torch.float32) * s_full).to(torch.bfloat16)
    return out[:out_dim, :in_dim].contiguous()


# FP8 checkpoint projections that the MODEL keeps in BF16, so the block-FP8
# (weight + weight_scale_inv) must be dequantized to BF16 on load.
# Maps checkpoint proj-suffix -> (buffer key, model target base, fused shard id
# or None for a direct projection, whether NoPE rope-padding applies).
_FP8_ATTN_PROJS = {
    ".q_a_proj.": ("q_a", "fused_qkv_a_proj", 0, False),
    ".kv_a_proj_with_mqa.": ("kv_a", "fused_qkv_a_proj", 1, True),
    ".q_b_proj.": ("q_b", "q_b_proj", None, False),
    ".o_proj.": ("o_proj", "o_proj", None, False),
}


def _try_load_fp8_attn_proj(
    name,
    tensor,
    buf,
    params_dict,
    loaded_params,
    kv_a_pad_size: int,
) -> bool:
    """Dequantize FP8 q_a_proj / kv_a_proj_with_mqa / o_proj to BF16 on load.

    The FP8 checkpoint stores these as block-FP8 (weight + weight_scale_inv),
    but the model holds them in BF16 (``fused_qkv_a_proj`` is always BF16 via
    DeepSeekV2FusedQkvAProjLinear; ``o_proj`` is excluded by
    modules_to_not_convert). When the model target is BF16 (no
    ``weight_scale_inv`` param) we dequantize; otherwise we return False so the
    normal stacked/direct path loads the FP8 tensor as-is.
    """
    matched = None
    for suffix, info in _FP8_ATTN_PROJS.items():
        if suffix in name:
            matched = (suffix, info)
            break
    if matched is None:
        return False
    suffix, (key, target_base, shard_id, is_kva) = matched
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False

    layer_prefix = name.rsplit(suffix, 1)[0]
    target_w = f"{layer_prefix}.{target_base}.weight"
    target_s = f"{layer_prefix}.{target_base}.weight_scale_inv"
    # If the model actually kept this projection in FP8, let the normal path
    # handle it (it has a weight_scale_inv param).
    if target_s in params_dict:
        return False

    entry = buf.setdefault(layer_prefix, {}).setdefault(key, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    buf[layer_prefix].pop(key, None)
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
    # NoPE: pad kv_a rope portion (kv_lora_rank -> kv_lora_rank + qk_rope_head_dim).
    if is_kva and kv_a_pad_size > 0:
        pad = torch.zeros(
            kv_a_pad_size,
            weight_bf16.shape[1],
            dtype=weight_bf16.dtype,
            device=weight_bf16.device,
        )
        weight_bf16 = torch.cat([weight_bf16, pad], dim=0)

    param = params_dict[target_w]
    if shard_id is None:
        param.weight_loader(param, weight_bf16)
    else:
        param.weight_loader(param, weight_bf16, shard_id)
    loaded_params.add(target_w)
    return True
