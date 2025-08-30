import torch
from torch import nn
import logging
import math
from collections.abc import Iterable
from typing import Optional
from contextlib import nullcontext

from vllm.model_executor.models.whisper import (
    WhisperEncoderLayer,
    WhisperEncoder,
    WhisperModel,
    WhisperForConditionalGeneration,
    WhisperMultiModalProcessor,
    WhisperDummyInputsBuilder,
    WhisperProcessingInfo,
    WhisperDecoder
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.utils import make_layers
from vllm.transformers_utils.configs.lite_whisper import LiteWhisperConfig
from vllm.attention import AttentionType
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.distributed import get_tensor_model_parallel_world_size

logger = logging.getLogger(__name__)

class LinearLowRank(nn.Module):
    """Low-rank linear layer compatible with HF checkpoint names weight1/weight2."""
    def __init__(self, in_features: int, out_features: int, low_rank_features: int):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(in_features, low_rank_features))
        self.weight2 = nn.Parameter(torch.randn(low_rank_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.weight1) @ self.weight2 + self.bias
    
    @property
    def weight(self):
        # Return as a tuple for loader compatibility
        return (self.weight1, self.weight2)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str | None = None):
        """Custom weight loader for low-rank matrices."""
        if shard_id is None:
            # This is a regular linear weight that needs to be decomposed
            # For now, we'll initialize with SVD decomposition
            U, S, V = torch.svd(loaded_weight)
            rank = min(self.weight1.shape[1], U.shape[1], V.shape[0])
            
            # Initialize weight1 and weight2 using SVD
            sqrt_s = torch.sqrt(S[:rank])
            self.weight1.data.copy_(U[:, :rank] * sqrt_s.unsqueeze(0))
            self.weight2.data.copy_((V[:rank, :] * sqrt_s.unsqueeze(1)))
        elif shard_id == "weight1":
            self.weight1.data.copy_(loaded_weight)
        elif shard_id == "weight2":
            self.weight2.data.copy_(loaded_weight)
        elif shard_id == "bias":
            self.bias.data.copy_(loaded_weight)

class LiteWhisperAttention(nn.Module):
    """Lite Whisper attention with separate low-rank q/k/v projections."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        standalone_encoder: bool = False,
        low_rank_config: dict[str, int] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            assert self.total_num_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        low_rank_config = low_rank_config or {}
        
        # Use separate q/k/v projections (lite-whisper style)
        if "q_proj" in low_rank_config:
            self.q_proj = LinearLowRank(
                in_features=embed_dim,
                out_features=embed_dim,
                low_rank_features=low_rank_config["q_proj"]
            )
        else:
            from vllm.model_executor.layers.linear import ColumnParallelLinear
            self.q_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=embed_dim,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            
        if "k_proj" in low_rank_config:
            self.k_proj = LinearLowRank(
                in_features=embed_dim,
                out_features=embed_dim,
                low_rank_features=low_rank_config["k_proj"]
            )
        else:
            from vllm.model_executor.layers.linear import ColumnParallelLinear
            self.k_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=embed_dim,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.k_proj",
            )
            
        if "v_proj" in low_rank_config:
            self.v_proj = LinearLowRank(
                in_features=embed_dim,
                out_features=embed_dim,
                low_rank_features=low_rank_config["v_proj"]
            )
        else:
            from vllm.model_executor.layers.linear import ColumnParallelLinear
            self.v_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=embed_dim,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.v_proj",
            )

        if "out_proj" in low_rank_config:
            self.out_proj = LinearLowRank(
                in_features=embed_dim,
                out_features=embed_dim,
                low_rank_features=low_rank_config["out_proj"]
            )
        else:
            from vllm.model_executor.layers.linear import RowParallelLinear
            self.out_proj = RowParallelLinear(
                input_size=embed_dim,
                output_size=embed_dim,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj",
            )

        if standalone_encoder:
            from vllm.attention.layer import MultiHeadAttention
            self.attn = MultiHeadAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
            )
        else:
            from vllm.attention import Attention
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=self.attn_type,
            )

    def forward(self, hidden_states: torch.Tensor):
        # Separate q, k, v projections
        if hasattr(self.q_proj, 'forward'):
            q = self.q_proj(hidden_states)
        else:
            q, _ = self.q_proj(hidden_states)
            
        if hasattr(self.k_proj, 'forward'):
            k = self.k_proj(hidden_states)
        else:
            k, _ = self.k_proj(hidden_states)
            
        if hasattr(self.v_proj, 'forward'):
            v = self.v_proj(hidden_states)
        else:
            v, _ = self.v_proj(hidden_states)

        # Split for multi-head attention
        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        v = v.view(*v.shape[:-1], self.num_kv_heads, self.head_dim)

        attn_output = self.attn(q.flatten(-2), k.flatten(-2), v.flatten(-2))

        if hasattr(self.out_proj, 'forward'):
            output = self.out_proj(attn_output)
        else:
            output, _ = self.out_proj(attn_output)

        return output


class LiteWhisperEncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_standalone_encoder: bool = False,
        low_rank_config: dict[str, int] | None = None,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        logger.info("Building LiteWhisperEncoderLayer with prefix=%s", prefix)
        self.embed_dim = config.d_model
        low_rank_config = low_rank_config or {}

        # Use LiteWhisperAttention with low-rank projections
        self.self_attn = LiteWhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            attn_type=AttentionType.ENCODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            standalone_encoder=is_standalone_encoder,
            low_rank_config=low_rank_config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Use low-rank MLP
        self.mlp = LiteWhisperMLP(
            embed_dim=config.d_model,
            ffn_dim=config.encoder_ffn_dim,
            act_fn=config.activation_function,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            low_rank_config=low_rank_config,
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Cast overflow tensors
        from vllm.model_executor.models.utils import cast_overflow_tensors
        hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states


class LiteWhisperMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        low_rank_config: dict[str, int] | None = None,
    ):
        super().__init__()
        from vllm.model_executor.layers.activation import get_act_fn
        
        self.activation_fn = get_act_fn(act_fn)
        low_rank_config = low_rank_config or {}
        
        if "fc1" in low_rank_config:
            self.fc1 = LinearLowRank(
                in_features=embed_dim,
                out_features=ffn_dim,
                low_rank_features=low_rank_config["fc1"]
            )
        else:
            from vllm.model_executor.layers.linear import ColumnParallelLinear
            self.fc1 = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=ffn_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1",
            )
            
        if "fc2" in low_rank_config:
            self.fc2 = LinearLowRank(
                in_features=ffn_dim,
                out_features=embed_dim,
                low_rank_features=low_rank_config["fc2"]
            )
        else:
            self.fc2 = RowParallelLinear(
                input_size=ffn_dim,
                output_size=embed_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2",
            )

    def forward(self, hidden_states: torch.Tensor):
        if hasattr(self.fc1, 'forward'):
            hidden_states = self.fc1(hidden_states)
        else:
            hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        if hasattr(self.fc2, 'forward'):
            hidden_states = self.fc2(hidden_states)
        else:
            hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class LiteWhisperEncoder(WhisperEncoder):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", is_standalone_encoder: bool = False):
        # Don't call super().__init__() to avoid duplicate layer creation
        # Instead, initialize manually with LiteWhisperEncoderLayer
        nn.Module.__init__(self)  # Call nn.Module.__init__ directly
        
        config = vllm_config.model_config.hf_config
        embed_dim = config.d_model
        self.is_standalone_encoder = is_standalone_encoder
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = (math.sqrt(embed_dim)
                            if config.scale_embedding else 1.0)

        # Copy conv layers and layer norm from WhisperEncoder
        self.conv1 = nn.Conv1d(self.num_mel_bins,
                               embed_dim,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv1d(embed_dim,
                               embed_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Initialize positional embeddings like WhisperEncoder
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype
        from transformers.models.whisper.modeling_whisper import sinusoids
        
        maybe_fp32_init_ctx = set_default_torch_dtype(torch.float32) if False else nullcontext()
        with (torch.no_grad(), maybe_fp32_init_ctx):
            self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
            self.embed_positions.weight.copy_(sinusoids(*self.embed_positions.weight.shape))

        # Get low_rank_config from config if it exists, else create default empty config for all layers
        low_rank_config_list = getattr(config, "low_rank_config", [])
        
        # If no config provided, assume all layers use low-rank with default config
        if not low_rank_config_list:
            # Default low-rank config for lite-whisper: all attention and MLP components
            default_config = {
                "q_proj": 32, "k_proj": 32, "v_proj": 32, "out_proj": 32,
                "fc1": 368, "fc2": 368
            }
            low_rank_config_list = [default_config] * config.encoder_layers
            
        low_rank_config_dict = {i: cfg for i, cfg in enumerate(low_rank_config_list)}

        # Create layers with make_layers using LiteWhisperEncoderLayer
        def create_layer(prefix: str):
            # Extract layer index from prefix (e.g., "layers.0" -> 0)
            try:
                index = int(prefix.split('.')[-1])
            except (ValueError, IndexError):
                index = 0
            
            low_rank_cfg = low_rank_config_dict.get(index, {})
            return LiteWhisperEncoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                is_standalone_encoder=is_standalone_encoder,
                low_rank_config=low_rank_cfg
            )
        
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.encoder_layers,
            create_layer,
            prefix=f"{prefix}.layers",
        )

# try to use specialized loader instead of the default one
class LiteWhisperModel(WhisperModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config

        # Encoder: use LiteWhisperEncoder
        self.encoder = LiteWhisperEncoder(
            vllm_config=vllm_config,
            prefix=f"{prefix}.encoder" if prefix else "encoder",
            is_standalone_encoder=False
        )

        # Decoder: keep original
        self.decoder = WhisperDecoder(
            vllm_config=vllm_config,
            prefix=f"{prefix}.decoder" if prefix else "decoder"
        )
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Custom weight loader for LiteWhisper using stacked_params_mapping approach."""
        
        # Define stacked_params_mapping for transforming HF weights to vLLM structure
        stacked_params_mapping = [
            # Decoder self-attention: separate q/k/v -> qkv_proj
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            # Decoder encoder-attention: separate k/v -> kv_proj  
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
        
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        # Debug: Print a few parameter names to understand the structure
        print("Sample model parameters:")
        for i, (param_name, _) in enumerate(params_dict.items()):
            if i < 20 or "proj_out" in param_name:  # Print first 20 and any proj_out
                print(f"  {param_name}")
            elif i == 20:
                print(f"  ... and {len(params_dict) - 20} more")
        
        # Check for proj_out specifically
        has_proj_out = "proj_out.weight" in params_dict or "model.proj_out.weight" in params_dict
        print(f"Model has proj_out.weight: {has_proj_out}")
        if not has_proj_out:
            # Check if it has a different name
            proj_variants = [name for name in params_dict.keys() if "proj_out" in name]
            print(f"Proj_out variants: {proj_variants}")
        
        for name, loaded_weight in weights:
            # Skip proj_out weights (they're tied to embed_tokens)
            if name.startswith("proj_out.") or name == "proj_out.weight":
                loaded_params.add(name)
                loaded_params.add(f"model.{name}")  # Also mark model. version as loaded
                print(f"  Skipped proj_out weight: {name}")
                continue
            
            # Handle encoder weights with low-rank decomposition
            # Check if this is a low-rank weight (weight1 or weight2)
            if ".weight1" in name or ".weight2" in name:
                # Try both with and without model. prefix
                param_name = name
                if param_name not in params_dict and f"model.{param_name}" in params_dict:
                    param_name = f"model.{param_name}"
                
                if param_name in params_dict:
                    param = params_dict[param_name]
                    if ".weight1" in name:
                        shard_id = "weight1"
                    else:  # .weight2
                        shard_id = "weight2"
                    
                    # For LinearLowRank modules, call the custom weight_loader
                    weight_loader = getattr(param, 'weight_loader', None)
                    if weight_loader and callable(weight_loader):
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(name)
                        loaded_params.add(param_name)  # Mark the actual param name as loaded
                    else:
                        # Fallback to default loader
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                        loaded_params.add(param_name)  # Mark the actual param name as loaded
                else:
                    print(f"  Warning: Low-rank param not found: {name}")
                continue
            
            # Check if this weight should be transformed via stacked_params_mapping
            # Only apply to decoder, not encoder (encoder uses separate low-rank weights)
            processed = False
            if name.startswith("decoder."):
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name in name:
                        # Transform name from HF format to vLLM format
                        vllm_name = name.replace(weight_name, param_name)
                        
                        # Try both with and without model. prefix
                        if vllm_name not in params_dict and f"model.{vllm_name}" in params_dict:
                            vllm_name = f"model.{vllm_name}"
                        
                        # Skip loading extra bias for GPTQ models
                        if vllm_name.endswith(".bias") and vllm_name not in params_dict:
                            print(f"  Skipping extra bias: {vllm_name}")
                            processed = True
                            break
                            
                        if vllm_name in params_dict:
                            param = params_dict[vllm_name]
                            weight_loader = getattr(param, 'weight_loader', None)
                            if weight_loader and callable(weight_loader):
                                weight_loader(param, loaded_weight, shard_id)
                                loaded_params.add(name)
                                loaded_params.add(vllm_name)  # Mark the actual param name as loaded
                                print(f"  Loaded QKV: {name} -> {vllm_name} ({shard_id})")
                            else:
                                print(f"  Warning: No weight_loader for stacked param: {vllm_name}")
                            processed = True
                            break
                        else:
                            print(f"  Warning: Stacked param not found: {vllm_name}")
                            processed = True
                            break
            else:
                print(f"  Debug: Non-decoder weight: {name}")
            
            if processed:
                continue
                
            # Handle other weights (direct mapping)
            param_name = name
            # Try both with and without model. prefix
            if param_name not in params_dict and f"model.{param_name}" in params_dict:
                param_name = f"model.{param_name}"
            
            # Skip loading extra bias for GPTQ models
            if param_name.endswith(".bias") and param_name not in params_dict:
                print(f"  Skipping extra bias: {param_name}")
                continue
            
            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                loaded_params.add(param_name)  # Mark the actual param name as loaded
            else:
                print(f"  Warning: Unmatched weight: {name}")
                    
        return loaded_params


@MULTIMODAL_REGISTRY.register_processor(
    WhisperMultiModalProcessor,
    info=WhisperProcessingInfo,
    dummy_inputs=WhisperDummyInputsBuilder
)
class LiteWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    config_class = LiteWhisperConfig

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Replace model with LiteWhisperModel
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        self.model = LiteWhisperModel(vllm_config=vllm_config, prefix=prefix) 
        self.unpadded_vocab_size = config.vocab_size
        self.proj_out = ParallelLMHead(config.vocab_size,
                                       config.d_model,
                                       quant_config=quant_config)
        self.proj_out = self.proj_out.tie_weights(
            self.model.decoder.embed_tokens)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)