import torch
from torch import nn
import logging

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
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.utils import make_layers
from vllm.transformers_utils.configs.lite_whisper import LiteWhisperConfig

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

class LiteWhisperEncoderLayer(WhisperEncoderLayer):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_standalone_encoder: bool = False,
        low_rank_config: dict[str, int] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            is_standalone_encoder=is_standalone_encoder,
        )

        logger.info("Building LiteWhisperEncoderLayer with prefix=%s", prefix)
        config = vllm_config.model_config.hf_config
        self.embed_dim = config.d_model
        low_rank_config = low_rank_config or {}

        # Replace k_proj/v_proj/q_proj/out_proj with low-rank linear if configured
        if "k_proj" in low_rank_config:
            self.self_attn.k_proj = LinearLowRank(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                low_rank_features=low_rank_config["k_proj"]
            )
        if "v_proj" in low_rank_config:
            self.self_attn.v_proj = LinearLowRank(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                low_rank_features=low_rank_config["v_proj"]
            )
        if "q_proj" in low_rank_config:
            self.self_attn.q_proj = LinearLowRank(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                low_rank_features=low_rank_config["q_proj"]
            )
        if "out_proj" in low_rank_config:
            self.self_attn.out_proj = LinearLowRank(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                low_rank_features=low_rank_config["out_proj"]
            )

        if "fc1" in low_rank_config:
            self.fc1 = LinearLowRank(
                in_features=self.embed_dim,
                out_features=config.encoder_ffn_dim,
                low_rank_features=low_rank_config["fc1"]
            )
        if "fc2" in low_rank_config:
            self.fc2 = LinearLowRank(
                in_features=config.encoder_ffn_dim,
                out_features=self.embed_dim,
                low_rank_features=low_rank_config["fc2"]
            )

class LiteWhisperEncoder(WhisperEncoder):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", is_standalone_encoder: bool = False):
        super().__init__(vllm_config=vllm_config, prefix=prefix, is_standalone_encoder=is_standalone_encoder)

        config: LiteWhisperConfig = vllm_config.model_config.hf_config
        low_rank_config_list = getattr(config, "low_rank_config", [])
        low_rank_config_dict = {i: cfg for i, cfg in enumerate(low_rank_config_list)}

        # Rebuild encoder layers using LiteWhisperEncoderLayer
        def create_layer(*args, **kwargs):
            prefix_layer = kwargs.get("prefix", "layers.0")
            index = kwargs.get("index", 0)

            low_rank_cfg = low_rank_config_dict.get(index, {})
            return LiteWhisperEncoderLayer(
                vllm_config=vllm_config,
                prefix=prefix_layer,
                is_standalone_encoder=is_standalone_encoder,
                low_rank_config=low_rank_cfg
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.encoder_layers,
            create_layer,
            prefix=f"{prefix}.layers" if prefix else "layers"
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
        self.model = LiteWhisperModel(vllm_config=vllm_config, prefix=prefix)
