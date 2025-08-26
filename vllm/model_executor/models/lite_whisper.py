# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
LiteWhisper model implementation for vLLM.

LiteWhisper is a compressed version of Whisper that uses low-rank decomposition
for some weights. This implementation reconstructs the full weights from the
low-rank components and uses the standard Whisper processor but loads the
processor from the tokenizer path.
"""

from typing import Iterable, Tuple, TYPE_CHECKING
import torch
import logging

logger = logging.getLogger(__name__)

from vllm.config import VllmConfig
from vllm.model_executor.models.whisper import (
    WhisperForConditionalGeneration,
    WhisperProcessingInfo,
    WhisperMultiModalProcessor,
    WhisperDummyInputsBuilder
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.configs.lite_whisper import LiteWhisperConfig
from transformers import WhisperProcessor

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SpeechToTextConfig


class LiteWhisperProcessingInfo(WhisperProcessingInfo):
    # Class-level flag to track if fallback message has been logged
    _fallback_logged = False
    
    def get_hf_processor(self, **kwargs: object) -> WhisperProcessor:
        """Load processor from tokenizer path with fallback to openai/whisper-large-v3."""
        from vllm.transformers_utils.processor import cached_get_processor
        
        processor_class = WhisperProcessor
        tokenizer_class = ("WhisperTokenizer", "WhisperTokenizerFast")
        if processor_class.tokenizer_class != tokenizer_class:
            processor_class.tokenizer_class = tokenizer_class
        
        # Build kwargs that include model config
        processor_kwargs = {
            "revision": self.ctx.model_config.revision,
            "trust_remote_code": self.ctx.model_config.trust_remote_code,
            **kwargs
        }
        
        tokenizer_path = self.ctx.model_config.tokenizer
        
        # First try tokenizer path if it's different from model path
        if tokenizer_path and tokenizer_path != self.ctx.model_config.model:
            try:
                return cached_get_processor(
                    tokenizer_path,
                    processor_cls=processor_class,
                    **processor_kwargs
                )
            except Exception:
                pass  # Silently fail and try next option
        
        # Try original model path
        try:
            return cached_get_processor(
                self.ctx.model_config.model,
                processor_cls=processor_class,
                **processor_kwargs
            )
        except Exception:
            pass  # Silently fail and try fallback
            
        # Final fallback to openai/whisper-large-v3
        if not LiteWhisperProcessingInfo._fallback_logged:
            logger.info("Using openai/whisper-large-v3 as processor fallback for LiteWhisper")
            LiteWhisperProcessingInfo._fallback_logged = True
        
        return cached_get_processor(
            "openai/whisper-large-v3",
            processor_cls=processor_class,
            **processor_kwargs
        )


# Register with custom processing info that uses tokenizer path
@MULTIMODAL_REGISTRY.register_processor(WhisperMultiModalProcessor,
                                        info=LiteWhisperProcessingInfo,
                                        dummy_inputs=WhisperDummyInputsBuilder)
class LiteWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    """
    LiteWhisper model that supports low-rank weight reconstruction.
    
    This model is identical to Whisper except it can load weights that have been
    decomposed into weight1 and weight2 components for compression. The weights
    are reconstructed during loading.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights, prioritizing regular weights over low-rank reconstruction."""
        # Convert to dict for easier processing
        weight_dict = dict(weights)

        # First, collect all existing regular weights
        final_weights = []
        processed_keys = set()
        reconstructed_count = 0

        # Add all non-low-rank weights first
        for name, weight in weight_dict.items():
            if ".weight1" not in name and ".weight2" not in name:
                final_weights.append((name, weight))
                processed_keys.add(name)

        # Now process low-rank weights - find weight1/weight2 pairs
        weight1_keys = {k for k in weight_dict.keys() if ".weight1" in k}

        for weight1_key in weight1_keys:
            # Get corresponding weight2 key
            base_name = weight1_key.replace(".weight1", "")
            weight2_key = weight1_key.replace(".weight1", ".weight2")
            regular_key = base_name + ".weight"

            # Skip if we already have the regular weight
            if regular_key in processed_keys:
                continue

            if weight2_key in weight_dict:
                weight1 = weight_dict[weight1_key]  
                weight2 = weight_dict[weight2_key]

                # Reconstruct: weight = weight1 @ weight2 (standard low-rank reconstruction)
                reconstructed = torch.mm(weight1, weight2)

                # Always transpose to match linear layer weight shape: [out_features, in_features]
                reconstructed = reconstructed.T

                final_weights.append((regular_key, reconstructed))
                processed_keys.add(regular_key)
                reconstructed_count += 1

        print(f"Lite-Whisper: Reconstructed {reconstructed_count} weights" + 
            " from low-rank decomposition")
        logger.info(f"LiteWhisper: Successfully reconstructed {reconstructed_count} weights from low-rank decomposition")

        # Load weights using parent method
        return super().load_weights(final_weights)

    @classmethod
    def get_speech_to_text_config(cls, model_config: "ModelConfig",
                                  task_type: str) -> "SpeechToTextConfig":
        """Override to use fallback processor loading."""
        from vllm.transformers_utils.processor import cached_get_processor
        from vllm.config import SpeechToTextConfig
        from transformers import WhisperProcessor
        
        processor_class = WhisperProcessor
        tokenizer_class = ("WhisperTokenizer", "WhisperTokenizerFast")
        if processor_class.tokenizer_class != tokenizer_class:
            processor_class.tokenizer_class = tokenizer_class
        
        # Build kwargs that include model config
        processor_kwargs = {
            "revision": model_config.revision,
            "trust_remote_code": model_config.trust_remote_code,
        }
        
        # Try the same fallback sequence as in get_hf_processor
        tokenizer_path = model_config.tokenizer
        
        processor = None
        
        # First try tokenizer path if it's different from model path
        if tokenizer_path and tokenizer_path != model_config.model:
            try:
                processor = cached_get_processor(
                    tokenizer_path,
                    processor_cls=processor_class,
                    **processor_kwargs
                )
            except Exception:
                pass  # Silently fail and try next option
        
        # Try original model path
        if processor is None:
            try:
                processor = cached_get_processor(
                    model_config.model,
                    processor_cls=processor_class,
                    **processor_kwargs
                )
            except Exception:
                pass  # Silently fail and try fallback
                
        # Final fallback to openai/whisper-large-v3
        if processor is None:
            processor = cached_get_processor(
                "openai/whisper-large-v3",
                processor_cls=processor_class,
                **processor_kwargs
            )
        
        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
            # audio-related config for speech2text
        )

    @classmethod
    def get_num_audio_tokens(cls, audio_duration_s: float,
                             stt_config: "SpeechToTextConfig",
                             model_config: "ModelConfig"):
        """Override to use fallback processor loading."""
        import math
        from typing import Optional
        from vllm.transformers_utils.processor import cached_get_processor
        from transformers import WhisperProcessor
        
        processor_class = WhisperProcessor
        tokenizer_class = ("WhisperTokenizer", "WhisperTokenizerFast")
        if processor_class.tokenizer_class != tokenizer_class:
            processor_class.tokenizer_class = tokenizer_class
        
        # Build kwargs that include model config
        processor_kwargs = {
            "revision": model_config.revision,
            "trust_remote_code": model_config.trust_remote_code,
        }
        
        # Try the same fallback sequence as in get_hf_processor
        tokenizer_path = model_config.tokenizer
        
        processor = None
        
        # First try tokenizer path if it's different from model path
        if tokenizer_path and tokenizer_path != model_config.model:
            try:
                processor = cached_get_processor(
                    tokenizer_path,
                    processor_cls=processor_class,
                    **processor_kwargs
                )
            except Exception:
                pass  # Silently fail and try next option
        
        # Try original model path
        if processor is None:
            try:
                processor = cached_get_processor(
                    model_config.model,
                    processor_cls=processor_class,
                    **processor_kwargs
                )
            except Exception:
                pass  # Silently fail and try fallback
                
        # Final fallback to openai/whisper-large-v3
        if processor is None:
            processor = cached_get_processor(
                "openai/whisper-large-v3",
                processor_cls=processor_class,
                **processor_kwargs
            )
        
        hop_length = processor.feature_extractor.hop_length
        assert hop_length is not None
        # NOTE(NickLucche) user can't pass encoder
        # prompts directly at least not to Whisper.
        # One indicator of the encoder amount of processing
        # is the log-mel spectogram length.
        return math.ceil(audio_duration_s * stt_config.sample_rate /
                         hop_length)


