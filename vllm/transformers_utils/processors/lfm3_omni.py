# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .lfm3_audio import Lfm3AudioFeatureExtractor
from .lfm3_vl import Lfm3VlProcessor


class Lfm3OmniProcessor(Lfm3VlProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        audio_token: str = "<|audio|>",
        **kwargs,
    ):
        # Not a constructor attribute: the Qwen3-VL base passes its three
        # sub-processors positionally.
        self.feature_extractor: Lfm3AudioFeatureExtractor = kwargs.pop(
            "feature_extractor"
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template)
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        vl_processor = Lfm3VlProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        kwargs.pop("tokenizer", None)
        return cls(
            image_processor=vl_processor.image_processor,
            tokenizer=vl_processor.tokenizer,
            video_processor=vl_processor.video_processor,
            feature_extractor=Lfm3AudioFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            ),
        )

    def replace_audio_token(self, audio_inputs, audio_idx, **kwargs) -> str:
        num_frames = int(audio_inputs["feature_attention_mask"][audio_idx].sum())
        num_tokens = self.feature_extractor.num_encoder_tokens(num_frames)
        return self.audio_token * num_tokens
