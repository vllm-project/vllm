# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin
from vllm.logger import init_logger

logger = init_logger(__name__)


class FireRedASRProcessor(ProcessorMixin):
    r"""
    Constructs a FunASR processor which wraps a FunASR feature extractor and
    a FunASR tokenizer into a single processor.

    [`FireRedASRProcessor`] offers all the functionalities of
    [`FireRedASRFeatureExtractor`] and [`Qwen2Tokenizer`]. See the
    [`~FireRedASRProcessor.__call__`] and [`~FireRedASRProcessor.decode`] for more
    information.

    Args:
        feature_extractor (`FireRedASRFeatureExtractor`): An instance of
            [`FireRedASRFeatureExtractor`].
            The feature extractor is a required input.
        tokenizer (`Qwen2Tokenizer`):
            An instance of [`Qwen2Tokenizer`]. The tokenizer is a required
            input.
    """

    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("PreTrainedTokenizerFast")

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_token="<|AUDIO|>",
    ):
        super().__init__(feature_extractor, tokenizer)
        self.audio_token = (
            tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        )
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(
            task=task, language=language, no_timestamps=no_timestamps
        )

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to FunASRFeatureExtractor's
        [`~FunASRFeatureExtractor.__call__`] and the `text` argument to
        [`~Qwen2Tokenizer.__call__`]. Please refer to the docstring of the
        above two methods for more information.
        """

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is not None:

            inputs = self.feature_extractor(
                audio, *args, sampling_rate=sampling_rate, **kwargs
            )

        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs

        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)


AutoProcessor.register("FireRedASRProcessor", FireRedASRProcessor)
