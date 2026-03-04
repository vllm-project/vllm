# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.audio_parser import AudioParser, AudioParserManager
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger

logger = init_logger(__name__)


@AudioParserManager.register_module("step_audio_2_tts_ta4")
class StepAudio2TTSTA4Parser(AudioParser):
    audio_start_token_id: int
    audio_end_token_id: int
    first_audio_token_id: int
    tts_pad_token_id: int
    audio_pad_token_id: int

    audio_start_token: str = "<tts_start>"
    audio_end_token: str = "<tts_end>"

    first_audio_token: str = "<audio_0>"
    tts_pad_token: str = "<tts_pad>"
    audio_pad_token: str = "<audio_6561>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the AudioParser "
                "constructor during construction."
            )

        audio_start_token_id = self.vocab.get(self.audio_start_token)
        audio_end_token_id = self.vocab.get(self.audio_end_token)
        if audio_start_token_id is None or audio_end_token_id is None:
            raise RuntimeError(
                "Step audio parser could not locate think start/end "
                "tokens in the tokenizer!"
            )
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id

        first_audio_token_id = self.vocab.get(self.first_audio_token)
        tts_pad_token_id = self.vocab.get(self.tts_pad_token)
        audio_pad_token_id = self.vocab.get(self.audio_pad_token)
        if (
            first_audio_token_id is None
            or tts_pad_token_id is None
            or audio_pad_token_id is None
        ):
            raise RuntimeError(
                "Step audio parser could not locate audio/pad tokens in the tokenizer!"
            )
        self.first_audio_token_id = first_audio_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.audio_pad_token_id = audio_pad_token_id

    def is_step_audio_token(self, token_id: int) -> bool:
        return token_id >= self.first_audio_token_id

    def extract_tts_content(
        self,
        input_token_ids: Sequence[int],
        has_tts_start: bool,
        has_tts_end: bool,
        is_text_audio_section: bool = True,
        include_pad_token: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        other_token_ids: list[int] = []
        tts_text_token_ids: list[int] = []
        tts_audio_token_ids: list[int] = []
        in_tts_content_section = False

        if has_tts_start and has_tts_end:
            # <tts_start> and <tts_end> tokens are both in delta message:
            # input: [<tts_start>def<audio_0><audio_1><tts_end>abc]
            # return:
            #    tts_text_token_ids: [def]
            #    tts_audio_token_ids: [<audio_0><audio_1>]
            #    other_token_ids: [abc]

            for token_id in input_token_ids:
                if token_id == self.audio_start_token_id:
                    in_tts_content_section = True
                    continue

                if token_id == self.audio_end_token_id:
                    in_tts_content_section = False
                    continue

                if not in_tts_content_section:
                    other_token_ids.append(token_id)
                else:
                    if not include_pad_token and (
                        token_id == self.audio_pad_token_id
                        or token_id == self.tts_pad_token_id
                    ):
                        continue
                    if self.is_step_audio_token(token_id):
                        tts_audio_token_ids.append(token_id)
                    else:
                        tts_text_token_ids.append(token_id)

            return tts_text_token_ids, tts_audio_token_ids, other_token_ids

        elif has_tts_start and not has_tts_end:
            # <tts_start> token in delta message:
            # input: [abc<tts_start>def<audio_0><audio_1>]
            # return:
            #    tts_text_token_ids: [def]
            #    tts_audio_token_ids: [<audio_0><audio_1>]
            #    other_token_ids: [abc]

            for token_id in input_token_ids:
                if token_id == self.audio_start_token_id:
                    in_tts_content_section = True
                    continue

                if not in_tts_content_section:
                    other_token_ids.append(token_id)
                else:
                    if not include_pad_token and (
                        token_id == self.audio_pad_token_id
                        or token_id == self.tts_pad_token_id
                    ):
                        continue
                    if self.is_step_audio_token(token_id):
                        tts_audio_token_ids.append(token_id)
                    else:
                        tts_text_token_ids.append(token_id)
            return tts_text_token_ids, tts_audio_token_ids, other_token_ids

        elif not has_tts_start and has_tts_end:
            # <tts_end> token in delta message:
            # input: [def<audio_0><audio_1><tts_end>abc]
            # return:
            #    tts_text_token_ids: [def]
            #    tts_audio_token_ids: [<audio_0><audio_1>]
            #    other_token_ids: [abc]
            in_tts_content_section = True
            for token_id in input_token_ids:
                if token_id == self.audio_end_token_id:
                    in_tts_content_section = False
                    continue

                if not in_tts_content_section:
                    other_token_ids.append(token_id)
                else:
                    if not include_pad_token and (
                        token_id == self.audio_pad_token_id
                        or token_id == self.tts_pad_token_id
                    ):
                        continue
                    if self.is_step_audio_token(token_id):
                        tts_audio_token_ids.append(token_id)
                    else:
                        tts_text_token_ids.append(token_id)

            return tts_text_token_ids, tts_audio_token_ids, other_token_ids
        else:
            # <tts_start> and <tts_end> tokens are both not in delta message:
            if is_text_audio_section:
                # input: [def<audio_0><audio_1>]
                # is_text_audio_section is true: assume all message in text audio section # noqa: E501
                # return:
                #    tts_text_token_ids: [def]
                #    tts_audio_token_ids: [<audio_0><audio_1>]
                for token_id in input_token_ids:
                    if not include_pad_token and (
                        token_id == self.audio_pad_token_id
                        or token_id == self.tts_pad_token_id
                    ):
                        continue
                    if self.is_step_audio_token(token_id):
                        tts_audio_token_ids.append(token_id)
                    else:
                        tts_text_token_ids.append(token_id)
                return tts_text_token_ids, tts_audio_token_ids, other_token_ids
            else:
                # input: [abcdef]
                # is_text_audio_section is false: assume all message not in text audio section # noqa: E501
                # return:
                #    tts_text_token_ids: []
                #    tts_audio_token_ids: []
                #    other_token_ids: []
                return [], [], list(input_token_ids)

    def is_audio_output(self, prompt_token_ids: Sequence[int]) -> bool:
        """Check if the last prompt token is '<tts_start>', indicating
        TA4 audio output format."""
        return prompt_token_ids[-1] == self.audio_start_token_id

    def extract_audio_content_nonstreaming(
        self,
        output_token_ids: Sequence[int],
        request: ChatCompletionRequest,
        is_audio_output: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Extract audio content from the model output.

        For text <tts_start>abc<audio_1475><audio_1978><audio_4218><tts_end>:
        - 'abc' goes to text content
        - '<audio_1475><audio_1978><audio_4218>' goes to audio content

        Returns:
            (text_token_ids, audio_token_ids, other_token_ids)
        """

        if not is_audio_output:
            return [], [], list(output_token_ids)
        else:
            return self.extract_tts_content(
                output_token_ids, has_tts_start=False, has_tts_end=True
            )

    def extract_audio_content_streaming(
        self,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        is_audio_output: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        if not is_audio_output:
            return [], [], list(delta_token_ids)

        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.audio_start_token_id, self.audio_end_token_id]
        ):
            return [], [], []

        if self.audio_end_token_id in previous_token_ids:
            # <tts_end> in previous, extract text/audio content
            return [], [], list(delta_token_ids)
        elif self.audio_end_token_id in delta_token_ids:
            # <tts_end> in delta, extract text/audio content
            return self.extract_tts_content(
                delta_token_ids, has_tts_start=False, has_tts_end=True
            )
        else:
            # <tts_end> not in delta, extract text/audio content
            return self.extract_tts_content(
                delta_token_ids,
                has_tts_start=False,
                has_tts_end=False,
                is_text_audio_section=True,
            )
