# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prompt builder helpers for the Kimi-Audio text-output subset."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


class _KimiAudioTokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


@dataclass
class KimiAudioTokenContent:
    audio_token_ids: list[int] = field(default_factory=list)
    text_token_ids: list[int] = field(default_factory=list)
    is_continuous_mask: list[bool] = field(default_factory=list)

    def audio_append(self, token_id: int, *, is_continuous: bool = False) -> None:
        self.audio_token_ids.append(token_id)
        self.is_continuous_mask.append(is_continuous)

    def text_append(self, token_id: int) -> None:
        self.text_token_ids.append(token_id)

    def audio_extend(
        self,
        token_ids: Sequence[int],
        *,
        is_continuous: bool = False,
    ) -> None:
        self.audio_token_ids.extend(token_ids)
        self.is_continuous_mask.extend([is_continuous] * len(token_ids))

    def text_extend(self, token_ids: Sequence[int]) -> None:
        self.text_token_ids.extend(token_ids)

    def merge(self, other: KimiAudioTokenContent) -> None:
        self.audio_token_ids.extend(other.audio_token_ids)
        self.text_token_ids.extend(other.text_token_ids)
        self.is_continuous_mask.extend(other.is_continuous_mask)

    def validate(self) -> None:
        if not (
            len(self.audio_token_ids)
            == len(self.text_token_ids)
            == len(self.is_continuous_mask)
        ):
            raise ValueError(
                "Kimi-Audio packed content lengths diverged: "
                f"audio={len(self.audio_token_ids)} "
                f"text={len(self.text_token_ids)} "
                f"mask={len(self.is_continuous_mask)}"
            )


@dataclass(frozen=True)
class KimiAudioSpecialTokenIds:
    msg_end: int
    media_begin: int
    media_end: int
    text_blank: int
    text_eos: int
    user_msg_start: int
    assistant_msg_start: int
    speech_ct: int
    speech_ctd: int

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: _KimiAudioTokenizerLike,
    ) -> KimiAudioSpecialTokenIds:
        return cls(
            msg_end=tokenizer.convert_tokens_to_ids("<|im_msg_end|>"),
            media_begin=tokenizer.convert_tokens_to_ids("<|im_media_begin|>"),
            media_end=tokenizer.convert_tokens_to_ids("<|im_media_end|>"),
            text_blank=tokenizer.convert_tokens_to_ids("<|im_kimia_text_blank|>"),
            text_eos=tokenizer.convert_tokens_to_ids("<|im_kimia_text_eos|>"),
            user_msg_start=tokenizer.convert_tokens_to_ids(
                "<|im_kimia_user_msg_start|>"
            ),
            assistant_msg_start=tokenizer.convert_tokens_to_ids(
                "<|im_kimia_assistant_msg_start|>"
            ),
            speech_ct=tokenizer.convert_tokens_to_ids("<|im_kimia_speech_ct_id|>"),
            speech_ctd=tokenizer.convert_tokens_to_ids("<|im_kimia_speech_ctd_id|>"),
        )


class KimiAudioPromptBuilder:
    """Build official-style text-output prompts for Kimi-Audio.

    This builder intentionally targets the subset currently supported by vLLM:
    user text, user audio, assistant text, and assistant generation prompts.
    """

    USER_START = "<|im_kimia_user_msg_start|>"
    ASSISTANT_START = "<|im_kimia_assistant_msg_start|>"
    MSG_END = "<|im_msg_end|>"
    MEDIA_BEGIN = "<|im_media_begin|>"
    MEDIA_END = "<|im_media_end|>"
    TEXT_BLANK = "<|im_kimia_text_blank|>"
    TEXT_EOS = "<|im_kimia_text_eos|>"
    SPEECH_CT = "<|im_kimia_speech_ct_id|>"
    SPEECH_CTD = "<|im_kimia_speech_ctd_id|>"

    AUDIO_PLACEHOLDER = f"{MEDIA_BEGIN}{TEXT_BLANK}{MEDIA_END}{SPEECH_CT}"

    @classmethod
    def build_audio_placeholder(cls, *, audio_count: int = 1) -> str:
        if audio_count < 0:
            raise ValueError("audio_count must be non-negative")
        return cls.AUDIO_PLACEHOLDER * audio_count

    @classmethod
    def build_user_audio_content(
        cls,
        request_prompt: str = "",
        *,
        audio_count: int = 1,
    ) -> str:
        audio_placeholder = cls.build_audio_placeholder(audio_count=audio_count)
        request_prompt = request_prompt.strip()
        if request_prompt:
            return f"{request_prompt}\n{audio_placeholder}"
        return audio_placeholder

    @classmethod
    def build_message(
        cls,
        *,
        role: str,
        content: str = "",
        message_type: str = "text",
        add_text_eos: bool | None = None,
        add_msg_end: bool = True,
        output_type: str = "text",
        audio_count: int = 1,
    ) -> str:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported role: {role}")
        if message_type not in {"text", "audio", None}:
            raise ValueError(f"Unsupported message_type: {message_type}")
        if output_type not in {"text", "both"}:
            raise ValueError(f"Unsupported output_type: {output_type}")

        role_prefix = cls.USER_START if role == "user" else cls.ASSISTANT_START

        if add_text_eos is None:
            add_text_eos = role == "assistant" and message_type == "text"

        body = content.strip() if message_type == "text" and content else content
        if message_type == "audio":
            control_token = cls.SPEECH_CT if output_type == "text" else cls.SPEECH_CTD
            body = (
                f"{cls.MEDIA_BEGIN}{cls.TEXT_BLANK}{cls.MEDIA_END}{control_token}"
                * audio_count
            )
            if content:
                text_content = content.strip()
                body = f"{text_content}\n{body}" if text_content else body
        elif message_type is None:
            body = ""

        suffix = ""
        if add_text_eos:
            suffix += cls.TEXT_EOS
        if add_msg_end:
            suffix += cls.MSG_END

        return f"{role_prefix}{body}{suffix}"

    @classmethod
    def build_prompt_from_messages(
        cls,
        messages: Sequence[dict[str, object]],
        *,
        add_generation_prompt: bool = True,
        output_type: str = "text",
    ) -> str:
        prompt_parts: list[str] = []

        for message in messages:
            role = str(message["role"])
            message_type = str(message.get("message_type") or "text")
            content = str(message.get("content") or "")
            audio_count = int(message.get("audio_count", 1) or 1)
            prompt_parts.append(
                cls.build_message(
                    role=role,
                    content=content,
                    message_type=message_type,
                    add_msg_end=True,
                    output_type=output_type,
                    audio_count=audio_count,
                )
            )

        if add_generation_prompt:
            prompt_parts.append(
                cls.build_message(
                    role="assistant",
                    message_type=None,
                    add_text_eos=False,
                    add_msg_end=False,
                    output_type=output_type,
                )
            )

        return "".join(prompt_parts)

    @classmethod
    def build_transcription_prompt(
        cls,
        request_prompt: str = "",
        *,
        audio_count: int = 1,
    ) -> str:
        return cls.build_prompt_from_messages(
            [
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": request_prompt,
                    "audio_count": audio_count,
                }
            ],
            add_generation_prompt=True,
            output_type="text",
        )

    @classmethod
    def _encode_text(
        cls,
        tokenizer: _KimiAudioTokenizerLike,
        text: str,
    ) -> list[int]:
        if not text:
            return []
        return list(tokenizer.encode(text, add_special_tokens=False))

    @classmethod
    def build_token_content(
        cls,
        *,
        tokenizer: _KimiAudioTokenizerLike,
        messages: Sequence[dict[str, object]],
        output_type: str = "text",
        add_generation_prompt: bool = True,
    ) -> KimiAudioTokenContent:
        special = KimiAudioSpecialTokenIds.from_tokenizer(tokenizer)
        packed_messages: list[KimiAudioTokenContent] = []
        previous_role: str | None = None

        for idx, message in enumerate(messages):
            role = str(message["role"])
            message_type = message.get("message_type") or "text"
            content = message.get("content")
            tokenize_role = previous_role is None or role != previous_role

            if idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                next_role = str(messages[idx + 1]["role"])
                has_ct_token = next_role != role
                has_msg_end_token = next_role != role

            packed_messages.append(
                cls.build_tokenized_message(
                    tokenizer=tokenizer,
                    special_tokens=special,
                    role=role,
                    message_type=message_type,
                    content=content,
                    tokenize_role=tokenize_role,
                    has_ct_token=has_ct_token,
                    has_msg_end_token=has_msg_end_token,
                    output_type=output_type,
                )
            )
            previous_role = role

        if add_generation_prompt:
            packed_messages.append(
                cls.build_tokenized_message(
                    tokenizer=tokenizer,
                    special_tokens=special,
                    role="assistant",
                    message_type=None,
                    content=None,
                    tokenize_role=True,
                    has_ct_token=False,
                    has_msg_end_token=False,
                    output_type=output_type,
                )
            )

        merged = KimiAudioTokenContent()
        for message in packed_messages:
            merged.merge(message)
        merged.validate()
        return merged

    @classmethod
    def build_tokenized_message(
        cls,
        *,
        tokenizer: _KimiAudioTokenizerLike,
        special_tokens: KimiAudioSpecialTokenIds,
        role: str,
        message_type: object,
        content: object,
        tokenize_role: bool,
        has_ct_token: bool,
        has_msg_end_token: bool,
        output_type: str,
    ) -> KimiAudioTokenContent:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported role: {role}")
        if message_type not in {"text", "audio", None}:
            raise ValueError(f"Unsupported message_type: {message_type}")
        if output_type not in {"text", "both"}:
            raise ValueError(f"Unsupported output_type: {output_type}")

        packed = KimiAudioTokenContent()
        if tokenize_role:
            if role == "user":
                packed.audio_append(special_tokens.user_msg_start)
            else:
                packed.audio_append(special_tokens.assistant_msg_start)
            packed.text_append(special_tokens.text_blank)

        if message_type == "text":
            text = str(content or "")
            text_token_ids = cls._encode_text(tokenizer, text)
            packed.text_extend(text_token_ids)
            packed.audio_extend(
                [special_tokens.text_blank] * len(text_token_ids),
                is_continuous=False,
            )
            if role == "assistant":
                packed.text_append(special_tokens.text_eos)
                packed.audio_append(special_tokens.text_blank)
        elif message_type == "audio":
            speech_token_ids = [int(token) for token in (content or [])]
            packed.audio_append(special_tokens.media_begin)
            packed.audio_extend(speech_token_ids, is_continuous=True)
            packed.audio_append(special_tokens.media_end)
            packed.text_extend(
                [special_tokens.text_blank] * (len(speech_token_ids) + 2)
            )
            if has_ct_token:
                control_token = (
                    special_tokens.speech_ct
                    if output_type == "text"
                    else special_tokens.speech_ctd
                )
                packed.audio_append(control_token)
                packed.text_append(special_tokens.text_blank)
        elif message_type is None:
            pass

        if has_msg_end_token:
            packed.audio_append(special_tokens.msg_end)
            packed.text_append(special_tokens.text_blank)

        packed.validate()
        return packed
