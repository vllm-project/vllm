# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Callable
from functools import cached_property, partial
from typing import Any, Final, override

import torch
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from PIL import Image

from vllm import envs
from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY, MediaConnector
from vllm.renderers.embed_utils import (
    safe_load_prompt_embeds,
    safe_load_prompt_embeds_async,
)

from .item_tracker import (
    AsyncMultiModalItemTracker,
    MultiModalItemTracker,
)
from .types import ModalityStr, MultiModalEmbedsPayload

MODALITY_PLACEHOLDERS_MAP = {
    "image": "<##IMAGE##>",
    "audio": "<##AUDIO##>",
    "video": "<##VIDEO##>",
    "prompt_embeds": "<##PROMPT_EMBEDS##>",
}


PROMPT_EMBEDS_PLACEHOLDER_TOKEN: Final[str] = "<prompt_embeds>"
"""The special token used as a placeholder for each embedding
position during chat template rendering.

Registered as an additional special token when `--enable-prompt-embeds` is set.
See `_ensure_prompt_embeds_placeholder_token` in `vllm/renderers/hf.py`.
"""

_ENABLE_PROMPT_EMBEDS_ERROR: Final[str] = (
    "You must set `--enable-prompt-embeds` to input `prompt_embeds`"
)


class BaseMultiModalContentParser(ABC):
    def __init__(self) -> None:
        super().__init__()

        # stores model placeholders list with corresponding
        # general MM placeholder:
        # {
        #   "<##IMAGE##>": ["<image>", "<image>", "<image>"],
        #   "<##AUDIO##>": ["<audio>", "<audio>"],
        #   "<##PROMPT_EMBEDS##>": ["<prompt_embeds>", "<prompt_embeds>"]
        # }
        self._placeholder_storage: dict[str, list] = defaultdict(list)

    @property
    @abstractmethod
    def model_config(self) -> ModelConfig:
        raise NotImplementedError

    def _add_placeholder(self, modality: ModalityStr, placeholder: str | None):
        mod_placeholder = MODALITY_PLACEHOLDERS_MAP[modality]
        if placeholder:
            self._placeholder_storage[mod_placeholder].append(placeholder)

    def mm_placeholder_storage(self) -> dict[str, list]:
        return dict(self._placeholder_storage)

    @abstractmethod
    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_image_embeds(
        self,
        image_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_image_pil(
        self, image_pil: Image.Image | None, uuid: str | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_input_audio(
        self, input_audio: InputAudio | None, uuid: str | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_audio_embeds(
        self,
        audio_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_prompt_embeds(self, data: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_video_embeds(
        self,
        video_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None = None,
    ) -> None:
        raise NotImplementedError


class MultiModalContentParser(BaseMultiModalContentParser):
    def __init__(
        self,
        tracker: MultiModalItemTracker,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self._tracker = tracker
        self._mm_processor_kwargs = mm_processor_kwargs

    @cached_property
    def _connector(self) -> MediaConnector:
        # Connector setup may probe VLLM_MEDIA_CACHE. Defer it until a request
        # actually contains media so text-only parsing never blocks on that I/O.
        return MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=self._tracker.media_io_kwargs,
            allowed_local_media_path=self._tracker.allowed_local_media_path,
            allowed_media_domains=self._tracker.allowed_media_domains,
        )

    @property
    def model_config(self) -> ModelConfig:
        return self._tracker.model_config

    @override
    def parse_prompt_embeds(self, data: str) -> None:
        """Decode a base64 prompt embeds tensor and store it in the tracker.

        Emits a single `PROMPT_EMBEDS_PLACEHOLDER_TOKEN` sentinel per
        content part. The renderer later expands each sentinel to a span of
        `tensor.shape[0]` placeholder tokens after tokenization.
        """
        if not self.model_config.enable_prompt_embeds:
            raise VLLMValidationError(
                _ENABLE_PROMPT_EMBEDS_ERROR, parameter="prompt_embeds"
            )

        tensor = safe_load_prompt_embeds(self.model_config, data.encode())
        self._tracker.add("prompt_embeds", (tensor, None))
        self._add_placeholder("prompt_embeds", PROMPT_EMBEDS_PLACEHOLDER_TOKEN)

    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None:
        image = self._connector.fetch_image(image_url) if image_url else None

        placeholder = self._tracker.add("image", (image, uuid))
        self._add_placeholder("image", placeholder)

    def parse_image_embeds(
        self,
        image_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise VLLMValidationError(
                "You must set `--enable-mm-embeds` to input `image_embeds`",
                parameter="image_embeds",
            )

        if isinstance(image_embeds, dict):
            embeds = {
                k: self._connector.fetch_image_embedding(v) if isinstance(v, str) else v
                for k, v in image_embeds.items()
            }
            placeholder = self._tracker.add("image_embeds", (embeds, uuid))

        if isinstance(image_embeds, str):
            embedding = self._connector.fetch_image_embedding(image_embeds)
            placeholder = self._tracker.add("image_embeds", (embedding, uuid))

        if image_embeds is None:
            placeholder = self._tracker.add("image_embeds", (None, uuid))

        self._add_placeholder("image", placeholder)

    def parse_audio_embeds(
        self,
        audio_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise VLLMValidationError(
                "You must set `--enable-mm-embeds` to input `audio_embeds`",
                parameter="audio_embeds",
            )

        if isinstance(audio_embeds, dict):
            embeds = {
                k: self._connector.fetch_audio_embedding(v) if isinstance(v, str) else v
                for k, v in audio_embeds.items()
            }
            placeholder = self._tracker.add("audio_embeds", (embeds, uuid))
        elif isinstance(audio_embeds, str):
            embedding = self._connector.fetch_audio_embedding(audio_embeds)
            placeholder = self._tracker.add("audio_embeds", (embedding, uuid))
        else:
            placeholder = self._tracker.add("audio_embeds", (None, uuid))

        self._add_placeholder("audio", placeholder)

    def parse_image_pil(
        self, image_pil: Image.Image | None, uuid: str | None = None
    ) -> None:
        placeholder = self._tracker.add("image", (image_pil, uuid))
        self._add_placeholder("image", placeholder)

    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None:
        audio = self._connector.fetch_audio(audio_url) if audio_url else None

        placeholder = self._tracker.add("audio", (audio, uuid))
        self._add_placeholder("audio", placeholder)

    def parse_input_audio(
        self, input_audio: InputAudio | None, uuid: str | None = None
    ) -> None:
        if input_audio:
            audio_data = input_audio.get("data", "")
            audio_format = input_audio.get("format", "")
            if audio_data:
                audio_url = f"data:audio/{audio_format};base64,{audio_data}"
            else:
                # If a UUID is provided, audio data may be empty.
                audio_url = None
        else:
            audio_url = None

        return self.parse_audio(audio_url, uuid)

    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
        video = (
            self._connector.fetch_video(
                video_url=video_url,
                video_processor=self._tracker.video_processor_name,
            )
            if video_url
            else None
        )

        placeholder = self._tracker.add("video", (video, uuid))
        self._add_placeholder("video", placeholder)

        # Extract audio from video if use_audio_in_video is True
        if (
            video_url
            and self._mm_processor_kwargs
            and self._mm_processor_kwargs.get("use_audio_in_video", False)
        ):
            audio = self._connector.fetch_audio(video_url) if video_url else None
            audio_placeholder = self._tracker.add("audio", (audio, uuid))
            self._add_placeholder("audio", audio_placeholder)

    def parse_video_embeds(
        self,
        video_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise VLLMValidationError(
                "You must set `--enable-mm-embeds` to input `video_embeds`",
                parameter="video_embeds",
            )

        if isinstance(video_embeds, dict):
            embeds = {
                k: self._connector.fetch_video_embedding(v) if isinstance(v, str) else v
                for k, v in video_embeds.items()
            }
            placeholder = self._tracker.add("video_embeds", (embeds, uuid))
        elif isinstance(video_embeds, str):
            embedding = self._connector.fetch_video_embedding(video_embeds)
            placeholder = self._tracker.add("video_embeds", (embedding, uuid))
        else:
            placeholder = self._tracker.add("video_embeds", (None, uuid))

        self._add_placeholder("video", placeholder)


class AsyncMultiModalContentParser(BaseMultiModalContentParser):
    def __init__(
        self,
        tracker: AsyncMultiModalItemTracker,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self._tracker = tracker
        self._mm_processor_kwargs: dict[str, Any] | None = mm_processor_kwargs

    @cached_property
    def _connector(self) -> MediaConnector:
        # Connector setup may probe VLLM_MEDIA_CACHE. Defer it until a request
        # actually contains media so text-only parsing never blocks on that I/O.
        return MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=self._tracker.media_io_kwargs,
            allowed_local_media_path=self._tracker.allowed_local_media_path,
            allowed_media_domains=self._tracker.allowed_media_domains,
        )

    @property
    def model_config(self) -> ModelConfig:
        return self._tracker.model_config

    async def _item_with_uuid_async(self, item: object, uuid: str | None):
        return item, uuid

    @override
    def parse_prompt_embeds(self, data: str) -> None:
        """Schedule async prompt embeds decode and store the coroutine in the tracker.

        Like the sync variant, emits a single sentinel `PROMPT_EMBEDS_PLACEHOLDER_TOKEN`
        per content part. Unlike the sync variant, the tensor decode is deferred to a
        thread-pool executor via `safe_load_prompt_embeds_async`.
        """
        if not self.model_config.enable_prompt_embeds:
            raise VLLMValidationError(
                _ENABLE_PROMPT_EMBEDS_ERROR, parameter="prompt_embeds"
            )

        self._tracker.add(
            "prompt_embeds", partial(self._load_prompt_embeds_async, data.encode())
        )
        self._add_placeholder("prompt_embeds", PROMPT_EMBEDS_PLACEHOLDER_TOKEN)

    async def _load_prompt_embeds_async(
        self, data_bytes: bytes
    ) -> tuple[torch.Tensor, None]:
        # Second tuple slot fills the tracker's generic `(item, uuid | None)`
        # contract. prompt_embeds has no UUID concept, so it's always `None`.
        tensor = await safe_load_prompt_embeds_async(self.model_config, data_bytes)
        return tensor, None

    async def _image_with_uuid_async(self, image_url: str | None, uuid: str | None):
        image = (
            await self._connector.fetch_image_async(image_url) if image_url else None
        )
        return image, uuid

    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None:
        placeholder = self._tracker.add(
            "image", partial(self._image_with_uuid_async, image_url, uuid)
        )
        self._add_placeholder("image", placeholder)

    def parse_image_embeds(
        self,
        image_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise VLLMValidationError(
                "You must set `--enable-mm-embeds` to input `image_embeds`",
                parameter="image_embeds",
            )

        placeholder = self._tracker.add(
            "image_embeds",
            partial(self._image_embeds_with_uuid_async, image_embeds, uuid),
        )
        self._add_placeholder("image", placeholder)

    async def _image_embeds_with_uuid_async(
        self,
        image_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None,
    ):
        if isinstance(image_embeds, dict):
            embeds = await _load_embeds_dict(
                image_embeds, self._connector.fetch_image_embedding_async
            )
        elif isinstance(image_embeds, str):
            embeds = await self._connector.fetch_image_embedding_async(image_embeds)
        else:
            embeds = None
        return embeds, uuid

    def parse_audio_embeds(
        self,
        audio_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise VLLMValidationError(
                "You must set `--enable-mm-embeds` to input `audio_embeds`",
                parameter="audio_embeds",
            )

        placeholder = self._tracker.add(
            "audio_embeds",
            partial(self._audio_embeds_with_uuid_async, audio_embeds, uuid),
        )
        self._add_placeholder("audio", placeholder)

    async def _audio_embeds_with_uuid_async(
        self,
        audio_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None,
    ):
        if isinstance(audio_embeds, dict):
            embeds = await _load_embeds_dict(
                audio_embeds, self._connector.fetch_audio_embedding_async
            )
        elif isinstance(audio_embeds, str):
            embeds = await self._connector.fetch_audio_embedding_async(audio_embeds)
        else:
            embeds = None
        return embeds, uuid

    def parse_image_pil(
        self,
        image_pil: Image.Image | None,
        uuid: str | None = None,
    ) -> None:
        placeholder = self._tracker.add(
            "image", partial(self._item_with_uuid_async, image_pil, uuid)
        )
        self._add_placeholder("image", placeholder)

    async def _audio_with_uuid_async(self, audio_url: str | None, uuid: str | None):
        audio = (
            await self._connector.fetch_audio_async(audio_url) if audio_url else None
        )
        return audio, uuid

    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None:
        placeholder = self._tracker.add(
            "audio", partial(self._audio_with_uuid_async, audio_url, uuid)
        )
        self._add_placeholder("audio", placeholder)

    def parse_input_audio(
        self, input_audio: InputAudio | None, uuid: str | None = None
    ) -> None:
        if input_audio:
            audio_data = input_audio.get("data", "")
            audio_format = input_audio.get("format", "")
            if audio_data:
                audio_url = f"data:audio/{audio_format};base64,{audio_data}"
            else:
                # If a UUID is provided, audio data may be empty.
                audio_url = None
        else:
            audio_url = None

        return self.parse_audio(audio_url, uuid)

    async def _video_with_uuid_async(self, video_url: str | None, uuid: str | None):
        video = (
            await self._connector.fetch_video_async(
                video_url,
                video_processor=self._tracker.video_processor_name,
            )
            if video_url
            else None
        )
        return video, uuid

    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
        placeholder = self._tracker.add(
            "video", partial(self._video_with_uuid_async, video_url, uuid)
        )
        self._add_placeholder("video", placeholder)

        # Extract audio from video if use_audio_in_video is True
        if (
            video_url
            and self._mm_processor_kwargs
            and self._mm_processor_kwargs.get("use_audio_in_video", False)
        ):
            audio_placeholder = self._tracker.add(
                "audio", partial(self._audio_with_uuid_async, video_url, uuid)
            )
            self._add_placeholder("audio", audio_placeholder)

    def parse_video_embeds(
        self,
        video_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise VLLMValidationError(
                "You must set `--enable-mm-embeds` to input `video_embeds`",
                parameter="video_embeds",
            )

        placeholder = self._tracker.add(
            "video_embeds",
            partial(self._video_embeds_with_uuid_async, video_embeds, uuid),
        )
        self._add_placeholder("video", placeholder)

    async def _video_embeds_with_uuid_async(
        self,
        video_embeds: MultiModalEmbedsPayload | None,
        uuid: str | None,
    ):
        if isinstance(video_embeds, dict):
            embeds = await _load_embeds_dict(
                video_embeds, self._connector.fetch_video_embedding_async
            )
        elif isinstance(video_embeds, str):
            embeds = await self._connector.fetch_video_embedding_async(video_embeds)
        else:
            embeds = None
        return embeds, uuid


async def _load_embeds_dict(
    data: dict[str, str | list[int | float]],
    fetch: Callable[[str], Awaitable["torch.Tensor"]],
) -> dict[str, Any]:
    encoded = {key: value for key, value in data.items() if isinstance(value, str)}
    tensors = await asyncio.gather(*(fetch(value) for value in encoded.values()))
    return {**data, **dict(zip(encoded, tensors))}
