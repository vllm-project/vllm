# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any, Literal

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth

_FIXED_PARAMS: dict[str, Any] = {
    "temperature": 0,
    "repetition_penalty": 1.1,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.0,
}

# Match both the user-facing model_name (LiteLLM Proxy request "model") and
# the provider model string (litellm_params.model).
_LOCKED_MODELS: set[str] = {
    "vibevoice-asr",
    "hosted_vllm/vibevoice",
    "hosted_vllm/vibevoice-asr",
}


class VibeVoiceASRFixedParams(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> dict:
        model = data.get("model")
        if isinstance(model, str) and (
            model in _LOCKED_MODELS
            or model.endswith("/vibevoice")
            or model.endswith("/vibevoice-asr")
        ):
            # Force overwrite (ignore upstream values).
            data.update(_FIXED_PARAMS)
        return data


proxy_handler_instance = VibeVoiceASRFixedParams()
