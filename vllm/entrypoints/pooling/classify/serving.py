# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypeAlias

import numpy as np

from vllm import ClassificationOutput
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.engine.serving import ServeContext
from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.logger import init_logger
from vllm.renderers import BaseRenderer

from .io_processor import ClassifyIOProcessor
from .protocol import (
    ClassificationData,
    ClassificationRequest,
    ClassificationResponse,
)

logger = init_logger(__name__)


ClassificationServeContext: TypeAlias = ServeContext[ClassificationRequest]


class ServingClassification(PoolingServing):
    request_id_prefix = "classify"

    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        *,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        trust_request_chat_template: bool = False,
    ) -> ClassifyIOProcessor:
        return ClassifyIOProcessor(
            model_config=model_config,
            renderer=renderer,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            trust_request_chat_template=trust_request_chat_template,
        )

    async def _build_response(
        self,
        ctx: ClassificationServeContext,
    ) -> ClassificationResponse:
        id2label = getattr(self.model_config.hf_config, "id2label", {})

        items: list[ClassificationData] = []
        num_prompt_tokens = 0

        final_res_batch_checked = ctx.final_res_batch

        for idx, final_res in enumerate(final_res_batch_checked):
            classify_res = ClassificationOutput.from_base(final_res.outputs)

            probs = classify_res.probs
            predicted_index = int(np.argmax(probs))
            label = id2label.get(predicted_index)

            item = ClassificationData(
                index=idx,
                label=label,
                probs=probs,
                num_classes=len(probs),
            )

            items.append(item)
            prompt_token_ids = final_res.prompt_token_ids
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return ClassificationResponse(
            id=ctx.request_id,
            created=ctx.created_time,
            model=ctx.model_name,
            data=items,
            usage=usage,
        )
