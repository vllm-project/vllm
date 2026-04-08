# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypeAlias

import numpy as np
from fastapi.responses import JSONResponse

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.logger import init_logger
from vllm.outputs import ClassificationOutput
from vllm.renderers import BaseRenderer

from .io_processor import ClassifyIOProcessor
from .protocol import (
    ClassificationData,
    ClassificationRequest,
    ClassificationResponse,
)

logger = init_logger(__name__)


ClassificationServeContext: TypeAlias = PoolingServeContext[ClassificationRequest]


class ServingClassification(PoolingServing):
    request_id_prefix = "classify"

    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> ClassifyIOProcessor:
        return ClassifyIOProcessor(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )

    async def _build_response(
        self,
        ctx: ClassificationServeContext,
    ) -> JSONResponse:
        id2label = getattr(self.model_config.hf_config, "id2label", {})
        num_prompt_tokens = 0
        items: list[ClassificationData] = []
        for idx, final_res in enumerate(ctx.final_res_batch):
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

        response = ClassificationResponse(
            id=ctx.request_id,
            created=ctx.created_time,
            model=ctx.model_name,
            data=items,
            usage=usage,
        )

        return JSONResponse(content=response.model_dump())
