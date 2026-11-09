# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests

from tests.models.language.pooling.embed_utils import float_embeddings
from tests.models.utils import check_embeddings_close
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.embed.protocol import EmbeddingResponse
from vllm.multimodal.utils import encode_image_url, fetch_image

image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
image_base64 = {"url": encode_image_url(fetch_image(image_url))}


@pytest.mark.parametrize(
    "dual_encoder_model",
    [
        "openai/clip-vit-base-patch32",
        "google/siglip-base-patch16-224",
        "google/siglip2-base-patch16-224",
    ],
)
def test_dual_encoder_chat_text_matches_input(dual_encoder_model: str):
    args = ["--runner", "pooling", "--enforce-eager", "--max-num-seqs", "5"]
    text = "a photo of a cat"

    with RemoteOpenAIServer(dual_encoder_model, args) as server:

        def embed(payload: dict) -> EmbeddingResponse:
            response = requests.post(
                server.url_for("v1/embeddings"),
                json={"model": dual_encoder_model, **payload},
            )
            response.raise_for_status()
            return EmbeddingResponse.model_validate(response.json())

        def user_message(part: dict) -> dict:
            return {"messages": [{"role": "user", "content": [part]}]}

        from_image = embed(
            user_message({"type": "image_url", "image_url": image_base64})
        )
        from_input = embed({"input": text})
        from_messages = embed(user_message({"type": "text", "text": text}))

    assert from_image.usage.prompt_tokens == 1
    assert from_messages.usage.prompt_tokens == from_input.usage.prompt_tokens
    check_embeddings_close(
        embeddings_0_lst=float_embeddings(from_input),
        embeddings_1_lst=float_embeddings(from_messages),
        name_0="input",
        name_1="messages",
    )
