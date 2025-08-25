# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from ...utils import CLSPoolingEmbedModelInfo, EmbedModelInfo
from .mteb_utils import mteb_test_embed_models

# ST models with projector (Dense) layers
ST_PROJECTOR_MODELS = [
    CLSPoolingEmbedModelInfo(
        "TencentBAC/Conan-embedding-v1",
        architecture="BertModel",
        enable_test=True,
    ),
]


@pytest.mark.parametrize("model_info", ST_PROJECTOR_MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner,
                           model_info: EmbedModelInfo) -> None:

    mteb_test_embed_models(hf_runner, vllm_runner, model_info)
