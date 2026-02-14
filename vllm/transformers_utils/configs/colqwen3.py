# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ColQwen3 config that inherits from Qwen3VLConfig.

TomoroAI ColQwen3 checkpoints use ``model_type: "colqwen3"`` which is not
recognised by HuggingFace Transformers.  By subclassing ``Qwen3VLConfig``
and setting ``model_type = "colqwen3"`` we let ``from_pretrained`` parse
the config JSON correctly without ``trust_remote_code``.
"""

from transformers import Qwen3VLConfig


class ColQwen3Config(Qwen3VLConfig):
    model_type = "colqwen3"
