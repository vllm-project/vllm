# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import AsyncGenerator
from typing import Union

from vllm.entrypoints.openai.protocol import (ChatCompletionResponse,
                                              ErrorResponse)

# TODO generateresponse
GenerationResponseT = Union[AsyncGenerator[str, None], ChatCompletionResponse,
                            ErrorResponse]