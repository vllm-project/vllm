# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

GenerationTask = Literal["generate", "transcription"]
PoolingTask = Literal["encode", "embed", "classify", "score"]

SupportedTask = Literal[GenerationTask, PoolingTask]
