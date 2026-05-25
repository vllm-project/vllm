# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Literal, TypeAlias

import torch

## Protocols for Audio
AudioResponseFormat: TypeAlias = Literal["json", "text", "srt", "verbose_json", "vtt"]
_LONG_INFO = torch.iinfo(torch.long)
