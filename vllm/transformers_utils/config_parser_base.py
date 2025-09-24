# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from transformers import PretrainedConfig


class ConfigParserBase(ABC):

    @abstractmethod
    def parse(self,
              model: Union[str, Path],
              trust_remote_code: bool,
              revision: Optional[str] = None,
              code_revision: Optional[str] = None,
              **kwargs) -> tuple[dict, PretrainedConfig]:
        raise NotImplementedError
