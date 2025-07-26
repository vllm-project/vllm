# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Union

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest

logger = init_logger(__name__)


@dataclass
class AdditionalHeadsProcessor:
    """Processor for additional head outputs from the model.
    
    This class handles storing and managing additional head outputs
    for generated tokens, similar to how LogprobsProcessor handles logprobs.
    """

    # Additional head outputs for this request
    additional_head_outputs: list[Union[list[float], dict[str, float]]]

    @classmethod
    def from_new_request(
        cls,
        request: EngineCoreRequest,
    ) -> "AdditionalHeadsProcessor":
        """Create a new AdditionalHeadsProcessor for a request.
        
        Args:
            request: The engine core request to process additional heads for.
        """
        return cls(additional_head_outputs=[], )

    def update_from_output(self, output: EngineCoreOutput) -> None:
        """Update with additional head outputs from EngineCore.
        
        Args:
            output: The engine core output containing new additional 
                head outputs.
        """
        if output.new_additional_head_outputs is not None:
            self.additional_head_outputs.append(
                output.new_additional_head_outputs.additional_head_outputs)
