from typing import List, cast

from vllm.model_executor.encode_only.engine_io import EncodeOnlyRequestOutput
from vllm.model_executor.encode_only.execute_io import EncodeOnlyExecuteOutput
from vllm.model_executor.prefill_only.engine_io import (
    PrefillOnlySchedulerOutput, SchedulerOutput)
from vllm.model_executor.prefill_only.output_processor import (OutputProcessor,
                                                               RequestOutput)


class PrefillOnlyModelOutputProcessor(OutputProcessor):

    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine):
        return cls()

    def __call__(
            self, scheduler_output: SchedulerOutput,
            execute_output: EncodeOnlyExecuteOutput) -> List[RequestOutput]:
        assert isinstance(scheduler_output, PrefillOnlySchedulerOutput)
        scheduler_output = cast(PrefillOnlySchedulerOutput, scheduler_output)

        if execute_output.pooled_output is not None:
            request_outputs = []
            for request, outputs in zip(scheduler_output.scheduled_requests,
                                        execute_output.pooled_output):
                prompt_token_ids = request.inputs.prompt_token_ids
                request_outputs.append(
                    EncodeOnlyRequestOutput(request_id=request.request_id,
                                            arrival_time=request.arrival_time,
                                            prompt_token_ids=prompt_token_ids,
                                            finished=True,
                                            outputs=outputs))
            return request_outputs
        else:
            request_outputs = []
            offset = 0
            for request in scheduler_output.scheduled_requests:
                prompt_token_ids = request.inputs.prompt_token_ids
                n_tokens = len(prompt_token_ids)
                request_outputs.append(
                    EncodeOnlyRequestOutput(request_id=request.request_id,
                                            arrival_time=request.arrival_time,
                                            prompt_token_ids=prompt_token_ids,
                                            finished=True,
                                            outputs=execute_output.
                                            last_hidden_states[offset:offset +
                                                               n_tokens]))
                offset += n_tokens
            return request_outputs
