from typing import List, Optional

import torch

from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SamplerOutput)
from vllm.worker.worker import Worker
from vllm.worker.worker_base import WorkerInput
from vllm.worker.model_runner_base import ModelRunnerInputBase


class SeparatedWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_input = None

    @torch.inference_mode()
    def get_logits(
        self,
        hidden_or_intermediate_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.model_runner.get_logits(hidden_or_intermediate_states, self.model_input)

    @torch.inference_mode()
    def compute_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.model_runner.compute_logits(logits, self.model_input)

    @torch.inference_mode()
    def do_sample(
        self,
        logits: torch.Tensor,
    ) -> List[SamplerOutput]:
        return self.model_runner.do_sample(logits, self.model_input)

    @torch.inference_mode()
    def execute_model_part(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:

        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None

            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req)
            self.model_input: ModelRunnerInputBase = (
                self.model_runner.prepare_model_input(
                    execute_model_req.seq_group_metadata_list,
                    execute_model_req.virtual_engine,
                    execute_model_req.finished_requests_ids))
            num_steps = execute_model_req.num_steps

            if self.do_metadata_broadcast:
                broadcast_data = worker_input.as_broadcastable_tensor_dict()
                broadcast_data.update(
                    self.model_input.as_broadcastable_tensor_dict())
                broadcast_data["num_steps"] = num_steps
                broadcast_tensor_dict(broadcast_data, src=0)
        else:
            assert self.do_metadata_broadcast
            broadcast_data = broadcast_tensor_dict(src=0)
            if not broadcast_data:
                return None

            num_steps = broadcast_data.pop("num_steps")
            worker_input = WorkerInput.from_broadcasted_tensor_dict(
                broadcast_data)
            self.model_input = (
                self.model_runner.
                make_model_input_from_broadcasted_tensor_dict(broadcast_data))

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict())

        hidden_or_intermediate_states = self.model_runner.model_execute(
            self.model_input, 
            self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None, 
            intermediate_tensors,
            num_steps
        )

        logits = self.get_logits(hidden_or_intermediate_states)
        # logits = self.compute_logits(logits, model_input)
        # output = self.do_sample(logits)

        if not self.is_driver_worker:
            return []

        if not get_pp_group().is_last_rank:
            # output is IntermediateTensors
            get_pp_group().send_tensor_dict(logits.tensors)
            return [None]

        return logits
