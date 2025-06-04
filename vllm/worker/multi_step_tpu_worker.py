# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from typing import Dict, Optional, Tuple

import torch

from vllm.distributed import broadcast_tensor_dict
from vllm.sequence import ExecuteModelRequest
from vllm.worker.tpu_model_runner import ModelInputForTPU
from vllm.worker.tpu_worker import TPUWorker
from vllm.worker.worker_base import WorkerInput


class MultiStepTPUWorker(TPUWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_model_input: Optional[ModelInputForTPU] = None

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[ModelInputForTPU, WorkerInput, Dict[str, torch.Tensor]]:
        assert self.is_driver_worker
        assert execute_model_req.virtual_engine == 0

        is_first_multi_step = execute_model_req.is_first_multi_step
        is_last_step = execute_model_req.is_last_step
        if is_first_multi_step:
            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req)
            worker_input = dataclasses.replace(
                worker_input,
                num_steps=execute_model_req.num_lookahead_slots + 1)
            model_input: ModelInputForTPU = (
                self.model_runner.prepare_model_input(
                    execute_model_req.seq_group_metadata_list,
                    execute_model_req.virtual_engine,
                    execute_model_req.finished_requests_ids))

            if execute_model_req.async_callback:
                model_input = dataclasses.replace(
                    model_input,
                    async_callback=execute_model_req.async_callback)
        else:
            assert self.cached_model_input is not None
            model_input = self.cached_model_input
            worker_input = WorkerInput()
        model_input = dataclasses.replace(
            model_input,
            is_first_multi_step=is_first_multi_step,
            is_last_step=is_last_step)

        if self.do_metadata_broadcast:
            if is_first_multi_step:
                broadcast_data = worker_input.as_broadcastable_tensor_dict()
                broadcast_data.update(
                    model_input.as_broadcastable_tensor_dict())
                broadcast_tensor_dict(broadcast_data, src=0)
            else:
                broadcast_data = {
                    "is_first_multi_step": is_first_multi_step,
                    "is_last_step": is_last_step,
                }
                broadcast_tensor_dict(broadcast_data, src=0)

        # Retuning empty dict here to keep this compatible with
        # `LocalOrDistributedWorkerBase._get_driver_input_and_broadcast`
        return model_input, worker_input, {}

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[Tuple[ModelInputForTPU, WorkerInput, Dict[str,
                                                            torch.Tensor]]]:
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    broadcast_tensor_dict({}, src=0)
                return None

            model_input, worker_input, _ = self._get_driver_input_and_broadcast(
                execute_model_req)
            if model_input.is_first_multi_step:
                self.cached_model_input = model_input
            return model_input, worker_input, {}
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            if not broadcast_data:
                return None

            if len(broadcast_data) == 2:
                assert self.cached_model_input is not None
                self.cached_model_input = dataclasses.replace(
                    self.cached_model_input,
                    is_first_multi_step=broadcast_data["is_first_multi_step"],
                    is_last_step=broadcast_data["is_last_step"])
                empty_worker_input = WorkerInput()
                return self.cached_model_input, empty_worker_input, {}

            worker_input = WorkerInput.from_broadcasted_tensor_dict(
                broadcast_data)
            model_input = (
                self.model_runner.
                make_model_input_from_broadcasted_tensor_dict(broadcast_data))
            self.cached_model_input = model_input
            return model_input, worker_input, {}
