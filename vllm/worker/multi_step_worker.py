from vllm.worker.worker import Worker
from dataclasses import dataclass
from vllm.worker.worker import WorkerInput
from vllm.worker.model_runner_base import BroadcastableModelInput
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.distributed import broadcast_tensor_dict, get_pp_group
from typing import Tuple, Optional, List
from dataclasses import field

from vllm.worker.multi_step_model_runner import (
    MutableModelInputForGPUWithMultiStepMetadata)


@dataclass
class MultiStepState:
    worker_input: WorkerInput
    model_input: MutableModelInputForGPUWithMultiStepMetadata


class MultiStepWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        self.multi_step_states: List[
            Optional[MultiStepState]] = [None] * pipeline_parallel_size
        self.temp_output = None

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[BroadcastableModelInput, WorkerInput]:
        """
        Get the driver input and broadcast it to other workers.
        """
        assert self.is_driver_worker
        virtual_engine = execute_model_req.virtual_engine
        is_first_multi_step = execute_model_req.is_first_multi_step
        if is_first_multi_step:
            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req)
            model_input: MutableModelInputForGPUWithMultiStepMetadata = (
                self.model_runner.prepare_model_input(
                    execute_model_req.seq_group_metadata_list,
                    execute_model_req.virtual_engine,
                    execute_model_req.finished_requests_ids))
        else:
            multi_step_state = self.multi_step_states[virtual_engine]
            worker_input = multi_step_state.worker_input
            model_input = multi_step_state.model_input

        model_input.is_first_multi_step = is_first_multi_step
        model_input.is_last_step = execute_model_req.is_last_step

        # we broadcast the last sampled token ids to all TP workers so they can
        # update their model input metadata inplace.
        if not is_first_multi_step:
            if get_pp_group().is_last_rank:
                assert model_input.outputs[
                    -1].sampler_output.sampled_token_ids is None
                assert model_input.outputs[-1].sampled_token_ids is not None
                model_input.last_sampled_token_ids = model_input.outputs[
                    -1].sampled_token_ids
                # free sampled token ids from the previous step if it has been
                # pythonized. Cannot free the last sampled token ids because
                # we need it for GPU advance_step.
                for output in model_input.outputs[:-1]:
                    if output.pythonized:
                        output.sampled_token_ids = None
            else:
                # otherwise we need to get the cached sampled token ids from the
                # execute_model_req
                assert execute_model_req.last_sampled_token_ids is not None
                model_input.last_sampled_token_ids = execute_model_req.last_sampled_token_ids.cuda(
                )
                model_input.add_sampler_output(
                    SamplerOutput(outputs=[], sampled_token_ids=None),
                    model_input.last_sampled_token_ids)

                # free sampled token ids from the previous step.
                # TODO(will) we could reuse the sampled token ids tensor from
                # the previous step instead.
                for output in model_input.outputs[:-1]:
                    output.sampled_token_ids = None
                assert model_input.outputs[-1].sampled_token_ids is not None

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_tensor_dict(broadcast_data, src=0)

        return model_input, worker_input

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[Tuple[MutableModelInputForGPUWithMultiStepMetadata,
                        WorkerInput]]:
        """
        Depending on the current state of the request and multi step worker,
        this method may skip the normal _prepare_model_input and
        _prepare_worker_input methods and instead used cached values.
        """
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

            virtual_engine = execute_model_req.virtual_engine
            model_input, worker_input = self._get_driver_input_and_broadcast(
                execute_model_req)
            assert isinstance(model_input,
                              MutableModelInputForGPUWithMultiStepMetadata)
            if execute_model_req.is_first_multi_step:
                # cache the worker input and model input for the next steps
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input=worker_input, model_input=model_input)
        # if TP workers
        else:
            broadcast_data = self._get_worker_input_from_broadcast()
            # if the driver has sent an empty input, we should stop the worker
            # loop
            if broadcast_data is None:
                return None
            model_input, worker_input = broadcast_data
            assert isinstance(model_input,
                              MutableModelInputForGPUWithMultiStepMetadata)
            virtual_engine = worker_input.virtual_engine
            if model_input.is_first_multi_step:
                pass
                # cache the worker input and model input for the next steps
                # TODO(will) see below

                # self.multi_step_states[virtual_engine] = MultiStepState(
                #     worker_input=worker_input, model_input=model_input)
            else:
                # TODO(will) possible to also use the cached worker input and model input
                # this can be done if we want to optimize the broadcast to only send
                # the last sampled token ids for non-first multi steps

                # multi_step_state = self.multi_step_states[virtual_engine]
                # cached_model_input = multi_step_state.model_input
                # cached_worker_input = multi_step_state.worker_input
                assert isinstance(
                    model_input, MutableModelInputForGPUWithMultiStepMetadata)
                # we need to update the last sampled token ids in the model input
                # for the workers so that they can run inplace advance_step
                model_input.add_sampler_output(
                    SamplerOutput(outputs=[], sampled_token_ids=None),
                    model_input.last_sampled_token_ids)
                # self.multi_step_states[virtual_engine] = MultiStepState(
                #     worker_input=worker_input, model_input=model_input)
                # model_input = cached_model_input
                # worker_input = cached_worker_input

        assert model_input is not None
        assert worker_input is not None
        return model_input, worker_input
