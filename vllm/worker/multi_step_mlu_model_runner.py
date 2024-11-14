import dataclasses
import functools
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Union)

import torch

from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (IntermediateTensors, SequenceGroupMetadata)

from ..model_executor.model_loader.tensorizer import TensorizerConfig

logger = init_logger(__name__)

from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
from vllm.worker.multi_step_model_runner import (ModelOutput,
                                                 StatefulModelInput,
                                                 PythonizationCache,
                                                 _pythonize_sampler_output,
                                                 MULTI_STEP_ATTENTION_BACKENDS,
                                                 _get_supported_attention_backends)

from vllm.worker.mlu_model_runner import MLUModelRunnerBase

MULTI_STEP_ATTENTION_BACKENDS += ["MLU_FLASH_ATTN"]

@dataclass
class MLUModelOutput(ModelOutput):

    def _pythonize_sampler_output(self, input_metadata: "MLUStatefulModelInput",
                                  copy_stream: torch.mlu.Stream,
                                  pinned_sampled_token_buffer: torch.Tensor,
                                  blocking: bool) -> bool:
        """
        If blocking is set, will block until the forward pass for the output is
        ready and pythonize the output. Upon completing Pythonization, erases
        self.logprobs (note that a non-blocking call that is performed when
        the sampler output is not yet ready, will not erase self.logprobs.)
        """
        assert self.sampled_token_ids is not None
        if not blocking and not self.sampler_output_ready_event.query():
            return False

        if blocking:
            self.sampler_output_ready_event.synchronize()
        with torch.mlu.stream(copy_stream):
            _pythonize_sampler_output(input_metadata, self.sampler_output,
                                      pinned_sampled_token_buffer,
                                      self.sampled_token_ids, self.logprobs,
                                      self.pythonization_cache)

        # Erase the logprobs MLU-side tensor.
        # Note that although _pythonize_sampler_output() runs in its
        # own CUDA stream, nonetheless _pythonize_sampler_output()
        # cannot return until Pythonization is complete; therefore
        # we know that by the time the CPU reaches this point,
        # `self.logprobs` is no longer needed.
        self.logprobs = None
        return True


@dataclass(frozen=False)
class MLUStatefulModelInput(StatefulModelInput):
    # ping-pong data structures for multi-step to wait on the previous step
    step_cuda_events: List[torch.mlu.Event] = field(
        default_factory=lambda: [torch.mlu.Event(blocking=False)] * 2)

    def record_step_event(self, current_stream: torch.mlu.Stream):
        # record the event for the current step so that the next step can sync
        # on it. We modulo by 2 to keep the events in a circular buffer and
        # support any attn backends that may be supported in the future. ie
        # Flashinfer would want two DecodeWrappers to overlap the CPU and GPU.
        self.step_cuda_events[self.current_step & 1] = \
            torch.mlu.Event(blocking=False)
        self.step_cuda_events[self.current_step & 1].record(current_stream)

    def add_sampler_output(self,
                           sampler_output: SamplerOutput,
                           sampled_token_ids: Optional[torch.Tensor] = None):
        self.cached_outputs.append(
            MLUModelOutput(sampler_output=sampler_output,
                        sampler_output_ready_event=None,
                        sampled_token_ids=sampled_token_ids,
                        pythonized=False))


# MutableModelInputForGPUWithMultiStepMetadata is not subclass of
# ModelInputForGPU but it wraps the actual input dataclass and adds multi-step
# metadata
# mypy: disable-error-code=type-var
class MLUMultiStepModelRunner(MLUModelRunnerBase[MLUStatefulModelInput]):

    def __init__(self, base_model_runner: MLUModelRunnerBase, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check attention backend support.
        supported_attention_backends: List[str] = \
            _get_supported_attention_backends(
                self.scheduler_config.chunked_prefill_enabled)
        if self.attn_backend.get_name() not in supported_attention_backends:
            ms_config_str: str = "Multi-Step + Chunked-Prefill" \
                if self.scheduler_config.chunked_prefill_enabled \
                      else "Multi-Step"
            raise ValueError(
                f"{ms_config_str} not supported for attention backend: "
                f"{self.attn_backend.get_name()}. Set VLLM_ATTENTION_BACKEND "
                f"to a value from {supported_attention_backends}.")


        # uses the base model runner to execute the model and wraps it with
        # multi-step logic
        self._base_model_runner: MLUModelRunnerBase = base_model_runner

        self.is_multi_step = self.scheduler_config.is_multi_step
        self.pinned_sampled_token_ids: Optional[torch.Tensor] = None

        # Using the PythonizationCache in Pipeline-Parallel clobbers the
        # SequenceOutput and CompletionSequenceGroupOutput object.
        # When cache-reset happens at the last step of a multi-step
        # execution, there may be other on-going single-step/multi-step
        # executions. The current caching implementation does not check
        # for this.
        self.pythonization_cache = PythonizationCache() \
            if self.parallel_config.pipeline_parallel_size == 1 else None

    @functools.cached_property
    def _copy_stream(self):
        # used to copy tensors from GPU to CPU asynchronously
        return torch.mlu.Stream()

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> MLUStatefulModelInput:
        model_input = (MLUStatefulModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        ))
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> MLUStatefulModelInput:
        frozen_model_input: ModelInputForGPUWithSamplingMetadata = \
              self._base_model_runner.prepare_model_input(
                    seq_group_metadata_list,
                    virtual_engine,
                    finished_requests_ids)

        assert frozen_model_input.query_lens is not None
        assert frozen_model_input.seq_lens is not None
        assert frozen_model_input.attn_metadata is not None
        num_queries = len(frozen_model_input.query_lens)
        num_seqs = len(frozen_model_input.seq_lens)
        num_single_step_prefills = frozen_model_input.attn_metadata.num_prefills

        model_input = MLUStatefulModelInput(
            frozen_model_input=frozen_model_input,
            num_seqs=num_seqs,
            num_queries=num_queries,
            num_single_step_prefills=num_single_step_prefills)

        return model_input

    def _async_process_outputs(self, model_input: MLUStatefulModelInput,
                               output_proc_callback: Callable):
        # Proceed with pythonization and output_proc in order.
        # Stop on the first one that fails to pythonize
        output_proc_callback()

        cont = True
        for step_num, model_output in enumerate(model_input.cached_outputs):
            if not model_output.pythonized:
                model_output.maybe_pythonize(model_input, self._copy_stream,
                                             self.pinned_sampled_token_ids)
                if model_output.pythonized:
                    ctx = output_proc_callback.keywords["ctx"]
                    ctx.append_output(
                        outputs=[model_output.sampler_output],
                        seq_group_metadata_list=ctx.seq_group_metadata_list,
                        scheduler_outputs=ctx.scheduler_outputs,
                        is_async=False,
                        is_last_step=False,
                        is_first_step_output=step_num == 0)

                    output_proc_callback()
                else:
                    cont = False

            if not cont:
                break

    def _final_process_outputs(self, model_input: MLUStatefulModelInput,
                               output_proc_callback: Optional[Callable]):
        assert model_input.frozen_model_input is not None

        has_async_callback = output_proc_callback is not None

        outputs = []
        for step_num, output in enumerate(model_input.cached_outputs):
            is_last_step = step_num == len(model_input.cached_outputs) - 1

            # For non-async case:
            #   -- We simply add the outputs
            # For async case:
            #   -- Invoke callback, pythonize, add to callback queue and repeat
            #   -- For last output, just add to callback queue
            if has_async_callback:
                assert output_proc_callback is not None

                # Invoke callback before pythonize (to overlap with GPU)
                output_proc_callback()

                # Pythonize
                if not output.pythonized:
                    output.pythonize(model_input, self._copy_stream,
                                     self.pinned_sampled_token_ids)

                    # For non last step, add to callback queue to chain
                    # callbacks=>pythonize pairs (for GPU overlap)
                    if not is_last_step:
                        ctx = output_proc_callback.keywords[  # type: ignore
                            "ctx"]  # type: ignore
                        ctx.append_output(
                            outputs=[output.sampler_output],
                            seq_group_metadata_list=ctx.
                            seq_group_metadata_list,
                            scheduler_outputs=ctx.scheduler_outputs,
                            is_async=False,
                            is_last_step=False,
                            is_first_step_output=step_num == 0)
                    else:
                        outputs.append(output.sampler_output)
            else:
                output.pythonize(model_input, self._copy_stream,
                                 self.pinned_sampled_token_ids)
                outputs.append(output.sampler_output)

        return outputs

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: MLUStatefulModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        """ 
        Execute the model for a single step and update multi-step
        metadata
        """
        assert num_steps == 1, "MLUMultiStepModelRunner only supports num_steps=1"
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None

        # path for warm up runs
        if not model_input.is_multi_step:
            return self._base_model_runner.execute_model(
                frozen_model_input, kv_caches, intermediate_tensors, num_steps)

        # make sure we skip the sampler on the lask rank and only pythonize
        # if CPU is ahead.
        if self.is_driver_worker and get_pp_group().is_last_rank:
            if self.pinned_sampled_token_ids is None:
                self.pinned_sampled_token_ids = torch.zeros(
                    (self.scheduler_config.max_num_seqs, 1),
                    dtype=torch.long,
                    device="cpu",
                    pin_memory=True)

            self._base_model_runner.model.sampler.include_gpu_probs_tensor = (
                True)
            if frozen_model_input.sampling_metadata:
                frozen_model_input.sampling_metadata.skip_sampler_cpu_output = (
                    True)

        # some pre-execute model logic for multi-step:
        #   - if it's the first step, we need to reset the sampling tensors
        #   - if it's not the first step, we need to advance the step using the
        #   appended sampler output from last iteration
        #   - also maybe pythonize if CPU is ahead of MLU

        current_stream = torch.mlu.current_stream()
        if not model_input.is_first_multi_step:
            # Explicitly block on the previous step's forward to make sure we
            # don't clobber any MLU tensors still in use.
            # This is not needed for flashattn backend, but for other attn
            # backends such as flashinfer that performs extra CPU operations on
            # input metadata we may need to synchronize any CPU operations that
            # might clobber enqueued forwards. (prevents CPU from running too
            # far ahead if needed)
            model_input.wait_previous_step()
            model_input = self._advance_step(
                model_input, model_input.cached_outputs[-1].sampler_output)

            # frozen_model_input may have been updated
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None

        if model_input.base_output_proc_callback is None:
            assert frozen_model_input is not None
            model_input.base_output_proc_callback = \
                        frozen_model_input.async_callback

        if frozen_model_input.async_callback is not None:
            assert model_input.base_output_proc_callback is not None
            async_callback = functools.partial(
                self._async_process_outputs,
                model_input=model_input,
                output_proc_callback=model_input.base_output_proc_callback)

            model_input.frozen_model_input = dataclasses.replace(  # type: ignore
                model_input.frozen_model_input,
                async_callback=async_callback)
            # Update the local instance
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None

        # Execute the model
        output = self._base_model_runner.execute_model(frozen_model_input,
                                                       kv_caches,
                                                       intermediate_tensors,
                                                       num_steps=1)

        # record the event for the current step so that the next step can sync
        model_input.record_step_event(current_stream)

        if get_pp_group().is_last_rank and self.is_driver_worker:
            assert len(
                output
            ) == 1, "MultiStepModelRunner requires single-step base_models"

            # event for the pythonization so that we only pythonize if the
            # tensors are ready. May be able to be combined with the step event
            output_ready_event = torch.mlu.Event()
            output_ready_event.record(current_stream)
            if self.parallel_config.pipeline_parallel_size > 1:
                output[0].sampled_token_ids_cpu = output[
                    0].sampled_token_ids.cpu()
            model_input.cached_outputs.append(
                MLUModelOutput(output[0], output_ready_event,
                            output[0].sampled_token_ids, False,
                            output[0].logprobs, self.pythonization_cache))

            # These MLU tensors are not required by multi-step;
            # erase them to ensure they are not pythonized or
            # transferred to CPU
            output[0].sampled_token_ids = None
            output[0].sampled_token_probs = None
            output[0].logprobs = None

            # Pythonize the output if CPU is ahead and the previous step is
            # ready.
            if frozen_model_input.async_callback is None:
                for model_output in model_input.cached_outputs:
                    model_output.maybe_pythonize(model_input,
                                                 self._copy_stream,
                                                 self.pinned_sampled_token_ids)

        model_input.current_step += 1

        if not get_pp_group().is_last_rank:
            # Should be IntermediateTensors
            assert isinstance(output, IntermediateTensors)
            return output
        if not self.is_driver_worker:
            return []

        # Pythonize the output and block if needed since it is the last step
        if model_input.is_last_step:
            outputs = self._final_process_outputs(
                model_input, model_input.base_output_proc_callback)
            if self.pythonization_cache:
                self.pythonization_cache.reset()
            return outputs

        # should be [SamplerOutput]
        return output

    def _update_sampling_metadata(self, sampling_metadata, num_seqs,
                                  num_queries):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (
            num_queries, )
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed # noqa: E501

        # Verify that all sequences are decodes
        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt is False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple
            assert seq_group.seq_len is None  # Decode
            assert seq_group.query_len is None  # Decode

    def _advance_step(self, model_input: MLUStatefulModelInput,
                      out: SamplerOutput) -> MLUStatefulModelInput:

        model_input.maybe_advance_frozen_model_input(self.device,
                                                     self.pin_memory)
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None
        assert frozen_model_input.input_tokens is not None
        assert frozen_model_input.input_tokens.shape[0] == model_input.num_seqs
        assert frozen_model_input.attn_metadata is not None

        sampled_token_ids = model_input.cached_outputs[-1].sampled_token_ids
        num_seqs = model_input.num_seqs
        num_queries = model_input.num_queries
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None
        attn_metadata = frozen_model_input.attn_metadata
        assert attn_metadata is not None

        turn_prefills_into_decodes: bool = model_input.current_step == 1 and \
                                    model_input.num_single_step_prefills != 0
        attn_metadata.advance_step(
            frozen_model_input,
            sampled_token_ids,
            self.block_size,
            num_seqs,
            num_queries,
            turn_prefills_into_decodes=turn_prefills_into_decodes)

        return model_input

    def load_model(self) -> None:
        return self._base_model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        return self._base_model_runner.save_sharded_state(
            path, pattern, max_size)

    def save_tensorized_model(self,
                              tensorizer_config: TensorizerConfig) -> None:
        return self._base_model_runner.save_tensorized_model(tensorizer_config)

    def profile_run(self) -> None:
        return self._base_model_runner.profile_run()

    def remove_all_loras(self):
        return self._base_model_runner.remove_all_loras()

    def capture_model(self, kv_caches: List[List]) -> None:
        return self._base_model_runner.capture_model(kv_caches)

    @property
    def vocab_size(self) -> int:
        return self._base_model_runner.vocab_size