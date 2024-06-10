import copy
from typing import List, Tuple, Set, Optional
import logging

import torch
import torch.distributed

from vllm.sequence import (ExecuteModelRequest, SamplerOutput, SequenceGroupMetadata)
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.lora.request import LoRARequest

from vllm.distributed.parallel_state import patch_tensor_parallel_group
from vllm.config import ParallelConfig
from vllm.worker.worker_base import WorkerBase

logger = logging.getLogger(__name__)


class SingleTpWorker(WorkerBase):
    """Class which allows a speculative draft model to run with tensor parallel
    degree of 1, while target model runs with larger tensor parallel degree.
    This reduces the overhead of small draft models.

    This is implemented by changing vLLM's tensor parallel group to a group of
    size 1 during forward passes.
    """

    @classmethod
    def maybe_wrap_worker(cls, worker, draft_parallel_config: ParallelConfig,
                          target_parallel_config: ParallelConfig):
        """Wrap the worker in a SingleTpWorker if necessary.
        """
        draft_tp = draft_parallel_config.tensor_parallel_size
        if draft_tp == target_parallel_config.tensor_parallel_size:
            return worker

        if draft_tp != 1:
            raise ValueError("{cls} only supports tp=1, found "
                             f"{draft_tp=}")

        logger.info(f"Wrapping {type(worker)} in {cls}")
        return cls(worker)

    def __init__(
        self,
        worker: WorkerBase, # MultiStepWorker
    ):
        self._worker = worker
        self._single_tp_group = None

        # Lazy initialization list.
        self._proposer: Top1Proposer

    def is_driver(self) -> bool:
        return self._worker.is_driver()

    def init_device(self):
        """Initialize the model on all ranks.

        This also creates a single-rank process group containing only the
        self process.
        """
        world_rank = torch.distributed.get_rank()
        self._single_tp_group = torch.distributed.new_group(ranks=[world_rank])
        self._single_tp_cpu_group = torch.distributed.new_group(ranks=[world_rank],
                                                                backend="gloo")
 
        logger.info(f"init_device. world_rank: {world_rank}, single_tp_group: {self._single_tp_group}, single_tp_cput_group: {self._single_tp_cpu_group}")

        with patch_tensor_parallel_group(self._single_tp_group, self._single_tp_cpu_group):
            self._worker.init_device()
 
        self._proposer = Top1Proposer(
            self,
            self._worker.device,
            self.vocab_size,
            max_proposal_len=self.max_model_len,
        )


    def set_include_gpu_probs_tensor(self):
        # Need include_gpu_probs_tensor for multi_step_worker
        self._worker.set_include_gpu_probs_tensor()

    def load_model(self):
        logger.info("SingleTPWorker.load_model()")
        with patch_tensor_parallel_group(self._single_tp_group, self._single_tp_cpu_group):
            self._worker.load_model()

    def determine_num_available_blocks(self):
        """Profile the model on all ranks.
        """
        with patch_tensor_parallel_group(self._single_tp_group, self._single_tp_cpu_group):
            return self._worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int):
        """Initialize the cache engine on all ranks.
        """
        with patch_tensor_parallel_group(self._single_tp_group, self._single_tp_cpu_group):
            self._worker.initialize_cache(num_gpu_blocks,
                                          num_cpu_blocks)

    @torch.inference_mode()
    def sampler_output(
        self,
        sample_len: int,
        execute_model_req: ExecuteModelRequest
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.

        For multi step worker, this indicator shall be True.
        """

        ## Worker-side logic: skip
        if not self.is_driver():
            logger.info("Workers should not make proposals")
            return None

        ## Driver-side logic
        self._raise_if_unsupported(execute_model_req)

        # Shallow copy input data so modifications (such as appending tokens)
        # do not cause side-effects.
        copied_seq_group_metadata_list = self._shallow_copy_inputs(
            execute_model_req.seq_group_metadata_list)
        copied_execute_model_req = execute_model_req.clone(
            copied_seq_group_metadata_list)

        # Assert enough KV space for sample_len tokens per sequence.
        self._assert_enough_kv_space(execute_model_req.seq_group_metadata_list,
                                     sample_len)

        # Run model sample_len times.
        model_outputs = []
        for i in range(sample_len):
            logger.info(f"Driver runs multiple draft steps. {i+1}/{sample_len}")
            model_output = self._execute_model_tp1(
                execute_model_req=copied_execute_model_req)
            assert (len(model_output) == 1
                    ), "composing multistep workers not supported"
            model_output = model_output[0]
            self._append_new_tokens(model_output,
                                    copied_seq_group_metadata_list)
            model_outputs.append(model_output)

        return model_outputs, True

    def get_spec_proposals(
        self,
        proposal_len: int,
        execute_model_req: ExecuteModelRequest) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """
        with patch_tensor_parallel_group(self._single_tp_group, self._single_tp_cpu_group):
            return self._proposer.get_proposals(proposal_len, execute_model_req)

    @torch.inference_mode()
    def execute_model(self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        with patch_tensor_parallel_group(self._single_tp_group, self._single_tp_cpu_group):
            return self._execute_model_tp1(execute_model_req)

    def _execute_model_tp1(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        logger.info("SingleTPWorker.execute_model_prefill()")

        if execute_model_req is None:
            seq_group_metadata_list = None
        else:
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        if not self._worker.is_driver():
            logger.info("Draft worker returns []")
            return []

        assert seq_group_metadata_list is not None
        assert execute_model_req is not None
        num_seq_groups = len(seq_group_metadata_list)
        blocks_to_swap_in = execute_model_req.blocks_to_swap_in
        blocks_to_swap_out = execute_model_req.blocks_to_swap_out
        blocks_to_copy = execute_model_req.blocks_to_copy

        self._worker.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        logger.info("SingleTPWorker._worker.model_runner.execute_model()")
        output = self._worker.model_runner.execute_model(seq_group_metadata_list,
                                                self._worker.gpu_cache)

        logger.info("SingleTPWorker.execute_model_prefill() output:")
        if output is not None:
            for seq_group_output in output.outputs:
                for sample in seq_group_output.samples:
                    logger.info(f"SamplerOutput: {sample}")

        # Worker only supports single-step execution. Wrap the output in a list
        # to conform to interface.
        return [output]


    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        return self._worker.get_cache_block_size_bytes()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    @property
    def max_model_len(self) -> int:
        return self._worker.max_model_len

    @property
    def vocab_size(self) -> int:
        return self._worker.vocab_size

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        """MultiStepWorker does not yet implement support for cache swap
        operations or beam search.
        """
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "MultiStepWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "MultiStepWorker does not support beam search.")

    def _shallow_copy_inputs(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> List[SequenceGroupMetadata]:
        """Copy input data structures to remove side-effects when input data
        structures are shared with other modules.

        Helpful when the vLLM scheduler runs in the same process as the worker.
        The alternative is deep-copying (or other form of deep copy); this has
        performance downsides.
        """

        # Shallow-copy the list of SequenceGroupMetadata. This allows us to
        # append tokens and change is_prompt without external side-effects.
        new_seq_group_metadata_list = []

        for old_seq_group_metadata in seq_group_metadata_list:
            # We must shallow-copy seq_group_metadata as is_prompt could change.
            seq_group_metadata = copy.copy(old_seq_group_metadata)
            new_seq_group_metadata_list.append(seq_group_metadata)

            # We must shallow-copy seq_data as we will append token ids
            new_seq_data = {}
            for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
                new_seq_data[seq_id] = copy.copy(old_seq_data)
                new_seq_data[
                    seq_id].output_token_ids = old_seq_data.output_token_ids[:]

            seq_group_metadata.seq_data = new_seq_data

        return new_seq_group_metadata_list

    def _assert_enough_kv_space(
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            num_steps: int) -> None:
        """Assert there are enough physical blocks per sequence to store the
        current KV plus additional KV from num_steps tokens.
        """
        assert self._worker.model_runner.block_size is not None
        for seq_group_metadata in seq_group_metadata_list:
            # Only one seq_id is guaranteed because there is no beam search.
            seq_id = list(seq_group_metadata.seq_data.keys())[0]
            seq = seq_group_metadata.seq_data[seq_id]

            # After num_steps, the seq len will be the current seq len
            # plus one token per step.
            final_seq_len = seq.get_len() + num_steps

            # We will have final_seq_len - 1 KV because vLLM saves KV for a
            # token in the iteration after the token was generated.
            required_num_kv_slots = final_seq_len - 1

            # The allocated number of kv slots is the number of allocated blocks
            # times the number of slots of block.
            number_physical_blocks = len(
                seq_group_metadata.block_tables[seq_id])
            allocated_kv_slots = (number_physical_blocks *
                                  self._worker.model_runner.block_size)

            if required_num_kv_slots > allocated_kv_slots:
                request_id = seq_group_metadata.request_id
                raise ValueError(
                    "The worker attempted to run "
                    f"{num_steps} times but found insufficient KV space for "
                    f"{request_id=} {seq_id=}. ({allocated_kv_slots=} "
                    f"{required_num_kv_slots=}).")

    def _append_new_tokens(
            self, model_output: SamplerOutput,
            seq_group_metadata_list: SequenceGroupMetadata) -> None:
        """Given model output from a single run, append the tokens to the
        sequences. This is normally done outside of the worker, but it is
        required if the worker is to perform multiple forward passes.
        """
        for seq_group_metadata, sequence_group_outputs in zip(
                seq_group_metadata_list, model_output):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                # NOTE: Beam search is not supported, so we can assume that
                # parent_seq_id == seq_id.
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]

                seq.append_token_id(token_id, token_logprob.logprob)
