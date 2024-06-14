import copy
import weakref
from typing import List, Tuple, Optional

import torch

from vllm.distributed.parallel_state import (_ENABLE_CUSTOM_ALL_REDUCE,
                                             GroupCoordinator, get_tp_group,
                                             get_world_group,
                                             patch_tensor_parallel_group)
from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker import Worker
from vllm.logger import init_logger

logger = init_logger(__name__)


class MultiStepWorker(Worker, ProposerWorkerBase):
    """The MultiStepWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less.

    The MultiStepWorker does not support cache swap operations, or beam search.
    Cache swap operations do not require large modifications. On the other hand,
    beam search requires memory allocations during sequence forks and thus
    requires more thought for MultiStepWorker support.
    """

    def __init__(self, draft_ranks: Optional[List[int]], **kwargs):
        """Create a MultiStepWorker.

        Args:
            draft_ranks (Optional[List[int]]): if this value is given, only some of
             the GPU ranks written in this value participaten in draft generation
        """
        rank = kwargs['rank']
        self.is_dummy = False
        self._draft_ranks = draft_ranks
        if draft_ranks is not None:
            self.is_dummy = rank not in draft_ranks
            self._tp_groups = None
            logger.info(f"{self._draft_ranks=}, {self._tp_groups=}")
        logger.info(f"{rank=}, {draft_ranks=}, {self.is_dummy=}")

        super().__init__(**kwargs)

        # Lazy initialization list.
        self._proposer: SpeculativeProposer

    def _patch_tensor_parallel_group(self):
        if self._tp_groups is not None:
            return patch_tensor_parallel_group(self._tp_groups[0],
                                               self._tp_groups[1])
        return None

    def init_device(self):
        if self.is_dummy:
            return

        if self._draft_ranks:
            local_rank = get_world_group().local_rank
            world_backend = torch.distributed.get_backend(
                get_world_group().device_group)
            tp_backend = torch.distributed.get_backend(get_tp_group().device_group)

            world_group = GroupCoordinator(
                group_ranks=[self._draft_ranks],
                local_rank=local_rank,
                torch_distributed_backend=world_backend,
                use_pynccl=False,
                use_custom_allreduce=False,
            )
            tp_group = GroupCoordinator(
                group_ranks=[self._draft_ranks],
                local_rank=local_rank,
                torch_distributed_backend=tp_backend,
                use_pynccl=True,
                use_custom_allreduce=_ENABLE_CUSTOM_ALL_REDUCE,
            )
            self._tp_groups = world_group, tp_group

        with self._patch_tensor_parallel_group():
            super().init_device()

        self._proposer = Top1Proposer(
            weakref.proxy(self),  # type: ignore[arg-type]
            self.device,
            self.vocab_size,
            max_proposal_len=self.max_model_len,
        )

    def set_include_gpu_probs_tensor(self):
        # Need include_gpu_probs_tensor for multi_step_worker
        self.model_runner.model.sampler.include_gpu_probs_tensor = True

    def load_model(self):
        if not self.is_dummy:
            with self._patch_tensor_parallel_group():
                super().load_model()

    def determine_num_available_blocks(self):
        if not self.is_dummy:
            with self._patch_tensor_parallel_group():
                return super().determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int):
        if not self.is_dummy:
            with self._patch_tensor_parallel_group():
                super().initialize_cache(num_gpu_blocks, num_cpu_blocks)

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.

        For multi step worker, this indicator shall be True.
        """
        if not self.is_dummy:
            return [], True

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
        model_outputs: List[SamplerOutput] = []
        for _ in range(sample_len):
            model_output: List[SamplerOutput] = super().execute_model(
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
        execute_model_req: ExecuteModelRequest,
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """
        if not self.is_dummy:
            return SpeculativeProposals(None, None, None)

        with self._patch_tensor_parallel_group():
            return self._proposer.get_spec_proposals(execute_model_req)

    @staticmethod
    def _append_new_tokens(
            model_output: List[SamplerOutput],
            seq_group_metadata_list: List[SequenceGroupMetadata]) -> None:
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
                seq.update_num_computed_tokens(1)

    @staticmethod
    def _shallow_copy_inputs(
        seq_group_metadata_list: List[SequenceGroupMetadata]
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
        assert self.model_runner.block_size is not None
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
                                  self.model_runner.block_size)

            if required_num_kv_slots > allocated_kv_slots:
                request_id = seq_group_metadata.request_id
                raise ValueError(
                    "The worker attempted to run "
                    f"{num_steps} times but found insufficient KV space for "
                    f"{request_id=} {seq_id=}. ({allocated_kv_slots=} "
                    f"{required_num_kv_slots=}).")

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

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        if self.is_dummy:
            return []

        with self._patch_tensor_parallel_group():
            return super().execute_model(execute_model_req)

    def get_cache_block_size_bytes(self) -> int:
        if self.is_dummy:
            return 0

        return super().get_cache_block_size_bytes()
