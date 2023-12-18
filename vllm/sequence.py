"""Sequence and its related classes."""
import copy
import enum
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
import torch

import msgspec

from vllm.block import LogicalTokenBlock
from vllm.anyscale.lora.utils import LoRARequest
from vllm.sampling_params import SamplingParams

PromptLogprobs = List[Optional[Dict[int, float]]]
SampleLogprobs = List[Dict[int, float]]


class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceData(msgspec.Struct, array_like=True, omit_defaults=True):
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        token_ids: The token IDs so far (prompt+output).
        num_prompt_tokens: The number of prompt tokens. If not specified,
            will be set to the length of token_ids.
        cumulative_logprob: The cumulative log probability of the output.
        num_processed_token_ids: The number of token ids that have been
            processed by the workers.
        prefill_start: The start index of the chunked prefill.
        prefill_end: The end index of the chunked prefill.
    """

    token_ids: List[int]
    num_prompt_tokens: int = -1
    cumulative_logprob: float = 0.0
    prefill_start: int = 0
    prefill_end: int = 0
    _prompt_token_id_count: Optional[Dict[int, int]] = None
    _output_token_id_count: Optional[Dict[int, int]] = None

    # The number of tokens that have been processed.
    # Processed means that the KV for the tokens has been computed and stored
    # to the KV cache.
    num_processed_token_ids: int = 0

    @property
    def prompt_token_id_count(self) -> Counter[int, int]:
        if self._prompt_token_id_count is None:
            self._prompt_token_id_count = Counter(self.get_prompt_token_ids())
        return self._prompt_token_id_count

    @property
    def output_token_id_count(self) -> Counter[int, int]:
        if self._output_token_id_count is None:
            self._output_token_id_count = Counter(self.get_output_token_ids())
        return self._output_token_id_count

    def __post_init__(self):
        if self.num_prompt_tokens < 0:
            self.num_prompt_tokens = len(self.token_ids)

        if (self.num_processed_token_ids >
                self.get_len()) or (self.num_processed_token_ids < 0):
            raise ValueError(f"{self.num_processed_token_ids=} must be in the "
                             "interval [0, {self.get_len()=}]")

    def append_token_ids(self, token_ids: List[int],
                         logprobs: List[float]) -> None:
        """Append token ids to the output token ids and update the cumulative
        logprob. Also updates the number of processed token ids to the sequence
        length before the new tokens.
        """
        self.num_processed_token_ids = self.get_len()

        self.token_ids.extend(token_ids)
        self.cumulative_logprob += sum(logprobs)

        for token_id in token_ids:
            self.output_token_id_count[
                token_id] = self.output_token_id_count.get(token_id, 0) + 1

    def reset_processed_tokens(self) -> None:
        """Set the number of processed tokens to zero. Used when a sequence is
        preempted by recomputation. This reset the prefill range as well.
        """
        self.num_processed_token_ids = 0
        self.prefill_start = 0
        self.prefill_end = 0

    def get_num_processed_token_ids(self) -> int:
        return self.num_processed_token_ids

    def get_unprocessed_token_ids(self) -> List[int]:
        return self.token_ids[self.get_unprocessed_token_start_idx():]

    def get_unprocessed_token_start_idx(self) -> int:
        seq_len = self.get_len()
        num_unprocessed_token_ids = seq_len - self.num_processed_token_ids
        return seq_len - num_unprocessed_token_ids

    def get_unprocessed_token_positions(self) -> List[int]:
        return list(
            range(self.get_unprocessed_token_start_idx(), self.get_len()))

    def get_len(self) -> int:
        return len(self.token_ids)

    def get_prompt_len(self) -> int:
        return self.num_prompt_tokens

    def get_prompt_token_ids(self) -> int:
        return self.token_ids[:self.num_prompt_tokens]

    def get_output_len(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    def get_token_ids(self) -> List[int]:
        return self.token_ids

    def advance_prefill_range(self, prefill_range: int) -> int:
        """Advance the prefill range by the specified amount

        Args:
            prefill_range: The amount to advance the prefill range.
        Returns:
            The actual number of advanced tokens.
        """
        self.prefill_start = self.prefill_end
        # The increased range could be larger than the seq length.
        # Clamp it to the seq length.
        # Note that we use prompt_len + output_len instead of
        # prompt_len here. This is because during recompute
        # we need to prefill for both prompt and output.
        self.prefill_end = min(self.prefill_end + prefill_range,
                               self.get_len())
        return self.prefill_end - self.prefill_start

    def get_prefill_range(self) -> Tuple[int, int]:
        """Returns the prefill range."""
        return self.prefill_start, self.prefill_end

    def get_num_unprefilled(self) -> int:
        return self.get_len() - self.prefill_end

    def get_output_token_ids(self) -> List[int]:
        return self.token_ids[self.num_prompt_tokens:]

    def get_token_id(self, index: int) -> int:
        return self.token_ids[index]

    def get_new_token_ids(self) -> List[int]:
        return self.get_unprocessed_token_ids()

    def get_last_token_id(self) -> int:
        return self.token_ids[-1]

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.get_prompt_token_ids()}, "
                f"output_token_ids={self.get_output_token_ids()}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"num_processed_token_ids={self.num_processed_token_ids}), "
                f"prompt_token_id_count={self.prompt_token_id_count}, "
                f"output_token_id_count={self.output_token_id_count}")


class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
        num_processed_tokens: The number of prompt tokens to be considered
            processed by SequenceData.
    """

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
        lora_request: Optional[LoRARequest] = None,
        num_processed_token_ids: int = 0,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size
        self.lora_request = lora_request

        self.data = SequenceData(
            prompt_token_ids, num_processed_token_ids=num_processed_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = []

        # Keep track of the first logical block index that has empty slots.
        # Used to determine which block new tokens should be appended to.
        self.block_index_for_new_tokens = 0

        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prompt_token_ids)
        self.status = SequenceStatus.WAITING

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        cursor = 0
        while cursor < len(token_ids):
            if not self.logical_token_blocks:
                self._append_logical_block()

            last_block = self.logical_token_blocks[
                self.block_index_for_new_tokens]
            if last_block.is_full():
                self._append_logical_block()
                self.block_index_for_new_tokens += 1
                last_block = self.logical_token_blocks[
                    self.block_index_for_new_tokens]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[cursor:cursor +
                                               num_empty_slots])
            cursor += num_empty_slots

    def ensure_num_empty_slots(self, num_desired_empty_slots: int) -> None:
        """Ensure the specified number of empty slots are present in the logical
        token blocks, allocating additional blocks if necessary.
        """
        if not self.logical_token_blocks:
            self._append_logical_block()

        num_empty_slots = sum(
            block.get_num_empty_slots() for block in
            self.logical_token_blocks[self.block_index_for_new_tokens:])
        num_empty_remaining = num_desired_empty_slots - num_empty_slots

        while num_empty_remaining > 0:
            self._append_logical_block()
            last_block = self.logical_token_blocks[-1]
            num_empty_remaining -= last_block.get_num_empty_slots()

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        return self.append_token_ids([token_id], [logprobs])

    def append_token_ids(
        self,
        token_ids: List[int],
        logprobs: List[Dict[int, float]],
    ) -> None:
        self._append_tokens_to_blocks(token_ids)
        self.output_logprobs.extend(logprobs)
        self.data.append_token_ids(token_ids, [
            logprob[token_id]
            for logprob, token_id in zip(logprobs, token_ids)
        ])

    def reset_processed_tokens(self):
        self.data.reset_processed_tokens()

    def get_num_processed_token_ids(self) -> int:
        return self.data.get_num_processed_token_ids()

    def get_num_unprocessed_token_ids(self) -> int:
        return self.get_len() - self.get_num_processed_token_ids()

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_new_token_ids(self) -> List[int]:
        return self.data.get_new_token_ids()

    def get_output_token_ids(self) -> List[int]:
        return self.data.get_output_token_ids()

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    # def get_beam_search_score(self,
    #                           length_penalty: float = 0.0,
    #                           seq_len: Optional[int] = None,
    #                           eos_token_id: Optional[int] = None) -> float:
    #     """Calculate the beam search score with length penalty.

    #     Adapted from

    #     https://github.com/huggingface/transformers/blob/ccb92be23def445f2af
    # dea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    #     """
    #     if seq_len is None:
    #         seq_len = self.get_len()
    #         # NOTE: HF implementation does not count the EOS token
    #         # towards the length, we align with that here for testing.
    #         if (eos_token_id is not None
    #                 and self.get_last_token_id() == eos_token_id):
    #             seq_len -= 1
    #     return self.get_cumulative_logprob() / (seq_len**length_penalty)

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float,
        arrival_time_perf_counter: float,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.arrival_time_perf_counter = arrival_time_perf_counter
        self.lora_request = lora_request
        self.first_scheduled_time = None
        self.first_token_time = None
        self.time_in_queue = None
        self.prompt_logprobs: Optional[PromptLogprobs] = None

    @property
    def prompt(self) -> str:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).data.get_prompt_token_ids()

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.sampling_params.use_beam_search:
            # For beam search, maximally there will always be `best_of` beam
            # candidates running in the future.
            return self.sampling_params.actual_best_of
        else:
            if self.sampling_params.actual_best_of > self.num_seqs():
                # At prompt stage, the sequence group is not yet filled up
                # and only have one sequence running. However, in the
                # generation stage, we will have `best_of` sequences running.
                return self.sampling_params.actual_best_of
            # At sampling stages, return the number of actual sequences
            # that are not finished yet.
            return self.num_unfinished_seqs()

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return list(self.seqs_dict.values())
        else:
            return [
                seq for seq in self.seqs_dict.values() if seq.status == status
            ]

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [
            seq for seq in self.seqs_dict.values() if not seq.is_finished()
        ]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]

    def advance_prefill_range(self, size: int) -> int:
        """Advance the prefill range by the specified amount.

        Args:
            size: The amount to advance the prefill range.
        Returns:
            The actual number of advanced tokens.
        """
        return [
            seq.data.advance_prefill_range(size)
            for seq in self.seqs_dict.values()
        ][0]

    def get_num_unprefilled(self) -> int:
        # All sequences in the group should have the same prompt.
        return list(self.seqs_dict.values())[0].data.get_num_unprefilled()

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.get_seqs())

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs_dict)})")


class SequenceGroupMetadataDelta(msgspec.Struct,
                                 tag=True,
                                 array_like=True,
                                 omit_defaults=True):
    request_id: str
    block_tables: Optional[Dict[int, List[int]]]

    @property
    def is_prompt(self):
        return False

    @property
    def is_chunked_prefill(self):
        # A Delta should always be decoding (not chunk-prefiling).
        return False


class SequenceGroupMetadata(msgspec.Struct,
                            tag=True,
                            array_like=True,
                            omit_defaults=True):
    """Metadata for a sequence group. Used to create `InputMetadata`.


    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        is_chunked_prefill: Whether the request is at chunked prefill stage.
            Note that chunked_prefill is also a prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
    """

    request_id: str
    is_chunked_prefill: bool
    is_prompt: bool
    seq_data: Dict[int, SequenceData]
    sampling_params: SamplingParams
    block_tables: Optional[Dict[int, List[int]]]
    lora_request: Optional[LoRARequest]

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def update_from_delta(self, delta: "SequenceGroupMetadataDelta"):
        self.block_tables = delta.block_tables
        self.is_prompt = delta.is_prompt
        return self


class SequenceOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    parent_seq_id: int
    output_token: int
    logprobs: Dict[int, float]

    def __repr__(self) -> str:
        return (f"SequenceOutputs(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutputs):
            raise NotImplementedError()
        return (self.parent_seq_id == other.parent_seq_id
                and self.output_token == other.output_token
                and self.logprobs == other.logprobs)


class SequenceGroupOutputs(msgspec.Struct, array_like=True,
                           omit_defaults=True):
    """The model outputs associated with a sequence group."""

    samples: List[SequenceOutputs]
    prompt_logprobs: Optional[PromptLogprobs]

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, self.__class__)
                and self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


class SamplerOutput(msgspec.Struct, array_like=True, omit_defaults=True):
    outputs: List[SequenceGroupOutputs]

    # Used to store an on-GPU tensor containing the batch probabilities.
    probs: Optional[torch.Tensor] = None

    # Used to store an on-GPU tensor containing the sampled token ids.
    sampled_tokens: Optional[torch.Tensor] = None

    # Used to store an on-CPU tensor containing the batch logits
    # for the full sequence.
    logits: Optional[torch.Tensor] = None

    draft_target_worker_metrics: Optional["DraftTargetWorkerMetrics"] = None

    def __getitem__(self, idx: int):
        return self.outputs[idx]

    def __setitem__(self, idx: int, value):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other,
                          self.__class__) and self.outputs == other.outputs


class DraftTargetWorkerMetrics(msgspec.Struct,
                               array_like=True,
                               omit_defaults=True):
    num_spec_tokens: int
    draft_acceptance_rate: float
    system_efficiency: float
    accepted_tokens: int
    draft_tokens: int
    emitted_tokens: int

    def __repr__(self) -> str:
        return (
            f"DraftTargetWorkerMetrics(num_spec_tokens={self.num_spec_tokens},"
            f"draft_acceptance_rate={self.draft_acceptance_rate:.3f}, "
            f"system_efficiency={self.system_efficiency:.3f}, "
            f"accepted_tokens={self.accepted_tokens}, "
            f"draft_tokens={self.draft_tokens}, "
            f"emitted_tokens={self.emitted_tokens})")


class ExecuteModelData(msgspec.Struct, array_like=True, omit_defaults=True):
    seq_group_metadata_list: List[Union[SequenceGroupMetadata,
                                        SequenceGroupMetadataDelta]]
    finished_request_ids_list: List[str]
    blocks_to_swap_in: Dict[int, int]
    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
    num_preallocated_slots: int
    return_logits: bool = False
