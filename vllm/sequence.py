"""Sequence and its related classes."""
import copy
import enum
from typing import Dict, List, Optional, Union

from vllm.block import LogicalTokenBlock
from vllm.sampling_params import SamplingParams


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
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceData:
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")


class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size

        self.data = SequenceData(prompt_token_ids)
        self.output_logprobs: List[Dict[int, float]] = []
        self.output_tokens: List[str] = []
        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prompt_token_ids)
        self.status = SequenceStatus.WAITING

    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        while token_ids:
            if not self.logical_token_blocks:
                self._append_logical_block()

            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[:num_empty_slots])
            token_ids = token_ids[num_empty_slots:]

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id])

    def get_len(self) -> int:
        return self.data.get_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, child_seq: "Sequence") -> None:
        child_seq.logical_token_blocks = copy.deepcopy(
            self.logical_token_blocks)
        child_seq.output_logprobs = copy.deepcopy(self.output_logprobs)
        child_seq.data = copy.deepcopy(self.data)

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
    ) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return self.seqs
        else:
            return [seq for seq in self.seqs if seq.status == status]

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    def find(self, seq_id: int) -> Sequence:
        for seq in self.seqs:
            if seq.seq_id == seq_id:
                return seq
        raise ValueError(f"Sequence {seq_id} not found.")

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.seqs)

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs)})")


class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `InputMetadata`.


    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
    """

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: Dict[int, SequenceData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables


class SequenceOutputs:
    """The model output associated with a sequence.

    Args:
        seq_id: The ID of the sequence.
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
        self,
        seq_id: int,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, float],
    ) -> None:
        self.seq_id = seq_id
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SequenceOutputs(seq_id={self.seq_id}, "
                f"parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}), "
                f"logprobs={self.logprobs}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutputs):
            return NotImplemented
        return (self.seq_id == other.seq_id
                and self.parent_seq_id == other.parent_seq_id
                and self.output_token == other.output_token
                and self.logprobs == other.logprobs)
