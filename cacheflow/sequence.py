import copy
import enum
from typing import Dict, List, Optional

from cacheflow.block import LogicalTokenBlock
from cacheflow.sampling_params import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED = enum.auto()


class SequenceData:

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids

        self.output_token_ids: List[int] = []
        self.cumulative_logprobs = 0.0

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt={self.prompt}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids})")


class Sequence:

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

    def append_token(self, token_id: int, logprobs: Dict[int, float]) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.output_token_ids.append(token_id)
        self.data.cumulative_logprobs += logprobs[token_id]

    def get_len(self) -> int:
        return self.data.get_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def fork(self, child_seq: 'Sequence') -> 'Sequence':
        child_seq.logical_token_blocks = copy.deepcopy(self.logical_token_blocks)
        child_seq.output_logprobs = copy.deepcopy(self.output_logprobs)
        child_seq.data = copy.deepcopy(self.data)

    def __repr__(self) -> str:
        return (f'Sequence(seq_id={self.seq_id}, '
                f'status={self.status.name}, '
                f'num_blocks={len(self.logical_token_blocks)})')


class SequenceGroup:

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
        raise ValueError(f'Sequence {seq_id} not found.')

    def is_finished(self) -> bool:
        return all(seq.status == SequenceStatus.FINISHED for seq in self.seqs)

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs)})")


class SequenceGroupMetadata:

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: Dict[int, SequenceData],      # Seq id -> sequence data.
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],     # Seq id -> list of physical block numbers.
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables


class SequenceOutputs:

    def __init__(
        self,
        seq_id: int,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, float],         # Token id -> logP(x_i+1 | x_0, ..., x_i).
    ) -> None:
        self.seq_id = seq_id
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f'SequenceOutputs(seq_id={self.seq_id}, '
                f'parent_seq_id={self.parent_seq_id}, '
                f'output_token={self.output_token}), '
                f'logprobs={self.logprobs}')

    def __eq__(self, other: 'SequenceOutputs') -> bool:
        return (self.seq_id == other.seq_id and
                self.parent_seq_id == other.parent_seq_id and
                self.output_token == other.output_token and
                self.logprobs == other.logprobs)
