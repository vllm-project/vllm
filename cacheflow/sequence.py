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


class Sequence:

    def __init__(
        self,
        seq_id: int,
        token_ids: List[int],
        block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.block_size = block_size

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the given token ids.
        self.add(token_ids)

        self.prompt_len = len(token_ids)
        self.status = SequenceStatus.WAITING
        self.output_logprobs: List[Dict[int, float]] = []
        self.cumulative_logprobs = 0.0

    def add_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def add(self, token_ids: List[int]) -> None:
        while token_ids:
            if not self.logical_token_blocks:
                self.add_block()

            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self.add_block()
                last_block = self.logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append(token_ids[:num_empty_slots])
            token_ids = token_ids[num_empty_slots:]

    def append(self, token_id: int, logprobs: Dict[int, float]) -> None:
        assert token_id in logprobs
        self.add([token_id])
        self.output_logprobs.append(logprobs)
        self.cumulative_logprobs += logprobs[token_id]

    def get_len(self) -> int:
        return sum(block.num_tokens for block in self.logical_token_blocks)

    def get_token_ids(self) -> List[int]:
        token_ids: List[int] = []
        for block in self.logical_token_blocks:
            token_ids.extend(block.get_token_ids())
        return token_ids

    def get_last_token_id(self) -> int:
        return self.logical_token_blocks[-1].get_last_token_id()

    def fork(self, child_seq: 'Sequence') -> 'Sequence':
        child_seq.logical_token_blocks = copy.deepcopy(self.logical_token_blocks)
        child_seq.output_logprobs = copy.deepcopy(self.output_logprobs)
        child_seq.cumulative_logprobs = self.cumulative_logprobs

    def __repr__(self) -> str:
        return (f'Sequence(seq_id={self.seq_id}, '
                f'status={self.status.name}, '
                f'num_blocks={len(self.logical_token_blocks)})')


class SequenceGroup:

    def __init__(
        self,
        group_id: int,
        seqs: List[Sequence],
        arrival_time: float,
    ) -> None:
        self.group_id = group_id
        self.seqs = seqs
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
        return (f'SequenceGroup(group_id={self.group_id}, '
                f'num_seqs={len(self.seqs)})')


class SequenceGroupInputs:

    def __init__(
        self,
        group_id: int,
        is_prompt: bool,
        input_tokens: Dict[int, List[int]],     # Seq id -> token ids.
        context_len: int,
        seq_logprobs: Dict[int, float],         # Seq id -> cumulative logprobs.
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],     # Seq id -> List of physical block numbers.
    ) -> None:
        self.group_id = group_id
        self.is_prompt = is_prompt
        self.input_tokens = input_tokens
        self.context_len = context_len
        self.seq_logprobs = seq_logprobs
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
