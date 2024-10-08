from typing import List
from typing import Sequence as GenericSequence
from typing import Union

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import PoolerOutput, SequenceGroupOutput


def create_output_by_sequence_group(
        outputs: GenericSequence[Union[SamplerOutput, PoolerOutput]],
        num_seq_groups: int) -> List[List[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].
    """
    output_by_sequence_group: List[List[SequenceGroupOutput]] = [
        [] for _ in range(num_seq_groups)
    ]
    for step in outputs:
        for i, sequence_group_output in enumerate(step):
            output_by_sequence_group[i].append(sequence_group_output)

    return output_by_sequence_group
