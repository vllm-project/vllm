from typing import List

from vllm.sequence import SamplerOutput, SequenceGroupOutput


def create_output_by_sequence_group(
        sampler_outputs: List[SamplerOutput],
        num_seq_groups: int) -> List[List[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].
    """
    output_by_sequence_group: List[List[SamplerOutput]] = [
        [] for _ in range(num_seq_groups)
    ]
    for step in sampler_outputs:
        for i, sequence_group_output in enumerate(step):
            output_by_sequence_group[i].append(sequence_group_output)

    return output_by_sequence_group
