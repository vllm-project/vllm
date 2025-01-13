from typing import Sequence as GenericSequence
from typing import cast

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import CompletionSequenceGroupOutput, SequenceGroupOutput


def create_output_by_sequence_group(
        outputs: GenericSequence[SamplerOutput],
        num_seq_groups: int) -> list[list[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].
    """
    output_by_sequence_group: list[list[CompletionSequenceGroupOutput]] = [
        [] for _ in range(num_seq_groups)
    ]
    for step in outputs:
        sequence_group_output: CompletionSequenceGroupOutput
        for i, sequence_group_output in enumerate(step):
            output_by_sequence_group[i].append(sequence_group_output)

    # Cast to the more generic type that CompletionSequenceGroupOutput
    # inherits from.
    return cast(list[list[SequenceGroupOutput]], output_by_sequence_group)
