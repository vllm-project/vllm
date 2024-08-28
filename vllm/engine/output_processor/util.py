from typing import List
from typing import Sequence as GenericSequence
from typing import Union

from vllm.sequence import PoolerOutput, SamplerOutput, SequenceGroupOutput

def create_output_by_sequence_group(
        outputs: GenericSequence[Union[SamplerOutput, PoolerOutput]],
        num_seq_groups: int,
        return_hidden_states: bool = False) -> List[List[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].
    """
    output_by_sequence_group: List[List[SequenceGroupOutput]] = [
        [] for _ in range(num_seq_groups)
    ]
    for step in outputs:
        for i, sequence_group_output in enumerate(step):
            if return_hidden_states and isinstance(step, SamplerOutput):
                sequence_group_output.prompt_hidden_states = step.prefill_hidden_states
                # `SamplerOutput.hidden_states` are of the shape [n_seqs, hidden_size].
                sequence_group_output.hidden_state = step.hidden_states[i, :]

            output_by_sequence_group[i].append(sequence_group_output)

    return output_by_sequence_group
