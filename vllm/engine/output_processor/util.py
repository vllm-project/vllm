from typing import List
from typing import Sequence as GenericSequence
from typing import Union

from vllm.core.scheduler import ScheduledSequenceGroup
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import PoolerOutput, SequenceGroupOutput


def create_output_by_sequence_group(
        outputs: GenericSequence[Union[SamplerOutput, PoolerOutput]],
        scheduled_seq_groups: List[ScheduledSequenceGroup],
        return_hidden_states: bool = False) -> List[List[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].

    Also passes hidden states from the 
    `SamplerOutputs` to the `SequenceGroupOutputs`
    if `return_hidden_states` is `True`.
    """
    output_by_sequence_group: List[List[SequenceGroupOutput]] = [
        [] for _ in scheduled_seq_groups
    ]
    input_lengths: List[int] = []
    for step in outputs:
        for i, sequence_group_output in enumerate(step):
            if return_hidden_states and isinstance(step, SamplerOutput):
                assert len(scheduled_seq_groups[i].seq_group.seqs) == 1
                if step.prefill_hidden_states is not None:
                    # Prefill tokens are concatenated in the order that their
                    # sequence group is scheduled.
                    seq_group_offset = sum(input_lengths[:i])
                    seq_group_input_length = len(
                        scheduled_seq_groups[i].seq_group.seqs[0].
                        inputs['prompt_token_ids'])
                    seq_group_end = seq_group_offset + seq_group_input_length
                    sequence_group_output.prompt_hidden_states = (
                        step.prefill_hidden_states[
                            seq_group_offset:seq_group_end].clone().cpu())
                    input_lengths.append(seq_group_input_length)
                # `SamplerOutput.hidden_states` are shape [n_seqs, hidden_size].
                sequence_group_output.hidden_state = (
                    step.hidden_states[i, :].clone().cpu().unsqueeze(0))

            output_by_sequence_group[i].append(sequence_group_output)

    return output_by_sequence_group
