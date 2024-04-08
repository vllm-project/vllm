from vllm.sequence import SequenceGroupOutput, SamplerOutput
from typing import List

def create_output_by_sequence_group(sampler_outputs: List[SamplerOutput], num_seq_groups: int):
    output_by_sequence_group = [ 
        [] for _ in range(num_seq_groups)
    ]
    for step in sampler_outputs:
        for i, sequence_group_output in enumerate(step):
            output_by_sequence_group[i].append(sequence_group_output) 

    return output_by_sequence_group
