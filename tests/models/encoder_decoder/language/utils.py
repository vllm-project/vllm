from typing import List, Optional, Tuple

from vllm.sequence import SampleLogprobs

from ....conftest import DecoderPromptType


def vllm_to_hf_output(
    vllm_output: Tuple[List[int], str, Optional[SampleLogprobs]],
    decoder_prompt_type: DecoderPromptType,
):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "</s>"
    if decoder_prompt_type == DecoderPromptType.NONE:
        hf_output_str = "<s>" + hf_output_str

    return output_ids, hf_output_str, out_logprobs
