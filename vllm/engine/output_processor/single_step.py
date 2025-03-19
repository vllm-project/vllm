# SPDX-License-Identifier: Apache-2.0

from typing import List

from vllm.config import SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.sequence import (CompletionSequenceGroupOutput, SequenceGroup,
                           SequenceGroupOutput)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

logger = init_logger(__name__)


def single_step_process_prompt_logprob(
        sg_output_proc: SequenceGroupOutputProcessor, seq_group: SequenceGroup,
        output: CompletionSequenceGroupOutput) -> None:
    """Process prompt logprobs associated with the :class:`SequenceGroupOutput`
    for a given step.

    Do nothing if the output has no prompt logprobs.

    Account for the fact that transformers do not compute first-token logprobs.
    
    Args:
      sg_output_proc: :class:`SequenceGroupOutputProcessor` instance
      seq_group: the output is associated with this :class:`SequenceGroup`
      output: the :class:`SequenceGroupOutput` for a single scheduler step
    """
    prompt_logprobs = output.prompt_logprobs

    # If this is the first (or only) "chunk" of the prefill, we need
    # to prepend None to the list of prompt logprobs. The reason for this
    # is that for N prompt tokens, the Sampler will generate N-1 total
    # prompt logprobs during prefill since the token at idx 0 will not
    # have a logprob associated with it.
    if prompt_logprobs is not None:
        if not seq_group.prompt_logprobs:
            prompt_logprobs = [None] + prompt_logprobs
            seq_group.prompt_logprobs = []

        assert hasattr(sg_output_proc, 'detokenizer')
        if (seq_group.sampling_params.detokenize
                and sg_output_proc.detokenizer):
            sg_output_proc.detokenizer.decode_prompt_logprobs_inplace(
                seq_group,
                prompt_logprobs,
                position_offset=len(seq_group.prompt_logprobs))

        seq_group.prompt_logprobs.extend(prompt_logprobs)


class SingleStepOutputProcessor(SequenceGroupOutputProcessor):
    """SequenceGroupOutputProcessor which handles "output processing" logic,
    which happens after the model returns generated token ids and before
    scheduling of the next batch. Output processing logic includes
    detokenization, and determining if a sequence is finished (e.g. via max len
    or eos token).

    The SingleStepOutputProcessor is specialized to the case where the model
    emits at most a single token per invocation, which precludes configurations
    such as speculative decoding or multi-step decoding. This enables beam
    search sampling, which requires forking/finishing/freeing sequences in a way
    that is currently difficult to schedule multiple steps ahead of time.
    """

    def __init__(self, scheduler_config: SchedulerConfig,
                 detokenizer: Detokenizer, scheduler: List[Scheduler],
                 seq_counter: Counter, stop_checker: StopChecker):
        self.scheduler_config = scheduler_config
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.stop_checker = stop_checker

    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput],
                        is_async: bool) -> None:
        """Append all new tokens to sequences in the sequence group. Fork any
        surviving beam candidates; free any unsurviving ones.

        Invokes detokenizer to detokenize new tokens, and also marks sequences
        as finished if they meet stop conditions.
        
        is_async - Indicates whether this postprocessor runs in 
            parallel with the GPU forward pass and is processing 
            tokens from the previous step. If this is true, then
            no tokens need to be appended since it is already done
            externally (before the next schedule() call)
        """
        assert (len(outputs) == 1
                ), f"{type(self)} does not support multiple outputs per step"
        return self._process_sequence_group_outputs(sequence_group, outputs[0],
                                                    is_async)

    def process_prompt_logprob(self, seq_group: SequenceGroup,
                               outputs: List[SequenceGroupOutput]) -> None:
        """Process prompt logprobs associated with one step of a single-step-
        scheduled computation.
        
        Args:
          seq_group: the output is associated with this :class:`SequenceGroup`
          outputs: the :class:`SequenceGroupOutput` for a single scheduler step
        """
        assert len(outputs) == 1, "Single step should only have 1 output."
        output = outputs[0]
        assert isinstance(output, CompletionSequenceGroupOutput)
        single_step_process_prompt_logprob(self, seq_group, output)

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput,
                                        is_async: bool) -> None:
        sampling_params = seq_group.sampling_params

        sample = outputs.samples[0]
        seq = seq_group.first_seq
        if not is_async:
            seq.append_token_id(sample.output_token, sample.logprobs)
        if sampling_params.detokenize and self.detokenizer:
            new_char_count = self.detokenizer.decode_sequence_inplace(
                seq, sampling_params)
        else:
            new_char_count = 0
        self.stop_checker.maybe_stop_sequence(
            seq,
            new_char_count,
            sampling_params,
            lora_req=seq_group.lora_request,
        )
        if seq.is_finished():
            for scheduler in self.scheduler:
                scheduler.free_seq(seq)
