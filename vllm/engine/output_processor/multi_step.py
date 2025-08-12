from typing import Callable, List

from transformers import PreTrainedTokenizer

from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Logprob, Sequence, SequenceGroup,
                           SequenceGroupOutput, SequenceOutput, SequenceStatus)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

logger = init_logger(__name__)


class MultiStepOutputProcessor(SequenceGroupOutputProcessor):
    """SequenceGroupOutputProcessor which handles logic related to
    detokenization and stopping conditions. It specializes to "multi-step
    decoding", where vLLM's worker may generate multiple tokens per invocation.
    This is currently mutually exclusive with advanced sampling techniques like
    beam search, which motivates the separation of this logic from the single
    step output processor.

    This class is responsible for things such as correctly appending all new
    token ids to their sequence, detokenizing new token ids, truncating new
    output tokens after an eos token, and correctly handling the case where the
    number of new output tokens per sequence differs in a single batch.
    """

    def __init__(
        self,
        detokenizer: Detokenizer,
        scheduler: Scheduler,
        seq_counter: Counter,
        get_tokenizer_for_seq: Callable[[Sequence], PreTrainedTokenizer],
        stop_checker: StopChecker,
    ):
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.get_tokenizer_for_seq = get_tokenizer_for_seq
        self.stop_checker = stop_checker

    def process_prompt_logprob(self, seq_group: SequenceGroup,
                               outputs: List[SequenceGroupOutput]) -> None:
        # TODO(sang): Prompt logprob currently not implemented in multi step
        # workers.
        logger.warning(
            "Prompt logprob is not supported by multi step workers. "
            "(e.g., speculative decode uses multi step workers).")
        pass

    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput]) -> None:
        """Append new tokens in the outputs to sequences in the sequence group.

        This only supports sequence groups of size 1. It supports greater than
        one new token per sequence.

        This applies logic like stop condition checking and detokenization,
        including freeing finished sequences. It also handles cases where there
        are tokens emitted after the EOS token.
        """
        seqs = sequence_group.get_seqs(status=SequenceStatus.RUNNING)

        assert seqs, "expected running sequences"
        assert len(seqs) == 1, (
            "Beam search not supported in multi-step decoding.")
        seq = seqs[0]

        # Since there's only one sequence per sequence group, we can take the
        # first sample.
        samples = [outputs[step].samples[0] for step in range(len(outputs))]

        # -1 means the output token is not valid (eg. due to spec decode
        # rejecting tokens).
        valid_samples = [
            sample for sample in samples if sample.output_token != -1
        ]
        assert valid_samples

        self._process_seq_outputs(seq, valid_samples,
                                  sequence_group.sampling_params)

    def _process_seq_outputs(self, seq: Sequence,
                             valid_samples: List[SequenceOutput],
                             sampling_params: SamplingParams) -> None:
        output_token_ids = [sample.output_token for sample in valid_samples]

        # Truncate to max_tokens if necessary.
        remaining_tokens = sampling_params.max_tokens - (seq.get_output_len() +
                                                         len(output_token_ids))
        if remaining_tokens < 0:
            valid_samples = valid_samples[:remaining_tokens]
            output_token_ids = output_token_ids[:remaining_tokens]

        # Truncate any tokens after EOS. This is required as spec decode
        # generates a fixed number of tokens without evaluating stopping
        # conditions within the block. This can cause an eos token to be
        # unintentionally ignored.
        if not sampling_params.ignore_eos:
            eos_token_id = self.get_tokenizer_for_seq(seq).eos_token_id
            # Avoiding .index calls as exception throwing in the happy path
            # is expensive.
            for i in range(len(output_token_ids)):
                if output_token_ids[i] == eos_token_id:
                    output_token_ids = output_token_ids[:i + 1]
                    valid_samples = valid_samples[:i + 1]
                    break

        # Incrementally append tokens to the sequence, as if we had only one new
        # token.
        for output_token_id in output_token_ids:
            seq.append_token_id(
                token_id=output_token_id,
                # TODO emit logprobs in multi-step decoding.
                logprobs={output_token_id: Logprob(0.0)},
            )

            new_char_count = 0
            if sampling_params.detokenize:
                new_char_count = self.detokenizer.decode_sequence_inplace(
                    seq, sampling_params)

            self.stop_checker.maybe_stop_sequence(
                seq,
                new_char_count=new_char_count,
                sampling_params=sampling_params)
            if seq.is_finished():
                break

        if seq.is_finished():
            self.scheduler.free_seq(seq)
