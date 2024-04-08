from typing import List

from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Logprob, Sequence, SequenceGroup,
                           SequenceGroupOutput, SequenceOutput, SequenceStatus)

logger = init_logger(__name__)


class BlockDecodeOutputProcessor(SequenceGroupOutputProcessor):

    def __init__(
        self,
        detokenizer,
        scheduler,
        seq_counter,
        get_tokenizer_for_seq,
        stop_checker,
    ):
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.get_tokenizer_for_seq = get_tokenizer_for_seq
        self.stop_checker = stop_checker

    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput]) -> None:
        seqs = sequence_group.get_seqs(status=SequenceStatus.RUNNING)

        assert seqs, "expected running sequences"
        assert len(seqs) == 1, ("Beam search not supported in block decoding.")
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
        # generates tokens in fixed blocks, which may go beyond the EOS token.
        if not sampling_params.ignore_eos:
            eos_token_id = self.get_tokenizer_for_seq(seq).eos_token_id
            # Avoiding .index calls as exception throwing in the happy path
            # is expensive.
            for i in range(len(output_token_ids)):
                if output_token_ids[i] == eos_token_id:
                    output_token_ids = output_token_ids[:i + 1]
                    valid_samples = valid_samples[:i + 1]
                    break

        for output_token_id in output_token_ids:
            seq.append_token_id(
                token_id=output_token_id,
                # TODO emit logprobs in block decoding.
                logprobs={output_token_id: Logprob(0.0)},
            )
            self.detokenizer.decode_sequence_inplace(seq, sampling_params)

        self.stop_checker.maybe_stop_sequence(seq,
                                              sampling_params,
                                              new_token_ids=output_token_ids)

        if seq.is_finished():
            self.scheduler.free_seq(seq)
