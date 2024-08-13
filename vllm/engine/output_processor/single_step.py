from typing import Dict, List, Optional, Tuple, Union

from vllm.config import SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Sequence, SequenceGroup, SequenceGroupOutput,
                           SequenceOutput, SequenceStatus)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

logger = init_logger(__name__)


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

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        detokenizer: Detokenizer,
        scheduler: List[Scheduler],
        seq_counter: Counter,
        stop_checker: StopChecker,
    ):
        self.scheduler_config = scheduler_config
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.stop_checker = stop_checker

    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput]) -> None:
        """Append all new tokens to sequences in the sequence group. Fork any
        surviving beam candidates; free any unsurviving ones.

        Invokes detokenizer to detokenize new tokens, and also marks sequences
        as finished if they meet stop conditions.
        """
        assert (len(outputs) == 1
                ), f"{type(self)} does not support multiple outputs per step"
        return self._process_sequence_group_outputs(sequence_group, outputs[0])

    def process_prompt_logprob(self, seq_group: SequenceGroup,
                               outputs: List[SequenceGroupOutput]) -> None:
        assert len(outputs) == 1, ("Single step should only has 1 output.")
        output = outputs[0]
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

            if seq_group.sampling_params.detokenize and self.detokenizer:
                self.detokenizer.decode_prompt_logprobs_inplace(
                    seq_group,
                    prompt_logprobs,
                    position_offset=len(seq_group.prompt_logprobs))

            seq_group.prompt_logprobs.extend(prompt_logprobs)

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput) -> None:
        sampling_params = seq_group.sampling_params
        if sampling_params.n == 1 and not sampling_params.use_beam_search:
            # only have one output sample
            sample = outputs.samples[0]
            # only have one sequence
            seq = seq_group.seqs[0]
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
            return

        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict: Dict[int, List[SequenceOutput]] = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            # Guard against a KeyError which can occur if the request was
            # aborted while the output was generated
            if (child_list :=
                    parent_child_dict.get(sample.parent_seq_id)) is not None:
                child_list.append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[
                parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                for scheduler in self.scheduler:
                    scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id: int = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token,
                                      child_sample.logprobs)
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token,
                                   last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
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

        # Non-beam search case
        if not sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        for scheduler in self.scheduler:
                            scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    for scheduler in self.scheduler:
                        scheduler.free_seq(seq)
            return

        # Beam search case
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs: List[Tuple[Sequence, Optional[Sequence]]] = []
        unselected_child_seqs: List[Tuple[Sequence, Optional[Sequence]]] = []
        beam_width = sampling_params.best_of
        length_penalty = sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False)
                                  for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                             if seq.is_finished()]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=x[0].eos_token_id),
                               reverse=True)
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                              if not seq.is_finished()]
        # Sort the running sequences by their scores.
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=x[0].eos_token_id),
                                reverse=True)

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                sampling_params.early_stopping, sampling_params,
                best_running_seq, current_worst_seq)

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    for scheduler in self.scheduler:
                        scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                for scheduler in self.scheduler:
                    scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                for scheduler in self.scheduler:
                    scheduler.free_seq(seq)

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=current_worst_seq.eos_token_id)
        if early_stopping is False:
            highest_attainable_score = best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=best_running_seq.eos_token_id)
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() +
                    sampling_params.max_tokens,
                    self.scheduler_config.max_model_len)
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=best_running_seq.eos_token_id,
                        seq_len=max_possible_length))
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=best_running_seq.eos_token_id))
        return current_worst_score >= highest_attainable_score
