import time
from typing import Iterable, List, Optional, Tuple, Type, Union

from transformers import PreTrainedTokenizer

import vllm
from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VisionLanguageConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.ray_utils import initialize_ray_cluster
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (MultiModalData, SamplerOutput, Sequence,
                           SequenceGroup, SequenceGroupOutput, SequenceOutput,
                           SequenceStatus)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer_group import (BaseTokenizerGroup,
                                                     get_tokenizer_group)
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import Counter
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor

logger = init_logger(__name__)


class BlockDecodeOutputProcessor(SequenceGroupOutputProcessor):
    
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        detokenizer,
        scheduler,
        seq_counter,
        get_tokenizer_for_seq,
        stop_checker,
    ):
        self.scheduler_config = scheduler_config
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.get_tokenizer_for_seq = get_tokenizer_for_seq
        self.stop_checker = stop_checker

    def process_outputs(self, sequence_group: SequenceGroup, outputs: SequenceGroupOutput) -> None:
        return self._process_sequence_group_outputs_multi_step(sequence_group, outputs)

    def _process_sequence_group_outputs_multi_step(self, seq_group, outputs):
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)

        assert seqs
        #if not seqs:
        #    return []

        assert len(seqs) == 1, ("Beam search not supported in speculative "
                                "decoding.")
        seq = seqs[0]

        # Since there's only one sequence per sequence group, we can take the
        # first sample.
        samples = [outputs[step].samples[0] for step in range(len(outputs))]

        # -1 means the output token is not valid (eg. due to spec decode
        # rejecting tokens).
        valid_samples = [
            sample for sample in samples if sample.output_token != -1
        ]

        # Draft target worker pads all outputs with -1 to have same length.
        output_token_ids = [sample.output_token for sample in valid_samples]
        #successes = [sample.success for sample in samples]

        ## Truncate to max_tokens if necessary.
        #remaining_tokens = seq_group.sampling_params.max_tokens - (
        #    seq.get_output_len() + len(output_token_ids))
        #if remaining_tokens < 0:
        #    valid_samples = valid_samples[:remaining_tokens]
        #    output_token_ids = output_token_ids[:remaining_tokens]

        ## Truncate any tokens after EOS. This is required as spec decode
        ## generates tokens in fixed blocks, which may go beyond the EOS token.
        #if not seq_group.sampling_params.ignore_eos:
        #    eos_token_id = self.tokenizer.get_lora_tokenizer(
        #        seq.lora_request).eos_token_id
        #    # Avoiding .index calls as exception throwing in the happy path
        #    # is expensive.
        #    for i in range(len(output_token_ids)):
        #        if output_token_ids[i] == eos_token_id:
        #            output_token_ids = output_token_ids[:i + 1]
        #            valid_samples = valid_samples[:i + 1]
        #            break

        #output_logprobs = [sample.logprobs for sample in valid_samples]

        ## Use the last sample for the sequence as it will have
        ## the speculation and num_unprocessed_tokens for all the
        ## previous samples (they are cumulative when it comes
        ## to those two attributes).
        #speculation = valid_samples[-1].speculation
        #num_unprocessed_tokens = valid_samples[-1].num_unprocessed_tokens

        for output_token_id in output_token_ids:
            from vllm.sequence import Logprob
            seq.append_token_id(
                token_id=output_token_id,
                logprobs={output_token_id: Logprob(0.0)},
            )

        #seq.append_token_ids(output_token_ids,
        #                     output_logprobs,
        #                    )
        #                     #num_unprocessed_tokens=num_unprocessed_tokens)
        ##seq.set_last_speculation(speculation)

        #if not all(successes):
        #    seq.set_status_to_failed()

        #if decode:
        #    self._decode_sequence(seq,
        #                          seq_group.sampling_params,
        #                          token_ids=seq.get_token_ids(),
        #                          unseen_token_ids=output_token_ids,
        #                          prefix_offset=seq.prefix_offset,
        #                          read_offset=seq.read_offset)
        #self._check_stop(seq, seq_group.sampling_params, seq.lora_request,
        #                 output_token_ids)
        # TODO pass output token ids
        self.stop_checker.maybe_stop_sequence(seq, seq_group.sampling_params)

        if seq.is_finished():
            self.scheduler.free_seq(seq)
