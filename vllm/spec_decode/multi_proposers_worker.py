from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase

logger = init_logger(__name__)


class MultiProposersWorker(ProposerWorkerBase, LoraNotSupportedWorkerBase):
    def __init__(self, *args, **kwargs):
        self.vocab_size = kwargs["model_config"].get_vocab_size()
        self._workers = kwargs.pop('worker_list', {})

        draft_parallel_config: ParallelConfig = kwargs['parallel_config']
        draft_tp = draft_parallel_config.tensor_parallel_size
        
        # TP>1 is not supported currently because DraftModelRunner does
        # not support TP>1.
        # TODO: Remove this when TP>1 is supported and #5814 is fixed.
        if draft_tp != 1:
            raise ValueError(
                f"speculative_draft_tensor_parallel_size cannot be "
                f"other value than 1 when using MultiProposersWorker. "
                f"Got {draft_tp} instead.")

    def init_device(self) -> None:
        for worker in self._workers.values():
            worker.init_device()

    def load_model(self) -> None:
        for worker in self._workers.values():
            worker.load_model()

    def set_include_gpu_probs_tensor(self) -> None:
        for worker in self._workers.values():
            if self.is_multi_step_worker_instance(worker):
                worker.set_include_gpu_probs_tensor()

    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[Optional[List[Optional[SamplerOutput]]], bool]:
        """No need to implement sampler_output for MultiProposersWorker,
        as the optional proposers of MultiProposersWorker will use their
        own Top1Proposers to call their sampler_output functions.
        """
        raise NotImplementedError
    
    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. Ideally, we
        recieve and process all sequences in the same batch with one specified
        proposer. However, if multiple proposers are specified, we currently
        use the proposer with the lowest proposal latency for the whole batch.

        If we use different proposers for different sequences in the same
        batch, all proposers will need to wait for the slowest proposer to
        finish on each batch for further scoring. It means those proposers
        with lower acceptance rates but faster speed, like Ngram, will be
        dragged down by the slowest proposer for each step when there remain
        more steps for them to complete. Therefore, a better strategy is to
        use the fastest proposer adaptively among all specified proposers for
        the current batch. This could be optimized when we have multiple
        scorers.
        """
        chosen_proposer = self._get_proposer_for_this_step(
            execute_model_req,
            # Once we support more policies, we can make this configurable.
            scheduling_policy="proposal_latency")
        
        if chosen_proposer == "disable":
            batch_size = len(execute_model_req.seq_group_metadata_list)
            proposal_len = execute_model_req.num_lookahead_slots
            proposal_tokens = torch.tensor(-1,
                                           dtype=torch.long).expand(
                                               batch_size, proposal_len)
            proposal_probs = torch.tensor(0,
                                          dtype=torch.float32).expand(
                                              batch_size, proposal_len,
                                              self.vocab_size)
            proposal_lens_tensor = torch.tensor(0, dtype=torch.long)
            return SpeculativeProposals(
                proposal_token_ids=proposal_tokens,
                proposal_probs=proposal_probs,
                proposal_lens=proposal_lens_tensor,
                no_proposals=True
                )
        return self._workers[chosen_proposer].get_spec_proposals(
            execute_model_req, seq_ids_with_bonus_token_in_last_step)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform speculative decoding on the input batch.
        """

        # To perform KV operations, the 'non_driver_ranks' of SpecDecodeWorker
        # might call this function with execute_model_req set to None many
        # times.
        if execute_model_req is None:
            return []
        
        # Currently, if one seq_group requires to perform execute_model through
        # MultiStepWorker, all seq_groups in the same batch have to perform
        # execute_model together. We have not found a good way to avoid this.
        proposer: str = '[ngram]'
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        valid_proposers = list(self._workers.keys())
        for _, seq in enumerate(seq_group_metadata_list):
            sd_params = seq.spec_decode_params
            if sd_params is not None:
                proposer = sd_params.get_proposer()
                if proposer not in valid_proposers:
                    if proposer == "disable":
                        return []
                    logger.info(
                        "proposer_name must be in %s, or set to None. "
                        "Got '%s' instead. Use '[ngram]' as replacement.",
                        valid_proposers, proposer)
                    proposer = '[ngram]'
                    sd_params.set_proposer(proposer)
                if self.is_multi_step_worker_instance(self._workers[proposer]):
                    break
        else:
            return []
        
        return self._workers[proposer].execute_model(execute_model_req)

    def get_cache_block_size_bytes(self) -> int:
        for worker in self._workers.values():
            if self.is_multi_step_worker_instance(worker):
                return worker.get_cache_block_size_bytes()
            
        return 0
    
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        for worker in self._workers.values():
            if self.is_multi_step_worker_instance(worker):
                return worker.determine_num_available_blocks()
            
        return -1, -1

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        for worker in self._workers.values():
            if self.is_multi_step_worker_instance(worker):
                worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
            
        return
    
    def _get_proposer_for_this_step(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
        scheduling_policy: Optional[str] = "proposal_latency",
    ) -> str:
        """Get the current proposer for the given sequence batch according to
        required scheduling_policy.
        """
        chosen_proposer = '[ngram]'
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        valid_proposers = list(self._workers.keys())

        if scheduling_policy == "popularity":
            proposer_count: Dict[str, int] = {}
            for seq in seq_group_metadata_list:
                sd_params = seq.spec_decode_params
                if sd_params is not None:
                    proposer = sd_params.get_proposer()
                    if proposer not in valid_proposers:
                        if proposer == "disable":
                            return "disable"
                        continue
                    if proposer not in proposer_count:
                        proposer_count[proposer] = 0
                    proposer_count[proposer] += 1
            if len(proposer_count.keys()) != 0:
                chosen_proposer = max(proposer_count, key=proposer_count.get)

        elif scheduling_policy == "proposal_latency":
            for _, seq in enumerate(seq_group_metadata_list):
                sd_params = seq.spec_decode_params
                if sd_params:
                    proposer = sd_params.get_proposer()
                    if proposer not in valid_proposers:
                        if proposer == "disable":
                            return "disable"
                        continue
                    else:
                        chosen_proposer = proposer
                        # Since MultiProposersWorker only supports Ngram as the
                        # backup proposer currently, we should use Ngram for
                        # the whole batch if any seq_group specifies it.
                        # TODO: Refactor this when flexible backup speculative
                        # model choices and latency metrics are supported.
                        if chosen_proposer == '[ngram]':
                            break

        elif scheduling_policy == "proposal_quality":
            # TODO: Use SpecDecodeWorkerMetrics to select the proposer with the
            # best draft_acceptance_rate dynamically.
            raise NotImplementedError(
                f"scheduling_policy: '{scheduling_policy}' has not been "
                f"implemented yet.")
        elif scheduling_policy == "given_priority":
            # TODO: Select the proposer according to a given priority order
            raise NotImplementedError(
                f"scheduling_policy: '{scheduling_policy}' has not been "
                f"implemented yet.")
        
        else:
            raise ValueError(
                f"Invalid scheduling_policy: '{scheduling_policy}'.")

        return chosen_proposer

    def is_multi_step_worker_instance(self, obj: ProposerWorkerBase) -> bool:
        if isinstance(obj, MultiStepWorker):
            return True
        elif isinstance(obj, SmallerTpProposerWorker):
            if hasattr(obj, '_worker'):
                return self.is_multi_step_worker_instance(obj._worker)
        else:
            return False
