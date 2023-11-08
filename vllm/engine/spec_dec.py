from vllm.config import SpecDecConfig
from vllm.model_executor import get_model
from vllm.sequence import SequenceGroupMetadata
from transformers import AutoModel
import torch
from typing import List
from vllm.sequence import SamplerOutput
from vllm.worker.worker import Worker

class SpecDecWorker(Worker):
    def __init__(self, config: SpecDecConfig) -> None:
        self.propose_cnt = config.propose_cnt
        self.draft_model_config = config.draft_model_config
        
        # self.draft_model = get_model(self.draft_model_config)
        self.draft_model = AutoModel(self.draft_model_config.model)
        
        ##### values to be set
        self.draft_probs = None
        self.draft_kvs = None # if we use hf stype kvs
    
    def _prepare_inputs(self, 
                        seq_group_metadata_list: List[SequenceGroupMetadata]) -> List[torch.Tensor]:
        input_ids_list = []
        for seq_group_metadata in seq_group_metadata_list:
            seq = seq_group_metadata.seq_data[0]
            assert len(seq) == 1, "Speculative Decoding does nor beam search for now"
            input_ids_list.append(seq.get_token_ids())
        return input_ids_list 
    
    def set_draft_tokens(self,
                seq_group_list: List[SequenceGroupMetadata]) -> torch.Tensor:
        input_ids = self._prepare_inputs(seq_group_list)
        # recompute for now
        draft_tokens = self.draft_model.generate(input_ids=input_ids,
                                  attention_mask=(input_ids != -1),
                                  max_new_tokens = self.propose_cnt)[:, input_ids.shape[-1]:]
        for i, seq_group_metadata in enumerate(seq_group_list):
            seq = seq_group_metadata.seq_data[0]
            seq.draft_token_ids = draft_tokens[i]
        
        return draft_tokens
    
    def accept(self,
               target_output: List[SamplerOutput]):
        def extract_probs(output: List[SamplerOutput]):
            pass
        
        def sample_accept(draft_probs: torch.Tensor, 
                          target_probs: torch.Tensor):
            pass
        
        target_probs = extract_probs(target_output)
        sample_accept(self.draft_probs, target_probs)
        
    
    def invalidate_draft_kv(self):
        pass
    
    def invalidate_target_kv(self):
        pass
    
    