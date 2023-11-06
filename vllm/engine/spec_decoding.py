from vllm.config import SpecDecodingConfig
from vllm.model_executor import get_model
import torch
from typing import List
from vllm.sequence import SamplerOutput

class SpeculativeHandler:
    def __init__(self, config: SpecDecodingConfig) -> None:
        self.draft_cnt = config.draft_cnt
        self.draft_model_config = config.draft_model_config
        self.draft_model = get_model(self.draft_model_config)
    
        ##### values to be set
        self.draft_probs = None
        self.draft_kvs = None # if we use hf stype kvs
        
    def propose(self) -> torch.Tensor:
        # propose draft_cnt tokens
        pass
    
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
    
    