from vllm.config import SpecDecConfig
from vllm.model_executor import get_model
from vllm.sequence import SequenceGroupMetadata
from transformers import AutoModelForCausalLM
import torch
from typing import List
from vllm.sequence import SamplerOutput
from vllm.worker.worker import Worker
from vllm.logger import init_logger

logger = init_logger(__name__)

# FIXME: we should get pad_token_id from tokenizer
PAD_TOKEN_ID = 0

class SpecDecWorker(Worker):
    def __init__(self, config: SpecDecConfig) -> None:
        self.propose_cnt = config.propose_cnt
        self.draft_model_config = config.draft_model_config
        
        # self.draft_model = get_model(self.draft_model_config)
        logger.info(
            "Initializing speculative decoding worker: "
            f"model={self.draft_model_config.model!r}, "
            f"tokenizer={self.draft_model_config.tokenizer!r}, "
            f"propose_cnt={self.propose_cnt}, "
            f"seed={self.draft_model_config.seed})")
        self.draft_model = AutoModelForCausalLM.from_pretrained(self.draft_model_config.model).cuda()
        
        ##### values to be set
        self.draft_probs = None
        self.draft_kvs = None # if we use hf stype kvs

    def _prepare_inputs(self, 
                        seq_group_metadata_list: List[SequenceGroupMetadata]) -> List[torch.Tensor]:
        input_ids_list = []
        for seq_group_metadata in seq_group_metadata_list:
            assert len(seq_group_metadata.seq_data) == 1, f"Speculative Decoding does nor beam search for now: {len(seq_group_metadata.seq_data)}"
            seq_id = next(iter(seq_group_metadata.seq_data))
            seq = seq_group_metadata.seq_data[seq_id]
            input_ids_list.append(seq.get_token_ids())
        max_len = max([len(input_ids) for input_ids in input_ids_list])
        input_ids_list = [_pad_left_to_max(input_ids, max_len, PAD_TOKEN_ID) for input_ids in input_ids_list]
        return torch.tensor(input_ids_list, dtype=torch.long, device='cuda')
    
    def set_draft_tokens(self,
                seq_group_list: List[SequenceGroupMetadata]) -> torch.Tensor:
        logger.info(f"# of input request: {len(seq_group_list)}")
        input_tensor = self._prepare_inputs(seq_group_list)
        # recompute for now
        attention_mask=(input_tensor != PAD_TOKEN_ID)
        draft_tokens = self.draft_model.generate(input_ids=input_tensor,
                                  attention_mask=attention_mask,
                                  max_new_tokens=self.propose_cnt)[:, input_tensor.shape[1]:]
        logger.info(f"Input tokens: {input_tensor}")
        logger.info(f"Draft tokens: {draft_tokens}")
        for i, seq_group_metadata in enumerate(seq_group_list):
            seq_id = next(iter(seq_group_metadata.seq_data))
            seq = seq_group_metadata.seq_data[seq_id]
            seq.draft_token_ids = draft_tokens[i].tolist()
        
        return draft_tokens
    
    def accept(self,
               target_output: List[SamplerOutput]):
        def extract_probs(output: List[SamplerOutput]):
            logprobs = []
            logger.info(f"# of output: {len(output)}")
            for seq_group_output in output:
                assert len(seq_group_output.samples) == 1
                print(seq_group_output.prompt_logprobs)
                sample = seq_group_output.samples[0]
                print(sample.logprobs)  
                exit(0)
        target_probs = extract_probs(target_output)
        _prob_accept(self.draft_probs, target_probs)
        
    
    def invalidate_draft_kv(self):
        pass
    
    def invalidate_target_kv(self):
        pass
    
def _prob_accept(draft_probs: torch.Tensor, 
                target_probs: torch.Tensor):
    p = draft_probs
    q = target_probs[:, :-1, :]
    # shape: [batch_size, propose_cnt, vocab_size]
    assert p.shape == q.shape
    accept_draft_prob = torch.minimum(torch.ones(()), q / p)
    rejected_locations = (
        torch.rand_like(accept_draft_prob) > accept_draft_prob
    ).nonzero()
    
def _pad_left_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    return [pad] * (max_len - len(x)) + x