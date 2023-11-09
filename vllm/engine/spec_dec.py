from vllm.config import SpecDecConfig
from vllm.model_executor import get_model
from vllm.sequence import SequenceGroupMetadata
from transformers import AutoModelForCausalLM
import torch
from typing import List, Dict
from vllm.sequence import SamplerOutput, SequenceGroupOutputs
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
    
    # TODO: we need to align draft and target model's sampler
    def _sample_method(self, logits):
        temperature = 1.0
        return torch.softmax(logits / temperature, dim=-1)
    
    def set_draft_tokens(self,
                seq_group_list: List[SequenceGroupMetadata]) -> None:
        logger.info(f"# of input request: {len(seq_group_list)}")
        input_tensor = self._prepare_inputs(seq_group_list)
        draft_logits, draft_distributions, draft_tokens = [], [], []
        # recompute for now
        attention_mask=(input_tensor != PAD_TOKEN_ID)
        past_key_values = None
        for i in range(self.propose_cnt):
            with torch.no_grad():
                outputs = self.draft_model(input_tensor,
                                    past_key_values=past_key_values,
                                    attention_mask=attention_mask,
                                    use_cache=True)
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            distribution = self._sample_method(next_token_logits)
            attention_mask = torch.cat([attention_mask, torch.ones(input_tensor.shape[0], 1, device='cuda')], dim=1)
            input_tensor = torch.multinomial(distribution, num_samples=1)
            
            draft_logits.append(next_token_logits)
            draft_distributions.append(distribution)
            draft_tokens.append(input_tensor)

        for i, seq_group_metadata in enumerate(seq_group_list):
            seq_id = next(iter(seq_group_metadata.seq_data))
            seq = seq_group_metadata.seq_data[seq_id]
            for j in range(self.propose_cnt):
                draft_token = draft_tokens[j][i].item()
                seq.draft_token_probs[draft_token] = draft_distributions[j][i]
                seq.draft_token_ids.append(draft_token)
            logger.info(f"Seq draft tokens: {seq.draft_token_ids}")
            logger.info(f"Seq draft prob: {seq.draft_token_probs}")
        
    def accept(self,
               target_outputs: List[SamplerOutput]):
        def extract_target_prob(output: SequenceGroupOutputs,
                                token_id: int):
            def add_dict(dst_dict: Dict[int, float], 
                         src_dict: Dict[int, float]):
                for k in src_dict:
                    assert k not in dst_dict, f"{src_dict} || {dst_dict}"
                    dst_dict[k] = src_dict[k]
                return dst_dict
                
            all_logprobs = {}
            for logprob in output.prompt_logprobs:
                if logprob is None:
                    continue
                all_logprobs = add_dict(all_logprobs, logprob)
            sample = seq_group_output.samples[0]
            all_logprobs = add_dict(all_logprobs, sample.logprobs)
            assert token_id in all_logprobs
            logprob = all_logprobs[token_id]
            return torch.exp(logprob)
        
        # Rejecting Sampling
        for seq_group_output in target_outputs:
            assert len(seq_group_output.samples) == 1
            sample = seq_group_output.samples[0]
            print(sample.sd_draft_probs)
            
            accept_token_ids = []
            for token_id in sample.sd_draft_ids:
                p = sample.sd_draft_probs[token_id][token_id].item()
                q = extract_target_prob(seq_group_output, token_id)
                r = torch.rand()
                if r <= p/q: # accept
                    accept_token_ids.append(token_id)
                else: # reject and resample
                    break
            
            # all proposed tokens are accepted
            if len(accept_token_ids) == sample.sd_draft_ids:
                accept_token_ids.append(sample.output_token)
        
        self.invalidate_draft_kv()
        self.invalidate_target_kv()    
    
    def invalidate_draft_kv(self):
        pass
    
    def invalidate_target_kv(self):
        pass
    
def _pad_left_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    return [pad] * (max_len - len(x)) + x