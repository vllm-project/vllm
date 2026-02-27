# mla_path_selector.py

from typing import Any, Literal

MLA_PATH_MHA_UNABSORBED = "UNABSORBED_MHA"
MLA_PATH_ABSORBED = "ABSORBED_MLA"

class MLAPathSelector:
    """
    Evaluates execution bounds relying on the 80/20 heuristic.
    Routes short context prefill queries to un-absorbed MHA, and long-contexts/decoding to absorbed MLA.
    """
    
    def __init__(self, threshold: int = 1024):
        self.threshold = threshold

    def select_path(self, attn_metadata: Any) -> Literal["UNABSORBED_MHA", "ABSORBED_MLA"]:
        if attn_metadata is None:
            return MLA_PATH_ABSORBED

        max_seq_len = 0
        
        # In typical vLLM AttentionMetadata shapes, prefills are explicitly handled.
        if hasattr(attn_metadata, 'prefill_seq_lens') and attn_metadata.prefill_seq_lens is not None:
            seq_lens = attn_metadata.prefill_seq_lens
            if len(seq_lens) > 0:
                max_seq_len = max(seq_lens)
        
        elif hasattr(attn_metadata, 'seq_lens') and attn_metadata.seq_lens is not None:
            seq_lens = attn_metadata.seq_lens
            if hasattr(seq_lens, 'max'):
                max_seq_len = seq_lens.max().item()
            elif isinstance(seq_lens, list) and len(seq_lens) > 0:
                max_seq_len = max(seq_lens)
        
        # When parsing v1 metadata
        elif hasattr(attn_metadata, 'max_seq_len'):
            max_seq_len = attn_metadata.max_seq_len
            
        elif hasattr(attn_metadata, 'num_prefills') and attn_metadata.num_prefills > 0:
            if hasattr(attn_metadata, 'num_prefill_tokens'):
                # Heuristic: average prefill seq len
                max_seq_len = attn_metadata.num_prefill_tokens // attn_metadata.num_prefills
        
        if max_seq_len > 0 and max_seq_len < self.threshold:
            return MLA_PATH_MHA_UNABSORBED
            
        return MLA_PATH_ABSORBED
