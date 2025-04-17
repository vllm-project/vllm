

from typing import Union
from vllm.sequence import Sequence
from typing import Sequence as GenericSequence


class ZeroOverheadSequence(Sequence):
    def __init__(self, seq_id, inputs, block_size, eos_token_id = None, lora_request = None, prompt_adapter_request = None):
        super().__init__(seq_id, inputs, block_size, eos_token_id, lora_request, prompt_adapter_request)
        self.effective_output_len : int = 0

    def fix_last_token_id(self, token_id: int) -> None:
        effect_offset = self.effective_output_len - len(self.data.output_token_ids)
        assert effect_offset < 0
        self.data._output_token_ids[effect_offset] = token_id
        if len(self.data._new_appended_tokens) >= effect_offset * -1:
            self.data._new_appended_tokens[effect_offset] = token_id
        self.data._cached_all_token_ids[effect_offset] = token_id
        self.effective_output_len += 1
    

    def zero_overhead_get_output_token_ids(self) -> tuple[int, ...]:
        return self.data.output_token_ids[:self.effective_output_len]
    
    def zero_overhead_get_output_len(self) -> int:
        return self.effective_output_len
    
    def zero_overhead_get_last_token_id(self) -> int:
        if self.effective_output_len == 0:
            return self.data._prompt_token_ids[-1]
        return self.data._output_token_ids[self.effective_output_len - 1]
    
    def zero_overhead_get_len(self) -> int:
        return self.effective_output_len + len(self.data._prompt_token_ids)
    
    def get_output_token_ids_to_return(
            self, delta: bool) -> Union[GenericSequence[int], int]:
        """If delta is True, only new tokens since the last call to
        this method are returned"""
        if not delta:
            return self.zero_overhead_get_output_token_ids()

        output_len = self.zero_overhead_get_output_len()

        # Get the number of new tokens
        num_new_tokens = output_len - self._last_output_token_ids_offset
        self._last_output_token_ids_offset = output_len

        # Return new tokens
        if num_new_tokens == 1:
            # Optimization for single decode token case
            # (which is what we have most of the time)
            return self.data._cached_all_token_ids[self.effective_output_len - 1]

        if num_new_tokens == 0:
            return []

        effect_offset = self.effective_output_len - len(self.data.output_token_ids)
        return self.data._cached_all_token_ids[-num_new_tokens : effect_offset]