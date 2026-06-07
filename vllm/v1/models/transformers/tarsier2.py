from transformers import Tarsier2ForConditionalGeneration, Tarsier2Config
from vllm.v1.models.base import LLM

class Tarsier2ForConditionalGeneration(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = Tarsier2Config.from_pretrained(self.model_name)

    def _get_config(self):
        # Workaround for malformed config.json in Tarsier2 checkpoint
        config = super()._get_config()
        if not hasattr(config, 'num_attention_heads'):
            config.num_attention_heads = self.config.num_attention_heads
        return config