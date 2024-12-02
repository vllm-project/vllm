from vllm.config import LoRAConfig

class LoRAScheduler:
    """Round robin scheduler for LoRAs."""

    def __init__(self, lora_config: LoRAConfig):
        self.lora_config = lora_config
        self.max_loras_each_iter = lora_config.max_loras
    
    def _register_lora(self, lora_id: int):
        pass

    def _unregister_lora(self, lora_id: int):
        pass

    def update_loras(self, all_lora_ids: list[int]):
        """Update the list of LoRAs that are available for scheduling."""
        pass

    def schedule_loras(self):
        """Schedule which LoRAs requests can belong to for the next iteration."""
        pass