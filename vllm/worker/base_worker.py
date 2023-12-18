from abc import ABC, abstractmethod


class BaseWorker(ABC):
    """Base class for Workers.
    See Worker implementation for details.
    """

    @abstractmethod
    def init_model(self):
        raise NotImplementedError

    @abstractmethod
    def profile_num_available_blocks(self):
        raise NotImplementedError

    @abstractmethod
    def init_cache_engine(self):
        raise NotImplementedError

    @abstractmethod
    def execute_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_metadata_cache_len(self):
        raise NotImplementedError

    @abstractmethod
    def get_runtime_context(self):
        raise NotImplementedError


class BaseLoraWorker(BaseWorker):
    """Base class for LoRA-enabled Workers.
    See the Worker implememntation for details.
    """

    @abstractmethod
    def add_lora(self):
        raise NotImplementedError

    @abstractmethod
    def remove_lora(self):
        raise NotImplementedError

    @abstractmethod
    def list_loras(self):
        raise NotImplementedError


class LoraNotSupportedWorker(BaseLoraWorker):
    """Implementation of BaseLoraWorker which raises
    an error on LoRA calls.
    """

    def add_lora(self):
        raise ValueError("LoRA not supported")

    def remove_lora(self):
        raise ValueError("LoRA not supported")

    def list_loras(self):
        raise ValueError("LoRA not supported")
