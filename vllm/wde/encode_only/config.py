from dataclasses import dataclass, fields
from typing import Optional

from vllm.logger import init_logger
from vllm.wde.core.config import EngineConfig, ModelConfig, SchedulerConfig

logger = init_logger(__name__)

_GB = 1 << 30


class EncodeOnlyModelConfig(ModelConfig):
    pass


class EncodeOnlySchedulerConfig(SchedulerConfig):

    def __init__(self,
                 max_model_len: int,
                 max_num_batched_tokens: Optional[int] = None,
                 max_num_requests: Optional[int] = None,
                 max_num_seqs: Optional[int] = None,
                 max_num_on_the_fly: Optional[int] = 3,
                 scheduling: str = "sync") -> None:
        self.max_model_len = max_model_len
        self.max_num_requests: int = 0
        self.max_num_batched_tokens: int = 0
        self.max_num_on_the_fly: int = max_num_on_the_fly
        self.scheduling = scheduling

        self.set_args(max_num_batched_tokens, max_num_requests, max_num_seqs)

    def set_args(self,
                 max_num_batched_tokens: Optional[int] = None,
                 max_num_requests: Optional[int] = None,
                 max_num_seqs: Optional[int] = None):
        if max_num_seqs is not None:
            self.max_num_requests = max_num_seqs
        else:
            self.max_num_requests = max_num_requests

        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            self.max_num_batched_tokens = (self.max_model_len *
                                           self.max_num_requests)

        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_model_len "
                f"({self.max_model_len}).")

        if self.max_num_on_the_fly < 2:
            raise ValueError(
                f"max_num_on_the_fly {self.max_num_on_the_fly} must "
                "be greater than 1")

        if self.scheduling not in ["sync", "async", "double_buffer"]:
            raise ValueError(f"scheduling {self.scheduling} must "
                             f"in sync, async double_buffer")


@dataclass(frozen=True)
class EncodeOnlyEngineConfig(EngineConfig):
    model_config: EncodeOnlyModelConfig
    scheduler_config: EncodeOnlySchedulerConfig

    def to_dict(self):
        """Return the configs as a dictionary, for use in **kwargs.
        """
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))

    def log_config(self):
        from vllm.version import __version__ as VLLM_VERSION
        logger.info(
            "Initializing an Encode Only engine (v%s) with config: "
            "model=%r, tokenizer=%r, "
            "tokenizer_mode=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, "
            "device_config=%s, served_model_name=%s, "
            "max_num_on_the_fly=%d, scheduling=%s)", VLLM_VERSION,
            self.model_config.model, self.model_config.tokenizer,
            self.model_config.tokenizer_mode,
            self.model_config.trust_remote_code, self.model_config.dtype,
            self.model_config.max_model_len, self.load_config.download_dir,
            self.load_config.load_format, self.device_config.device,
            self.model_config.served_model_name,
            self.scheduler_config.max_num_on_the_fly,
            self.scheduler_config.scheduling)
