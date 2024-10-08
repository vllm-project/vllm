from dataclasses import dataclass, fields

from vllm.logger import init_logger
from vllm.model_executor.prefill_only.config import (  # noqa: E501
    EngineConfig, ModelConfig, PrefillOnlySchedulerConfig)

logger = init_logger(__name__)

_GB = 1 << 30


@dataclass(frozen=True)
class EncodeOnlyEngineConfig(EngineConfig):
    model_config: ModelConfig
    scheduler_config: PrefillOnlySchedulerConfig

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
