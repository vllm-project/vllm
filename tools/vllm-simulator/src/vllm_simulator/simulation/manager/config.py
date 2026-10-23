import json
import typing
from copy import deepcopy
from typing import Optional

from vllm_simulator.simulation.manager.env import Envs
from vllm_simulator.simulation.types import SchedulerConfig
from vllm_simulator.simulation.vllm.utils import resolve_scheduler_config
from vllm_simulator.spec import AcceleratorInfo
from vllm_simulator.time_predictor import (
    AISimulateTimePredictor,
    InferTimePredictor,
)
from vllm_simulator.utils import get_logger

if typing.TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = get_logger()


class ConfigManager:
    """Centralized configuration manager with caching."""

    _model_config: Optional["ModelConfig"] = None
    _scheduler_config: Optional[SchedulerConfig] = None
    _raw_config: Optional[dict] = None

    @classmethod
    def _get_raw_config(cls) -> dict:
        if cls._raw_config is None:
            with open(Envs.config_path()) as f:
                cls._raw_config = json.load(f)
        return cls._raw_config

    @classmethod
    def init_from_vllm_config(cls, vllm_config: "VllmConfig"):
        """Resolve the model and scheduler configs; called once per engine
        from Worker.init_device."""
        cls._model_config = vllm_config.model_config
        cls._scheduler_config = resolve_scheduler_config(vllm_config)
        if cls._scheduler_config.pp_size != 1:
            raise ValueError("Pipeline-parallel simulation supports only pp_size=1")

        # The configuration from the external config file has higher priority.
        external_config = cls._get_raw_config().get("scheduler", {})
        external_pp_size = external_config.get("pp_size", 1)
        if external_pp_size != 1:
            raise ValueError("Pipeline-parallel simulation supports only pp_size=1")
        for field_name in [
            "tp_size",
            "dp_size",
            "ep_size",
            "backend_name",
            "backend_version",
        ]:
            field_value = external_config.get(field_name)
            if field_value is not None:
                setattr(cls._scheduler_config, field_name, field_value)

    @classmethod
    def get_model_config(cls) -> Optional["ModelConfig"]:
        return cls._model_config

    @classmethod
    def get_scheduler_config(cls) -> Optional[SchedulerConfig]:
        return cls._scheduler_config

    @classmethod
    def get_accelerator_info(cls) -> AcceleratorInfo:
        config = cls._get_raw_config()
        platform_config = config.get("platform", {})
        acc_info = platform_config.get("accelerator", {})
        hw = AcceleratorInfo.find_by_hw_name(acc_info.get("name"))
        if hw is None:
            logger.debug(
                f"Failed to initialize device info with {acc_info.get('name')}. All available devices are: {AcceleratorInfo.list_all_hws().keys()}"
            )
            hw = AcceleratorInfo(
                name=acc_info.get("name"),
                hbm_capacity_gb=acc_info.get("hbm_capacity_gb"),
            )
        else:
            hw = deepcopy(hw)
            if acc_info.get("hbm_capacity_gb") is not None:
                hw.hbm_capacity_gb = acc_info["hbm_capacity_gb"]
            logger.info("Device info initialized: %s", hw)
        return hw

    @classmethod
    def cpu_overhead_enabled(cls) -> bool:
        """Enable measured native CPU time in NON_BLOCKING engine steps."""
        enabled = cls._get_raw_config().get("cpu_overhead", {}).get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("cpu_overhead.enabled must be a boolean")
        return enabled

    @classmethod
    def get_inference_time_predictor(cls) -> InferTimePredictor:
        model = cls._model_config
        hw = cls.get_accelerator_info()
        sched_config = cls._scheduler_config
        predictor_config = cls._get_raw_config().get("predictor", {})
        if predictor_config.get("name") == "aisimulate":
            return AISimulateTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=predictor_config.get("database_path"),
                database_mode=predictor_config.get("database_mode", "SILICON"),
                prefill_scale_factor=predictor_config.get("prefill_scale_factor", 1),
                decode_scale_factor=predictor_config.get("decode_scale_factor", 1),
                prefill_min_latency=predictor_config.get("prefill_min_latency", 0),
                workload_distribution=predictor_config.get(
                    "workload_distribution", "balanced"
                ),
                enable_oom_check=predictor_config.get("enable_oom_check", False),
            )
        elif predictor_config.get("name") == "ml":
            from vllm_simulator.time_predictor.ml import MLTimePredictor

            return MLTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=predictor_config.get("database_path"),
                latency_scale=predictor_config.get("latency_scale", 1.0),
                logprobs_cost_us_per_token=predictor_config.get(
                    "logprobs_cost_us_per_token", 0.0
                ),
                sample_tokens_base_ms=predictor_config.get(
                    "sample_tokens_base_ms", 0.0
                ),
                sample_tokens_lo_us_per_token=predictor_config.get(
                    "sample_tokens_lo_us_per_token", 0.0
                ),
                sample_tokens_hi_us_per_token=predictor_config.get(
                    "sample_tokens_hi_us_per_token", 0.0
                ),
                sample_tokens_breakpoint_tokens=predictor_config.get(
                    "sample_tokens_breakpoint_tokens", 0
                ),
            )
        elif predictor_config.get("name") == "replay":
            from vllm_simulator.time_predictor.replay import ReplayTimePredictor

            return ReplayTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=predictor_config.get("database_path"),
                miss_fallback_seconds=predictor_config.get(
                    "miss_fallback_seconds", 0.0
                ),
                miss_strategy=predictor_config.get("miss_strategy", "zero"),
                miss_knn_k=predictor_config.get("miss_knn_k", 3),
            )
        elif predictor_config.get("name") == "linear":
            from vllm_simulator.time_predictor.linear import LinearTimePredictor

            return LinearTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                base_ms=predictor_config.get("base_ms", 0.0),
                per_token_ms=predictor_config.get("per_token_ms", 0.0),
            )
        else:
            raise ValueError(
                f"Unknown predictor name: {predictor_config.get('name')}. "
                "Supported: aisimulate, ml, replay, linear"
            )
