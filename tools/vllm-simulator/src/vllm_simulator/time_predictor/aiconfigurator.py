import typing
from typing import Optional

import numpy as np
from aiconfigurator.sdk import models
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import (
    DatabaseMode,
    KVCacheQuantMode,
)
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.perf_database import (
    get_database,
    get_systems_paths,
    resolve_query_version,
)
from vllm_simulator.simulation.types import (
    SchedulerConfig,
)
from vllm_simulator.spec.accelerator import AcceleratorInfo
from vllm_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)
from vllm_simulator.utils import get_logger

if typing.TYPE_CHECKING:
    import vllm.config

logger = get_logger()


def get_perf_model(
    sched_config: SchedulerConfig,
    model: "vllm.config.ModelConfig",
    workload_distribution: str = "balanced",
) -> models.BaseModel:
    # Quant modes left as None are inferred by AIConfigurator from the
    # checkpoint's quantization_config. Only a runtime --kv-cache-dtype is
    # invisible to the checkpoint, so it is forwarded explicitly.
    model_config = ModelConfig(
        pp_size=sched_config.pp_size,
        tp_size=sched_config.attn_tp_size,
        moe_tp_size=sched_config.moe_tp_size,
        moe_ep_size=sched_config.moe_ep_size,
        attention_dp_size=sched_config.attn_dp_size,
        kvcache_quant_mode=(
            KVCacheQuantMode.fp8
            if sched_config.kv_cache_dtype.startswith("fp8")
            else None
        ),
        workload_distribution=workload_distribution,
    )

    return models.get_model(
        model_path=model.model,
        model_config=model_config,
        backend_name=sched_config.backend_name,
    )


class AIConfiguratorTimePredictor(InferTimePredictor):
    def __init__(
        self,
        model: "vllm.config.ModelConfig",
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: Optional[str] = None,
        database_mode: DatabaseMode | str = DatabaseMode.SILICON,
        prefill_scale_factor: float = 1,
        decode_scale_factor: float = 1,
        prefill_min_latency: float = 0,
        workload_distribution: str = "balanced",
        enable_oom_check: bool = False,
    ):
        super().__init__(model, hw, config)

        self.prefill_scale_factor = prefill_scale_factor
        self.decode_scale_factor = decode_scale_factor
        self.prefill_min_latency = prefill_min_latency
        self.enable_oom_check = enable_oom_check

        if isinstance(database_mode, str):
            database_mode = self._get_database_mode(database_mode)

        systems_paths = (
            [database_path] if database_path is not None else get_systems_paths()
        )
        database = get_database(
            system=hw.name,
            backend=config.backend_name,
            version=self._resolve_database_version(
                hw.name, config.backend_name, config.backend_version, systems_paths
            ),
            systems_paths=systems_paths,
        )

        if database is None:
            raise ValueError("Failed to initialize the database.")

        database.set_default_database_mode(database_mode)

        self._session = InferenceSession(
            model=get_perf_model(config, model, workload_distribution),
            backend=get_backend(self.config.backend_name),
            database=database,
        )

    def _resolve_database_version(
        self,
        system: str,
        backend: str,
        version: Optional[str],
        systems_paths: list[str],
    ) -> str:
        # AIC only serves versions within its queryable slots (current/previous/next);
        # fall back to the 'current' slot when the requested version is unlisted.
        if not version:
            return "current"
        try:
            return resolve_query_version(
                system, backend, version, systems_paths=systems_paths
            )
        except ValueError:
            resolved = resolve_query_version(
                system, backend, "current", systems_paths=systems_paths
            )
            logger.warning(
                "AIConfigurator has no queryable %s/%s on %s, falling back to "
                "'current' slot %s. Set AIC_ALLOW_UNLISTED_VERSIONS=1 to force "
                "the raw version.",
                backend,
                version,
                system,
                resolved,
            )
            return resolved

    def _get_database_mode(self, mode: str) -> DatabaseMode:
        return {
            "SILICON": DatabaseMode.SILICON,
            "HYBRID": DatabaseMode.HYBRID,
            "EMPIRICAL": DatabaseMode.EMPIRICAL,
            "SOL": DatabaseMode.SOL,
            "SOL_FULL": DatabaseMode.SOL_FULL,
        }.get(mode.upper(), DatabaseMode.SILICON)

    def ctx_attn_flops_ratio_with_avg(self, reqs: list[ScheduleRequest]) -> float:
        if len(reqs) == 1:
            return 1.0
        mean_past = np.mean([req.past_kv_length for req in reqs])
        mean_input = np.mean([req.extend_length for req in reqs])
        avg_flops = (mean_past + mean_past + mean_input) * mean_input / 2 * len(reqs)

        actual_flops = 0
        for req in reqs:
            actual_flops += (
                (req.past_kv_length + req.past_kv_length + req.extend_length)
                * req.extend_length
                / 2
            )

        return actual_flops / avg_flops

    def predict_infer_latency_dict(self, batch: ScheduleBatch) -> dict:
        # Returns latency details for debugging operators.
        if batch.is_decode():
            # Decode: output sequence length (osl) = 2, input sequence length (isl) = mean(past_kv_length)
            isl = int(np.mean([req.past_kv_length for req in batch.reqs]))
            runtime_config = RuntimeConfig(batch_size=batch.batch_size, isl=isl, osl=2)
            if self.enable_oom_check:
                summary = self._session.run_static(runtime_config, mode="static_gen")
                latency_dict = summary.get_generation_latency_dict()
            else:
                # faster path
                results = self._session._backend._run_static_breakdown(
                    self._session._model,
                    self._session._database,
                    runtime_config,
                    mode="static_gen",
                )
                latency_dict = results[2]
        else:
            # Prefill: output sequence length (osl) = 1, input sequence length (isl) = mean(past_kv + input), prefix = mean(past_kv)
            mean_past = np.mean([req.past_kv_length for req in batch.reqs])
            mean_input = np.mean([req.extend_length for req in batch.reqs])
            isl = int(mean_past + mean_input)
            prefix = int(mean_past)

            seq_imbalance_correction_scale = self.ctx_attn_flops_ratio_with_avg(
                batch.reqs
            )
            extra_kwargs = (
                {"seq_imbalance_correction_scale": seq_imbalance_correction_scale}
                if seq_imbalance_correction_scale >= 0.4
                else {}
            )
            runtime_config = RuntimeConfig(
                batch_size=batch.batch_size,
                isl=isl,
                prefix=prefix,
                osl=1,
                **extra_kwargs,
            )

            if self.enable_oom_check:
                summary = self._session.run_static(runtime_config, mode="static_ctx")
                latency_dict = summary.get_context_latency_dict()
            else:
                # faster path
                results = self._session._backend._run_static_breakdown(
                    self._session._model,
                    self._session._database,
                    runtime_config,
                    mode="static_ctx",
                )
                latency_dict = results[0]
        return latency_dict

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        latency_dict = self.predict_infer_latency_dict(batch)
        infer_time = sum(latency_dict.values())

        if batch.is_decode():
            infer_time *= self.decode_scale_factor
        else:
            infer_time = max(
                infer_time * self.prefill_scale_factor, self.prefill_min_latency
            )

        return infer_time / 1e3
