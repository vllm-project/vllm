# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml

from vllm.logger import init_logger

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.cli.pdjob')

DEFAULT_TIMEOUT_IN_SECONDS = 600  # 10 minus


@dataclass
class VllmConfig:
    replicas: int
    envs: Dict[str, str]
    role_num_gpus: int
    role_params: Dict[str, Any]
    unique_role_extra_params: str


@dataclass
class Config:
    timeout: int
    num_gpus: int
    gpus_per_worker: int
    prefill_config: VllmConfig
    decode_config: VllmConfig
    scheduler_config: Dict[str, Any]
    working_dir: str
    kv_connector: str
    envs: Dict[str, str]


def _parse_config(config: Dict[str, Any]) -> Config:
    num_gpus = config["num_gpus"]
    gpus_per_worker = config["gpus_per_worker"]

    assert num_gpus % gpus_per_worker == 0, "num_gpus must be a multiple of gpus_per_worker"

    timeout = DEFAULT_TIMEOUT_IN_SECONDS
    if "timeout" in config:
        timeout = int(config["timeout"])

    working_dir = config["working_dir"]
    
    # make sure working_dir exists
    if not os.path.exists(working_dir):
        os.makedirs(working_dir, exist_ok=True)
    
    envs = {}
    if "envs" in config:
        envs = config["envs"] or {}

    general_params = config["general"]["params"]
    general_extra_params = general_params.pop("_extra_params", "")

    def _parse_vllm_config(vllm_config: Dict[str, Any]) -> Tuple[VllmConfig, str]:
        envs = {}
        if "envs" in vllm_config:
            envs = vllm_config["envs"] or {}

        envs = {str(k): str(v) for k, v in envs.items()}
        role_num_gpus = vllm_config["num_gpus"]

        replicas = vllm_config["replicas"]
        assert replicas > 0, "replicas must be greater than 0"

        kv_connector = None

        unique_role_param = vllm_config["params"]
        unique_role_extra_params = unique_role_param.pop("_extra_params", "")

        role_params = general_params.copy()
        role_params.update(unique_role_param)
        unique_role_extra_params = general_extra_params + " " + unique_role_extra_params
        for _ in range(replicas):
            # Use an external global variable for global incrementing rank_id
            logger.info(f"Parsing vllm config args for {role_params}")
            kv_cfg_str = role_params["kv_transfer_config"]
            if kv_cfg_str.strip().startswith("{"):
                # Directly a JSON string
                try:
                    kv_parsed = json.loads(kv_cfg_str)
                    current_kv_connector = kv_parsed["kv_connector"]
                    if kv_connector is None:
                        kv_connector = current_kv_connector.lower()
                    else:
                        assert kv_connector == current_kv_connector.lower(), "kv_connector must be equal"
                except Exception as e:
                    logger.error("Failed to parse kv-transfer-config: %s", e)
                    raise e
            else:
                raise ValueError("`kv_transfer_config` is not found in kv_parsed")

        return VllmConfig(replicas, envs, role_num_gpus, role_params, unique_role_extra_params), kv_connector

    prefill_config, prefill_kv_connector = _parse_vllm_config(config["prefill"])
    decode_config, decode_kv_connector = _parse_vllm_config(config["decode"])

    logger.info(f"prefill kv_connector: {prefill_kv_connector} and decode kv_connector: {decode_kv_connector} ")

    scheduler_config = config["scheduler"]

    return Config(timeout, num_gpus, gpus_per_worker, prefill_config, decode_config, scheduler_config, working_dir,
                  prefill_kv_connector, envs)


def read_config(yaml_path: str) -> Config:
    try:
        with open(yaml_path, encoding="utf-8") as config_file:
            load_config = yaml.safe_load(config_file)
            return _parse_config(load_config)
    except Exception as ex:
        logger.error(
            "Unable to read the config file at %s. \
            Make sure path is correct", yaml_path)
        raise ex
