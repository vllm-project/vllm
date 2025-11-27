# SPDX-License-Identifier: Apache-2.0
"""
Configuration module for Prefill-Decode disaggregated job management.

This module provides:
- Configuration data classes (VllmConfig, Config)
- YAML configuration file parsing
- Configuration validation and processing
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml

from vllm.logger import init_logger

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.cli.pdjob')

# Default timeout in seconds (10 minutes)
DEFAULT_TIMEOUT_IN_SECONDS = 600


@dataclass
class VllmConfig:
    """
    Configuration for a vLLM service role (Prefill or Decode).
    
    Attributes:
        replicas: Number of service replicas to start
        envs: Environment variables for the service
        role_num_gpus: Total number of GPUs required for this role
        role_params: Service-specific parameters
        unique_role_extra_params: Additional command-line parameters for this role
    """
    replicas: int
    envs: Dict[str, str]
    role_num_gpus: int
    role_params: Dict[str, Any]
    unique_role_extra_params: str


@dataclass
class Config:
    """
    Main configuration class for Prefill-Decode disaggregated job.
    
    Attributes:
        timeout: Timeout in seconds for various operations
        num_gpus: Total number of GPUs available
        gpus_per_worker: Number of GPUs per worker
        prefill_config: Configuration for Prefill services
        decode_config: Configuration for Decode services
        scheduler_config: Configuration for scheduler/proxy server
        working_dir: Working directory for the job
        kv_connector: KV connector type (e.g., "nixl", "p2p")
        envs: Global environment variables
    """
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
    """
    Parse configuration dictionary into Config object.
    
    Validates and processes the configuration, including:
    - GPU allocation validation
    - Working directory creation
    - Environment variable processing
    - Prefill and Decode configuration parsing
    
    Args:
        config: Configuration dictionary from YAML file
        
    Returns:
        Config: Parsed configuration object
        
    Raises:
        AssertionError: If num_gpus is not a multiple of gpus_per_worker
        ValueError: If kv_transfer_config is invalid
    """
    num_gpus = config["num_gpus"]
    gpus_per_worker = config["gpus_per_worker"]

    # Validate that num_gpus is divisible by gpus_per_worker
    assert num_gpus % gpus_per_worker == 0, "num_gpus must be a multiple of gpus_per_worker"

    # Get timeout, default to DEFAULT_TIMEOUT_IN_SECONDS if not specified
    timeout = DEFAULT_TIMEOUT_IN_SECONDS
    if "timeout" in config:
        timeout = int(config["timeout"])

    working_dir = config["working_dir"]
    
    # Make sure working_dir exists, create if it doesn't
    if not os.path.exists(working_dir):
        os.makedirs(working_dir, exist_ok=True)
    
    # Get global environment variables
    envs = {}
    if "envs" in config:
        envs = config["envs"] or {}

    # Extract general parameters and extra params
    general_params = config["general"]["params"]
    general_extra_params = general_params.pop("_extra_params", "")

    def _parse_vllm_config(vllm_config: Dict[str, Any]) -> Tuple[VllmConfig, str]:
        """
        Parse vLLM service configuration (Prefill or Decode).
        
        Processes role-specific configuration, validates replicas, merges parameters,
        and extracts KV connector information.
        
        Args:
            vllm_config: Role-specific configuration dictionary
            
        Returns:
            Tuple[VllmConfig, str]: Configuration object and KV connector type
            
        Raises:
            AssertionError: If replicas <= 0 or KV connectors don't match
            ValueError: If kv_transfer_config is invalid
        """
        # Get role-specific environment variables
        envs = {}
        if "envs" in vllm_config:
            envs = vllm_config["envs"] or {}

        # Convert all env values to strings
        envs = {str(k): str(v) for k, v in envs.items()}
        role_num_gpus = vllm_config["num_gpus"]

        # Validate replicas count
        replicas = vllm_config["replicas"]
        assert replicas > 0, "replicas must be greater than 0"

        kv_connector = None

        # Extract role-specific parameters and extra params
        unique_role_param = vllm_config["params"]
        unique_role_extra_params = unique_role_param.pop("_extra_params", "")

        # Merge general params with role-specific params
        role_params = general_params.copy()
        role_params.update(unique_role_param)
        # Combine extra params from general and role-specific configs
        unique_role_extra_params = general_extra_params + " " + unique_role_extra_params
        
        # Parse KV connector from kv_transfer_config
        # Use an external global variable for global incrementing rank_id
        for _ in range(replicas):
            logger.info(f"Parsing vllm config args for {role_params}")
            kv_cfg_str = role_params["kv_transfer_config"]
            if kv_cfg_str.strip().startswith("{"):
                # Directly a JSON string
                try:
                    kv_parsed = json.loads(kv_cfg_str)
                    current_kv_connector = kv_parsed["kv_connector"]
                    if kv_connector is None:
                        # First replica: set the KV connector
                        kv_connector = current_kv_connector.lower()
                    else:
                        # Subsequent replicas: verify consistency
                        assert kv_connector == current_kv_connector.lower(), "kv_connector must be equal"
                except Exception as e:
                    logger.error("Failed to parse kv-transfer-config: %s", e)
                    raise e
            else:
                raise ValueError("`kv_transfer_config` is not found in kv_parsed")

        return VllmConfig(replicas, envs, role_num_gpus, role_params, unique_role_extra_params), kv_connector

    # Parse Prefill and Decode configurations
    prefill_config, prefill_kv_connector = _parse_vllm_config(config["prefill"])
    decode_config, decode_kv_connector = _parse_vllm_config(config["decode"])

    logger.info(f"prefill kv_connector: {prefill_kv_connector} and decode kv_connector: {decode_kv_connector} ")

    # Get scheduler configuration
    scheduler_config = config["scheduler"]

    # Create and return Config object
    # Note: Using prefill_kv_connector as the main kv_connector
    return Config(timeout, num_gpus, gpus_per_worker, prefill_config, decode_config, scheduler_config, working_dir,
                  prefill_kv_connector, envs)


def read_config(yaml_path: str) -> Config:
    """
    Read and parse configuration from YAML file.
    
    Opens the YAML file, loads its contents, and parses into a Config object.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Config: Parsed configuration object
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        Exception: Any exception raised during parsing
    """
    try:
        with open(yaml_path, encoding="utf-8") as config_file:
            # Load YAML content into dictionary
            load_config = yaml.safe_load(config_file)
            return _parse_config(load_config)
    except Exception as ex:
        logger.error(
            "Unable to read the config file at %s. "
            "Make sure path is correct", yaml_path)
        raise ex
