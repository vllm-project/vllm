import os

from vllm_simulator.utils.logger import get_logger

logger = get_logger()


class Envs:
    @classmethod
    def config_path(cls) -> str:
        VLLM_SIMULATOR_CONFIG_PATH = os.getenv("VLLM_SIMULATOR_CONFIG_PATH")
        if not VLLM_SIMULATOR_CONFIG_PATH or not os.path.exists(
            VLLM_SIMULATOR_CONFIG_PATH
        ):
            raise RuntimeError(
                f"The configuration path is not set or does not exist({VLLM_SIMULATOR_CONFIG_PATH}). Please set it using the system variable VLLM_SIMULATOR_CONFIG_PATH"
            )
        return VLLM_SIMULATOR_CONFIG_PATH

    @classmethod
    def max_tokens(cls) -> int | None:
        """Return the output-length override, or None when unset."""
        value = os.getenv("VLLM_SIMULATOR_MAX_TOKENS")
        return int(value) if value is not None else None

    @classmethod
    def simulation_mode(cls) -> str:
        VLLM_SIMULATOR_OUTPUT_MODE = os.getenv(
            "VLLM_SIMULATOR_OUTPUT_MODE", "NON_BLOCKING"
        ).upper()
        assert VLLM_SIMULATOR_OUTPUT_MODE in ("BLOCKING", "NON_BLOCKING")
        return VLLM_SIMULATOR_OUTPUT_MODE
