"""`vllm` CLI with the collector hooks installed.

The baseline is collected on Model Runner V2, the runner the simulator
models.

Usage:
    python -m vllm_simulator.collector.launch_server serve <model> [vLLM args...]
"""

import os
import sys

from vllm_simulator.collector.hook import (
    C_EngineCoreHook,
    C_SchedulerHook,
    C_VLLMEngineArgsHook,
    C_WorkerHook,
)
from vllm_simulator.hook import install_class_hooks

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

install_class_hooks(
    [
        C_VLLMEngineArgsHook,
        C_WorkerHook,
        C_SchedulerHook,
        C_EngineCoreHook,
    ]
)

from vllm.entrypoints.cli.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
