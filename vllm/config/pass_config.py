# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, field
from pathlib import Path

from pydantic.dataclasses import dataclass

from vllm.compilation.inductor_pass import InductorPass
from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@config
@dataclass
class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config."""

    dump_graph_stages: list[str] = field(default_factory=list)
    """List of stages for which we want to dump the graph. Each pass defines
    its own stages (before, after, maybe in-between)."""
    dump_graph_dir: Path = Path(".")
    """Directory to dump the graphs."""
    # TODO(luka) better pass enabling system.
    enable_fusion: bool = True
    """Whether to enable the custom fusion pass."""
    enable_noop: bool = True
    """Whether to enable the custom no-op elimination pass."""
    enable_sequence_parallelism: bool = False
    """Whether to enable sequence parallelism."""
    enable_async_tp: bool = False
    """Whether to enable async TP."""

    def uuid(self):
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Do not include dump_graph_* in the hash - they don't affect
        compilation.
        """
        include = {
            "enable_fusion", "enable_noop", "enable_sequence_parallelism",
            "enable_async_tp"
        }
        dict_ = {k: v for k, v in asdict(self).items() if k in include}
        return InductorPass.hash_dict(dict_)

    def __post_init__(self) -> None:
        if not self.enable_noop and self.enable_fusion:
            logger.warning_once(
                "Fusion enabled but reshape elimination disabled. "
                "RMSNorm + quant (fp8) fusion might not work")
