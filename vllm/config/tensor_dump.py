from pydantic import Field

from vllm.config.utils import config


@config
class TensorDumpOptions:
    """Tensor capture options that may be changed at runtime."""

    enabled: bool = True
    """Capture real model forwards."""
    patterns: list[str] = Field(default_factory=lambda: ["*"])
    """Shell-style globs matched against tensor names."""
    ranks: list[int] | None = None
    """Distributed ranks to capture; None selects all ranks."""
    max_steps: int = Field(default=1, ge=1)
    """Maximum captured forwards per worker between resets."""


@config
class TensorDumpConfig(TensorDumpOptions):
    """Configuration for eager Model Runner V2 tensor dumping."""

    output_dir: str | None = Field(default=None, min_length=1)
    """Directory for worker dumps. None disables dumping.
    This path is local to each worker and cannot be changed at runtime."""
