# ABOUTME: Runtime configuration helpers for EPS execution.
# ABOUTME: Converts engine-level EPS settings into runner-friendly values.

from dataclasses import dataclass

from vllm.config.eps import EpsConfig


@dataclass
class EpsRuntimeConfig:
    enabled: bool
    scope: str
    method: str
    head_scope: str
    group_blocks: int
    last_n: int
    alpha: float
    dim: int
    top_pages: int | None = None
    strict: bool = False
    sentinel: int = 0
    metrics_path: str | None = None

    @property
    def union_enabled(self) -> bool:
        return self.enabled and self.scope == "union" and self.method != "off"


def to_runtime_config(cfg: EpsConfig) -> EpsRuntimeConfig:
    return EpsRuntimeConfig(
        enabled=cfg.enabled or cfg.method != "off",
        scope=cfg.scope,
        method=cfg.method,
        head_scope=cfg.heads,
        group_blocks=cfg.group_blocks,
        last_n=cfg.last_n,
        alpha=cfg.alpha,
        dim=cfg.dim,
        top_pages=cfg.top_pages,
        strict=cfg.strict,
        metrics_path=cfg.metrics_path,
    )
