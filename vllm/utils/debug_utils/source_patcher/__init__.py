from vllm.utils.debug_utils.source_patcher.code_patcher import (
    CodePatcher,
    apply_patches_from_config,
    patch_function,
)
from vllm.utils.debug_utils.source_patcher.types import (
    EditSpec,
    PatchApplicationError,
    PatchConfig,
    PatchSpec,
    PatchState,
)

__all__ = [
    "CodePatcher",
    "apply_patches_from_config",
    "patch_function",
    "EditSpec",
    "PatchApplicationError",
    "PatchConfig",
    "PatchSpec",
    "PatchState",
]
