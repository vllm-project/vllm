from typing import Optional

import vllm.envs as envs
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.logger import init_logger

logger = init_logger(__name__)

use_flux = False
if envs.VLLM_USE_FLUX:
    try:
        import flux  # noqa
        use_flux = True
        logger.info("Using flux kernels for collective communication fusion.")
    except ImportError:
        logger.info("Attempting to use flux but flux not installed.")
        use_flux = False

# Depends on arch, see auto_tile_shape in include/flux/gemm_hparams.h
# Can be 256 on sm80.
FLUX_TILE_SIZE: int = 128


# Heuristic to check if collective communication kernels should be used for a
# particular problem size.
def use_cc_kernels(m_shape: int, n_slices: Optional[int] = None) -> bool:
    if use_flux:
        if n_slices is None:
            n_slices = get_tensor_model_parallel_world_size()
        return (m_shape % (FLUX_TILE_SIZE * n_slices) == 0
                and m_shape >= FLUX_TILE_SIZE * n_slices)
    else:
        # For symmetric memory kernels.  TODO: Is this ok?
        return True
