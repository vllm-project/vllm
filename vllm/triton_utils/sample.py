import math

# This is a hardcoded limit in Triton (max block size).
MAX_TRITON_N_COLS = 131072


def get_num_triton_sampler_splits(n_cols: int) -> int:
    """Get the number of splits to use for Triton sampling.

    Triton has a limit on the number of columns it can handle, so we need to
    split the tensor and call the kernel multiple times if it's too large.
    """
    return math.ceil(n_cols / MAX_TRITON_N_COLS)
