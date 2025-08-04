# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def is_ray_initialized():
    """Check if Ray is initialized."""
    try:
        import ray
        return ray.is_initialized()
    except ImportError:
        return False


def is_in_ray_actor():
    """Check if we are in a Ray actor."""

    try:
        import ray
        return (ray.is_initialized()
                and ray.get_runtime_context().get_actor_id() is not None)
    except ImportError:
        return False
