# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def is_ray_initialized():
    """Check if Ray is initialized."""
    try:
        import ray
        if hasattr(ray, 'is_initialized'):
            return ray.is_initialized()
        else:
            # Fallback for older Ray versions or import issues
            return hasattr(
                ray, '_global_state') and ray._global_state.is_initialized
    except (ImportError, AttributeError):
        return False


def is_in_ray_actor():
    """Check if we are in a Ray actor."""

    try:
        import ray
        if hasattr(ray, 'is_initialized'):
            ray_initialized = ray.is_initialized()
        else:
            # Fallback for older Ray versions or import issues
            ray_initialized = hasattr(
                ray, '_global_state') and ray._global_state.is_initialized

        return (ray_initialized
                and ray.get_runtime_context().get_actor_id() is not None)
    except (ImportError, AttributeError):
        return False
