'''Utility functions for input types'''


def has_required_keys(
    d: dict,
    required_keys: set,
) -> bool:
    return required_keys.issubset(d.keys())