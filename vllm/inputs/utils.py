'''Utility functions for input types'''

def has_required_keys(d: dict, 
                      required_keys: list,
                      ) -> bool:
    return set(required_keys).issubset(d.keys())

def is_str(s,) -> bool:
    return isinstance(s, str)

def is_dict(d,) -> bool:
    return isinstance(d, dict)