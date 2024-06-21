from typing import Callable, Optional
from vllm.logger import init_logger

logger = init_logger(__name__)


class CodeCache:
    """
    The CodeCache is a simple map from mangled function names to Callables.

    The CodeCache can be used to store the results of compiled code so that the
    same Callable can be reused rather than needing to be recompiled.

    Mangled function names should be generated with (or be compatible with) the
    'utils.mangle_name' function.

    The 'disable' option turns off the cache, so it will always call the generate
    function in 'lookup_or_create'.

    Note: the CodeCache can be initialized with pre-compiled functions.
    """
    def __init__(self, disable: bool = False):
        self.cache = dict()
        self.disable = disable

    """
    Lookup a Callable for a function based on the 'mangled_name'.  If the name
    is not present in the cache, call the supplied 'generator' to create
    the Callable to be associated with the 'mangled_name'.  If the
    generator fails for any reason a None will be stored in the map and
    returned instead of a Callable.  This will prevent any failed generators
    from being called repeatedly.
    """
    def lookup_or_create(self, mangled_name: str,
                         generator: Callable) -> Optional[Callable]:
        if self.disable or not mangled_name in self.cache:
            try:
                logger.debug(f"generating code for {mangled_name}")
                self.cache[mangled_name] = generator()
            except Exception as ex:
                self.cache[mangled_name] = None
                raise ex
        else:
            logger.debug(f"cache hit for {mangled_name}")
        return self.cache[mangled_name]

    """
    Add a new entry to the cache.  Return False if an entry with the
    given name already exists.
    """
    def add(mangled_name: str, fn: Optional[Callable]) -> bool:
        if mangled_name in self.cache:
            return False
        self.cache[mangled_name] = fn
        return True
