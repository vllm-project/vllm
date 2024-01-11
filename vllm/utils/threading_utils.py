import os
import traceback
from threading import Lock
from functools import wraps


def synchronized(method):
    """ Synchronization decorator at the instance level """
    outer_lock = Lock()
    method.__locks__ = {}

    @wraps(method)
    def synched_method(self, *args, **kwargs):
        # This step ensures that there's a unique lock for each instance
        with outer_lock: # Protects access to the __locks__ directory
            if self not in method.__locks__:
                method.__locks__[self] = Lock()
            lock = method.__locks__[self]

        with lock:
            return method(self, *args, **kwargs)

        return synced_method


def exit_on_error(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            os._exit(1)
    
    return wrapper
