__version__ = "0.0.0"


def jit(cls):

    def disable_function(func):

        def disabled(*args, **kwargs):
            return

        return disabled

    return disable_function
