try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "dev"
    __version__tuple__ = (0, 0, __version__)
