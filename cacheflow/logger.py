import logging


def _init_logger():
    logging.basicConfig(
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
        level=logging.INFO,
    )


_init_logger()


def get_logger(name: str):
    return logging.getLogger(name)
