from .storage import PagedSHMStorage


class PagedSHMServer:
    def __init__(self, size: int, block_size: int):
        self.storage = PagedSHMStorage(size=size, block_size=block_size, pin=False)




class PagedSHMServerProcess:
    pass