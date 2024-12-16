from typing import List

from my_model_runner import DummyModelRunner


class DummyCacheEngine:
    pass


class DummyWorker:

    def __init__(self):
        self.cache_engine = List[DummyCacheEngine]
        self.model_runner = DummyModelRunner()
