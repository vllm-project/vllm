import threading
from concurrent.futures import ThreadPoolExecutor

from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group.tokenizer_group import (
    TokenizerGroup)
from vllm.utils import make_async

logger = init_logger(__name__)


class ThreadPoolTokenizerGroup(TokenizerGroup):
    """A threadpool of TokenizerGroups for async tokenization."""

    def __init__(self, *args, max_workers: int, **tokenizer_config):
        super().__init__(*args, **tokenizer_config)
        self.local = threading.local()

        def init_tokenizer():
            logger.info(
                f"Starting tokenizer thread {threading.current_thread().name}")
            self.local.tokenizer = TokenizerGroup(*args, **tokenizer_config)

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='tokenizer_thread',
            initializer=init_tokenizer,
        )

        self._encode_async = make_async(self._encode_local, self.executor)

    def _encode_local(self, *args, **kwargs):
        return self.local.tokenizer.encode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.executor.submit(self._encode_local, *args,
                                    **kwargs).result()

    async def encode_async(self, *args, **kwargs):
        return await self._encode_async(*args, **kwargs)
