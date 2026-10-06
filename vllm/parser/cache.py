import hashlib
import threading
from typing import Dict, Tuple, Optional
from collections import OrderedDict
from vllm.parser.abstract_parser import Parser

class ParserCacheManager:
    """
    LRU Cache for parsers to avoid redundant parsing in streaming chat derender
    and RL rollouts. Keyed by (request_id, choice_index).
    
    Verifies that the prompt digest, output_token_ids, and output_chunk_lens 
    match the history exactly to ensure streaming continuity.
    """
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: OrderedDict[Tuple[str, int], Tuple[str, Parser, list[int], list[int]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self,
        request_id: str,
        choice_index: int,
        prompt_digest: str,
        output_token_ids: list[int],
        output_chunk_lens: list[int]
    ) -> Optional[Parser]:
        if self.max_size <= 0:
            return None
            
        key = (request_id, choice_index)
        with self._lock:
            if key not in self._cache:
                return None
            
            cached_digest, parser, cached_token_ids, cached_chunk_lens = self._cache[key]
            
            # Verify prompt hasn't mutated under the same ID
            if cached_digest != prompt_digest:
                del self._cache[key]
                return None
                
            # Verify exact history match for continuity
            if cached_token_ids != output_token_ids or cached_chunk_lens != output_chunk_lens:
                del self._cache[key]
                return None
                
            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            return parser

    def put(
        self,
        request_id: str,
        choice_index: int,
        prompt_digest: str,
        output_token_ids: list[int],
        output_chunk_lens: list[int],
        parser: Parser
    ) -> None:
        if self.max_size <= 0:
            return
            
        key = (request_id, choice_index)
        with self._lock:
            self._cache[key] = (prompt_digest, parser, output_token_ids, output_chunk_lens)
            self._cache.move_to_end(key)
            
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
