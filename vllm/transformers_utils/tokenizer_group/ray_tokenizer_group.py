import asyncio
import os
from typing import List, Optional

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import PreTrainedTokenizer

from vllm.config import TokenizerPoolConfig
from vllm.engine.ray_utils import ray
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (
    BaseTokenizerGroup)
from vllm.transformers_utils.tokenizer_group.tokenizer_group import (
    TokenizerGroup)


class RayTokenizerGroupPool(BaseTokenizerGroup):
    """A Ray-based pool of TokenizerGroups for async tokenization."""

    # Class to use for workers making up the pool.
    _worker_cls = TokenizerGroup

    @classmethod
    def from_config(cls, tokenizer_pool_config: TokenizerPoolConfig,
                    **init_kwargs) -> "RayTokenizerGroupPool":
        ray_actor_options = (tokenizer_pool_config.extra_config or {
            "num_cpus": 0
        })
        ray_actor_options.setdefault(
            "scheduling_strategy",
            NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=True))

        # Carry over the env vars to the actors.
        # This is necessary for API keys and such.
        ray_actor_options.setdefault("runtime_env", {})
        _carry_over_env_vars_to_runtime_env(ray_actor_options["runtime_env"])

        init_kwargs["num_actors"] = tokenizer_pool_config.pool_size
        init_kwargs["ray_actor_options"] = ray_actor_options

        return cls(**init_kwargs)

    def __init__(self, tokenizer_id: str, enable_lora: bool, max_num_seqs: int,
                 max_input_length: Optional[int], num_actors: int,
                 ray_actor_options: dict, **tokenizer_config):
        # Store a local copy of the TokenizerGroup for quick access
        # to underlying HF tokenizers.
        self._local_tokenizer_group = self._worker_cls(
            tokenizer_id=tokenizer_id,
            enable_lora=enable_lora,
            max_num_seqs=max_num_seqs,
            max_input_length=max_input_length,
        )

        ray_tokenizer_group_cls = ray.remote(
            self._worker_cls).options(**ray_actor_options)
        self.tokenizer_actors = [
            ray_tokenizer_group_cls.remote(tokenizer_id, enable_lora,
                                           max_num_seqs, max_input_length,
                                           **tokenizer_config)
            for _ in range(num_actors)
        ]
        self._idle_actors: Optional[asyncio.Queue] = None

    @property
    def pool_size(self) -> int:
        return len(self.tokenizer_actors)

    def ping(self):
        return ray.get(
            [actor.ping.remote() for actor in self.tokenizer_actors])

    def _ensure_queue_initialized(self):
        if self._idle_actors is None:
            self._idle_actors = asyncio.Queue()
            for actor in self.tokenizer_actors:
                self._idle_actors.put_nowait(actor)

    def encode(self,
               prompt: str,
               request_id: Optional[str] = None,
               lora_request: Optional[LoRARequest] = None) -> List[int]:
        """Encode a prompt using the tokenizer group.

        We pick an idle actor and use it to encode the prompt.
        The actor is then put back in the queue for future use.
        This is blocking.
        """
        self._ensure_queue_initialized()

        if self._idle_actors.empty():
            raise RuntimeError("No idle actors available.")
        actor = self._idle_actors.get_nowait()
        try:
            ret = ray.get(
                actor.encode.remote(request_id=request_id,
                                    prompt=prompt,
                                    lora_request=lora_request))
        finally:
            # Put the actor back in the queue.
            # This is done in a finally block to ensure that the actor is
            # always put back in the queue, even if an exception/cancellation
            # is raised.
            self._idle_actors.put_nowait(actor)
        return ret

    async def encode_async(
            self,
            prompt: str,
            request_id: Optional[str] = None,
            lora_request: Optional[LoRARequest] = None) -> List[int]:
        """Encode a prompt using the tokenizer group.

        We pick an idle actor and use it to encode the prompt.
        If there are no idle actors, we wait until one becomes
        available.
        The actor is then put back in the queue for future use.
        This is non-blocking.
        """
        self._ensure_queue_initialized()

        actor = await self._idle_actors.get()
        try:
            ret = await actor.encode.remote(request_id=request_id,
                                            prompt=prompt,
                                            lora_request=lora_request)
        finally:
            # Put the actor back in the queue.
            # This is done in a finally block to ensure that the actor is
            # always put back in the queue, even if an exception/cancellation
            # is raised.
            self._idle_actors.put_nowait(actor)
        return ret

    def get_max_input_len(self,
                          lora_request: Optional[LoRARequest] = None
                          ) -> Optional[int]:
        """Get the maximum input length for the LoRA request."""
        return self._local_tokenizer_group.get_max_input_len(lora_request)

    def get_lora_tokenizer(
            self,
            lora_request: Optional[LoRARequest] = None
    ) -> "PreTrainedTokenizer":
        return self._local_tokenizer_group.get_lora_tokenizer(lora_request)

    async def get_lora_tokenizer_async(
            self,
            lora_request: Optional[LoRARequest] = None
    ) -> "PreTrainedTokenizer":
        return await self._local_tokenizer_group.get_lora_tokenizer_async(
            lora_request)


def _carry_over_env_vars_to_runtime_env(runtime_env: dict) -> None:
    """Copy over all current process environment variables to the runtime_env.

    The variables in runtime_env will take precedence over the current process
    environment variables.

    runtime_env will be modified in place."""
    env_vars = os.environ.copy()
    runtime_env.setdefault("env_vars", {})
    env_vars.update(runtime_env["env_vars"])
    runtime_env["env_vars"] = env_vars
