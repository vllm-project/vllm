# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from typing import List, Optional

try:
    from ray.exceptions import ActorDiedError  # type: ignore
except ImportError:
    # For older versions of Ray
    from ray.exceptions import RayActorError as ActorDiedError  # type: ignore
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from vllm.config import TokenizerPoolConfig
from vllm.executor.ray_utils import ray
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .base_tokenizer_group import BaseTokenizerGroup
from .tokenizer_group import TokenizerGroup

logger = init_logger(__name__)


class RayTokenizerGroupPool(BaseTokenizerGroup):
    """A Ray-based pool of TokenizerGroups for async tokenization."""

    # Class to use for workers making up the pool.
    _worker_cls = TokenizerGroup

    @classmethod
    def from_config(cls, tokenizer_pool_config: Optional[TokenizerPoolConfig],
                    **init_kwargs) -> "RayTokenizerGroupPool":
        if not tokenizer_pool_config:
            raise ValueError("tokenizer_pool_config must not be None.")
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
        self._tokenizer_config = {
            "tokenizer_id": tokenizer_id,
            "enable_lora": enable_lora,
            "max_num_seqs": max_num_seqs,
            "max_input_length": max_input_length,
            **tokenizer_config
        }
        self._local_tokenizer_group = self._worker_cls(
            **self._tokenizer_config, )

        self._ray_tokenizer_group_cls = ray.remote(
            self._worker_cls).options(**ray_actor_options)  # type: ignore
        self.tokenizer_actors = [self._init_actor() for _ in range(num_actors)]
        self._idle_actors: Optional[asyncio.Queue] = None

        # If set, actor is unhealthy. Will reraise on the next
        # check_health call.
        self._exception: Optional[ActorDiedError] = None

    def _init_actor(self) -> ray.ObjectRef:
        return self._ray_tokenizer_group_cls.remote(**self._tokenizer_config)

    @property
    def pool_size(self) -> int:
        return len(self.tokenizer_actors)

    def ping(self):
        return ray.get([
            actor.ping.remote()  # type: ignore
            for actor in self.tokenizer_actors
        ])

    def _ensure_queue_initialized(self):
        if self._idle_actors is None:
            self._idle_actors = asyncio.Queue()
            for actor in self.tokenizer_actors:
                self._idle_actors.put_nowait(actor)

    def _finalize_encode(self, actor: ray.ObjectRef,
                         original_actor: ray.ObjectRef, actor_is_alive: bool):
        assert self._idle_actors is not None
        # Cleanup the dead actor.
        if not actor_is_alive or original_actor is not actor:
            self.tokenizer_actors.remove(original_actor)
        if actor_is_alive:
            # Put the actor back in the queue.
            # This is done in a finally block to ensure that the actor is
            # always put back in the queue, even if an exception/cancellation
            # is raised.
            self._idle_actors.put_nowait(actor)
            # Add back the new actor.
            if original_actor is not actor:
                self.tokenizer_actors.append(actor)

    def encode(self,
               prompt: str,
               request_id: Optional[str] = None,
               lora_request: Optional[LoRARequest] = None,
               add_special_tokens: Optional[bool] = None) -> List[int]:
        """Encode a prompt using the tokenizer group.

        We pick an idle actor and use it to encode the prompt.
        The actor is then put back in the queue for future use.
        This is blocking.
        """
        self.check_health()
        self._ensure_queue_initialized()
        assert self._idle_actors is not None

        if self._idle_actors.empty():
            raise RuntimeError("No idle actors available.")
        actor = self._idle_actors.get_nowait()
        actor_is_alive = True
        original_actor = actor
        try:
            ret = ray.get(
                actor.encode.remote(request_id=request_id,
                                    prompt=prompt,
                                    lora_request=lora_request,
                                    add_special_tokens=add_special_tokens))
        except ActorDiedError as e:
            # If the actor is dead, we first try to reinitialize it.
            logger.warning("%s died with ActorDiedError, reinitializing.",
                           actor,
                           exc_info=e)
            actor = self._init_actor()
            try:
                ret = ray.get(
                    actor.encode.remote(request_id=request_id,
                                        prompt=prompt,
                                        lora_request=lora_request,
                                        add_special_tokens=add_special_tokens))
            except ActorDiedError as e:
                logger.error(
                    "%s died for second time in a row, marking "
                    "RayTokenizerGroupPool as unhealthy.", actor)
                actor_is_alive = False
                if not self._exception:
                    self._exception = e
                self.check_health()
        finally:
            self._finalize_encode(actor, original_actor, actor_is_alive)
        return ret

    async def encode_async(
            self,
            prompt: str,
            request_id: Optional[str] = None,
            lora_request: Optional[LoRARequest] = None,
            add_special_tokens: Optional[bool] = None) -> List[int]:
        """Encode a prompt using the tokenizer group.

        We pick an idle actor and use it to encode the prompt.
        If there are no idle actors, we wait until one becomes
        available.
        The actor is then put back in the queue for future use.
        This is non-blocking.
        """
        self.check_health()
        self._ensure_queue_initialized()
        assert self._idle_actors is not None

        actor = await self._idle_actors.get()
        actor_is_alive = True
        original_actor = actor
        try:
            ret = await actor.encode.remote(
                request_id=request_id,
                prompt=prompt,
                lora_request=lora_request,
                add_special_tokens=add_special_tokens)
        except ActorDiedError as e:
            # If the actor is dead, we first try to reinitialize it.
            logger.warning("%s died with ActorDiedError, reinitializing.",
                           actor,
                           exc_info=e)
            actor = self._init_actor()
            try:
                ret = await actor.encode.remote(
                    request_id=request_id,
                    prompt=prompt,
                    lora_request=lora_request,
                    add_special_tokens=add_special_tokens)
            except ActorDiedError as e:
                logger.error(
                    "%s died for second time in a row, marking "
                    "RayTokenizerGroupPool as unhealthy.", actor)
                actor_is_alive = False
                if not self._exception:
                    self._exception = e
                self.check_health()
        finally:
            self._finalize_encode(actor, original_actor, actor_is_alive)
        return ret

    def get_max_input_len(self,
                          lora_request: Optional[LoRARequest] = None
                          ) -> Optional[int]:
        """Get the maximum input length for the LoRA request."""
        return self._local_tokenizer_group.get_max_input_len(lora_request)

    def get_lora_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self._local_tokenizer_group.get_lora_tokenizer(lora_request)

    async def get_lora_tokenizer_async(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return await self._local_tokenizer_group.get_lora_tokenizer_async(
            lora_request)

    def check_health(self):
        if self._exception:
            raise RuntimeError(
                "TokenizerGroupPool is unhealthy.") from self._exception


def _carry_over_env_vars_to_runtime_env(runtime_env: dict) -> None:
    """Copy over all current process environment variables to the runtime_env.

    The variables in runtime_env will take precedence over the current process
    environment variables.

    runtime_env will be modified in place."""
    env_vars = os.environ.copy()
    runtime_env.setdefault("env_vars", {})
    env_vars.update(runtime_env["env_vars"])
    runtime_env["env_vars"] = env_vars
