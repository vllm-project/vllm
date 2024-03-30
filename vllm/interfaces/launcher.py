# launcher interface, as proposed in
# https://github.com/vllm-project/vllm/issues/3587
# `Launcher` is responsible for creating workers.

import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Type, TypeVar


class DistributedTask(ABC):

    def __init__(self, env: Dict[str, str], args, kwargs):
        self.update_env(env)
        self.run(*args, **kwargs)

    def update_env(self, env: Dict[str, str]):
        for k, v in env.items():
            if k in os.environ:
                warnings.warn(
                    f"Overwriting environment variable {k} "
                    f"from {os.environ[k]} to {v}",
                    stacklevel=2)
            os.environ[k] = v

    @abstractmethod
    def run(self, *args, **kwargs):
        # usually:
        # initialize coordinator and communicator
        # initialize device
        # initialize model
        # warmup model
        # run model
        pass


T = TypeVar('T', bound=DistributedTask)
SubClassOfDistributedTask = Type[T]


class Launcher(ABC):

    @abstractmethod
    def launch(self, task_type: SubClassOfDistributedTask):
        # this is a dunmmy implementation, but captures the idea
        # 1. prepare environment variables, args, kwargs for each task
        n_tasks = 4
        envs = [{} for _ in range(n_tasks)]
        args = [() for _ in range(n_tasks)]
        kwargs = [{} for _ in range(n_tasks)]
        # 2. create tasks (typically these tasks should be run in parallel)
        # note that creating a task will also run it. This is designed for
        # simple launcher like multiprocessing, where we can only pass a
        # function to run, and cannot do any further operations on the task.
        for env, arg, kwarg in zip(envs, args, kwargs):
            task_type(env, arg, kwarg)
        # 3. wait for tasks to finish
