import abc
import dataclasses
import gc
import itertools
import time
from typing import Callable

import numpy as np
import torch


class Benchmark(abc.ABC):

  def setup(self):
    pass

  def before_run(self):
    pass

  @abc.abstractmethod
  def run(self):
    pass

  def after_run(self):
    pass

  def teardown(self):
    pass


class wrap_benchmark(Benchmark):

  def __init__(self, fn_run: Callable[[], None]):
    self.fn_run = fn_run

  def run(self):
    self.fn_run()


@dataclasses.dataclass
class BenchResult:
  warmup: int
  repeat: int
  latency: np.ndarray

  def avg(self) -> np.ndarray:
    return np.mean(self.latency)

  def std(self) -> np.ndarray:
    return np.std(self.latency)

  def avg_std(self) -> np.ndarray:
    return self.avg(), self.std()


def bench(
    f: Callable[[], None],
    warmup: int = 100,
    repeat: int = 500,
) -> BenchResult:
  if isinstance(f, Benchmark):
    b = f
  else:
    b = wrap_benchmark(f)

  cache = torch.empty(256 * 2**20, dtype=torch.int8, device="cuda:0")
  b.setup()

  latency = np.zeros(repeat, dtype=np.float64)
  for i in range(-warmup, repeat):
    b.before_run()
    cache.zero_()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    b.run()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    b.after_run()

    if i >= 0:
      latency[i] = t1 - t0

  b.teardown()
  del cache
  return BenchResult(warmup, repeat, latency)


def gc_torch():
  gc.collect()
  torch.cuda.empty_cache()


def batched(iterable, n):
  "Batch data into tuples of length n. The last batch may be shorter."
  # batched('ABCDEFG', 3) --> ABC DEF G
  if n < 1:
    raise ValueError('n must be at least one')
  it = iter(iterable)
  while batch := list(itertools.islice(it, n)):
    yield batch
