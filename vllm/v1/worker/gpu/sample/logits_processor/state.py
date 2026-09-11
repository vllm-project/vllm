# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator
from itertools import chain

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor


class LogitsProcessors:
    """Encapsulates initialized V2 logits processor objects.

    Mirrors the V1 container but is bound to the V2 interface: V1 processors
    keep their state via ``BatchUpdate`` ledgers, which do not exist in V2.
    """

    def __init__(self, logitsprocs: Iterable[LogitsProcessor] | None = None) -> None:
        self.argmax_invariant: list[LogitsProcessor] = []
        self.non_argmax_invariant: list[LogitsProcessor] = []
        if logitsprocs:
            for logitproc in logitsprocs:
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)

    @property
    def all(self) -> Iterator[LogitsProcessor]:
        """Iterator over all logits processors."""
        return chain(self.argmax_invariant, self.non_argmax_invariant)
