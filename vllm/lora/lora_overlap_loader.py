# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import torch


class LoRAOverlapLoader:

    def __init__(self, device: torch.device):
        self._stream = torch.cuda.Stream(device=device)
        self._pending: torch.cuda.Event | None = None

    @contextmanager
    def load_context(self):
        with torch.cuda.stream(self._stream):
            yield
        event = torch.cuda.Event()
        event.record(self._stream)
        self._pending = event

    def synchronize(self) -> None:
        if self._pending is not None:
            torch.cuda.current_stream().wait_event(self._pending)
            self._pending = None

    @property
    def has_pending(self) -> bool:
        return self._pending is not None
