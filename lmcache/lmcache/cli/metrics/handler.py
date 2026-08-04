# SPDX-License-Identifier: Apache-2.0
"""Metrics handlers — control *where* metrics are written.

Each handler pairs a destination (stream, file, …) with a
:class:`~lmcache.cli.metrics.formatter.MetricsFormatter` that controls
the rendering.
"""

# Standard
from typing import IO, Optional, Union
import abc
import os
import sys

# First Party
from lmcache.cli.metrics.formatter import (
    JsonFormatter,
    MetricsFormatter,
    TerminalFormatter,
)
from lmcache.cli.metrics.section import Section


class MetricsHandler(abc.ABC):
    """Base class for metrics handlers."""

    def __init__(self, formatter: MetricsFormatter) -> None:
        self.formatter = formatter

    @abc.abstractmethod
    def emit(self, title: str, sections: list[Section]) -> None:
        """Format and write metrics to the destination.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.
        """


class StreamHandler(MetricsHandler):
    """Writes formatted metrics to a text stream.

    Args:
        formatter: The formatter to use for rendering.
        stream: Writable text stream. Defaults to ``sys.stdout``.
    """

    def __init__(
        self,
        formatter: Optional[MetricsFormatter] = None,
        stream: Optional[IO[str]] = None,
    ) -> None:
        super().__init__(formatter or TerminalFormatter())
        self._stream = stream

    def emit(self, title: str, sections: list[Section]) -> None:
        """Format and write metrics to the stream.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.
        """
        stream = self._stream or sys.stdout
        stream.write(self.formatter.format(title, sections))
        stream.write("\n")


class FileHandler(MetricsHandler):
    """Writes formatted metrics to a file.

    Args:
        path: Destination file path.
        formatter: The formatter to use for rendering. Defaults to
            :class:`~lmcache.cli.metrics.formatter.JsonFormatter`.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        formatter: Optional[MetricsFormatter] = None,
    ) -> None:
        super().__init__(formatter or JsonFormatter())
        self.path = path

    def emit(self, title: str, sections: list[Section]) -> None:
        """Format and write metrics to the file.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.
        """
        with open(self.path, "w") as f:
            f.write(self.formatter.format(title, sections))
            f.write("\n")
