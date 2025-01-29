import time
from abc import ABC, abstractmethod

from vllm.logger import init_logger
from vllm.v1.metrics.stats import SchedulerStats

logger = init_logger(__name__)

_LOCAL_LOGGING_INTERVAL_SEC = 5.0


class StatLoggerBase(ABC):

    @abstractmethod
    def log(self, scheduler_stats: SchedulerStats):
        ...


class LoggingStatLogger(StatLoggerBase):

    def __init__(self):
        self.last_log_time = time.monotonic()

    def log(self, scheduler_stats: SchedulerStats):
        """Log Stats to standard output."""

        # Log every _LOCAL_LOGGING_INTERVAL_SEC.
        now = time.monotonic()
        if now - self.last_log_time < _LOCAL_LOGGING_INTERVAL_SEC:
            return
        self.last_log_time = now

        # Format and print output.
        logger.info(
            "Running: %d reqs, Waiting: %d reqs ",
            scheduler_stats.num_running_reqs,
            scheduler_stats.num_waiting_reqs,
        )
