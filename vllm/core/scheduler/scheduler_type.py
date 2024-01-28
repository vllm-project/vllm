from vllm.utils.base_int_enum import BaseIntEnum


class SchedulerType(BaseIntEnum):
    VLLM = 1
    SARATHI = 2
    DSARATHI = 3