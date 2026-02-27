from dataclasses import dataclass
from typing import List, Optional

from dllm.entities import Role, SchedulerPolicy

required_vllm_options = [
    ("host", ),
    ("port", ),
    ("tensor-parallel-size", "tp"),
    ("data-parallel-size", "dp"),
    ("data-parallel-size-local", "dpl"),
    ("data-parallel-start-rank", "dpr"),
    ("data-parallel-address", "dpa"),
    ("data-parallel-rpc-port", "dpp"),
    ("headless", ),
    ("enable-expert-parallel", ),
    ("disable-expert-parallel", ),
    ("kv-transfer-config", ),
]


class AutoValidator:

    def __post_init__(self):
        for name, f in self.__dataclass_fields__.items(  # type: ignore # get all data fields from a dataclass
        ):
            if method := getattr(self, f"_validate_{name}", None):
                method()


@dataclass
class InferenceInstanceConfig(AutoValidator):
    startup_params: List[str]
    startup_env: Optional[str]
    tp: int
    dp: int
    ep: int

    def _validate_startup_params(self):

        def __contain_long_options(opname, params):
            underline_op = opname.replace("-", "_")
            return any(
                p == f"--{opname}" or p.startswith(f"--{opname}=") or p ==
                f"--{underline_op}" or p.startswith(f"--{underline_op}=")
                for p in params)

        def __contain_short_options(opname, params):
            underline_op = opname.replace("-", "_")
            return any(
                p == f"-{opname}" or p.startswith(f"-{opname}=")
                or p == f"-{underline_op}" or p.startswith(f"-{underline_op}=")
                for p in params)

        bad_options = []
        for option in required_vllm_options:
            if len(option) > 0 and __contain_long_options(
                    option[0], self.startup_params):
                bad_options.append(option[0])
            if len(option) > 1 and __contain_short_options(
                    option[1], self.startup_params):
                bad_options.append(option[1])

        if bad_options:
            raise ValueError(
                f"{bad_options} should not be specified in start up commands, instead, dllm will populate options after verification"
            )

    def _validate_ep(self):
        if self.ep < 0:  # type: ignore
            raise ValueError(
                "expert parallel size should be 0 (EP disabled) or >1 (EP enabled)"
            )

    def _validate_dp(self):
        if not self.dp > 0:  # type: ignore
            raise ValueError("data parallel size should be greater than 0")

    def _validate_tp(self):
        if not self.tp > 0:  # type: ignore
            raise ValueError("tensor parallel size should be greater than 0")


@dataclass
class ControllerConfig(AutoValidator):
    scheduler_policy: SchedulerPolicy
    num_prefill_instances: int
    num_decode_instances: int
    prefill_instance_config: InferenceInstanceConfig
    decode_instance_config: InferenceInstanceConfig

    def _validate_num_prefill_instances(self):
        if self.num_prefill_instances < 0:
            raise ValueError(
                "number of prefill instances should be equal to or greater than 0"
            )

    def _validate_num_decode_instances(self):
        if self.num_decode_instances < 0:
            raise ValueError(
                "number of decode instances should be equal to or greater than 0"
            )


@dataclass
class PDDistConfig(AutoValidator):
    role: Role
    pd_rank: int = 0
    pd_size: int = 0

    def is_pd_dist(self):
        return self.role != Role.MIXED


@dataclass
class DPConfig(AutoValidator):
    dp_rank: int = 0
    dp_size: int = 1
    dp_local_size: int = 1
    dp_master_ip: str = ""
    dp_master_port: int = 0

    def is_dp_enabled(self):
        return self.dp_size and self.dp_size > 1


@dataclass
class EPConfig(AutoValidator):
    ep_size: int = 0

    def is_ep_enabled(self):
        return self.ep_size and self.ep_size > 0


@dataclass
class VllmInstanceConfig(AutoValidator):
    exec_cmd: list[str]
    env: Optional[str] = None
    tp: int = 1
    dp_config: Optional[DPConfig] = None
    ep_config: Optional[EPConfig] = None
    pd_config: Optional[PDDistConfig] = None
