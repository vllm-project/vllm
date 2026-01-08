# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import override


@dataclass
class SLACriterionBase(ABC):
    target: float

    @abstractmethod
    def validate(self, actual: float) -> bool:
        """Return `True` if this criterion is met; otherwise `False`."""
        raise NotImplementedError

    @abstractmethod
    def format_cond(self, lhs: str) -> str:
        raise NotImplementedError

    def print_and_validate(
        self,
        metrics: dict[str, float],
        metrics_key: str,
    ) -> bool:
        metric = metrics[metrics_key]
        result = self.validate(metric)

        cond = self.format_cond(f"{metrics_key} = {metric:.2f}")
        print(f"Validating SLA: {cond} | " + ("PASSED" if result else "FAILED"))

        return result


@dataclass
class SLALessThan(SLACriterionBase):
    @override
    def validate(self, actual: float) -> bool:
        return actual < self.target

    @override
    def format_cond(self, lhs: str) -> str:
        return f"{lhs}<{self.target:.2f}"


@dataclass
class SLALessThanOrEqualTo(SLACriterionBase):
    @override
    def validate(self, actual: float) -> bool:
        return actual <= self.target

    @override
    def format_cond(self, lhs: str) -> str:
        return f"{lhs}<={self.target:.2f}"


@dataclass
class SLAGreaterThan(SLACriterionBase):
    @override
    def validate(self, actual: float) -> bool:
        return actual > self.target

    @override
    def format_cond(self, lhs: str) -> str:
        return f"{lhs}>{self.target:.2f}"


@dataclass
class SLAGreaterThanOrEqualTo(SLACriterionBase):
    @override
    def validate(self, actual: float) -> bool:
        return actual >= self.target

    @override
    def format_cond(self, lhs: str) -> str:
        return f"{lhs}>={self.target:.2f}"


# NOTE: The ordering is important! Match longer op_keys first
SLA_CRITERIA: dict[str, type[SLACriterionBase]] = {
    "<=": SLALessThanOrEqualTo,
    ">=": SLAGreaterThanOrEqualTo,
    "<": SLALessThan,
    ">": SLAGreaterThan,
}


class SLASweep(list["SLASweepItem"]):
    @classmethod
    def read_json(cls, filepath: os.PathLike):
        with open(filepath, "rb") as f:
            records = json.load(f)

        return cls.from_records(records)

    @classmethod
    def from_records(cls, records: list[dict[str, str]]):
        if not isinstance(records, list):
            raise TypeError(
                f"The SLA sweep should be a list of dictionaries, "
                f"but found type: {type(records)}"
            )

        return cls(SLASweepItem.from_record(record) for record in records)


class SLASweepItem(dict[str, SLACriterionBase]):
    @classmethod
    def from_record(cls, record: dict[str, str]):
        sla_criteria: dict[str, SLACriterionBase] = {}

        for metric_key, metric_value in record.items():
            for op_key in SLA_CRITERIA:
                if metric_value.startswith(op_key):
                    sla_criteria[metric_key] = SLA_CRITERIA[op_key](
                        float(metric_value.removeprefix(op_key))
                    )
                    break
            else:
                raise ValueError(
                    f"Invalid operator for "
                    f"SLA constraint '{metric_key}={metric_value}'. "
                    f"Valid operators are: {sorted(SLA_CRITERIA)}",
                )

        return cls(sla_criteria)

    def as_text(self, sep: str = ", ") -> str:
        return sep.join(v.format_cond(k) for k, v in self.items())
