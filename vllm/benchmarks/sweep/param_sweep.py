# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from typing import Any


class ParameterSweep(list["ParameterSweepItem"]):
    @classmethod
    def read_json(cls, filepath: os.PathLike):
        with open(filepath, "rb") as f:
            records = json.load(f)

        return cls.from_records(records)

    @classmethod
    def from_records(cls, records: list[dict[str, object]]):
        if not isinstance(records, list):
            raise TypeError(
                f"The parameter sweep should be a list of dictionaries, "
                f"but found type: {type(records)}"
            )

        return cls(ParameterSweepItem.from_record(record) for record in records)


class ParameterSweepItem(dict[str, object]):
    @classmethod
    def from_record(cls, record: dict[str, object]):
        if not isinstance(record, dict):
            raise TypeError(
                f"Each item in the parameter sweep should be a dictionary, "
                f"but found type: {type(record)}"
            )

        return cls(record)

    def __or__(self, other: dict[str, Any]):
        return type(self)(super().__or__(other))

    # In JSON, we prefer "_"
    def _iter_param_key_candidates(self, param_key: str):
        # Inner config arguments are not converted by the CLI
        if "." in param_key:
            prefix, rest = param_key.split(".", 1)
            for prefix_candidate in self._iter_param_key_candidates(prefix):
                yield prefix_candidate + "." + rest

            return

        yield param_key
        yield param_key.replace("-", "_")
        yield param_key.replace("_", "-")

    # In CLI, we prefer "-"
    def _iter_cmd_key_candidates(self, param_key: str):
        for k in reversed(tuple(self._iter_param_key_candidates(param_key))):
            yield "--" + k

    def _normalize_cmd_key(self, param_key: str):
        return next(self._iter_cmd_key_candidates(param_key))

    def has_param(self, param_key: str) -> bool:
        return any(k in self for k in self._iter_param_key_candidates(param_key))

    def apply_to_cmd(self, cmd: list[str]) -> list[str]:
        cmd = list(cmd)

        for k, v in self.items():
            for k_candidate in self._iter_cmd_key_candidates(k):
                try:
                    k_idx = cmd.index(k_candidate)

                    if isinstance(v, bool):
                        cmd[k_idx] = self._normalize_cmd_key(k if v else "no-" + k)
                    else:
                        cmd[k_idx + 1] = str(v)

                    break
                except ValueError:
                    continue
            else:
                if isinstance(v, bool):
                    cmd.append(self._normalize_cmd_key(k if v else "no-" + k))
                else:
                    cmd.extend([self._normalize_cmd_key(k), str(v)])

        return cmd

    def as_text(self, sep: str = ", ") -> str:
        return sep.join(f"{k}={v}" for k, v in self.items())
