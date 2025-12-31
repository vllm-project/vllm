# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from typing import Any


class ParameterSweep(list["ParameterSweepItem"]):
    @classmethod
    def read_json(cls, filepath: os.PathLike):
        with open(filepath, "rb") as f:
            data = json.load(f)

        # Support both list and dict formats
        if isinstance(data, dict):
            return cls.read_from_dict(data)

        return cls.from_records(data)

    @classmethod
    def read_from_dict(cls, data: dict[str, dict[str, object]]):
        """
        Read parameter sweep from a dict format where keys are names.

        Example:
            {
                "experiment1": {"max_tokens": 100, "temperature": 0.7},
                "experiment2": {"max_tokens": 200, "temperature": 0.9}
            }
        """
        records = [{"_benchmark_name": name, **params} for name, params in data.items()]
        return cls.from_records(records)

    @classmethod
    def from_records(cls, records: list[dict[str, object]]):
        if not isinstance(records, list):
            raise TypeError(
                f"The parameter sweep should be a list of dictionaries, "
                f"but found type: {type(records)}"
            )

        # Validate that all _benchmark_name values are unique if provided
        names = [r["_benchmark_name"] for r in records if "_benchmark_name" in r]
        if names and len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(
                f"Duplicate _benchmark_name values found: {set(duplicates)}. "
                f"All _benchmark_name values must be unique."
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

    @property
    def name(self) -> str:
        """
        Get the name for this parameter sweep item.

        Returns the '_benchmark_name' field if present, otherwise returns a text
        representation of all parameters.
        """
        if "_benchmark_name" in self:
            return self["_benchmark_name"]
        return self.as_text(sep="-")

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

    def _normalize_cmd_kv_pair(self, k: str, v: object) -> list[str]:
        """
        Normalize a key-value pair into command-line arguments.

        Returns a list containing either:
        - A single element for boolean flags (e.g., ['--flag'] or ['--flag=true'])
        - Two elements for key-value pairs (e.g., ['--key', 'value'])
        """
        if isinstance(v, bool):
            # For nested params (containing "."), use =true/false syntax
            if "." in k:
                return [f"{self._normalize_cmd_key(k)}={'true' if v else 'false'}"]
            else:
                return [self._normalize_cmd_key(k if v else "no-" + k)]
        else:
            return [self._normalize_cmd_key(k), str(v)]

    def apply_to_cmd(self, cmd: list[str]) -> list[str]:
        cmd = list(cmd)

        for k, v in self.items():
            # Skip the '_benchmark_name' field, not a parameter
            if k == "_benchmark_name":
                continue

            # Serialize dict values as JSON
            if isinstance(v, dict):
                v = json.dumps(v)

            for k_candidate in self._iter_cmd_key_candidates(k):
                try:
                    k_idx = cmd.index(k_candidate)

                    # Replace existing parameter
                    normalized = self._normalize_cmd_kv_pair(k, v)
                    if len(normalized) == 1:
                        # Boolean flag
                        cmd[k_idx] = normalized[0]
                    else:
                        # Key-value pair
                        cmd[k_idx] = normalized[0]
                        cmd[k_idx + 1] = normalized[1]

                    break
                except ValueError:
                    continue
            else:
                # Add new parameter
                cmd.extend(self._normalize_cmd_kv_pair(k, v))

        return cmd

    def as_text(self, sep: str = ", ") -> str:
        return sep.join(f"{k}={v}" for k, v in self.items() if k != "_benchmark_name")
