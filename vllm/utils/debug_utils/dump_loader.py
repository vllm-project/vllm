import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch

LOAD_FAILED: object = object()


def parse_meta_from_filename(path: Path) -> dict[str, Any]:
    stem = Path(path).stem
    result: dict[str, Any] = {}
    for kv in stem.split("___"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            result[k] = v
    for field_name, converter in _TYPED_FIELDS:
        if field_name in result:
            result[field_name] = converter(result[field_name])
    return result


@dataclass
class ValueWithMeta:
    value: Any
    meta: dict[str, Any]

    @staticmethod
    def load(path: Path) -> "ValueWithMeta":
        path = Path(path)
        meta_from_filename = parse_meta_from_filename(path)

        try:
            raw = torch.load(path, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"Skip load {path} since error {e}")
            return ValueWithMeta(
                value=LOAD_FAILED, meta={**meta_from_filename, "filename": path.name}
            )

        value, meta_from_embedded = _unwrap_dict_format(raw)
        return ValueWithMeta(
            value=value,
            meta={**meta_from_filename, **meta_from_embedded, "filename": path.name},
        )


def _unwrap_dict_format(obj: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(obj, dict) and "value" in obj:
        meta = obj.get("meta", {})
        assert isinstance(meta, dict), f"Expected meta to be dict, got {type(meta)}"
        return obj["value"], meta
    return obj, {}


class DumpLoader:
    def __init__(self):
        directory = os.environ.get("VLLM_DUMP_LOADER_DIR")

        self._enable = directory is not None
        if self._enable:
            self._directory = Path(directory)
            self._df = read_meta(directory)

    @property
    def enable(self):
        return self._enable

    def load(self, name, **kwargs):
        assert self._enable, "Please call DumpLoader.load only when it is enabled"

        from vllm.utils.debug_utils.dumper import dumper

        step = dumper._state.step
        conditions = dict(name=name, step=step, **kwargs)
        row = find_row(self._df, conditions=conditions)
        assert row is not None, (
            f"DumpLoader cannot find row given query {name=} {kwargs=} "
            f"{self._directory=}"
        )

        path = self._directory / row["filename"]
        output = torch.load(path, weights_only=False)
        if isinstance(output, dict) and "value" in output:
            output = output["value"]

        print(
            f"[DumpLoader] load from {path=} (query: {name=} {kwargs=}, output: "
            f"{type(output)})"
        )
        return output


def read_meta(directory):
    directory = Path(directory)
    assert directory.is_dir(), f"{directory=} should be a directory"

    rows = []
    for p in directory.glob("*.pt"):
        try:
            full_kwargs = parse_meta_from_filename(p)
            rows.append(
                {
                    "filename": str(p.name),
                    **full_kwargs,
                }
            )
        except Exception as e:
            print(f"[DumpLoader] skip loading {p} due to error {e}")

    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.col("step").cast(int),
        pl.col("rank").cast(int),
        pl.col("dump_index").cast(int),
    )
    df = _add_duplicate_index(df)
    df = df.sort("rank", "dump_index")
    return df


def _add_duplicate_index(df: pl.DataFrame) -> pl.DataFrame:
    group_cols = [c for c in df.columns if c not in ["filename", "dump_index"]]
    df = df.sort(group_cols + ["dump_index"])
    df = df.with_columns(
        pl.cum_count("dump_index").over(group_cols).sub(1).alias("duplicate_index")
    )
    return df


def filter_rows(df: pl.DataFrame, conditions: dict[str, Any]) -> list[dict]:
    filter_exprs = [
        (
            pl.col(col) == _cast_to_polars_dtype(conditions[col], df.schema[col])
            if conditions[col] is not None
            else pl.col(col).is_null()
        )
        for col in conditions
        if col in df.columns
    ]
    if not filter_exprs:
        return []
    return df.filter(functools.reduce(lambda a, b: a & b, filter_exprs)).to_dicts()


def find_row(df: pl.DataFrame, conditions: dict[str, Any]):
    rows = filter_rows(df, conditions)
    if len(rows) > 1:
        print(f"find_row find ambiguous results: {rows=}")
        return None
    return rows[0] if rows else None


def _cast_to_polars_dtype(value, target_dtype):
    if target_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        return int(value)
    elif target_dtype in (pl.Float64, pl.Float32):
        return float(value)
    elif target_dtype == pl.Boolean:
        return bool(value)
    elif target_dtype == pl.String:
        return str(value)
    else:
        return value


def read_tokenizer_path(directory: Path) -> str | None:
    """Read tokenizer_path from any .pt file's embedded metadata in a dump directory."""
    for p in directory.glob("*.pt"):
        item: ValueWithMeta = ValueWithMeta.load(p)
        tokenizer_path: str | None = item.meta.get("tokenizer_path")
        if tokenizer_path is not None:
            return str(tokenizer_path)
    return None


_TYPED_FIELDS: list[tuple[str, Callable[[str], Any]]] = [
    ("rank", int),
]


dump_loader = DumpLoader()
