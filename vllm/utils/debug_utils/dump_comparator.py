"""Simplified dump comparator — a self-contained single-file script for comparing
two dump directories tensor-by-tensor.

For advanced features (unshard, token alignment, per-dimension annotations), see the
SGLang's ``sglang.srt.debug_utils.comparator`` package can read these dumps.
"""

import argparse
import functools
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from vllm.utils.debug_utils.dumper import get_device, get_truncated_value


def main(args):
    import polars as pl

    from vllm.utils.debug_utils.dump_loader import find_row, read_meta

    df_target = read_meta(args.target_path)
    df_target = df_target.filter(
        (pl.col("step") >= args.start_step) & (pl.col("step") <= args.end_step)
    )
    if args.filter:
        df_target = df_target.filter(pl.col("filename").str.contains(args.filter))
    assert all(c in df_target.columns for c in ["rank", "step", "dump_index", "name"])

    df_baseline = read_meta(args.baseline_path)
    print("df_target", df_target)
    print("df_baseline", df_baseline)

    tensor_dim_descs: list[TensorDimDesc] = _get_tensor_dim_descs()

    for row in df_target.iter_rows(named=True):
        path_target = Path(args.target_path) / row["filename"]

        tensor_dim_desc: TensorDimDesc | None = None
        if tensor_dim_descs:
            matched: list[TensorDimDesc] = [
                desc
                for desc in tensor_dim_descs
                if re.search(desc.pattern, row["filename"]) is not None
            ]
            if matched:
                tensor_dim_desc = matched[0]

        row_baseline = find_row(
            df_baseline,
            conditions=dict(
                step=row["step"],
                **{
                    k: v
                    for k, v in row.items()
                    if k not in ["step", "dump_index", "filename"]
                },
            ),
        )

        if row_baseline is None:
            print(f"Skip: target={str(path_target)} since no baseline")
            x_target = _load_object(path_target)
            if x_target is not None:
                print(f"x_target(sample)={get_truncated_value(x_target)}")
            continue

        path_baseline = Path(args.baseline_path) / row_baseline["filename"]
        print(
            f"Check:\n"
            f"target={str(path_target)} (duplicate_index={row['duplicate_index']})\n"
            f"baseline={str(path_baseline)} "
            f"(duplicate_index={row_baseline['duplicate_index']})"
        )
        check_tensor_pair(
            path_baseline=path_baseline,
            path_target=path_target,
            diff_threshold=args.diff_threshold,
            name=row["name"],
            tensor_dim_desc=tensor_dim_desc,
        )
        print()


def check_tensor_pair(
    path_baseline,
    path_target,
    diff_threshold: float = 1e-3,
    name="",
    tensor_dim_desc: Optional["TensorDimDesc"] = None,
):
    x_baseline = _load_object(path_baseline)
    x_target = _load_object(path_target)

    if x_baseline is None or x_target is None:
        print(
            f"Skip comparison because of None: x_baseline={x_baseline}, "
            f"x_target={x_target}"
        )
        return

    print(
        f"Raw "
        f"[shape] {x_baseline.shape} vs {x_target.shape}\t"
        f"[{'' if x_baseline.dtype == x_target.dtype else '🟠'}dtype] "
        f"{x_baseline.dtype} vs {x_target.dtype}"
    )

    if tensor_dim_desc is not None:
        import einops

        x_baseline = einops.rearrange(
            x_baseline,
            tensor_dim_desc.baseline_desc + " -> " + tensor_dim_desc.target_desc,
        )
        if tensor_dim_desc.baseline_cropper is not None:
            print("Apply baseline_cropper")
            x_baseline = tensor_dim_desc.baseline_cropper(x_baseline)

    x_baseline, x_target = _comparison_preprocessor(x_baseline, x_target, name=name)
    x_baseline = _try_unify_shape(x_baseline, target_shape=x_target.shape)

    print(
        f"After preprocessor "
        f"[shape] {x_baseline.shape} vs {x_target.shape}\t"
        f"[dtype] {x_baseline.dtype} vs {x_target.dtype}"
    )

    x_baseline_original_dtype = x_baseline.dtype
    x_target_original_dtype = x_target.dtype

    x_target = x_target.float()
    x_baseline = x_baseline.float()

    for name, fn in [
        ("mean", torch.mean),
        ("std", torch.std),
        ("min", torch.min),
        ("max", torch.max),
        *(
            [
                ("p1", functools.partial(torch.quantile, q=0.01)),
                ("p5", functools.partial(torch.quantile, q=0.05)),
                ("p95", functools.partial(torch.quantile, q=0.95)),
                ("p99", functools.partial(torch.quantile, q=0.99)),
            ]
            if x_baseline.numel() < 10_000_000
            else []
        ),
    ]:
        value_baseline = fn(x_baseline).item()
        value_target = fn(x_target).item()
        print(
            f"[{name}] {value_baseline:.4f} vs {value_target:.4f} (diff: "
            f"{value_target - value_baseline:.4f})"
        )

    if x_baseline.shape != x_target.shape:
        print("⚠️ Shape mismatch")
        return

    diff_info = _compute_and_print_diff(
        x_baseline=x_baseline,
        x_target=x_target,
        diff_threshold=diff_threshold,
    )
    needs_print = diff_info["max_abs_diff"] > 1e-3

    if (x_baseline_original_dtype != x_target_original_dtype) and (
        (
            downcast_dtype := _compute_smaller_dtype(
                x_baseline_original_dtype, x_target_original_dtype
            )
        )
        is not None
    ):
        _compute_and_print_diff(
            x_baseline=x_baseline.to(downcast_dtype),
            x_target=x_target.to(downcast_dtype),
            diff_threshold=diff_threshold,
            prefix_text=f"When downcast to {downcast_dtype}: ",
        )

    if needs_print:
        print(f"x_baseline(sample)={get_truncated_value(x_baseline)}")
        print(f"x_target(sample)={get_truncated_value(x_target)}")


def _compute_and_print_diff(
    x_baseline, x_target, diff_threshold: float, prefix_text=""
):
    raw_abs_diff = (x_target - x_baseline).abs()

    max_abs_diff = raw_abs_diff.max().item()
    mean_abs_diff = raw_abs_diff.mean().item()
    rel_diff = _calc_rel_diff(x_target, x_baseline)

    rel_diff_marker: str = "❌" if rel_diff > diff_threshold else "✅"
    print(
        prefix_text
        + f"{rel_diff_marker} rel_diff={rel_diff}\t"
        + f"max_abs_diff={max_abs_diff}\t"
        + f"mean_abs_diff={mean_abs_diff}"
    )

    max_diff_coord = _argmax_coord(raw_abs_diff)
    print(
        f"max_abs_diff happens at coord={max_diff_coord} with "
        f"baseline={x_baseline[max_diff_coord].item()} "
        f"target={x_target[max_diff_coord].item()}"
    )

    return dict(max_abs_diff=max_abs_diff)


def _argmax_coord(x: torch.Tensor) -> tuple:
    flat_idx = x.argmax()
    return tuple(idx.item() for idx in torch.unravel_index(flat_idx, x.shape))


def _compute_smaller_dtype(dtype_a, dtype_b):
    info_dict = {
        (torch.float32, torch.bfloat16): torch.bfloat16,
        # ... add more ...
    }
    return info_dict.get((dtype_a, dtype_b)) or info_dict.get((dtype_b, dtype_a))


def _try_unify_shape(x: torch.Tensor, target_shape):
    x_shape = x.shape
    num_dim_to_remove = len(x_shape) - len(target_shape)
    if (x_shape[num_dim_to_remove:] == target_shape) and all(
        val == 1 for val in x_shape[:num_dim_to_remove]
    ):
        out = functools.reduce(lambda a, _: a.squeeze(0), range(num_dim_to_remove), x)
        print(f"Unify shape: {x_shape} -> {out.shape} (to match {target_shape})")
        return out

    return x


# Copied from DeepGEMM
def _calc_rel_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _load_object(path):
    try:
        x = torch.load(path, weights_only=False)
    except Exception as e:
        print(f"Skip load {path} since error {e}")
        return None

    if isinstance(x, dict) and "value" in x:
        x = x["value"]

    if not isinstance(x, torch.Tensor):
        print(f"Skip load {path} since {type(x)=} is not a Tensor ({x=})")
        return None
    return x.to(get_device())


def _comparison_preprocessor(x_baseline, x_target, name):
    """Customization endpoint. Can insert arbitrary adhoc postprocessing logic here."""
    return x_baseline, x_target


@dataclass
class TensorDimDesc:
    pattern: str
    baseline_desc: str
    target_desc: str
    baseline_cropper: Callable[[torch.Tensor], torch.Tensor] | None = None


def _get_tensor_dim_descs() -> list[TensorDimDesc]:
    """Customization endpoint. Return a list of TensorDimDesc to rearrange baseline
    dimensions to match target layout via einops before comparison."""
    return []


if __name__ == "__main__":
    # python -m vllm.utils.debug_utils.dump_comparator --baseline-path ... --target-path
    # ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=1000000)
    parser.add_argument("--diff-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--filter", type=str, default=None, help="Regex to filter filenames"
    )
    args = parser.parse_args()
    main(args)
