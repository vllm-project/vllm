# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts quantized with compressed-tensors (llm-compressor).

llm-compressor saves the experts one by one under the reference names,
`model.llm.layers.N.mlp.experts.E.{gate,up,down}_proj.{weight_packed | weight,
weight_scale, weight_zero_point}`. They load into this rank's expert shard of
`w13_*` / `w2_*`, with the rows of `gate_proj` and `up_proj` interleaved as in
the NVFP4 / MXFP8 `w13` (the SiTU kernel reads (gate, up) pairs), and run on
the Humming expert GEMMs of the Hopper path (`ops.moe_sm90.HummingExperts`),
which take compressed-tensors weights and activation schemes as they are.

Supported: int4 / int8 weights (`pack-quantized`, symmetric or with zero
points; W4A16, W8A16), int8 / fp8 weights with int8 / fp8 dynamic activations
(`int-quantized`, `float-quantized`; W8A8, FP8_DYNAMIC, FP8_BLOCK) and int4
weights with int8 dynamic activations (W4A8, `int-quantized` int4 values,
packed when loaded), per group, channel or block. Static activation scales
and per-tensor weight scales are not.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)

from .ops.moe import combine, combine_push
from .ops.moe_sm90 import HummingExperts
from .ops.seq_parallel import SeqParallel

# Checkpoint projection -> (fused parameter, row parity in it; None: all rows).
PROJECTIONS: dict[str, tuple[str, int | None]] = {
    "gate_proj": ("w13", 0),
    "up_proj": ("w13", 1),
    "down_proj": ("w2", None),
}
_WEIGHT_FORMATS = ("pack-quantized", "int-quantized", "float-quantized")


def _value(field: Any) -> Any:
    """The value of a compressed-tensors enum field (or the field itself)."""
    return getattr(field, "value", field)


@dataclass(frozen=True)
class Inkling2CTScheme:
    """The compressed-tensors scheme of one layer's routed experts."""

    weights: Any  # compressed_tensors QuantizationArgs
    input_activations: Any | None
    format: str

    @classmethod
    def from_config(
        cls, quant_config: CompressedTensorsConfig, prefix: str
    ) -> "Inkling2CTScheme":
        """The scheme of the experts of the MoE layer at `prefix`; all
        three projections of every expert must share it."""
        # Targets name modules by class (`Linear`) or by path.
        probe = nn.Linear(1, 1, bias=False, device="meta")
        schemes = {
            proj: quant_config.get_scheme_dict(probe, f"{prefix}.experts.0.{proj}")
            for proj in PROJECTIONS
        }
        if any(scheme is None for scheme in schemes.values()):
            raise ValueError(
                f"{prefix}: unquantized Inkling2 experts are not supported; "
                "quantize every routed expert projection (gate_proj, up_proj, "
                "down_proj) or none of the model"
            )
        keys: dict[str, tuple[Any, Any, Any]] = {}
        for proj, s in schemes.items():
            assert s is not None
            keys[proj] = (s["weights"], s.get("input_activations"), s["format"])
        if len(set(map(repr, keys.values()))) != 1:
            raise ValueError(
                f"{prefix}: the Inkling2 expert projections must share one "
                f"compressed-tensors scheme, got {keys}"
            )
        scheme = cls(*keys["gate_proj"])
        scheme.validate(prefix)
        return scheme

    @property
    def packs_weights(self) -> bool:
        """Whether the int4 weights come unpacked, one per int8 (W4A8's
        `int-quantized` format), and are packed into int32 when loaded."""
        return self.format == "int-quantized" and self.weights.num_bits == 4

    @property
    def block_rows(self) -> int:
        """Rows per weight scale block (1 unless the strategy is block)."""
        if _value(self.weights.strategy) != "block":
            return 1
        return self.weights.block_structure[0]

    def validate(self, prefix: str) -> None:
        w, a = self.weights, self.input_activations

        def unsupported(what: str) -> ValueError:
            return ValueError(
                f"{prefix}: compressed-tensors Inkling2 experts do not support "
                f"{what} (weights: {w}, input activations: {a})"
            )

        if self.format not in _WEIGHT_FORMATS:
            raise unsupported(f"the {self.format!r} format")
        weight_type = (_value(w.type), w.num_bits)
        if weight_type not in (("int", 4), ("int", 8), ("float", 8)):
            raise unsupported("these weight types")
        if (_value(w.type) == "float") != (self.format == "float-quantized"):
            raise unsupported(f"{_value(w.type)} weights in the {self.format!r} format")
        if w.dynamic:
            raise unsupported("dynamic weight quantization")
        strategy = _value(w.strategy)
        if strategy not in ("group", "channel", "block"):
            raise unsupported(f"the {strategy!r} weight strategy")
        if not w.symmetric and (
            self.format != "pack-quantized" or strategy not in ("group", "channel")
        ):
            raise unsupported("zero points outside group / channel packed weights")
        if a is not None:
            if a.dynamic is not True:
                raise unsupported("static activation scales")
            if not a.symmetric:
                raise unsupported("asymmetric activations")
            if (_value(a.type), a.num_bits) not in (("int", 8), ("float", 8)):
                raise unsupported("these activation types")
            if _value(a.strategy) not in ("token", "group"):
                raise unsupported(f"the {a.strategy!r} activation strategy")

    def humming_weight_schema(self) -> Any:
        """The Humming source schema of the checkpoint weights."""
        from vllm.utils.humming import CompressedTensorsWeightSchema

        w = self.weights
        strategy, group_size = _value(w.strategy), w.group_size
        if strategy == "block":
            # Block scales are loaded repeated over their rows (`block_rows`).
            strategy, group_size = "group", w.block_structure[1]
        return CompressedTensorsWeightSchema(
            format="pack-quantized" if self.packs_weights else self.format,
            type=_value(w.type),
            num_bits=w.num_bits,
            strategy=strategy,
            symmetric=w.symmetric,
            group_size=group_size,
        )

    def humming_input_schema(self, fallback: bool = True) -> Any | None:
        """The Humming schema of the activations (None: bf16). With
        `fallback`, as for vLLM's Humming MoE, activations the GPU has no
        MMA for (e.g. FP8 before SM89) run in bf16, with a warning."""
        from vllm.model_executor.layers.quantization.utils.humming.schema import (
            check_and_fallback_input_schema,
        )
        from vllm.utils.humming import CompressedTensorsInputSchema

        a = self.input_activations
        if a is None:
            return None
        input_schema = CompressedTensorsInputSchema(
            format=self.format,
            type=_value(a.type),
            num_bits=a.num_bits,
            dynamic=True,
            group_size=(a.group_size or 0) if _value(a.strategy) == "group" else 0,
        ).to_humming_schema(torch.bfloat16)
        if not fallback:
            return input_schema
        return check_and_fallback_input_schema(
            weight_schema=self.humming_weight_schema().to_humming_schema(
                torch.bfloat16
            ),
            input_schema=input_schema,
            param_dtype=torch.bfloat16,
            allow_fp8_sm100=False,
        )


def _unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """`[R, G]` int32 words of `32 / bits` rows each (first row in the low
    bits, as compressed-tensors packs along dim 0) -> `[R * 32 / bits, G]`."""
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=packed.device, dtype=torch.int32) * bits
    values = (packed[:, None, :] >> shifts[None, :, None]) & ((1 << bits) - 1)
    return values.reshape(-1, packed.shape[-1])


def _pack_rows(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of `_unpack_rows`."""
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=values.device, dtype=torch.int32) * bits
    words = values.reshape(-1, per_word, values.shape[-1]).to(torch.int32)
    return (words << shifts[None, :, None]).sum(1, dtype=torch.int32)


def _pack_cols(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Signed `[N, K]` int8 values -> the `pack-quantized` `[N, K * bits /
    32]` int32 words (offset to unsigned, first column in the low bits)."""
    unsigned = values.to(torch.int32) + (1 << (bits - 1))
    return _pack_rows(unsigned.T.contiguous(), bits).T.contiguous()


class Inkling2CTExperts(nn.Module):
    """Drop-in for `moe.Inkling2Experts` (same routing inputs and outputs,
    those of its Humming path) with compressed-tensors weights."""

    supports_lora = False
    # Always the Humming path: routing by `select_experts`, no fused
    # activation quantization.
    sm100 = False
    sm120 = False

    def __init__(self, config, scheme: Inkling2CTScheme) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter
        self.shapes = ((2 * inter, hidden), (hidden, inter))
        self.scheme = scheme
        self.weight_schema = weight_schema = scheme.humming_weight_schema()
        self.zero_point_bits = (
            None if scheme.weights.symmetric else weight_schema.num_bits
        )

        # (Humming, checkpoint) names of each projection's tensors. The
        # parameters keep the checkpoint names; `weight_shape` is only
        # metadata of the packed format and not kept.
        self.tensor_names: list[tuple[str, str]] = []
        for name, (n, k) in zip(("w13", "w2"), self.shapes):
            attrs = weight_schema.get_tensors_attrs(
                shape_n=n,
                shape_k=k,
                param_dtype=torch.bfloat16,
                num_experts=self.num_experts,
            )
            attrs.pop("weight_shape", None)
            names = [
                (
                    key,
                    "weight"
                    if key == "weight_packed" and scheme.packs_weights
                    else key,
                )
                for key in attrs
            ]
            for key, checkpoint_key in names:
                param = nn.Parameter(
                    torch.empty(attrs[key]["shape"], dtype=attrs[key]["dtype"]),
                    requires_grad=False,
                )
                self.register_parameter(f"{name}_{checkpoint_key}", param)
            self.tensor_names = names

        start = self.ep_rank * self.num_experts
        self.register_buffer(
            "expert_bounds",
            torch.tensor([start, start + self.num_experts], dtype=torch.int32),
            persistent=False,
        )
        self.humming: HummingExperts | None = None

    def load_expert(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        expert_id: int,
        projection: str,
        key: str,
    ) -> bool:
        """Loads tensor `key` of one checkpoint expert projection; returns
        whether the expert is one of this rank's."""
        local = expert_id - self.ep_rank * self.num_experts
        if not 0 <= local < self.num_experts:
            return False
        _, parity = PROJECTIONS[projection]
        loaded = loaded.to(param.device)
        dst = param.data[local]
        if key == "weight_scale":
            loaded = loaded.to(param.dtype)
            if self.scheme.block_rows > 1:
                n = dst.shape[0] if parity is None else dst.shape[0] // 2
                loaded = loaded.repeat_interleave(self.scheme.block_rows, 0)[:n]
        elif key == "weight" and self.scheme.packs_weights:
            loaded = _pack_cols(loaded, 4)
        if parity is None:
            rows = dst
        elif key == "weight_zero_point":
            # Packed along the rows: interleave unpacked rows and repack.
            bits = self.zero_point_bits
            assert bits is not None
            rows = _unpack_rows(dst, bits)
            half = rows[parity::2]
            assert half.shape == (loaded.shape[0] * 32 // bits, loaded.shape[1]), (
                f"{projection}.{key}: {tuple(loaded.shape)} vs {tuple(dst.shape)}"
            )
            half.copy_(_unpack_rows(loaded, bits))
            dst.copy_(_pack_rows(rows, bits))
            return True
        else:
            rows = dst[parity::2]
        assert loaded.shape == rows.shape and loaded.dtype == rows.dtype, (
            f"{projection}.{key}: {tuple(loaded.shape)} {loaded.dtype} != "
            f"{tuple(rows.shape)} {rows.dtype}"
        )
        rows.copy_(loaded)
        return True

    def process_weights_after_loading(self) -> None:
        weights = {
            name: (
                self.weight_schema,
                {
                    key: getattr(self, f"{name}_{checkpoint_key}").data
                    for key, checkpoint_key in self.tensor_names
                },
            )
            for name in ("w13", "w2")
        }
        self.humming = HummingExperts(
            weights,
            self.shapes,
            nvfp4=False,
            input_schema=self.scheme.humming_input_schema(),
        )
        # The Humming copies replace the checkpoint layout.
        for name in ("w13", "w2"):
            for _, checkpoint_key in self.tensor_names:
                param = getattr(self, f"{name}_{checkpoint_key}")
                param.data = param.data.new_empty((0,))

    def forward(
        self,
        x: torch.Tensor,
        routing,
        quantized,
        plan,
        reduce: bool = True,
        push: SeqParallel | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """See `moe.Inkling2Experts.forward` (`routing` is from
        `ops.moe_sm90.select_experts`; `quantized` and `plan` are None)."""
        assert self.humming is not None, "process_weights_after_loading"
        ids, gammas, src2dst = routing
        down = self.humming(
            x,
            ids,
            gammas,
            src2dst,
            self.num_experts * self.ep_size,
            self.gate_cap,
            self.up_cap,
        )
        if push is not None:
            combine_push(down, src2dst, self.topk, push)
            return None
        if not reduce:
            return down, src2dst
        out = combine(down, src2dst, self.topk, x.dtype)
        return out if self.ep_size == 1 else tensor_model_parallel_all_reduce(out)
