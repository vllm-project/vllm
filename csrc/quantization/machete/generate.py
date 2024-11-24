import itertools
import math
import os
import shutil
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import jinja2
# yapf conflicts with isort for this block
# yapf: disable
from vllm_cutlass_library_extension import (DataType, EpilogueScheduleTag,
                                            EpilogueScheduleType,
                                            MixedInputKernelScheduleType,
                                            TileSchedulerTag,
                                            TileSchedulerType, VLLMDataType,
                                            VLLMDataTypeNames,
                                            VLLMDataTypeSize, VLLMDataTypeTag,
                                            VLLMDataTypeTorchDataTypeTag,
                                            VLLMDataTypeVLLMScalarTypeTag,
                                            VLLMKernelScheduleTag)

# yapf: enable

#
#   Generator templating
#

DISPATCH_TEMPLATE = """
#include "../machete_mm_launcher.cuh"

namespace machete {

{% for impl_config in impl_configs %}
{% set type_sig = gen_type_sig(impl_config.types) -%}
{% for s in impl_config.schedules %}
extern torch::Tensor impl_{{type_sig}}_sch_{{gen_sch_sig(s)}}(MMArgs);
{%- endfor %}

torch::Tensor mm_dispatch_{{type_sig}}(MMArgs args) {
  [[maybe_unused]] auto M = args.A.size(0);
  [[maybe_unused]] auto N = args.B.size(1);
  [[maybe_unused]] auto K = args.A.size(1);
    
  if (!args.maybe_schedule) {
    {%- for cond, s in impl_config.heuristic %}
    {%if cond is not none%}if ({{cond}})
    {%- else %}else
    {%- endif %}
        return impl_{{type_sig}}_sch_{{ gen_sch_sig(s) }}(args);{% endfor %}
  }

  {%- for s in impl_config.schedules %}
  if (*args.maybe_schedule == "{{ gen_sch_sig(s) }}")
    return impl_{{type_sig}}_sch_{{ gen_sch_sig(s) }}(args);
  {%- endfor %}
  TORCH_CHECK_NOT_IMPLEMENTED(false, "machete_gemm(..) is not implemented for "
                                     "schedule = ", *args.maybe_schedule);
}
{%- endfor %}


static inline std::optional<at::ScalarType> maybe_scalartype(
    c10::optional<at::Tensor> const& t) {
    if (!t) {
      return std::nullopt;
    } else {
      return t->scalar_type();
    };
}

torch::Tensor mm_dispatch(MMArgs args) {
  auto out_type = args.maybe_out_type.value_or(args.A.scalar_type());
  auto a_type = args.A.scalar_type();
  auto maybe_g_scales_type = maybe_scalartype(args.maybe_group_scales);
  auto maybe_g_zeros_type = maybe_scalartype(args.maybe_group_zeros);
  auto maybe_ch_scales_type = maybe_scalartype(args.maybe_channel_scales);
  auto maybe_tok_scales_type = maybe_scalartype(args.maybe_token_scales);

  {% for impl_config in impl_configs %}
  {% set t = impl_config.types -%}
  {% set type_sig = gen_type_sig(t) -%}
  if (args.b_type == {{VLLMScalarTypeTag[t.b]}}
      && a_type == {{TorchTypeTag[t.a]}}
      && out_type == {{TorchTypeTag[t.out]}}
      && {%if t.b_group_scale != void -%}
      maybe_g_scales_type == {{TorchTypeTag[t.b_group_scale]}}
      {%- else %}!maybe_g_scales_type{%endif%}
      && {%if t.b_group_zeropoint != void -%}
      maybe_g_zeros_type == {{TorchTypeTag[t.b_group_zeropoint]}}
      {%- else %}!maybe_g_zeros_type{%endif%}
      && {%if t.b_channel_scale != void -%}
      maybe_ch_scales_type == {{TorchTypeTag[t.b_channel_scale]}}
      {%- else %}!maybe_ch_scales_type{%endif%}
      && {%if t.a_token_scale != void -%}
      maybe_tok_scales_type == {{TorchTypeTag[t.a_token_scale]}}
      {%- else %}!maybe_tok_scales_type{%endif%}
  ) {
      return mm_dispatch_{{type_sig}}(args);
  }
  {%- endfor %}
  
  TORCH_CHECK_NOT_IMPLEMENTED(
    false, "machete_mm(..) is not implemented for "
    "a_type=", args.A.scalar_type(),
    ", b_type=", args.b_type.str(),
    ", out_type=", out_type,
    ", with_group_scale_type=", maybe_g_scales_type
        ? toString(*maybe_g_scales_type) : "None",
    ", with_group_zeropoint_type=", maybe_g_zeros_type
        ? toString(*maybe_g_zeros_type) : "None",
    ", with_channel_scale_type=", maybe_ch_scales_type
        ? toString(*maybe_ch_scales_type) : "None",
    ", with_token_scale_type=", maybe_tok_scales_type
        ? toString(*maybe_tok_scales_type) : "None",
    "; implemented types are: \\n",
    {%- for impl_config in impl_configs %}
    {% set t = impl_config.types -%}
    "\\t{{gen_type_option_name(t)}}\\n",
    {%- endfor %}
    "");
}

std::vector<std::string> supported_schedules_dispatch(
    SupportedSchedulesArgs args) {
    auto out_type = args.maybe_out_type.value_or(args.a_type);
    
    {% for impl_config in impl_configs %}
    {% set t = impl_config.types -%}
    {% set schs = impl_config.schedules -%}
    if (args.b_type == {{VLLMScalarTypeTag[t.b]}}
        && args.a_type == {{TorchTypeTag[t.a]}}
        && out_type == {{TorchTypeTag[t.out]}}
        && {%if t.b_group_scale != void -%}
        args.maybe_group_scales_type == {{TorchTypeTag[t.b_group_scale]}}
        {%- else %}!args.maybe_group_scales_type{%endif%}
        && {%if t.b_group_zeropoint != void-%}
        args.maybe_group_zeros_type == {{TorchTypeTag[t.b_group_zeropoint]}}
        {%- else %}!args.maybe_group_zeros_type{%endif%}
    ) {
        return {
            {%- for s in impl_config.schedules %}
            "{{gen_sch_sig(s)}}"{% if not loop.last %},{% endif %}
            {%- endfor %}
        };
    }
    {%- endfor %}
    
    return {};
};

}; // namespace machete
"""

IMPL_TEMPLATE = """
#include "../machete_mm_launcher.cuh"

namespace machete {
    
{% for sch in unique_schedules(impl_configs) %}
{% set sch_sig = gen_sch_sig(sch) -%}
struct sch_{{sch_sig}} {
  using TileShapeNM = Shape<{{
      to_cute_constant(sch.tile_shape_mn)|join(', ')}}>;
  using ClusterShape = Shape<{{
      to_cute_constant(sch.cluster_shape_mnk)|join(', ')}}>;
  // TODO: Reimplement
  // using KernelSchedule   = {{KernelScheduleTag[sch.kernel_schedule]}};
  using EpilogueSchedule = {{EpilogueScheduleTag[sch.epilogue_schedule]}};
  using TileScheduler    = {{TileSchedulerTag[sch.tile_scheduler]}};
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};
{% endfor %}
    
{% for impl_config in impl_configs %}
{% set t = impl_config.types -%}
{% set schs = impl_config.schedules -%}
{% set type_sig = gen_type_sig(t) -%}

template<typename Sch>
using Kernel_{{type_sig}} = MacheteKernelTemplate<
  {{DataTypeTag[t.a]}},  // ElementA
  {{DataTypeTag[t.b]}},  // ElementB
  {{DataTypeTag[t.out]}},  // ElementD
  {{DataTypeTag[t.accumulator]}}, // Accumulator
  {{DataTypeTag[t.b_group_scale]}}, // GroupScaleT
  {{DataTypeTag[t.b_group_zeropoint]}}, // GroupZeroT
  {{DataTypeTag[t.b_channel_scale]}}, // ChannelScaleT
  {{DataTypeTag[t.a_token_scale]}}, // TokenScaleT
  cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput,
  Sch>;

{% for sch in schs %}
{% set sch_sig = gen_sch_sig(sch) -%}
torch::Tensor 
impl_{{type_sig}}_sch_{{sch_sig}}(MMArgs args) {
  return run_impl<Kernel_{{type_sig}}<sch_{{sch_sig}}>>(args);
}
{%- endfor %}
{%- endfor %}

}; // namespace machete
"""

PREPACK_TEMPLATE = """
#include "../machete_prepack_launcher.cuh"

namespace machete {

torch::Tensor prepack_B_dispatch(PrepackBArgs args) {
  auto convert_type = args.maybe_group_scales_type.value_or(args.a_type);
  {%- for t in types %}
  {% set b_type = unsigned_type_with_bitwidth(t.b_num_bits) %}
  if (args.a_type == {{TorchTypeTag[t.a]}}
      && args.b_type.size_bits() == {{t.b_num_bits}} 
      && convert_type == {{TorchTypeTag[t.convert]}}) {
    return prepack_impl<
      PrepackedLayoutBTemplate<
        {{DataTypeTag[t.a]}}, // ElementA
        {{DataTypeTag[b_type]}}, // ElementB
        {{DataTypeTag[t.convert]}}, // ElementConvert
        {{DataTypeTag[t.accumulator]}}, // Accumulator
        cutlass::layout::ColumnMajor,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput>
    >(args.B); 
  }
  {%- endfor %}
  
  TORCH_CHECK_NOT_IMPLEMENTED(false, 
    "prepack_B_dispatch(..) is not implemented for "
    "atype = ", args.a_type,
    ", b_type = ", args.b_type.str(),
    ", with_group_scales_type= ", args.maybe_group_scales_type ? 
        toString(*args.maybe_group_scales_type) : "None");
}

}; // namespace machete
"""

TmaMI = MixedInputKernelScheduleType.TmaWarpSpecializedCooperativeMixedInput
TmaCoop = EpilogueScheduleType.TmaWarpSpecializedCooperative


@dataclass(frozen=True)
class ScheduleConfig:
    tile_shape_mn: Tuple[int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    kernel_schedule: MixedInputKernelScheduleType
    epilogue_schedule: EpilogueScheduleType
    tile_scheduler: TileSchedulerType


@dataclass(frozen=True)
class TypeConfig:
    a: DataType
    b: Union[DataType, VLLMDataType]
    b_group_scale: DataType
    b_group_zeropoint: DataType
    b_channel_scale: DataType
    a_token_scale: DataType
    out: DataType
    accumulator: DataType


@dataclass(frozen=True)
class PrepackTypeConfig:
    a: DataType
    b_num_bits: int
    convert: DataType
    accumulator: DataType


@dataclass
class ImplConfig:
    types: TypeConfig
    schedules: List[ScheduleConfig]
    heuristic: List[Tuple[Optional[str], ScheduleConfig]]


def generate_sch_sig(schedule_config: ScheduleConfig) -> str:
    tile_shape = (
        f"{schedule_config.tile_shape_mn[0]}x{schedule_config.tile_shape_mn[1]}"
    )
    cluster_shape = (f"{schedule_config.cluster_shape_mnk[0]}" +
                     f"x{schedule_config.cluster_shape_mnk[1]}" +
                     f"x{schedule_config.cluster_shape_mnk[2]}")
    kernel_schedule = VLLMKernelScheduleTag[schedule_config.kernel_schedule]\
        .split("::")[-1]
    epilogue_schedule = EpilogueScheduleTag[
        schedule_config.epilogue_schedule].split("::")[-1]
    tile_scheduler = TileSchedulerTag[schedule_config.tile_scheduler]\
        .split("::")[-1]

    return (f"{tile_shape}_{cluster_shape}_{kernel_schedule}" +
            f"_{epilogue_schedule}_{tile_scheduler}")


# mostly unique shorter sch_sig
def generate_terse_sch_sig(schedule_config: ScheduleConfig) -> str:
    kernel_terse_names_replace = {
        "KernelTmaWarpSpecializedCooperativeMixedInput_": "TmaMI_",
        "TmaWarpSpecializedCooperative_": "TmaCoop_",
        "StreamKScheduler": "streamK",
    }

    sch_sig = generate_sch_sig(schedule_config)
    for orig, terse in kernel_terse_names_replace.items():
        sch_sig = sch_sig.replace(orig, terse)
    return sch_sig


# unique type_name
def generate_type_signature(kernel_types: TypeConfig):
    return str("".join([
        VLLMDataTypeNames[getattr(kernel_types, field.name)]
        for field in fields(TypeConfig)
    ]))


def generate_type_option_name(kernel_types: TypeConfig):
    return ", ".join([
        f"{field.name.replace('b_', 'with_')+'_type'}=" +
        VLLMDataTypeNames[getattr(kernel_types, field.name)]
        for field in fields(TypeConfig)
    ])


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def to_cute_constant(value: List[int]):

    def _to_cute_constant(value: int):
        if is_power_of_two(value):
            return f"_{value}"
        else:
            return f"Int<{value}>"

    if isinstance(value, Iterable):
        return [_to_cute_constant(value) for value in value]
    else:
        return _to_cute_constant(value)


def unique_schedules(impl_configs: List[ImplConfig]):
    return list(
        set(sch for impl_config in impl_configs
            for sch in impl_config.schedules))


def unsigned_type_with_bitwidth(num_bits):
    return {
        4: DataType.u4,
        8: DataType.u8,
        16: DataType.u16,
        32: DataType.u32,
        64: DataType.u64,
    }[num_bits]


template_globals = {
    "void": DataType.void,
    "DataTypeTag": VLLMDataTypeTag,
    "VLLMScalarTypeTag": VLLMDataTypeVLLMScalarTypeTag,
    "TorchTypeTag": VLLMDataTypeTorchDataTypeTag,
    "KernelScheduleTag": VLLMKernelScheduleTag,
    "EpilogueScheduleTag": EpilogueScheduleTag,
    "TileSchedulerTag": TileSchedulerTag,
    "to_cute_constant": to_cute_constant,
    "gen_sch_sig": generate_terse_sch_sig,
    "gen_type_sig": generate_type_signature,
    "unique_schedules": unique_schedules,
    "unsigned_type_with_bitwidth": unsigned_type_with_bitwidth,
    "gen_type_option_name": generate_type_option_name
}


def create_template(template_str):
    template = jinja2.Template(template_str)
    template.globals.update(template_globals)
    return template


mm_dispatch_template = create_template(DISPATCH_TEMPLATE)
mm_impl_template = create_template(IMPL_TEMPLATE)
prepack_dispatch_template = create_template(PREPACK_TEMPLATE)


def create_sources(impl_configs: List[ImplConfig], num_impl_files=8):
    sources = []

    sources.append((
        "machete_mm_dispatch",
        mm_dispatch_template.render(impl_configs=impl_configs),
    ))

    prepack_types = []
    for impl_config in impl_configs:
        convert_type = impl_config.types.a \
             if impl_config.types.b_group_scale == DataType.void \
             else impl_config.types.b_group_scale
        prepack_types.append(
            PrepackTypeConfig(
                a=impl_config.types.a,
                b_num_bits=VLLMDataTypeSize[impl_config.types.b],
                convert=convert_type,
                accumulator=impl_config.types.accumulator,
            ))

    def prepacked_type_key(prepack_type: PrepackTypeConfig):
        # For now we we can just use the first accumulator type seen since
        # the tensor core shapes/layouts don't vary based on accumulator
        # type so we can generate less code this way
        return (prepack_type.a, prepack_type.b_num_bits, prepack_type.convert)

    unique_prepack_types = []
    prepack_types_seen = set()
    for prepack_type in prepack_types:
        key = prepacked_type_key(prepack_type)
        if key not in prepack_types_seen:
            unique_prepack_types.append(prepack_type)
            prepack_types_seen.add(key)

    sources.append((
        "machete_prepack",
        prepack_dispatch_template.render(types=unique_prepack_types, ),
    ))

    # Split up impls across files
    num_impls = reduce(lambda x, y: x + len(y.schedules), impl_configs, 0)
    num_impls_per_file = math.ceil(num_impls / num_impl_files)

    files_impls: List[List[ImplConfig]] = [[]]

    curr_num_impls_assigned = 0
    curr_impl_in_file = 0
    curr_impl_configs = deepcopy(list(reversed(impl_configs)))

    while curr_num_impls_assigned < num_impls:
        room_left_in_file = num_impls_per_file - curr_impl_in_file
        if room_left_in_file == 0:
            files_impls.append([])
            room_left_in_file = num_impls_per_file
            curr_impl_in_file = 0

        curr_ic = curr_impl_configs[-1]
        if len(curr_ic.schedules) >= room_left_in_file:
            # Break apart the current impl config
            tmp_ic = deepcopy(curr_ic)
            tmp_ic.schedules = curr_ic.schedules[:room_left_in_file]
            curr_ic.schedules = curr_ic.schedules[room_left_in_file:]
            files_impls[-1].append(tmp_ic)
        else:
            files_impls[-1].append(curr_ic)
            curr_impl_configs.pop()
        curr_num_impls_assigned += len(files_impls[-1][-1].schedules)
        curr_impl_in_file += len(files_impls[-1][-1].schedules)

    for part, file_impls in enumerate(files_impls):
        sources.append((
            f"machete_mm_impl_part{part+1}",
            mm_impl_template.render(impl_configs=file_impls),
        ))

    return sources


def generate():
    # See csrc/quantization/machete/Readme.md, the Codegeneration for more info
    # about how this works
    SCRIPT_DIR = os.path.dirname(__file__)

    sch_common_params = dict(
        kernel_schedule=TmaMI,
        epilogue_schedule=TmaCoop,
        tile_scheduler=TileSchedulerType.StreamK,
    )

    # Stored as "condition": ((tile_shape_mn), (cluster_shape_mnk))
    default_tile_heuristic_config = {
        #### M = 257+
        "M > 256 && K <= 16384 && N <= 4096": ((128, 128), (2, 1, 1)),
        "M > 256": ((128, 256), (2, 1, 1)),
        #### M = 129-256
        "M > 128 && K <= 4096 && N <= 4096": ((128, 64), (2, 1, 1)),
        "M > 128 && K <= 8192 && N <= 8192": ((128, 128), (2, 1, 1)),
        "M > 128": ((128, 256), (2, 1, 1)),
        #### M = 65-128
        "M > 64 && K <= 4069 && N <= 4069": ((128, 32), (2, 1, 1)),
        "M > 64 && K <= 4069 && N <= 8192": ((128, 64), (2, 1, 1)),
        "M > 64 && K >= 8192 && N >= 12288": ((256, 128), (2, 1, 1)),
        "M > 64": ((128, 128), (2, 1, 1)),
        #### M = 33-64
        "M > 32 && K <= 6144 && N <= 6144": ((128, 16), (1, 1, 1)),
        "M > 32 && K >= 16384 && N >= 12288": ((256, 64), (2, 1, 1)),
        "M > 32": ((128, 64), (2, 1, 1)),
        #### M = 17-32
        "M > 16 && K <= 12288 && N <= 8192": ((128, 32), (2, 1, 1)),
        "M > 16": ((256, 32), (2, 1, 1)),
        #### M = 1-16
        "N >= 26624": ((256, 16), (1, 1, 1)),
        None: ((128, 16), (1, 1, 1)),
    }

    # For now we use the same heuristic for all types
    # Heuristic is currently tuned for H100s
    default_heuristic = [
        (cond, ScheduleConfig(*tile_config,
                              **sch_common_params))  # type: ignore
        for cond, tile_config in default_tile_heuristic_config.items()
    ]

    def get_unique_schedules(heuristic: Dict[str, ScheduleConfig]):
        # Do not use schedules = list(set(...)) because we need to make sure
        # the output list is deterministic; otherwise the generated kernel file
        # will be non-deterministic and causes ccache miss.
        schedules = []
        for _, schedule_config in heuristic:
            if schedule_config not in schedules:
                schedules.append(schedule_config)
        return schedules

    impl_configs = []

    GPTQ_kernel_type_configs = list(
        TypeConfig(
            a=a,
            b=b,
            b_group_scale=a,
            b_group_zeropoint=DataType.void,
            b_channel_scale=DataType.void,
            a_token_scale=DataType.void,
            out=a,
            accumulator=DataType.f32,
        ) for b in (VLLMDataType.u4b8, VLLMDataType.u8b128)
        for a in (DataType.f16, DataType.bf16))

    impl_configs += [
        ImplConfig(x[0], x[1], x[2])
        for x in zip(GPTQ_kernel_type_configs,
                     itertools.repeat(get_unique_schedules(default_heuristic)),
                     itertools.repeat(default_heuristic))
    ]

    AWQ_kernel_type_configs = list(
        TypeConfig(
            a=a,
            b=b,
            b_group_scale=a,
            b_group_zeropoint=a,
            b_channel_scale=DataType.void,
            a_token_scale=DataType.void,
            out=a,
            accumulator=DataType.f32,
        ) for b in (DataType.u4, DataType.u8)
        for a in (DataType.f16, DataType.bf16))

    impl_configs += [
        ImplConfig(x[0], x[1], x[2])
        for x in zip(AWQ_kernel_type_configs,
                     itertools.repeat(get_unique_schedules(default_heuristic)),
                     itertools.repeat(default_heuristic))
    ]

    # Stored as "condition": ((tile_shape_mn), (cluster_shape_mnk))
    # TODO (LucasWilkinson): Further tuning required
    qqq_tile_heuristic_config = {
        #### M = 257+
        # ((128, 256), (2, 1, 1)) Broken for QQQ types
        # TODO (LucasWilkinson): Investigate further
        # "M > 256 && K <= 16384 && N <= 4096": ((128, 128), (2, 1, 1)),
        # "M > 256": ((128, 256), (2, 1, 1)),
        "M > 256": ((128, 128), (2, 1, 1)),
        #### M = 129-256
        "M > 128 && K <= 4096 && N <= 4096": ((128, 64), (2, 1, 1)),
        "M > 128 && K <= 8192 && N <= 8192": ((128, 128), (2, 1, 1)),
        # ((128, 256), (2, 1, 1)) Broken for QQQ types
        # TODO (LucasWilkinson): Investigate further
        # "M > 128": ((128, 256), (2, 1, 1)),
        "M > 128": ((128, 128), (2, 1, 1)),
        #### M = 65-128
        "M > 64 && K <= 4069 && N <= 4069": ((128, 32), (2, 1, 1)),
        "M > 64 && K <= 4069 && N <= 8192": ((128, 64), (2, 1, 1)),
        "M > 64 && K >= 8192 && N >= 12288": ((256, 128), (2, 1, 1)),
        "M > 64": ((128, 128), (2, 1, 1)),
        #### M = 33-64
        "M > 32 && K <= 6144 && N <= 6144": ((128, 16), (1, 1, 1)),
        # Broken for QQQ types
        # TODO (LucasWilkinson): Investigate further
        #"M > 32 && K >= 16384 && N >= 12288": ((256, 64), (2, 1, 1)),
        "M > 32": ((128, 64), (2, 1, 1)),
        #### M = 17-32
        "M > 16 && K <= 12288 && N <= 8192": ((128, 32), (2, 1, 1)),
        "M > 16": ((256, 32), (2, 1, 1)),
        #### M = 1-16
        "N >= 26624": ((256, 16), (1, 1, 1)),
        None: ((128, 16), (1, 1, 1)),
    }

    # For now we use the same heuristic for all types
    # Heuristic is currently tuned for H100s
    qqq_heuristic = [
        (cond, ScheduleConfig(*tile_config,
                              **sch_common_params))  # type: ignore
        for cond, tile_config in qqq_tile_heuristic_config.items()
    ]

    QQQ_kernel_types = [
        *(TypeConfig(
            a=DataType.s8,
            b=VLLMDataType.u4b8,
            b_group_scale=b_group_scale,
            b_group_zeropoint=DataType.void,
            b_channel_scale=DataType.f32,
            a_token_scale=DataType.f32,
            out=DataType.f16,
            accumulator=DataType.s32,
        ) for b_group_scale in (DataType.f16, DataType.void)),
        *(TypeConfig(
            a=DataType.e4m3,
            b=VLLMDataType.u4b8,
            b_group_scale=b_group_scale,
            b_group_zeropoint=DataType.void,
            b_channel_scale=DataType.f32,
            a_token_scale=DataType.f32,
            out=DataType.f16,
            accumulator=DataType.f32,
        ) for b_group_scale in (DataType.f16, DataType.void)),
    ]

    impl_configs += [
        ImplConfig(x[0], x[1], x[2])
        for x in zip(QQQ_kernel_types,
                     itertools.repeat(get_unique_schedules(qqq_heuristic)),
                     itertools.repeat(qqq_heuristic))
    ]

    output_dir = os.path.join(SCRIPT_DIR, "generated")

    # Delete the "generated" directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create the "generated" directory
    os.makedirs(output_dir)

    # Render each group of configurations into separate files
    for filename, code in create_sources(impl_configs):
        filepath = os.path.join(output_dir, f"{filename}.cu")
        with open(filepath, "w") as output_file:
            output_file.write(code)
        print(f"Rendered template to {filepath}")


if __name__ == "__main__":
    generate()
