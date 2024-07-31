import itertools
import math
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Tuple

import jinja2
from vllm_cutlass_library_extension import (DataType, DataTypeNames,
                                            DataTypeTag, EpilogueScheduleTag,
                                            EpilogueScheduleType,
                                            KernelScheduleTag,
                                            KernelScheduleType,
                                            MixedInputKernelScheduleType,
                                            TileSchedulerTag,
                                            TileSchedulerType, VLLMDataType,
                                            VLLMTileSchedulerType)

#
#   Generator templating
#

DISPATCH_TEMPLATE = """
#include "../machete_mm_launcher.cuh"

namespace machete {
using KernelDispatcher_ = KernelDispatcher<
    {{DataTypeTag[type_config.element_a]}},  // ElementA
    {{DataTypeTag[type_config.element_b]}},  // ElementB
    {{DataTypeTag[type_config.element_d]}},  // ElementD
    {{DataTypeTag[type_config.accumulator]}}, // Accumulator
    {{DataTypeTag[type_config.element_b_scale]}}, // Scales
    {{DataTypeTag[type_config.element_b_zeropoint]}}>; // Zeropoints

{% for _, schedule_name in schedules %}extern torch::Tensor 
impl_{{type_name}}_sch_{{schedule_name}}(PytorchArguments args);
{% endfor %}
template <>
torch::Tensor KernelDispatcher_::dispatch(PytorchArguments args) {
  if (!args.schedule) {
    return impl_{{ type_name }}_sch_{{ schedules[0][1] }}(args);
  }
  {% for _, schedule_name in schedules %}
  if (*args.schedule == "{{ schedule_name }}") {
    return impl_{{ type_name }}_sch_{{ schedule_name }}(args);
  }
  {% endfor %}
  TORCH_CHECK_NOT_IMPLEMENTED(false, "machete_gemm(..) is not implemented for "
                                     "schedule = ", *args.schedule);
}

template <>
std::vector<std::string> KernelDispatcher_::supported_schedules() {
  return { 
    {% for _, schedule_name in schedules -%}
    "{{ schedule_name }}"{{ ",
    " if not loop.last }}{%- endfor %}
  };
}

}; // namespace machete
"""

IMPL_TEMPLATE = """
#include "../machete_mm_launcher.cuh"

namespace machete {
template <typename Config, bool with_C, bool with_scales, bool with_zeropoints>
using Kernel = KernelTemplate<
    {{DataTypeTag[type_config.element_a]}},  // ElementA
    {{DataTypeTag[type_config.element_b]}},  // ElementB
    {{DataTypeTag[type_config.element_d]}},  // ElementD
    {{DataTypeTag[type_config.accumulator]}}, // Accumulator
    {{DataTypeTag[type_config.element_b_scale]}}, // Scales
    {{DataTypeTag[type_config.element_b_zeropoint]}}, // Zeropoints
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput
>::Speacialization<Config, with_C, with_scales, with_zeropoints>;

{% for schedule, schedule_name in schedules %}
struct sch_{{schedule_name}} {
  using TileShapeNM = Shape<{{
      to_cute_constant(schedule.tile_shape_mn)|join(', ')}}>;
  using ClusterShape = Shape<{{
      to_cute_constant(schedule.cluster_shape_mnk)|join(', ')}}>;
  // TODO: Reimplement
  // using KernelSchedule   = {{KernelScheduleTag[schedule.kernel_schedule]}};
  using EpilogueSchedule = {{EpilogueScheduleTag[schedule.epilogue_schedule]}};
  using TileScheduler    = {{TileSchedulerTag[schedule.tile_scheduler]}};
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

torch::Tensor 
impl_{{type_name}}_sch_{{schedule_name}}(PytorchArguments args) {
  bool with_C = args.C.has_value(), with_scales = args.scales.has_value(),
       with_zeropoints = args.zeros.has_value();

  {% for s in specializations %}
  if (with_C == {{s.with_C|lower}}
      && with_zeropoints == {{s.with_zeropoints|lower}}
      && with_scales == {{s.with_scales|lower}}) {
      return run_impl<Kernel<sch_{{schedule_name}}, {{s.with_C|lower}},
        {{s.with_scales|lower}}, {{s.with_zeropoints|lower}}>>(args);
  }{% endfor %}

  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "for the sake of compile times and binary size machete_mm(..) is "
      " not implemented for with_C=", with_C, ", with_scales=", with_scales, 
      ", with_zeropoints=", with_zeropoints, 
      " (for {{type_name}}_sch_{{schedule_name}})");
}
{% endfor %}

}; // namespace machete
"""

PREPACK_TEMPLATE = """
#include "../machete_prepack_launcher.cuh"

namespace machete {
using PrepackBDispatcher_ = PrepackBDispatcher<
  {{DataTypeTag[type_config.element_a]}}, // ElementA
  {{DataTypeTag[type_config.element_b]}}, // ElementB
  {{DataTypeTag[type_config.element_d]}}, // ElementD
  {{DataTypeTag[type_config.accumulator]}}, // Accumulator
  {{DataTypeTag[type_config.element_b_scale]}}, // Scales
  {{DataTypeTag[type_config.element_b_zeropoint]}}>; // Zeropoints

using PrepackedLayoutB = PrepackedLayoutBBTemplate<
  {{DataTypeTag[type_config.element_a]}}, // ElementA
  {{DataTypeTag[type_config.element_b]}}, // ElementB
  {{DataTypeTag[type_config.element_d]}}, // ElementD
  {{DataTypeTag[type_config.accumulator]}}, // Accumulator
  cutlass::layout::ColumnMajor,
  cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput>;

template <>
torch::Tensor PrepackBDispatcher_::dispatch(torch::Tensor B) {
  return prepack_impl<PrepackedLayoutB>(B);
}
}; // namespace machete
"""

TmaMI = MixedInputKernelScheduleType.TmaWarpSpecializedCooperativeMixedInput
TmaCoop = EpilogueScheduleType.TmaWarpSpecializedCooperative


@dataclass
class ScheduleConfig:
    tile_shape_mn: Tuple[int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    kernel_schedule: KernelScheduleType
    epilogue_schedule: EpilogueScheduleType
    tile_scheduler: TileSchedulerType


@dataclass
class TypeConfig:
    element_a: DataType
    element_b: DataType
    element_b_scale: DataType
    element_b_zeropoint: DataType
    element_d: DataType
    accumulator: DataType


@dataclass
class Specialization:
    with_C: bool
    with_zeropoints: bool
    with_scales: bool


@dataclass
class ImplConfig:
    type_config: TypeConfig
    schedule_configs: List[ScheduleConfig]
    specializations: List[Specialization]


def generate_schedule_name(schedule_config: ScheduleConfig) -> str:
    tile_shape = (
        f"{schedule_config.tile_shape_mn[0]}x{schedule_config.tile_shape_mn[1]}"
    )
    cluster_shape = (f"{schedule_config.cluster_shape_mnk[0]}" +
                     f"x{schedule_config.cluster_shape_mnk[1]}" +
                     f"x{schedule_config.cluster_shape_mnk[2]}")
    kernel_schedule = KernelScheduleTag[schedule_config.kernel_schedule].split(
        "::")[-1]
    epilogue_schedule = EpilogueScheduleTag[
        schedule_config.epilogue_schedule].split("::")[-1]
    tile_scheduler = TileSchedulerTag[schedule_config.tile_scheduler].split(
        "::")[-1]

    return (f"{tile_shape}_{cluster_shape}_{kernel_schedule}" +
            f"_{epilogue_schedule}_{tile_scheduler}")


# mostly unique shorter schedule_name
def generate_terse_schedule_name(schedule_config: ScheduleConfig) -> str:
    kernel_terse_names_replace = {
        "KernelTmaWarpSpecializedCooperativeMixedInput_": "TmaMI_",
        "TmaWarpSpecializedCooperative_": "TmaCoop_",
        "VLLMStreamKScheduler": "NMstreamK",
    }

    schedule_name = generate_schedule_name(schedule_config)
    for orig, terse in kernel_terse_names_replace.items():
        schedule_name = schedule_name.replace(orig, terse)
    return schedule_name


# unique type_name
def generate_type_signature(kernel_type_config: TypeConfig):
    element_a = DataTypeNames[kernel_type_config.element_a]
    element_b = DataTypeNames[kernel_type_config.element_b]
    element_d = DataTypeNames[kernel_type_config.element_d]
    accumulator = DataTypeNames[kernel_type_config.accumulator]
    element_scale = DataTypeNames[kernel_type_config.element_b_scale]
    element_zeropoint = DataTypeNames[kernel_type_config.element_b_zeropoint]

    return (f"{element_a}{element_b}{element_d}"
            f"{accumulator}{element_scale}{element_zeropoint}")


# non-unique shorter type_name
def generate_terse_type_signature(kernel_type_config: TypeConfig):
    element_a = DataTypeNames[kernel_type_config.element_a]
    element_b = DataTypeNames[kernel_type_config.element_b]

    return f"{element_a}{element_b}"


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


template_globals = {
    "DataTypeTag": DataTypeTag,
    "KernelScheduleTag": KernelScheduleTag,
    "EpilogueScheduleTag": EpilogueScheduleTag,
    "TileSchedulerTag": TileSchedulerTag,
    "to_cute_constant": to_cute_constant,
}


def create_template(template_str):
    template = jinja2.Template(template_str)
    template.globals.update(template_globals)
    return template


mm_dispatch_template = create_template(DISPATCH_TEMPLATE)
mm_impl_template = create_template(IMPL_TEMPLATE)
prepack_dispatch_template = create_template(PREPACK_TEMPLATE)


def create_sources(impl_config: ImplConfig, num_impl_files=2):
    sources = []
    # Render the template with the provided configurations
    schedules_with_names = [(schedule, generate_terse_schedule_name(schedule))
                            for schedule in impl_config.schedule_configs]

    type_name = generate_type_signature(impl_config.type_config)
    terse_type_name = generate_terse_type_signature(impl_config.type_config)

    sources.append((
        f"machete_mm_{terse_type_name}",
        mm_dispatch_template.render(type_name=type_name,
                                    type_config=impl_config.type_config,
                                    schedules=schedules_with_names),
    ))

    sources.append((
        f"machete_prepack_{terse_type_name}",
        prepack_dispatch_template.render(
            type_name=type_name,
            type_config=impl_config.type_config,
        ),
    ))

    num_schedules = len(impl_config.schedule_configs)
    schedules_per_file = math.ceil(num_schedules / num_impl_files)
    for part, i in enumerate(range(0, num_schedules, schedules_per_file)):
        file_schedules = schedules_with_names[i:i + schedules_per_file]

        sources.append((
            f"machete_mm_{terse_type_name}_impl_part{part}",
            mm_impl_template.render(
                type_name=type_name,
                type_config=impl_config.type_config,
                schedules=file_schedules,
                specializations=impl_config.specializations,
            ),
        ))
    return sources


def generate():
    SCRIPT_DIR = os.path.dirname(__file__)

    schedules = list((
        ScheduleConfig(
            tile_shape_mn=tile_shape_mn,
            cluster_shape_mnk=cluster_shape_mnk,
            kernel_schedule=kernel_schedule,
            epilogue_schedule=epilogue_schedule,
            tile_scheduler=tile_scheduler,
        ) for tile_shape_mn, cluster_shape_mnk in (
            ((128, 16), (1, 1, 1)),
            ((128, 32), (1, 1, 1)),
            ((128, 64), (1, 1, 1)),
            ((128, 128), (1, 1, 1)),
            # ((128, 256), (1, 1, 1)),
        ) for kernel_schedule in (TmaMI, ) for epilogue_schedule in (TmaCoop, )
        for tile_scheduler in (
            TileSchedulerType.Default,
            VLLMTileSchedulerType.StreamK,
        )))

    impl_configs = []

    GPTQ_kernel_type_configs = list(
        (TypeConfig(
            element_a=element_a,
            element_b=element_b,
            element_b_scale=element_a,
            element_b_zeropoint=element_a,
            element_d=element_a,
            accumulator=DataType.f32,
        ) for element_b in (VLLMDataType.u4b8, VLLMDataType.u8b128)
         for element_a in (DataType.f16, DataType.bf16)))

    GPTQ_kernel_specializations = [
        Specialization(with_C=False, with_zeropoints=False, with_scales=True)
    ]

    impl_configs += [
        ImplConfig(x[0], x[1], x[2])
        for x in zip(GPTQ_kernel_type_configs, itertools.repeat(schedules),
                     itertools.repeat(GPTQ_kernel_specializations))
    ]

    AWQ_kernel_type_configs = list(
        (TypeConfig(
            element_a=element_a,
            element_b=element_b,
            element_b_scale=element_a,
            element_b_zeropoint=element_a,
            element_d=element_a,
            accumulator=DataType.f32,
        ) for element_b in (DataType.u4, DataType.u8)
         for element_a in (DataType.f16, DataType.bf16)))

    AWQ_kernel_specializations = [
        Specialization(with_C=False, with_zeropoints=True, with_scales=True)
    ]

    impl_configs += [
        ImplConfig(x[0], x[1], x[2])
        for x in zip(AWQ_kernel_type_configs, itertools.repeat(schedules),
                     itertools.repeat(AWQ_kernel_specializations))
    ]

    output_dir = os.path.join(SCRIPT_DIR, "generated")

    # Delete the "generated" directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create the "generated" directory
    os.makedirs(output_dir)

    # Render each group of configurations into separate files
    for impl_config in impl_configs:
        for filename, code in create_sources(impl_config):
            filepath = os.path.join(output_dir, f"{filename}.cu")
            with open(filepath, "w") as output_file:
                output_file.write(code)
            print(f"Rendered template to {filepath}")


if __name__ == "__main__":
    generate()
