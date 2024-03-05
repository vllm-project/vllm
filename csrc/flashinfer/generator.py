"""
Copyright (c) 2024 by vLLM team.
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import itertools
import os
import pathlib
from typing import List

root = pathlib.Path(__name__).cwd()
enable_bf16 = True


def get_instantiation_cu() -> List[str]:
    prefix = "generated"
    (root / prefix).mkdir(parents=True, exist_ok=True)
    dtypes = {"fp16": "nv_half"}
    if enable_bf16:
        dtypes["bf16"] = "nv_bfloat16"
    group_sizes = os.environ.get("FLASHINFER_GROUP_SIZES", "1,4,8").split(",")
    head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "64,128,256").split(",")
    group_sizes = [int(x) for x in group_sizes]
    head_dims = [int(x) for x in head_dims]
    causal_options = [True]  # NOTE(woosuk): Disabled non-causal for now.
    allow_fp16_qk_reduction_options = [False]  # NOTE(woosuk): Disabled for now.
    layout_options = ["NHD"]  # NOTE(woosuk): Disabled other layouts for now.
    pos_encoding_mode_options = ["None", "ALiBi"]  # NOTE(woosuk): Disabled RoPELlama for now.

    # dispatch.inc
    path = root / prefix / "dispatch.inc"
    if not path.exists():
        with open(root / prefix / "dispatch.inc", "w") as f:
            f.write("#define _DISPATCH_CASES_group_size(...)      \\\n")
            for x in group_sizes:
                f.write(f"  _DISPATCH_CASE({x}, GROUP_SIZE, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("#define _DISPATCH_CASES_head_dim(...)        \\\n")
            for x in head_dims:
                f.write(f"  _DISPATCH_CASE({x}, HEAD_DIM, __VA_ARGS__) \\\n")
            f.write("// EOL\n")
            f.write("\n")

    files = []
    for (
        group_size,
        head_dim,
        dtype,
        causal,
        allow_fp16_qk_reduction,
        layout,
        pos_encoding_mode,
    ) in itertools.product(
        group_sizes,
        head_dims,
        dtypes,
        causal_options,
        allow_fp16_qk_reduction_options,
        layout_options,
        pos_encoding_mode_options,
    ):
        # paged batch prefill
        fname = f"paged_batch_prefill_group{group_size}_head{head_dim}_causal{causal}_fp16qk{allow_fp16_qk_reduction}_layout{layout}_pe{pos_encoding_mode}_{dtype}.cu"
        files.append(prefix + "/" + fname)
        if not (root / prefix / fname).exists():
            with open(root / prefix / fname, "w") as f:
                f.write('#include <flashinfer_decl.h>\n\n')
                f.write(f"#include <flashinfer.cuh>\n\n")
                f.write(f"using namespace flashinfer;\n\n")
                f.write(
                    "INST_BatchPrefillPagedWrapper({}, {}, {}, {}, {}, {}, {})\n".format(
                        dtypes[dtype],
                        group_size,
                        head_dim,
                        str(causal).lower(),
                        str(allow_fp16_qk_reduction).lower(),
                        "QKVLayout::k" + layout,
                        "PosEncodingMode::k" + pos_encoding_mode,
                    )
                )

        # ragged batch prefill
        fname = f"ragged_batch_prefill_group{group_size}_head{head_dim}_causal{causal}_fp16qk{allow_fp16_qk_reduction}_layout{layout}_pe{pos_encoding_mode}_{dtype}.cu"
        files.append(prefix + "/" + fname)
        if not (root / prefix / fname).exists():
            with open(root / prefix / fname, "w") as f:
                f.write('#include "<flashinfer_decl.h>"\n\n')
                f.write(f"#include <flashinfer.cuh>\n\n")
                f.write(f"using namespace flashinfer;\n\n")
                f.write(
                    "INST_BatchPrefillRaggedWrapper({}, {}, {}, {}, {}, {}, {})\n".format(
                        dtypes[dtype],
                        group_size,
                        head_dim,
                        str(causal).lower(),
                        str(allow_fp16_qk_reduction).lower(),
                        "QKVLayout::k" + layout,
                        "PosEncodingMode::k" + pos_encoding_mode,
                    )
                )

        # single prefill
        fname = f"single_prefill_group{group_size}_head{head_dim}_causal{causal}_fp16qk{allow_fp16_qk_reduction}_layout{layout}_pe{pos_encoding_mode}_{dtype}.cu"
        files.append(prefix + "/" + fname)
        if not (root / prefix / fname).exists():
            with open(root / prefix / fname, "w") as f:
                f.write('#include <flashinfer_decl.h>\n\n')
                f.write(f"#include <flashinfer.cuh>\n\n")
                f.write(f"using namespace flashinfer;\n\n")
                f.write(
                    "INST_SinglePrefill({}, {}, {}, {}, {}, {}, {})\n".format(
                        dtypes[dtype],
                        group_size,
                        head_dim,
                        str(causal).lower(),
                        str(allow_fp16_qk_reduction).lower(),
                        "QKVLayout::k" + layout,
                        "PosEncodingMode::k" + pos_encoding_mode,
                    )
                )

    return files


files = get_instantiation_cu()
