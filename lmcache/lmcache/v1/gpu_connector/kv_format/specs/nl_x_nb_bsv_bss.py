# SPDX-License-Identifier: Apache-2.0
"""DSA indexer k-cache: ``NL x [NB, BS, 132]`` uint8, blocked scales.

Logically identical to the per-layer MLA layout (one plane, ``num_heads ==
1``), which is what vLLM's ``get_kv_cache_shape`` reports and how the tensor
registers. The PHYSICAL page layout differs (vllm
``indexer_k_quant_and_cache``): per block, all tokens' 128-byte fp8 values
first, then all tokens' 4-byte fp32 scales —
``[BS x 128 vals][BS x 4B scales]``. Only the transfer kernels care (they
address values and scales separately); every geometry accessor here matches
the MLA spec.
"""

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.nl_x_nb_bs_hs import (
    NL_X_NB_BS_HS_Spec,
)
import lmcache.c_ops as lmc_ops


class NL_X_NB_BSV_BSS_Spec(NL_X_NB_BS_HS_Spec):
    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NB_BSV_BSS
    attention_backends = ("vLLM MLA",)
