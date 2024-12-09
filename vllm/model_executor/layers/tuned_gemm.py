import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.envs import VLLM_USE_ROCM_SKINNY_GEMM
from vllm.platforms import current_platform
from vllm.utils import is_navi

support_tuned_gemms = False
if current_platform.is_rocm():
    import vllm._gradlib_C  # noqa: F401
    support_tuned_gemms = True


def hipb_mm(inp, weights, solidx, bias=None):
    return torch.ops._gradlib_C.hipb_mm(inp, weights, solidx, bias, None, None,
                                        None, None)


def rocb_mm(inp, weights, solidx):
    return torch.ops._gradlib_C.rocb_mm(inp, weights, solidx)


class TunedGemm:

    def __init__(self):
        self.extensions_created = False
        self.save_gemm = int(os.environ.get('VLLM_TUNE_GEMM', 0))
        self.untune_path = os.environ.get('VLLM_UNTUNE_FILE',
                                          "/tmp/vllm_untuned.csv")
        self.tune_path = os.environ.get('VLLM_TUNE_FILE', "tuned.csv")
        self.bestsols = {}
        self.load_best_sols()
        self.create_ds()
        self.cu_count = torch.cuda.get_device_properties(
            device='cuda').multi_processor_count

        self.use_skinny = (current_platform.is_rocm()
                           and VLLM_USE_ROCM_SKINNY_GEMM and not is_navi())

        if (self.save_gemm == 1):
            self.tuned_df = pd.DataFrame(
                columns=['M', 'N', 'K', 'bias', 'dtype'])
        else:
            self.tuned_df = None

    def load_best_sols(self):
        if self.tune_path is not None and Path(self.tune_path).is_file():
            self.bestsols = pd.read_csv(self.tune_path)

    def create_ds(self):
        df: pd.DataFrame = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = df.iloc[i]
            key = (ds['M'], ds['N'], ds['K'], ds['bias'], ds['dtype'])
            if ds['libtype'] == 'hipblaslt':
                soltype = 1
            elif ds['libtype'] == 'rocblas':
                soltype = 2
            solds[key] = (soltype, int(ds['solidx']))
        self.solids = solds

    def query_sol(self, m, n, k, bias, dtype):
        return self.solids.get((m, n, k, bias, str(dtype)), (0, 0))

    def apply_skinny(self, m, n, k, inp_view, weights):
        if not self.use_skinny:
            return None
        if inp_view.dtype != torch.float16 or k % 8 != 0:
            return None
        if m > 8 and 0 < n <= 4:
            out = torch.empty(inp_view.shape[0],
                              weights.shape[0],
                              dtype=inp_view.dtype,
                              device='cuda')
            ops.wvSpltK(weights, inp_view, out, n, self.cu_count)
            return out
        elif m % 4 == 0 and n == 1 and k <= 8192:
            out = torch.empty(inp_view.shape[0],
                              weights.shape[0],
                              dtype=inp_view.dtype,
                              device='cuda')
            ops.LLMM1(weights, inp_view, out, 4)
            return out
        else:
            return None

    def mm(self, inp, weights, bias=None):
        if not support_tuned_gemms:
            return F.linear(inp, weights, bias)
        # F.Linear can take a 3 dimensional input. vllm
        # uses this for linear units. However, sampler
        # will use torch.matmul with 2 dimensions only
        if inp.dim() == 3:
            try:
                inp_view = inp.view(-1, inp.size(-1))
                batched = True
            except RuntimeError:
                return F.linear(inp, weights, bias)
        else:
            inp_view = inp
            batched = False
        if self.extensions_created is False:
            torch.ops._gradlib_C.rocb_create_extension()
            torch.ops._gradlib_C.hipb_create_extension()
            self.extensions_created = True
        m = weights.shape[0]
        n = inp_view.shape[0]
        k = inp_view.shape[1]
        use_bias = bias is not None
        soltype, solidx = self.query_sol(m=m,
                                         n=n,
                                         k=k,
                                         bias=use_bias,
                                         dtype=inp.dtype)
        out = self.apply_skinny(m, n, k, inp_view, weights)
        if out is not None:
            if batched:
                out = out.view(inp.shape[0], inp.shape[1], weights.shape[0])
            if bias is not None:
                return out + bias
            return out
        elif soltype == 1:
            out = hipb_mm(inp_view, weights.t(), solidx, bias)
        elif soltype == 2:
            out = rocb_mm(inp_view, weights.t(), solidx)
            if bias is not None:
                out = out + bias
        else:
            if (self.save_gemm == 1):
                self.tuned_df = pd.concat([
                    self.tuned_df,
                    pd.DataFrame({
                        'M': [m],
                        'N': [n],
                        'K': [k],
                        'bias': [bias is not None],
                        'dtype': [inp.dtype],
                    })
                ]).drop_duplicates()
                self.tuned_df.to_csv(self.untune_path, index=False)
            return F.linear(inp, weights, bias)
        if batched:
            out = out.view(inp.shape[0], inp.shape[1], weights.shape[0])
        return out


tgemm = TunedGemm()
