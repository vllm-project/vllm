import sys

import pandas as pd
import torch
import torch.nn.functional as F

import vllm._gradlib_C  # noqa: F401

torch.ops._gradlib_C.rocb_create_extension()
torch.ops._gradlib_C.hipb_create_extension()


class TunedGemm:

    def __init__(self, tuned_csv_file):
        self.bestsols = pd.read_csv(tuned_csv_file, index_col=[0])
        self.create_ds()

    def create_ds(self):
        df = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = df.iloc[i]
            key = (ds['M'], ds['N'], ds['K'])
            if ds['libtype'] == 'hipblaslt':
                soltype = 1
            elif ds['libtype'] == 'rocblas':
                soltype = 2
            solds[key] = (soltype, int(ds['solidx']))
        #print(solds)
        self.solids = solds

    def query_sol(self, m, n, k):
        return self.solids.get((m, n, k), (0, 0))

    def mm(self, inp, weights):
        soltype, solidx = self.query_sol(m=weights.shape[0],
                                         n=inp.shape[0],
                                         k=inp.shape[1])
        if soltype == 1:
            out = torch.ops._gradlib_C.hipb_mm(inp, weights.t(), solidx, None,
                                               None, None, None, None)
        elif soltype == 2:
            out = torch.ops._gradlib_C.rocb_mm(inp, weights.t(), solidx)
        else:
            out = F.linear(inp, weights)
        return out

    def run_all_tuned_sols(self):
        for i in range(len(self.bestsols)):
            ds = self.bestsols.iloc[i]
            print('>>> Running tuned solution')
            print(ds)
            inp = torch.randn((ds['N'], ds['K']),
                              dtype=get_dtype(ds['dtype']),
                              device='cuda')
            weights = torch.randn((ds['M'], ds['K']),
                                  dtype=get_dtype(ds['dtype']),
                                  device='cuda')
            self.mm(inp, weights)


def get_dtype(dtype_csv):
    if dtype_csv == 'torch.float16':
        dtype = torch.float16
    elif dtype_csv == 'torch.bfloat16':
        dtype = torch.bfloat16
    elif dtype_csv == 'torch.float32':
        dtype = torch.float32
    elif dtype_csv == 'torch.float8_e4m3fnuz':
        dtype = torch.float8_e4m3fnuz
    return dtype


if __name__ == '__main__':
    tgemm = TunedGemm(sys.argv[1])  #csv file with tuned sols goes in argv[1]
    print(tgemm.bestsols)
    tgemm.run_all_tuned_sols()
