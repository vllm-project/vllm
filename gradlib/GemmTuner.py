import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

import vllm._gradlib_C  # noqa: F401

rtol = 1e-5
atol = 1

CACHE_INVALIDATE_BUFFERS = int(os.getenv("CACHE_INVALIDATE_BUFFERS", "37"))
ONE = torch.ones(1, dtype=torch.float32, device='cuda')


class Gemm:

    def __init__(self, m, n, k, bias, indtype, outdtype, rocblas_decode=False):
        self.m = m
        self.k = k
        self.n = n
        self.bias = torch.randn(m, device='cuda').to(indtype) if bias else None
        self.indtype = indtype
        self.outdtype = outdtype
        self.use_rocblas = (indtype == outdtype
                            and indtype is not torch.float8_e4m3fnuz)
        self.nb = CACHE_INVALIDATE_BUFFERS
        self.inp = torch.randn((self.n, self.k),
                               device='cuda').to(self.indtype)
        self.weights = torch.randn((self.m, self.k),
                                   device='cuda').to(self.indtype)
        # weights2 is used in measurement/warm iters to ensure
        # HBM fetch for weight tensors
        self.weights2 = torch.randn((self.nb, self.m, self.k),
                                    device='cuda').to(self.indtype)
        self.blob = torch.ones(128 * 1024 * 1024,
                               dtype=torch.float32,
                               device='cuda')
        self.topn = 20  #number of top solutions from each source
        self.hipb_sols = []
        self.rocb_sols = []
        self.rtol = 1e-5
        self.atol = 1
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        # prefer hipblaslt unless rocblas time is less than this
        # ratio of hipblaslt time
        self.hipb_prefer_ratio = 0.995
        self.rocblas_decode = rocblas_decode

    def find_hipblas_sols(self):
        sols = torch.ops._gradlib_C.hipb_findallsols(self.inp,
                                                     self.weights.t(),
                                                     self.bias, self.outdtype)
        print('M N K bias dtype',
              self.m,
              self.n,
              self.k,
              self.bias is not None,
              self.indtype,
              '>>> Total hipb solutions',
              len(sols),
              flush=True)
        #print(sols)
        self.hipb_sols = sols

    def check_gemm_ref(self, libtype, solidx):
        if self.indtype == torch.float8_e4m3fnuz:
            ref = torch._scaled_mm(self.inp,
                                   self.weights.t(),
                                   scale_a=ONE,
                                   scale_b=ONE,
                                   out_dtype=self.outdtype)
            if type(ref) is tuple and len(ref) == 2:
                ref = ref[0]
        else:
            ref = F.linear(self.inp, self.weights, self.bias)
        if libtype == 'hipblaslt':
            c = torch.ops._gradlib_C.hipb_mm(self.inp, self.weights.t(),
                                             solidx, self.bias, self.outdtype,
                                             None, None, None)
        elif libtype == 'rocblas':
            c = torch.ops._gradlib_C.rocb_mm(self.inp, self.weights.t(),
                                             solidx)
            if self.bias is not None:
                c += self.bias
        if torch.allclose(c.to(self.outdtype),
                          ref.to(self.outdtype),
                          atol=self.atol,
                          rtol=self.rtol):
            return True

        print('>>>',
              libtype,
              'Solidx',
              solidx,
              'FAILED reference test',
              flush=True)
        #print(ref, flush=True)
        #print(c, flush=True)
        return False

    def hipb_time_sol(self, solidx, cold_iters=2, warm_iters=10):
        #print('>>>hipbtime',solidx)
        for i in range(cold_iters):
            torch.ops._gradlib_C.hipb_mm(self.inp, self.weights.t(), solidx,
                                         None, self.outdtype, None, None, None)
        self.start.record()
        for i in range(warm_iters):
            torch.ops._gradlib_C.hipb_mm(
                self.inp, self.weights2[random.randint(0, self.nb - 1)].t(),
                solidx, None, self.outdtype, None, None, None)
        self.end.record()
        torch.cuda.synchronize()
        gtime = self.start.elapsed_time(self.end) / warm_iters
        #print('>>> Solidx GTime',solidx,gtime,'ms')
        return gtime

    def hipb_time_all_sols(self, fast_mode=0, top_sols=0):
        coldi = 20
        warmi = 20
        if fast_mode:
            coldi = 2
            warmi = 2
        solutions = self.hipb_sols
        if top_sols:
            solutions = self.hipb_top_sols
        gtimes = {}
        for solidx in solutions:
            gtimes[solidx] = self.hipb_time_sol(solidx,
                                                cold_iters=coldi,
                                                warm_iters=warmi)
        self.hipb_gtimedf = pd.DataFrame.from_dict(
            gtimes, orient='index',
            columns=['gtimems']).sort_values(by='gtimems')
        self.hipb_gtimedf.to_csv('/tmp/hipb_gtimedf.csv')
        print('>>> HipBlasLt top solutions, Fast Mode', fast_mode)
        print(self.hipb_gtimedf.head(self.topn))

    def rocb_time_sol(self, solidx, cold_iters=2, warm_iters=10):

        def rocb_mm_bias(inp, w, solidx, bias):
            return torch.ops._gradlib_C.rocb_mm(inp, w, solidx) + bias

        def rocb_mm_nobias(inp, w, solidx, _):
            return torch.ops._gradlib_C.rocb_mm(inp, w, solidx)

        rocb_fun = rocb_mm_bias if self.bias is not None else rocb_mm_nobias
        for _ in range(cold_iters):
            rocb_fun(self.inp, self.weights.t(), solidx, self.bias)

        self.start.record()
        for _ in range(warm_iters):
            rocb_fun(self.inp, self.weights2[random.randint(0,
                                                            self.nb - 1)].t(),
                     solidx, self.bias)

        self.end.record()
        torch.cuda.synchronize()
        gtime = self.start.elapsed_time(self.end) / warm_iters
        #print('>>> RocSolidx GTime',solidx,gtime,'ms')
        return gtime

    def find_rocblas_sols(self):
        sols = torch.ops._gradlib_C.rocb_findallsols(self.inp,
                                                     self.weights.t())
        print('M N K dtype',
              self.m,
              self.n,
              self.k,
              self.indtype,
              '>>> Total rocb solutions',
              len(sols),
              flush=True)
        #print(sols)
        self.rocb_sols = sols

    def rocb_time_all_sols(self, fast_mode=0, top_sols=0):
        coldi = 20
        warmi = 20
        if fast_mode:
            coldi = 2
            warmi = 2
        solutions = self.rocb_sols
        if top_sols:
            solutions = self.rocb_top_sols
        gtimes = {}
        for solidx in solutions:
            gtimes[solidx] = self.rocb_time_sol(solidx, coldi, warmi)
        self.rocb_gtimedf = pd.DataFrame.from_dict(
            gtimes, orient='index',
            columns=['gtimems']).sort_values(by='gtimems')
        self.rocb_gtimedf.to_csv('/tmp/rocb_gtimedf.csv')
        print('>>> Rocblas top solutions, Fast Mode', fast_mode, flush=True)
        print(self.rocb_gtimedf.head(self.topn), flush=True)

    def warmup(self, warmi=500):
        for i in range(warmi):
            self.blob = self.blob + 0.00001

    def functional_check_topn_fastest(self):
        rocb_topn = []
        for solidx in self.rocb_gtimedf.index[:self.topn]:
            if self.check_gemm_ref(libtype='rocblas', solidx=solidx):
                rocb_topn.append(solidx)
        self.rocb_top_sols = rocb_topn
        hipb_topn = []
        for solidx in self.hipb_gtimedf.index[:self.topn]:
            if self.check_gemm_ref(libtype='hipblaslt', solidx=solidx):
                hipb_topn.append(solidx)
        self.hipb_top_sols = hipb_topn

    def find_fastest_solution(self):
        if self.use_rocblas:
            self.find_rocblas_sols()
        if not (self.rocblas_decode and self.n == 1):
            self.find_hipblas_sols()
        self.warmup()
        self.rocb_time_all_sols(fast_mode=1)
        self.warmup()
        self.hipb_time_all_sols(fast_mode=1)
        self.functional_check_topn_fastest()
        self.warmup()
        self.rocb_time_all_sols(fast_mode=0, top_sols=1)
        self.warmup()
        self.hipb_time_all_sols(fast_mode=0, top_sols=1)
        if len(self.rocb_gtimedf) > 0 and len(self.hipb_gtimedf) > 0:
            best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
            best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
            if best_rocb_time < best_hipb_time * self.hipb_prefer_ratio:
                self.best_libtype = 'rocblas'
                self.best_solidx = self.rocb_gtimedf.index[0]
                self.best_soltime = best_rocb_time
            else:
                self.best_libtype = 'hipblaslt'
                self.best_solidx = self.hipb_gtimedf.index[0]
                self.best_soltime = best_hipb_time
            #self.check_gemm_ref(self.best_libtype,self.best_solidx)
        elif len(self.hipb_gtimedf) > 0:
            print('>>> Only hipblas solutions found!', flush=True)
            best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
            self.best_libtype = 'hipblaslt'
            self.best_solidx = self.hipb_gtimedf.index[0]
            self.best_soltime = best_hipb_time
        elif len(self.rocb_gtimedf) > 0:
            print('>>> Only rocblas solutions found!', flush=True)
            best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
            self.best_libtype = 'rocblas'
            self.best_solidx = self.rocb_gtimedf.index[0]
            self.best_soltime = best_rocb_time
        else:
            print('>>> No rocblas or hipblas solutions found!', flush=True)
            self.best_libtype = 'rocblas'
            self.best_solidx = 0
            self.best_soltime = 0
        print('>>> Fastest Solution is',
              self.best_libtype,
              self.best_solidx,
              self.best_soltime,
              flush=True)


class GemmTuner:

    def __init__(self,
                 indtype,
                 outdtype,
                 tuned_file=None,
                 rocblas_decode=False):
        self.gemm_problems = pd.DataFrame(columns=['M', 'N', 'K', 'bias'])
        self.indtype = indtype
        self.outdtype = outdtype
        self.rocblas_decode = rocblas_decode
        self.tuned_file = tuned_file
        if Path(tuned_file).is_file():
            self.tuned_shapes = pd.read_csv(tuned_file)
        else:
            self.tuned_shapes = None

    def add_gemm(self, m, n, k, indtype=None, bias=False):
        indtype = self.indtype if self.indtype is not None else indtype
        assert indtype is not None
        outdtype = self.outdtype if self.outdtype is not None else indtype
        assert outdtype is not None
        if (self.tuned_shapes is None or (self.tuned_shapes[
            (self.tuned_shapes['M'] == m) & (self.tuned_shapes['N'] == n) &
            (self.tuned_shapes['K'] == k) &
            (self.tuned_shapes['bias'] == bias) &
            (self.tuned_shapes['dtype'] == str(indtype)) &
            (self.tuned_shapes['outdtype'] == str(outdtype))].empty)):
            entry = {
                'M': [m],
                'N': [n],
                'K': [k],
                'bias': [bias],
                'dtype': [indtype],
                'outdtype': [outdtype]
            }
            df = pd.DataFrame(entry)
            self.gemm_problems = pd.concat([self.gemm_problems, df],
                                           ignore_index=True)
        else:
            print(f">>>Info: Found Duplicate shape(M:{m},"
                  " N:{n}, K:{k} bias:{bias}), skipping")

    def find_best_sols(self):
        df = self.gemm_problems
        soldf = pd.DataFrame(columns=['libtype', 'solidx', 'soltimes'])
        for i in range(len(df)):
            ds = df.loc[i, :]
            indtype = self.indtype or ds['dtype']
            outdtype = self.outdtype or indtype
            gemmobj = Gemm(ds['M'],
                           ds['N'],
                           ds['K'],
                           ds['bias'],
                           indtype=indtype,
                           outdtype=outdtype,
                           rocblas_decode=self.rocblas_decode)
            gemmobj.find_fastest_solution()
            soldf.loc[i, 'libtype'] = gemmobj.best_libtype
            soldf.loc[i, 'solidx'] = gemmobj.best_solidx
            soldf.loc[i, 'soltimes'] = gemmobj.best_soltime

            del gemmobj
            torch.cuda.empty_cache()

        finaldf = pd.concat([self.gemm_problems, soldf], axis=1)
        if self.tuned_shapes is not None:
            finaldf = pd.concat([finaldf, self.tuned_shapes])
        finaldf['solidx'] = finaldf['solidx'].convert_dtypes('int64')
        finaldf.to_csv(self.tuned_file, index=False)
        print(finaldf)
