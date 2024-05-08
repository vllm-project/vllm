import sys

import hipbsolidxgemm
import pandas as pd
#import gradlib
import rocsolidxgemm
import torch
import torch.nn.functional as F

#gradlib.create_extension()
rocsolidxgemm.rocb_create_extension()
hipbsolidxgemm.hipb_create_extension()

#m = 128; n = 192 ;k = 256
#m = 7168; k = 4096*2; n = 256
#m = int(1024*1.25); k = int(1024*8); n = 1
#m = 1; k = int(1024*8); n = int(1024*7)
#m=22016; k=4096 ; n=1
#m=int(27648/1);k=5120;n=8
#m=5120;k=13824;n=1
m = 3 * 5120
k = 5120
n = 1

rtol = 1e-5
atol = 1
dtype = torch.float16


class Gemm:

    def __init__(self, m, n, k, dtype=torch.float16):
        self.m = m
        self.k = k
        self.n = n
        self.dtype = dtype
        self.inp = torch.randn((self.n, self.k),
                               dtype=self.dtype,
                               device='cuda')
        self.weights = torch.randn((self.m, self.k),
                                   dtype=self.dtype,
                                   device='cuda')
        self.hipb_sols = []
        self.rtol = 1e-5
        self.atol = 1
        self.cold_iters = 2
        self.warm_iters = 10

    def find_hipblas_sols(self):
        sols = hipbsolidxgemm.hipb_findallsols(self.inp, self.weights.t())
        print('M N K', self.m, self.n, self.k, '>>> Total hipb solutions',
              len(sols))
        #print(sols)
        self.hipb_sols = sols

    def hipb_check_gemm_ref(self, user_solidxs=None):
        ref = F.linear(self.inp, self.weights)
        solidxs = user_solidxs if user_solidxs is not None else self.hipb_sols
        if len(solidxs) > 0:
            for solidx in solidxs:
                c = hipbsolidxgemm.hipb_mm(self.inp, self.weights.t(), solidx)
                if torch.allclose(c, ref, atol=self.atol, rtol=self.rtol):
                    print('>>> Hipb solidx', solidx, 'passed reference test')
                else:
                    print('>>> Hipb solidx', solidx, 'FAILED reference test')
                    print(ref)
                    print(c)

    def hipb_time_sol(self, solidx):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(self.cold_iters):
            hipbsolidxgemm.hipb_mm(self.inp, self.weights.t(), solidx)
        start.record()
        for i in range(self.warm_iters):
            hipbsolidxgemm.hipb_mm(self.inp, self.weights.t(), solidx)
        end.record()
        torch.cuda.synchronize()
        gtime = start.elapsed_time(end) / self.warm_iters
        #print('>>> Solidx GTime',solidx,gtime,'ms')
        return gtime

    def hipb_time_all_sols(self):
        gtimes = {}
        for solidx in self.hipb_sols:
            gtimes[solidx] = self.hipb_time_sol(solidx)
        self.gtimedf = pd.DataFrame.from_dict(
            gtimes, orient='index',
            columns=['gtimems']).sort_values(by='gtimems')
        self.gtimedf.to_csv('/tmp/gtimedf.csv')
        print(self.gtimedf.head(10))


gemmobj = Gemm(m=3 * 5120, n=1, k=5120)
gemmobj.find_hipblas_sols()
#gemmobj.hipb_check_gemm_ref()
#gemmobj.hipb_check_gemm_ref(user_solidxs=[131,8190])
#gemmobj.hipb_time_sol(gemmobj.hipb_sols[0])
gemmobj.hipb_time_all_sols()
gemmobj.hipb_check_gemm_ref(user_solidxs=gemmobj.gtimedf.head(5).index.values)

sys.exit()


def splitk_linear(inp, w, splitk=2):
    wsp = torch.chunk(w, splitk, dim=1)
    isp = torch.chunk(inp, splitk, dim=1)
    print('>>>', isp[0].shape, wsp[1].shape)
    cnew = []
    for i in range(splitk):
        cnew.append(F.linear(isp[i], wsp[i]))
    #cnew1 = F.linear(isp[1],wsp[1])
    c = cnew[0]
    for i in range(1, splitk):
        c.add_(cnew[i])
    #c = torch.add(cnew0,cnew1)

    return c


def splitm_linear(inp, w, splitm=2, splits=None, splitk=1):
    outputp = []
    #wsp = torch.chunk(F.pad(weights,(0,0,0,padm)),splitm)
    if splits is not None:
        wsp = torch.split(w, splits)
    else:
        wsp = torch.chunk(w, splitm)
    #cout = torch.empty(inp.shape[0], w.shape[0],
    #                   dtype=inp.dtype,device=inp.device)
    #csp = torch.chunk(cout,splitm,dim=1)

    for i, _ in enumerate(wsp):
        #print('>>>wspi',wsp[i].shape)
        if splitk == 1:
            outputp.append(F.linear(inp, wsp[i]))
            #cout[:,i*wsp[i].shape[0]:
            #     (i+1)*wsp[i].shape[0]] = F.linear(inp, wsp[i])
            #csp[i].copy_(F.linear(inp, wsp[i]))
        else:
            outputp.append(splitk_linear(inp, wsp[i], splitk))
    c = torch.cat((outputp), dim=1)
    #print('>>>',c.shape,cout.shape)
    return c


def splitn_linear(inp, w, splitn=2, splits=None):
    outputp = []
    if splits is not None:
        isp = torch.split(inp, splits)
    else:
        isp = torch.chunk(inp, splitn)
    torch.empty(inp.shape[0], w.shape[0], dtype=inp.dtype, device=inp.device)
    for i, _ in enumerate(isp):
        outputp.append(F.linear(isp[i], w))
        #cout[i*isp[i].shape[0]:
        #     (i+1)*isp[i].shape[0],:] = F.linear(isp[i], w)
    c = torch.cat((outputp), dim=0)
    #print('>>>',c.shape,cout.shape)
    return c


nncount = 0
for _ in range(10):
    #a = torch.randn((m, k), dtype=dtype, device='cuda')
    #b = torch.randn((k, n), dtype=dtype, device='cuda')
    inp = torch.randn((n, k), dtype=dtype, device='cuda')
    weights = torch.randn((m, k), dtype=dtype, device='cuda')
    #c = gradlib.mm(inp, weights.t())
    c = hipbsolidxgemm.hipb_mm(inp, weights.t(), 20053)
    c = hipbsolidxgemm.hipb_mm(inp, weights.t(), 20053)
    c = rocsolidxgemm.rocb_mm(inp, weights.t(), 60995)
    c = rocsolidxgemm.rocb_mm(inp, weights.t(), 60995)

    splitm = 2
    #padm=2
    outsp = []
    #wsp = torch.chunk(F.pad(weights,(0,0,0,padm)),splitm)
    #wsp = torch.chunk(weights,splitm)
    #wsp = torch.split(weights,(3*1024,4*1024))
    #c = torch.empty((n,m),dtype=dtype,device='cuda')
    #outtup = []
    #for i,_ in enumerate(wsp):
    #    print('>>>wspi',wsp[i].shape)
    #    outsp.append(F.linear(inp, wsp[i]))
    #    #outtup.append(splitk_linear(inp, wsp[i]))
    #outsp = [torch.add(a,b) for a,b in outtup]
    #c = torch.cat((outsp),dim=1)
    #c = c[:,:-padm]
    #c = splitm_linear(inp,weights,splitm=4,splits=None,splitk=1)
    #c = splitn_linear(inp,weights,splitn=2,splits=None)

    #wsp = torch.chunk(weights,2,dim=1)
    #isp = torch.chunk(inp,2,dim=1)
    #print('>>>',isp[0].shape,wsp[1].shape)
    #cnew0 = F.linear(isp[0],wsp[0])
    #cnew1 = F.linear(isp[1],wsp[1])
    #c = torch.add(cnew0,cnew1)
    #c = splitk_linear(inp, weights, splitk=4)

    #torch.cuda.synchronize()
    ref = F.linear(inp, weights)
    #ref = torch.matmul(a,b)
    if torch.allclose(c, ref, atol=atol, rtol=rtol):
        nncount += 1
    else:
        print(ref)
        print(c)
'''
tncount = 0
for _ in range(10):
    a = torch.randn((m, k), dtype=dtype, device='cuda')
    b = torch.randn((n, k), dtype=dtype, device='cuda')
    c = gradlib.mm(a, b.t())
    #torch.cuda.synchronize()
    ref = torch.matmul(a, b.t())
    if torch.allclose(c, ref,  atol=atol,  rtol=rtol):
        tncount += 1
    else:
        print(ref)
        print(c)
        #torch.save(c-ref, '/tmp/difference.pt')
        #np.savetxt('my_file.txt', (c-ref).cpu().numpy())
        dfs = ref - c
        nz = torch.nonzero(dfs,as_tuple=True)
        print(nz)
        print(dfs[nz])
        print(ref[nz])
        print(c[nz])
'''
'''
ntcount = 0
for _ in range(10):
    a = torch.randn((k, m), dtype=dtype, device='cuda')
    b = torch.randn((k, n), dtype=dtype, device='cuda')
    c = gradlib.mm(a.t(), b)
    #torch.cuda.synchronize()
    if torch.allclose(c, torch.matmul(a.t(), b),  atol=atol,  rtol=rtol):
        ntcount += 1

ttcount = 0
for _ in range(10):
    a = torch.randn((k, m), dtype=dtype, device='cuda')
    b = torch.randn((n, k), dtype=dtype, device='cuda')
    c = gradlib.mm(a.t(), b.t())
    torch.cuda.synchronize()
    if torch.allclose(c, torch.matmul(a.t(), b.t()),  atol=atol,  rtol=rtol):
        ttcount += 1
'''
print(f"GEMM (m, n, k) = {n}, {m}, {k}")
print(f"NN GEMMs: pass {nncount}/10, tol={rtol}")
#print(f"TN GEMMs: pass {tncount}/10, tol={rtol}")
#print(f"NT GEMMs: pass {ntcount}/10, tol={rtol}")
#print(f"TT GEMMs: pass {ttcount}/10, tol={rtol}")
