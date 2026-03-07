# E
v
ro
m

t Var
ab

s R
f
r

c

Th
s pag
 docum

ts a
 

v
ro
m

t var
ab

s support
d by vLLM.
## Cor
 E
v
ro
m

t Var
ab

s
### VLLM_LOGGING_LEVEL
**Typ
:** `str

g`
**D
fau
t:** `INFO`
**D
scr
pt
o
:** Co
tro
s th
 
ogg

g v
rbos
ty. Va

d va
u
s ar
 `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
```bash

xport VLLM_LOGGING_LEVEL=DEBUG
```
### VLLM_LOG_STATS_INTERVAL
**Typ
:** `f
oat`
**D
fau
t:** `10.0`
**D
scr
pt
o
:** I
t
rva
 

 s
co
ds b
t


 
ogg

g stat
st
cs.
```bash

xport VLLM_LOG_STATS_INTERVAL=1.0  # Log stats 
v
ry s
co
d
```
### VLLM_HOST_IP
**Typ
:** `str

g`
**D
fau
t:** Auto-d
t
ct
d
**D
scr
pt
o
:** Ov
rr
d
 th
 IP addr
ss us
d by vLLM for d
str
but
d commu

cat
o
.
```bash

xport VLLM_HOST_IP=192.168.1.100
```
### VLLM_PORT
**Typ
:** `

t`
**D
fau
t:** `8000`
**D
scr
pt
o
:** Port for th
 vLLM s
rv
r.
```bash

xport VLLM_PORT=8080
```
### VLLM_API_KEY
**Typ
:** `str

g`
**D
fau
t:** No


**D
scr
pt
o
:** API k
y for auth

t
cat
o
. Ca
 b
 comma-s
parat
d for mu
t
p

 k
ys.
```bash

xport VLLM_API_KEY=your-s
cr
t-k
y
```
## P
rforma
c
 & Opt
m
zat
o

### VLLM_USE_V1
**Typ
:** `0` or `1`
**D
fau
t:** `1` (s

c
 v0.7.0)
**D
scr
pt
o
:** E
ab

 th
 


 vLLM v1 arch
t
ctur
.
```bash

xport VLLM_USE_V1=1  # E
ab

 v1

xport VLLM_USE_V1=0  # Us
 

gacy arch
t
ctur

```
### VLLM_ENABLE_CUDA_COMPATIBILITY
**Typ
:** `0` or `1`
**D
fau
t:** `0`
**D
scr
pt
o
:** E
ab

 CUDA for
ard compat
b


ty for o
d
r dr
v
rs.
```bash

xport VLLM_ENABLE_CUDA_COMPATIBILITY=1
```
### VLLM_CUDA_COMPATIBILITY_PATH
**Typ
:** `str

g`
**D
fau
t:** No


**D
scr
pt
o
:** Path to CUDA compat
b


ty 

brar

s.
```bash

xport VLLM_CUDA_COMPATIBILITY_PATH=/usr/
oca
/cuda-12.9/compat
```
### VLLM_WORKER_MULTIPROC_METHOD
**Typ
:** `str

g`
**D
fau
t:** `fork`
**D
scr
pt
o
:** Mu
t
proc
ss

g start m
thod for 
ork
rs. Opt
o
s: `fork`, `spa

`.
```bash

xport VLLM_WORKER_MULTIPROC_METHOD=spa


```
### VLLM_ENABLE_V1_MULTIPROCESSING
**Typ
:** `0` or `1`
**D
fau
t:** `1`
**D
scr
pt
o
:** E
ab

 mu
t
proc
ss

g 

 v1 arch
t
ctur
.
```bash

xport VLLM_ENABLE_V1_MULTIPROCESSING=0  # D
sab

 for d
bugg

g
```
## D
str
but
d & Mu
t
-GPU
### NCCL_DEBUG
**Typ
:** `str

g`
**D
fau
t:** No


**D
scr
pt
o
:** S
t to `TRACE` or `INFO` for NCCL d
bugg

g.
```bash

xport NCCL_DEBUG=TRACE
```
### NCCL_SOCKET_IFNAME
**Typ
:** `str

g`
**D
fau
t:** Auto-d
t
ct
d
**D
scr
pt
o
:** N
t
ork 

t
rfac
 for NCCL commu

cat
o
.
```bash

xport NCCL_SOCKET_IFNAME=
th0
```
### GLOO_SOCKET_IFNAME
**Typ
:** `str

g`
**D
fau
t:** Auto-d
t
ct
d
**D
scr
pt
o
:** N
t
ork 

t
rfac
 for GLOO commu

cat
o
.
```bash

xport GLOO_SOCKET_IFNAME=
th0
```
### NCCL_P2P_DISABLE
**Typ
:** `0` or `1`
**D
fau
t:** `0`
**D
scr
pt
o
:** D
sab

 p
r-to-p
r commu

cat
o
 b
t


 GPUs.
```bash

xport NCCL_P2P_DISABLE=1
```
### NCCL_CUMEM_ENABLE
**Typ
:** `0` or `1`
**D
fau
t:** `1` (

 



r NCCL v
rs
o
s)
**D
scr
pt
o
:** E
ab

 CUDA m
mory a
ocat
o
 opt
m
zat
o
.
```bash

xport NCCL_CUMEM_ENABLE=0
```
## D
bugg

g
### CUDA_LAUNCH_BLOCKING
**Typ
:** `0` or `1`
**D
fau
t:** `0`
**D
scr
pt
o
:** Mak
 a
 CUDA op
rat
o
s sy
chro
ous for d
bugg

g.
```bash

xport CUDA_LAUNCH_BLOCKING=1
```
### VLLM_TRACE_FUNCTION
**Typ
:** `0` or `1`
**D
fau
t:** `0`
**D
scr
pt
o
:** R
cord a
 fu
ct
o
 ca
s for d
bugg

g. **WARNING: S
o
s do

 
x
cut
o
 by ov
r 100x.**
```bash

xport VLLM_TRACE_FUNCTION=1
```
### VLLM_ATTENTION_BACKEND
**Typ
:** `str

g`
**D
fau
t:** Auto-s


ct
d
**D
scr
pt
o
:** Forc
 a sp
c
f
c att

t
o
 back

d. Opt
o
s: `FLASH_ATTN`, `FLASHINFER`, `XFORMERS`, 
tc.
```bash

xport VLLM_ATTENTION_BACKEND=FLASH_ATTN
```
### TRITON_PTXAS_PATH
**Typ
:** `str

g`
**D
fau
t:** Auto-d
t
ct
d
**D
scr
pt
o
:** Path to ptxas b

ary for Tr
to
 comp

at
o
.
```bash

xport TRITON_PTXAS_PATH=/usr/
oca
/cuda/b

/ptxas
```
## Mod

 & Tok


z
r
### VLLM_USE_MODELSCOPE
**Typ
:** `0` or `1`
**D
fau
t:** `0`
**D
scr
pt
o
:** Do


oad mod

s from Mod

Scop
 

st
ad of Hugg

gFac
.
```bash

xport VLLM_USE_MODELSCOPE=1
```
### HF_HOME
**Typ
:** `str

g`
**D
fau
t:** `~/.cach
/hugg

gfac
`
**D
scr
pt
o
:** Cach
 d
r
ctory for Hugg

gFac
 mod

s.
```bash

xport HF_HOME=/path/to/cach

```
### TRANSFORMERS_CACHE
**Typ
:** `str

g`
**D
fau
t:** `~/.cach
/hugg

gfac
/tra
sform
rs`
**D
scr
pt
o
:** Cach
 d
r
ctory for Tra
sform
rs mod

s.
```bash

xport TRANSFORMERS_CACHE=/path/to/tra
sform
rs_cach

```
### VLLM_ALLOW_LONG_MAX_MODEL_LEN
**Typ
:** `0` or `1`
**D
fau
t:** `0`
**D
scr
pt
o
:** A
o
 ov
rr
d

g max mod

 


gth b
yo
d saf
ty 

m
ts.
```bash

xport VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```
## Pr
f
x Cach

g
### VLLM_PREFIX_CACHING_HASH_ALGO
**Typ
:** `str

g`
**D
fau
t:** `sha256`
**D
scr
pt
o
:** Hash

g a
gor
thm for pr
f
x cach

g. Opt
o
s: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`.
```bash

xport VLLM_PREFIX_CACHING_HASH_ALGO=sha256_cbor
```
## Commo
 Us
 Cas
s
### D
v

opm

t / D
bugg

g
```bash

xport VLLM_LOGGING_LEVEL=DEBUG

xport VLLM_TRACE_FUNCTION=1

xport CUDA_LAUNCH_BLOCKING=1

xport VLLM_LOG_STATS_INTERVAL=1.0
```
### Product
o
 D
p
oym

t
```bash

xport VLLM_LOGGING_LEVEL=INFO

xport VLLM_API_KEY=your-s
cr
t-k
y

xport VLLM_PORT=8000
```
### Mu
t
-Nod
 S
tup
```bash

xport VLLM_HOST_IP=10.0.0.1

xport NCCL_SOCKET_IFNAME=
th0

xport GLOO_SOCKET_IFNAME=
th0
```
### CUDA Compat
b


ty (O
d
r Dr
v
rs)
```bash

xport VLLM_ENABLE_CUDA_COMPATIBILITY=1

xport VLLM_CUDA_COMPATIBILITY_PATH=/usr/
oca
/cuda-12.9/compat
```
## S
 A
so
    - [Co
f
gurat
o
 Gu
d
](../co
f
gurat
o
/

g


_params.md)
    - [Troub

shoot

g Gu
d
](../usag
/troub

shoot

g.md)
    - [P
rforma
c
 Tu


g](../usag
/p
rforma
c
_tu


g.md)
