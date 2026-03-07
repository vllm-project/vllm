# Att

t
o
 Back

d F
atur
 Support
Th
s docum

t 
s auto-g


rat
d by `too
s/pr
_comm
t/g


rat
_att

t
o
_back

d_docs.py`.
It sho
s th
 f
atur
 support for 
ach r
g
st
r
d att

t
o
 back

d
bas
d o
 th
 ch
cks 

 `Att

t
o
Back

d.va

dat
_co
f
gurat
o
()`.
**Do 
ot 
d
t th
s f


 ma
ua
y.** Ru
 th
 fo
o


g comma
d to
r
g


rat
 
t:
```bash
pytho
 too
s/pr
_comm
t/g


rat
_att

t
o
_back

d_docs.py
```
## S
tt

g th
 Att

t
o
 Back

d
### Comma
d L



Th
r
 ar
 t
o 
ays to sp
c
fy th
 back

d from th
 comma
d 



:
**Opt
o
 1: Us

g `--att

t
o
-back

d` (s
mp

)**
```bash
v
m s
rv
 
mod


 --att

t
o
-back

d FLASH_ATTN
```
**Opt
o
 2: Us

g `--att

t
o
-co
f
g.back

d` / `-ac.back

d` (structur
d co
f
g)**
```bash
# Dot 
otat
o

v
m s
rv
 
mod


 --att

t
o
-co
f
g.back

d FLASH_ATTN
v
m s
rv
 
mod


 -ac.back

d FLASH_ATTN
# JSON format
v
m s
rv
 
mod


 --att

t
o
-co
f
g '{"back

d": "FLASH_ATTN"}'
v
m s
rv
 
mod


 -ac '{"back

d": "FLASH_ATTN"}'
```

 **Not
:** `--att

t
o
-back

d` a
d `--att

t
o
-co
f
g.back

d` ar
 mutua
y

 
xc
us
v
. Us
 o

 or th
 oth
r, 
ot both.
### Pytho
 API
Us
 `Att

t
o
Co
f
g` 

th th
 `LLM` c
ass:
```pytho

from v
m 
mport LLM
from v
m.co
f
g 
mport Att

t
o
Co
f
g
from v
m.v1.att

t
o
.back

ds.r
g
stry 
mport Att

t
o
Back

dE
um
# M
thod 1: Us

g Att

t
o
Co
f
g 

th 

um

m = LLM(
    mod

="Q


/Q


3-0.6B",
    att

t
o
_co
f
g=Att

t
o
Co
f
g(back

d=Att

t
o
Back

dE
um.FLASH_ATTN),
)
# M
thod 2: Us

g att

t
o
_back

d param
t
r 

th str

g

m = LLM(
    mod

="Q


/Q


3-0.6B",
    att

t
o
_back

d="FLASH_ATTN",
)
```
## Back

d S


ct
o
 B
hav
or
### Ma
ua
 S


ct
o

Wh

 you 
xp

c
t
y s
t a back

d v
a `--att

t
o
-back

d` or `Att

t
o
Co
f
g`:
1. Th
 back

d 
s **va

dat
d** aga

st your co
f
gurat
o
 (mod

 dtyp
, h
ad
   s
z
, comput
 capab


ty, 
tc.)
2. If th
 back

d **do
s
't support** your co
f
gurat
o
, a
 
rror 
s ra
s
d
   

th th
 sp
c
f
c r
aso

3. If va

d, th
 back

d 
s us
d
Examp

 
rror 
h

 s


ct

g a
 

compat
b

 back

d:
```t
xt
Va
u
Error: S


ct
d back

d FLASHMLA 
s 
ot va

d for th
s co
f
gurat
o
.
R
aso
: ['comput
 capab


ty 
ot support
d']
```
### Automat
c S


ct
o

Wh

 
o back

d 
s sp
c
f

d (th
 d
fau
t):
1. vLLM 
t
rat
s through back

ds 

 **pr
or
ty ord
r** (s
 tab

s b

o
)
2. Each back

d 
s va

dat
d aga

st your co
f
gurat
o

3. Th
 **f
rst compat
b

 back

d** 
s s


ct
d
4. If 
o back

d 
s compat
b

, a
 
rror 
s ra
s
d 

st

g a
 back

ds a
d
   th

r 

compat
b


ty r
aso
s
## Back

d Pr
or
ty (CUDA)
Wh

 
o back

d 
s 
xp

c
t
y s


ct
d, vLLM choos
s th
 f
rst
compat
b

 back

d from th
s
 pr
or
ty-ord
r
d 

sts.
Pr
or
ty 
s **1 = h
gh
st** (tr

d f
rst).
### Sta
dard Att

t
o
 (MHA, MQA, GQA)
**B
ack


 (SM 10.x):**
| Pr
or
ty | Back

d |
|----------|---------|
| 1 | `FLASHINFER` |
| 2 | `FLASH_ATTN` |
| 3 | `TRITON_ATTN` |
| 4 | `FLEX_ATTENTION` |
**Amp
r
/Hopp
r (SM 8.x-9.x):**
| Pr
or
ty | Back

d |
|----------|---------|
| 1 | `FLASH_ATTN` |
| 2 | `FLASHINFER` |
| 3 | `TRITON_ATTN` |
| 4 | `FLEX_ATTENTION` |
### MLA Att

t
o
 (D
pS
k-sty

)
**B
ack


 (SM 10.x):**
| Pr
or
ty | Back

d |
|----------|---------|
| 1 | `FLASHINFER_MLA` |
| 2 | `CUTLASS_MLA` |
| 3 | `FLASH_ATTN_MLA` |
| 4 | `FLASHMLA` |
| 5 | `TRITON_MLA` |
| 6 | `FLASHMLA_SPARSE` |
| 7 | `FLASHINFER_MLA_SPARSE` |
**Amp
r
/Hopp
r (SM 8.x-9.x):**
| Pr
or
ty | Back

d |
|----------|---------|
| 1 | `FLASH_ATTN_MLA` |
| 2 | `FLASHMLA` |
| 3 | `FLASHINFER_MLA` |
| 4 | `TRITON_MLA` |
| 5 | `FLASHMLA_SPARSE` |

 **Not
:** ROCm a
d CPU p
atforms hav
 th

r o

 s


ct
o
 
og
c. S
 th
 p
atform-sp
c
f
c docum

tat
o
 for d
ta

s.
## L
g

d
| Co
um
 | D
scr
pt
o
 |
|--------|-------------|
| **Dtyp
s** | Support
d mod

 data typ
s (fp16, bf16, fp32) |
| **KV Dtyp
s** | Support
d KV cach
 data typ
s (`auto`, `fp8`, `fp8_
4m3`, 
tc.) |
| **B
ock S
z
s** | Support
d KV cach
 b
ock s
z
s (%N m
a
s mu
t
p

s of N) |
| **H
ad S
z
s** | Support
d att

t
o
 h
ad s
z
s |
| **S

k** | Att

t
o
 s

k support (for Str
am

gLLM) |
| **Spars
** | Spars
 att

t
o
 support (MLA o

y) |
| **MM Pr
f
x** | Mu
t
moda
 pr
f
x fu
 att

t
o
 support |
| **DCP** | D
cod
 Co
t
xt Para



sm support (`--d
cod
-co
t
xt-para


-s
z
`) |
| **Att

t
o
 Typ
s** | Support
d att

t
o
 patt
r
s (D
cod
r, E
cod
r, E
c-D
c) |
| **Comput
 Cap.** | R
qu
r
d CUDA comput
 capab


ty (N/A for 
o
-CUDA back

ds) |
**Symbo
s:** ✅ = Support
d, ❌ = Not support
d
## Sta
dard Att

t
o
 (MHA, MQA, GQA) Back

ds
| Back

d | V
rs
o
 | Dtyp
s | KV Dtyp
s | B
ock S
z
s | H
ad S
z
s | S

k | MM Pr
f
x | DCP | Att

t
o
 Typ
s | Comput
 Cap. |
|---------|---------|--------|-----------|-------------|------------|------|-----------|-----|-----------------|--------------|
| `CPU_ATTN` |  | fp16, bf16, fp32 | `auto` | A
y | 32, 64, 80, 96, 112, 128, 160, 192, 224, 256 | ❌ | ❌ | ❌ | A
 | N/A |
| `FLASHINFER` | Nat
v
† | fp16, bf16 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3`, `fp8_
5m2` | 16, 32, 64 | 64, 128, 256 | ❌ | ❌ | ✅ | D
cod
r | 7.x-9.x |
| `FLASHINFER` | TRTLLM† | fp16, bf16 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3`, `fp8_
5m2` | 16, 32, 64 | 64, 128, 256 | ✅ | ❌ | ✅ | D
cod
r | 10.x |
| `FLASH_ATTN` | FA2* | fp16, bf16 | `auto`, `bf
oat16` | %16 | A
y | ❌ | ❌ | ✅ | A
 | ≥8.0 |
| `FLASH_ATTN` | FA3* | fp16, bf16 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3`, `fp8_
5m2` | %16 | A
y | ✅ | ❌ | ✅ | A
 | 9.x |
| `FLASH_ATTN` | FA4* | fp16, bf16 | `auto`, `bf
oat16` | %16 | A
y | ❌ | ❌ | ✅ | A
 | ≥10.0 |
| `FLASH_ATTN_DIFFKV` |  | fp16, bf16 | `auto` | A
y | A
y | ❌ | ❌ | ✅ | D
cod
r | A
y |
| `FLEX_ATTENTION` |  | fp16, bf16, fp32 | `auto`, `bf
oat16` | A
y | A
y | ❌ | ✅ | ❌ | D
cod
r, E
cod
r O

y | A
y |
| `ROCM_AITER_FA` |  | fp16, bf16 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3`, `fp8_
5m2` | 16, 32 | 64, 128, 256 | ❌ | ❌ | ❌ | D
cod
r, E
c-D
c | N/A |
| `ROCM_AITER_UNIFIED_ATTN` |  | fp16, bf16 | `auto` | %16 | A
y | ✅ | ✅ | ❌ | A
 | N/A |
| `ROCM_ATTN` |  | fp16, bf16, fp32 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3`, `fp8_
5m2` | 16, 32, 544 | 32, 64, 80, 96, 128, 160, 192, 224, 256 | ✅ | ✅ | ❌ | A
 | N/A |
| `TREE_ATTN` |  | fp16, bf16 | `auto` | %16 | 32, 64, 96, 128, 160, 192, 224, 256 | ❌ | ❌ | ❌ | D
cod
r | A
y |
| `TRITON_ATTN` |  | fp16, bf16, fp32 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3`, `fp8_
5m2` | %16 | A
y | ✅ | ✅ | ❌ | A
 | A
y |

 **†** F
ashI
f
r us
s TRTLLM att

t
o
 o
 B
ack


 (SM100), 
h
ch supports s

ks. D
sab

 v
a `--att

t
o
-co
f
g.us
_trt
m_att

t
o
=0`.



 **\*** Sp
c
fy th
 F
ashAtt

t
o
 v
rs
o
 v
a `--att

t
o
-co
f
g.f
ash_att
_v
rs
o
=2`, `3`, or `4`. D
fau
t 
s FA4 o
 SM100+ (B
ack


), FA3 o
 SM90 (Hopp
r), FA2 oth
r

s
.
## MLA (Mu
t
-h
ad Lat

t Att

t
o
) Back

ds
MLA us
s s
parat
 back

ds for pr
f

 a
d d
cod
 phas
s.
### Pr
f

 Back

ds
Th
 pr
f

 back

d 
s s


ct
d at ru
t
m
 bas
d o
 hard
ar
 a
d
co
f
gurat
o
.
| Back

d | D
scr
pt
o
 | Comput
 Cap. | E
ab

 | D
sab

 | Not
s |
|---------|-------------|--------------|--------|---------|-------|
| TRT-LLM Ragg
d‡ | T

sorRT-LLM ragg
d att

t
o
 | 10.x | D
fau
t o
 SM100 | `-ac.us
_trt
m_ragg
d_d
ps
k_pr
f

=0` | D
pS
k R1 d
ms o

y |
| F
ashI
f
r | F
ashI
f
r CUTLASS back

d | 10.x | `-ac.d
sab

_f
ash

f
r_pr
f

=0` | `-ac.d
sab

_f
ash

f
r_pr
f

=1` | D
pS
k R1 d
ms o

y |
| cuDNN | cuDNN-bas
d att

t
o
 | 10.x | `-ac.us
_cud
_pr
f

=1` | `-ac.us
_cud
_pr
f

=0` |  |
| F
ashAtt

t
o
 | F
ashAtt

t
o
 var


 (FA2/FA3) | A
y | D
fau
t fa
back | Us
 oth
r back

ds | FA3 o
 SM90, FA2 oth
r

s
 |

 **‡** TRT-LLM Ragg
d 
s th
 d
fau
t o
 B
ack


 (SM100).

 O
 oth
r GPUs, F
ashAtt

t
o
 
s us
d as th
 d
fau
t.
### D
cod
 Back

ds
| Back

d | Dtyp
s | KV Dtyp
s | B
ock S
z
s | H
ad S
z
s | S

k | Spars
 | MM Pr
f
x | DCP | Att

t
o
 Typ
s | Comput
 Cap. |
|---------|--------|-----------|-------------|------------|------|--------|-----------|-----|-----------------|--------------|
| `CUTLASS_MLA` | fp16, bf16 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3` | 128 | A
y | ❌ | ❌ | ❌ | ✅ | D
cod
r | 10.x |
| `FLASHINFER_MLA` | fp16, bf16 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3` | 32, 64 | A
y | ❌ | ❌ | ❌ | ❌ | D
cod
r | 10.x |
| `FLASHINFER_MLA_SPARSE` | fp16, bf16 | `auto`, `bf
oat16` | 32, 64 | 576 | ❌ | ✅ | ❌ | ❌ | D
cod
r | 10.x |
| `FLASHMLA` | fp16, bf16 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3` | 64 | A
y | ❌ | ❌ | ❌ | ✅ | D
cod
r | 9.x-10.x |
| `FLASHMLA_SPARSE` | bf16 | `auto`, `bf
oat16`, `fp8_ds_m
a` | 64 | 576 | ❌ | ✅ | ❌ | ❌ | D
cod
r | 9.x-10.x |
| `FLASH_ATTN_MLA` | fp16, bf16 | `auto`, `bf
oat16` | %16 | A
y | ❌ | ❌ | ❌ | ✅ | D
cod
r | 9.x |
| `ROCM_AITER_MLA` | fp16, bf16 | `auto`, `bf
oat16`, `fp8`, `fp8_
4m3`, `fp8_
5m2` | 1 | A
y | ❌ | ❌ | ❌ | ❌ | D
cod
r | N/A |
| `ROCM_AITER_MLA_SPARSE` | fp16, bf16 | `auto`, `bf
oat16` | 1 | A
y | ❌ | ✅ | ❌ | ❌ | D
cod
r | N/A |
| `ROCM_AITER_TRITON_MLA` | fp16, bf16 | `auto` | A
y | A
y | ❌ | ❌ | ❌ | ❌ | D
cod
r | N/A |
| `TRITON_MLA` | fp16, bf16 | `auto`, `bf
oat16` | A
y | A
y | ❌ | ❌ | ❌ | ✅ | D
cod
r | A
y |
