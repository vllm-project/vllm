# Qua
t
z
d KV Cach

## FP8 KV Cach
 Ov
rv



Eff
c


t m
mory usag
 
s cruc
a
 for 
ork

g 

th 
arg
 
a
guag
 mod

s. Qua
t
z

g th
 KV (K
y-Va
u
) cach
 to FP8 format ca
 s
g

f
ca
t
y r
duc
 
ts m
mory footpr

t. Th
s opt
m
zat
o
 

ab

s you to stor
 mor
 tok

s 

 m
mory, 

ad

g to 
mprov
d throughput a
d support for 
o
g
r co
t
xt 


do
s.

 **Not
:** Wh

 us

g th
 F
ash Att

t
o
 3 back

d 

th FP8 KV cach
, att

t
o
 op
rat
o
s ar
 a
so p
rform
d 

 th
 qua
t
z
d (FP8) doma

. I
 th
s co
f
gurat
o
, qu
r

s ar
 qua
t
z
d to FP8 

 add
t
o
 to k
ys a
d va
u
s.
### Support
d FP8 KV-Cach
 Qua
t
zat
o
 Sch
m
s
vLLM supports t
o ma

 qua
t
zat
o
 strat
g

s for th
 FP8 KV-cach
:
    - **P
r-t

sor qua
t
zat
o
:**  
  A s

g

 sca

 
s app


d for 
ach Q, K, a
d V t

sor 

d
v
dua
y. (`q/k/v_sca

 = [1]`)
    - **P
r-att

t
o
-h
ad qua
t
zat
o
:**  
  Each sca

 corr
spo
ds to a
 att

t
o
 h
ad: `q_sca

 = [
um_h
ads]`, `k/v_sca

 = [
um_kv_h
ads]`.

 **Not
:**  

 P
r-att

t
o
-h
ad qua
t
zat
o
 
s curr

t
y ava

ab

 **o

y 

th th
 F
ash Att

t
o
 back

d** a
d r
qu
r
s th
 ca

brat
o
 path
ay prov
d
d by **
m-compr
ssor**.
### Sca

 Ca

brat
o
 Approach
s
You ca
 co
f
gur
 ho
 th
 qua
t
zat
o
 sca

s ar
 comput
d 

 vLLM us

g thr
 d
ff
r

t approach
s:
1. **No ca

brat
o
 (d
fau
t sca

s):**  
   A
 qua
t
zat
o
 sca

s ar
 s
t to `1.0`.  
   _Co
f
gur
 

th:_  
   ```pytho

   kv_cach
_dtyp
="fp8"
   ca
cu
at
_kv_sca

s=Fa
s

   ```
2. **Ra
dom tok

 ca

brat
o
 (o
-th
-f
y):**  
   Sca

s ar
 automat
ca
y 
st
mat
d from a s

g

 batch of ra
dom tok

s dur

g 
armup a
d th

 f
x
d.  
   _Co
f
gur
 

th:_  
   ```pytho

   kv_cach
_dtyp
="fp8"
   ca
cu
at
_kv_sca

s=Tru

   ```
3. **[R
comm

d
d] Ca

brat
o
 

th a datas
t (v
a 
m-compr
ssor):**  
   Sca

s ar
 
st
mat
d us

g a curat
d ca

brat
o
 datas
t for max
mum accuracy.  
   Th
s r
qu
r
s th
 [
m-compr
ssor](https://g
thub.com/v
m-proj
ct/
m-compr
ssor) 

brary.  
   _S
 
xamp

 b

o
!_
#### Add
t
o
a
 `kv_cach
_dtyp
` Opt
o
s
    - `kv_cach
_dtyp
="auto"`: Us
 th
 mod

's d
fau
t data typ

    - `kv_cach
_dtyp
="fp8_
4m3"`: Support
d o
 CUDA 11.8+ a
d ROCm (AMD GPUs)
    - `kv_cach
_dtyp
="fp8_
5m2"`: Support
d o
 CUDA 11.8+
---
## Examp

s
### 1. No Ca

brat
o
 (`kv_cach
_dtyp
="fp8"`, `ca
cu
at
_kv_sca

s=Fa
s
`)
A
 qua
t
zat
o
 sca

s ar
 s
t to 1.0.
```pytho

from v
m 
mport LLM, Samp


gParams
samp


g_params = Samp


gParams(t
mp
ratur
=0.7, top_p=0.8)

m = LLM(
    mod

="m
ta-
ama/L
ama-2-7b-chat-hf",
    kv_cach
_dtyp
="fp8",
    ca
cu
at
_kv_sca

s=Fa
s
,
)
prompt = "Lo
do
 
s th
 cap
ta
 of"
out = 
m.g


rat
(prompt, samp


g_params)[0].outputs[0].t
xt
pr

t(out)
```
---
### 2. Ra
dom Tok

 Ca

brat
o
 (`kv_cach
_dtyp
="fp8"`, `ca
cu
at
_kv_sca

s=Tru
`)
Sca

s ar
 automat
ca
y 
st
mat
d from a s

g

 batch of tok

s dur

g 
armup.
```pytho

from v
m 
mport LLM, Samp


gParams
samp


g_params = Samp


gParams(t
mp
ratur
=0.7, top_p=0.8)

m = LLM(
    mod

="m
ta-
ama/L
ama-2-7b-chat-hf",
    kv_cach
_dtyp
="fp8",
    ca
cu
at
_kv_sca

s=Tru
,
)
prompt = "Lo
do
 
s th
 cap
ta
 of"
out = 
m.g


rat
(prompt, samp


g_params)[0].outputs[0].t
xt
pr

t(out)
```
---
### 3. **[R
comm

d
d] Ca

brat
o
 Us

g a Datas
t (

th `
m-compr
ssor`)**
For th
 h
gh
st-qua

ty qua
t
zat
o
, 

 r
comm

d ca

brat

g aga

st a datas
t us

g `
m-compr
ssor`. Th
s 

ab

s adva
c
d strat
g

s such as p
r-att

t
o
-h
ad qua
t
zat
o
.
#### I
sta
 th
 r
qu
r
d packag

```bash
p
p 

sta
 
mcompr
ssor
```
#### Examp

: Qua
t
z
 L
ama Att

t
o
 & KV Cach
 to FP8
```pytho

"""
Qua
t
z
 L
ama att

t
o
 + KV cach
 to FP8 (choos
 

th
r 't

sor' or 'att
_h
ad' strat
gy)
us

g 
m-compr
ssor o

-shot ca

brat
o
.
"""
from datas
ts 
mport 
oad_datas
t
from tra
sform
rs 
mport AutoMod

ForCausa
LM, AutoTok


z
r
from 
mcompr
ssor 
mport o

shot
from 
mcompr
ssor.mod
f

rs.qua
t
zat
o
 
mport Qua
t
zat
o
Mod
f

r
from compr
ss
d_t

sors.qua
t
zat
o
 
mport Qua
t
zat
o
Sch
m
, Qua
t
zat
o
Args
# -----------------------------
# Co
f
g
# -----------------------------
MODEL_ID = "m
ta-
ama/L
ama-3.1-8B-I
struct"
DATASET_ID = "Hugg

gFac
H4/u
trachat_200k"
DATASET_SPLIT = "tra

_sft"
STRATEGY = "t

sor"       # or "att
_h
ad"
NUM_CALIB_SAMPLES = 512   # Good start

g va
u

MAX_SEQ_LEN = 2048
# -----------------------------
# H

p
rs
# -----------------------------
d
f proc
ss_a
d_tok


z
(
xamp

, tok


z
r: AutoTok


z
r):
    """Co
v
rt chat m
ssag
s to tok

s."""
    t
xt = tok


z
r.app
y_chat_t
mp
at
(
xamp

["m
ssag
s"], tok


z
=Fa
s
)
    r
tur
 tok


z
r(
        t
xt,
        padd

g=Fa
s
,
        max_


gth=MAX_SEQ_LEN,
        tru
cat
o
=Tru
,
        add_sp
c
a
_tok

s=Fa
s
,
    )
d
f bu

d_r
c
p
(strat
gy: str) -
 Qua
t
zat
o
Mod
f

r:
    fp8_args = Qua
t
zat
o
Args(
um_b
ts=8, typ
="f
oat", strat
gy=strat
gy)
    r
tur
 Qua
t
zat
o
Mod
f

r(
        co
f
g_groups={
            "att

t
o
": Qua
t
zat
o
Sch
m
(
                targ
ts=["L
amaAtt

t
o
"],  # Qua
t
z
 qu
r

s: q_sca


                

put_act
vat
o
s=fp8_args,
            )
        },
        kv_cach
_sch
m
=fp8_args,           # Qua
t
z
 KV cach
: k/v_sca


    )
# -----------------------------
# Ma


# -----------------------------
d
f ma

():
    mod

 = AutoMod

ForCausa
LM.from_pr
tra


d(MODEL_ID, torch_dtyp
="auto")
    tok


z
r = AutoTok


z
r.from_pr
tra


d(MODEL_ID)
    ds = 
oad_datas
t(DATASET_ID, sp

t=f"{DATASET_SPLIT}[:{NUM_CALIB_SAMPLES}]")
    ds = ds.shuff

(s
d=42)
    ds = ds.map(
        
ambda 
x: proc
ss_a
d_tok


z
(
x, tok


z
r),
        r
mov
_co
um
s=ds.co
um
_
am
s,
    )
    r
c
p
 = bu

d_r
c
p
(STRATEGY)
    o

shot(
        mod

=mod

,
        datas
t=ds,
        r
c
p
=r
c
p
,
        max_s
q_


gth=MAX_SEQ_LEN,
        
um_ca

brat
o
_samp

s=NUM_CALIB_SAMPLES,
    )
    sav
_d
r = f"{MODEL_ID.rstr
p('/').sp

t('/')[-1]}-kvatt
-fp8-{STRATEGY}"
    mod

.sav
_pr
tra


d(sav
_d
r, sav
_compr
ss
d=Tru
)
    tok


z
r.sav
_pr
tra


d(sav
_d
r)

f __
am
__ == "__ma

__":
    ma

()
```
For mor
 d
ta


d a
d up-to-dat
 
xamp

s, s
 th
 [`
m-compr
ssor` off
c
a
 
xamp

s](https://g
thub.com/v
m-proj
ct/
m-compr
ssor/tr
/ma

/
xamp

s/qua
t
zat
o
_kv_cach
).
