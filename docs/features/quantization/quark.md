# AMD Quark
Qua
t
zat
o
 ca
 
ff
ct
v

y r
duc
 m
mory a
d ba
d

dth usag
, acc


rat
 computat
o
 a
d 
mprov

throughput 
h


 

th m


ma
 accuracy 
oss. vLLM ca
 

v
rag
 [Quark](https://quark.docs.amd.com/
at
st/),
th
 f

x
b

 a
d po

rfu
 qua
t
zat
o
 too
k
t, to produc
 p
rforma
t qua
t
z
d mod

s to ru
 o
 AMD GPUs. Quark has sp
c
a

z
d support for qua
t
z

g 
arg
 
a
guag
 mod

s 

th 


ght,
act
vat
o
 a
d kv-cach
 qua
t
zat
o
 a
d cutt

g-
dg
 qua
t
zat
o
 a
gor
thms 

k

AWQ, GPTQ, Rotat
o
 a
d SmoothQua
t.
## Quark I
sta
at
o

B
for
 qua
t
z

g mod

s, you 

d to 

sta
 Quark. Th
 
at
st r


as
 of Quark ca
 b
 

sta

d 

th p
p:
```bash
p
p 

sta
 amd-quark
```
You ca
 r
f
r to [Quark 

sta
at
o
 gu
d
](https://quark.docs.amd.com/
at
st/

sta
.htm
)
for mor
 

sta
at
o
 d
ta

s.
Add
t
o
a
y, 

sta
 `v
m` a
d `
m-
va
uat
o
-har

ss` for 
va
uat
o
:
```bash
p
p 

sta
 v
m "
m-
va
[ap
]
=0.4.11"
```
## Qua
t
zat
o
 Proc
ss
Aft
r 

sta


g Quark, 

 


 us
 a
 
xamp

 to 

ustrat
 ho
 to us
 Quark.
Th
 Quark qua
t
zat
o
 proc
ss ca
 b
 

st
d for 5 st
ps as b

o
:
1. Load th
 mod


2. Pr
par
 th
 ca

brat
o
 data
oad
r
3. S
t th
 qua
t
zat
o
 co
f
gurat
o

4. Qua
t
z
 th
 mod

 a
d 
xport
5. Eva
uat
o
 

 vLLM
### 1. Load th
 Mod


Quark us
s [Tra
sform
rs](https://hugg

gfac
.co/docs/tra
sform
rs/

/

d
x)
to f
tch mod

 a
d tok


z
r.
??? cod

    ```pytho

    from tra
sform
rs 
mport AutoTok


z
r, AutoMod

ForCausa
LM
    MODEL_ID = "m
ta-
ama/L
ama-2-70b-chat-hf"
    MAX_SEQ_LEN = 512
    mod

 = AutoMod

ForCausa
LM.from_pr
tra


d(
        MODEL_ID,
        d
v
c
_map="auto",
        dtyp
="auto",
    )
    mod

.
va
()
    tok


z
r = AutoTok


z
r.from_pr
tra


d(MODEL_ID, mod

_max_


gth=MAX_SEQ_LEN)
    tok


z
r.pad_tok

 = tok


z
r.
os_tok


    ```
### 2. Pr
par
 th
 Ca

brat
o
 Data
oad
r
Quark us
s th
 [PyTorch Data
oad
r](https://pytorch.org/tutor
a
s/b
g


r/bas
cs/data_tutor
a
.htm
)
to 
oad ca

brat
o
 data. For mor
 d
ta

s about ho
 to us
 ca

brat
o
 datas
ts 
ff
c


t
y, p

as
 r
f
r
to [Add

g Ca

brat
o
 Datas
ts](https://quark.docs.amd.com/
at
st/pytorch/ca

brat
o
_datas
ts.htm
).
??? cod

    ```pytho

    from datas
ts 
mport 
oad_datas
t
    from torch.ut

s.data 
mport DataLoad
r
    BATCH_SIZE = 1
    NUM_CALIBRATION_DATA = 512
    # Load th
 datas
t a
d g
t ca

brat
o
 data.
    datas
t = 
oad_datas
t("m
t-ha
-
ab/p


-va
-backup", sp

t="va

dat
o
")
    t
xt_data = datas
t["t
xt"][:NUM_CALIBRATION_DATA]
    tok


z
d_outputs = tok


z
r(
        t
xt_data,
        r
tur
_t

sors="pt",
        padd

g=Tru
,
        tru
cat
o
=Tru
,
        max_


gth=MAX_SEQ_LEN,
    )
    ca

b_data
oad
r = DataLoad
r(
        tok


z
d_outputs['

put_
ds'],
        batch_s
z
=BATCH_SIZE,
        drop_
ast=Tru
,
    )
    ```
### 3. S
t th
 Qua
t
zat
o
 Co
f
gurat
o

W
 

d to s
t th
 qua
t
zat
o
 co
f
gurat
o
, you ca
 ch
ck
[quark co
f
g gu
d
](https://quark.docs.amd.com/
at
st/pytorch/us
r_gu
d
_co
f
g_d
scr
pt
o
.htm
)
for furth
r d
ta

s. H
r
 

 us
 FP8 p
r-t

sor qua
t
zat
o
 o
 


ght, act
vat
o
,
kv-cach
 a
d th
 qua
t
zat
o
 a
gor
thm 
s AutoSmoothQua
t.
!!! 
ot

    Not
 th
 qua
t
zat
o
 a
gor
thm 

ds a JSON co
f
g f


 a
d th
 co
f
g f


 
s 
ocat
d 


    [Quark Pytorch 
xamp

s](https://quark.docs.amd.com/
at
st/pytorch/pytorch_
xamp

s.htm
),
    u
d
r th
 d
r
ctory `
xamp

s/torch/
a
guag
_mod



g/
m_ptq/mod

s`. For 
xamp

,
    AutoSmoothQua
t co
f
g f


 for L
ama 
s
    `
xamp

s/torch/
a
guag
_mod



g/
m_ptq/mod

s/
ama/autosmoothqua
t_co
f
g.jso
`.
??? cod

    ```pytho

    from quark.torch.qua
t
zat
o
 
mport (Co
f
g, Qua
t
zat
o
Co
f
g,
                                        FP8E4M3P
rT

sorSp
c,
                                        
oad_qua
t_a
go_co
f
g_from_f


)
    # D
f


 fp8/p
r-t

sor/stat
c sp
c.
    FP8_PER_TENSOR_SPEC = FP8E4M3P
rT

sorSp
c(
        obs
rv
r_m
thod="m

_max",
        
s_dy
am
c=Fa
s
,
    ).to_qua
t
zat
o
_sp
c()
    # D
f


 g
oba
 qua
t
zat
o
 co
f
g, 

put t

sors a
d 


ght app
y FP8_PER_TENSOR_SPEC.
    g
oba
_qua
t_co
f
g = Qua
t
zat
o
Co
f
g(
        

put_t

sors=FP8_PER_TENSOR_SPEC,
        


ght=FP8_PER_TENSOR_SPEC,
    )
    # D
f


 qua
t
zat
o
 co
f
g for kv-cach
 
ay
rs, output t

sors app
y FP8_PER_TENSOR_SPEC.
    KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
    kv_cach
_
ay
r_
am
s_for_
ama = ["*k_proj", "*v_proj"]
    kv_cach
_qua
t_co
f
g = {
        
am
: Qua
t
zat
o
Co
f
g(
            

put_t

sors=g
oba
_qua
t_co
f
g.

put_t

sors,
            


ght=g
oba
_qua
t_co
f
g.


ght,
            output_t

sors=KV_CACHE_SPEC,
        )
        for 
am
 

 kv_cach
_
ay
r_
am
s_for_
ama
    }
    
ay
r_qua
t_co
f
g = kv_cach
_qua
t_co
f
g.copy()
    # D
f


 a
gor
thm co
f
g by co
f
g f


.
    LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE = "
xamp

s/torch/
a
guag
_mod



g/
m_ptq/mod

s/
ama/autosmoothqua
t_co
f
g.jso
"
    a
go_co
f
g = 
oad_qua
t_a
go_co
f
g_from_f


(LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE)
    EXCLUDE_LAYERS = ["
m_h
ad"]
    qua
t_co
f
g = Co
f
g(
        g
oba
_qua
t_co
f
g=g
oba
_qua
t_co
f
g,
        
ay
r_qua
t_co
f
g=
ay
r_qua
t_co
f
g,
        kv_cach
_qua
t_co
f
g=kv_cach
_qua
t_co
f
g,
        
xc
ud
=EXCLUDE_LAYERS,
        a
go_co
f
g=a
go_co
f
g,
    )
    ```
### 4. Qua
t
z
 th
 Mod

 a
d Export
Th

 

 ca
 app
y th
 qua
t
zat
o
. Aft
r qua
t
z

g, 

 

d to fr
z
 th

qua
t
z
d mod

 f
rst b
for
 
xport

g. Not
 that 

 

d to 
xport mod

 

th format of
Hugg

gFac
 `saf
t

sors`, you ca
 r
f
r to
[Hugg

gFac
 format 
xport

g](https://quark.docs.amd.com/
at
st/pytorch/
xport/quark_
xport_hf.htm
)
for mor
 
xport

g format d
ta

s.
??? cod

    ```pytho

    
mport torch
    from quark.torch 
mport Mod

Qua
t
z
r, Mod

Export
r
    from quark.torch.
xport 
mport Export
rCo
f
g, Jso
Export
rCo
f
g
    # App
y qua
t
zat
o
.
    qua
t
z
r = Mod

Qua
t
z
r(qua
t_co
f
g)
    qua
t_mod

 = qua
t
z
r.qua
t
z
_mod

(mod

, ca

b_data
oad
r)
    # Fr
z
 qua
t
z
d mod

 to 
xport.
    fr
z
d_mod

 = qua
t
z
r.fr
z
(mod

)
    # D
f


 
xport co
f
g.
    LLAMA_KV_CACHE_GROUP = ["*k_proj", "*v_proj"]
    
xport_co
f
g = Export
rCo
f
g(jso
_
xport_co
f
g=Jso
Export
rCo
f
g())
    
xport_co
f
g.jso
_
xport_co
f
g.kv_cach
_group = LLAMA_KV_CACHE_GROUP
    # Mod

: L
ama-2-70b-chat-hf-
-fp8-a-fp8-kvcach
-fp8-p
rt

sor-autosmoothqua
t
    EXPORT_DIR = MODEL_ID.sp

t("/")[1] + "-
-fp8-a-fp8-kvcach
-fp8-p
rt

sor-autosmoothqua
t"
    
xport
r = Mod

Export
r(co
f
g=
xport_co
f
g, 
xport_d
r=EXPORT_DIR)
    

th torch.
o_grad():
        
xport
r.
xport_saf
t

sors_mod

(
            fr
z
d_mod

,
            qua
t_co
f
g=qua
t_co
f
g,
            tok


z
r=tok


z
r,
        )
    ```
### 5. Eva
uat
o
 

 vLLM
No
, you ca
 
oad a
d ru
 th
 Quark qua
t
z
d mod

 d
r
ct
y through th
 LLM 

trypo

t:
??? cod

    ```pytho

    from v
m 
mport LLM, Samp


gParams
    # Samp

 prompts.
    prompts = [
        "H

o, my 
am
 
s",
        "Th
 pr
s
d

t of th
 U

t
d Stat
s 
s",
        "Th
 cap
ta
 of Fra
c
 
s",
        "Th
 futur
 of AI 
s",
    ]
    # Cr
at
 a samp


g params obj
ct.
    samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.95)
    # Cr
at
 a
 LLM.
    
m = LLM(
        mod

="L
ama-2-70b-chat-hf-
-fp8-a-fp8-kvcach
-fp8-p
rt

sor-autosmoothqua
t",
        kv_cach
_dtyp
="fp8",
        qua
t
zat
o
="quark",
    )
    # G


rat
 t
xts from th
 prompts. Th
 output 
s a 

st of R
qu
stOutput obj
cts
    # that co
ta

 th
 prompt, g


rat
d t
xt, a
d oth
r 

format
o
.
    outputs = 
m.g


rat
(prompts, samp


g_params)
    # Pr

t th
 outputs.
    pr

t("\
G


rat
d Outputs:\
" + "-" * 60)
    for output 

 outputs:
        prompt = output.prompt
        g


rat
d_t
xt = output.outputs[0].t
xt
        pr

t(f"Prompt:    {prompt!r}")
        pr

t(f"Output:    {g


rat
d_t
xt!r}")
        pr

t("-" * 60)
    ```
Or, you ca
 us
 `
m_
va
` to 
va
uat
 accuracy:
```bash

m_
va
 --mod

 v
m \
  --mod

_args pr
tra


d=L
ama-2-70b-chat-hf-
-fp8-a-fp8-kvcach
-fp8-p
rt

sor-autosmoothqua
t,kv_cach
_dtyp
='fp8',qua
t
zat
o
='quark' \
  --tasks gsm8k
```
## Quark Qua
t
zat
o
 Scr
pt
I
 add
t
o
 to th
 
xamp

 of Pytho
 API abov
, Quark a
so off
rs a
[qua
t
zat
o
 scr
pt](https://quark.docs.amd.com/
at
st/pytorch/
xamp

_quark_torch_
m_ptq.htm
)
to qua
t
z
 
arg
 
a
guag
 mod

s mor
 co
v




t
y. It supports qua
t
z

g mod

s 

th var

ty
of d
ff
r

t qua
t
zat
o
 sch
m
s a
d opt
m
zat
o
 a
gor
thms. It ca
 
xport th
 qua
t
z
d mod


a
d ru
 
va
uat
o
 tasks o
 th
 f
y. W
th th
 scr
pt, th
 
xamp

 abov
 ca
 b
:
```bash
pytho
3 qua
t
z
_quark.py --mod

_d
r m
ta-
ama/L
ama-2-70b-chat-hf \
                          --output_d
r /path/to/output \
                          --qua
t_sch
m
 
_fp8_a_fp8 \
                          --kv_cach
_dtyp
 fp8 \
                          --qua
t_a
go autosmoothqua
t \
                          --
um_ca

b_data 512 \
                          --mod

_
xport hf_format \
                          --tasks gsm8k
```
## Us

g OCP MX (MXFP4, MXFP6) mod

s
vLLM supports 
oad

g MXFP4 a
d MXFP6 mod

s qua
t
z
d off



 through AMD Quark, comp

a
t 

th [Op

 Comput
 Proj
ct (OCP) sp
c
f
cat
o
](https://
.op

comput
.org/docum

ts/ocp-m
crosca


g-formats-mx-v1-0-sp
c-f

a
-pdf).
Th
 sch
m
 curr

t
y o

y supports dy
am
c qua
t
zat
o
 for act
vat
o
s.
Examp

 usag
, aft
r 

sta


g th
 
at
st AMD Quark r


as
:
```bash
v
m s
rv
 fxmarty/q


_1.5-mo
-a2.7b-mxfp4 --t

sor-para


-s
z
 1
# or, for a mod

 us

g fp6 act
vat
o
s a
d fp4 


ghts:
v
m s
rv
 fxmarty/q


1.5_mo
_a2.7b_chat_
_fp4_a_fp6_
2m3 --t

sor-para


-s
z
 1
```
A s
mu
at
o
 of th
 matr
x mu
t
p

cat
o
 
x
cut
o
 

 MXFP4/MXFP6 ca
 b
 ru
 o
 d
v
c
s that do 
ot support OCP MX op
rat
o
s 
at
v

y (
.g. AMD I
st

ct MI325, MI300 a
d MI250), d
qua
t
z

g 


ghts from FP4/FP6 to ha
f pr
c
s
o
 o
 th
 f
y, us

g a fus
d k
r


. Th
s 
s us
fu
 
.g. to 
va
uat
 FP4/FP6 mod

s us

g vLLM, or a
t
r
at
v

y to b


f
t from th
 ~2.5-4x m
mory sav

gs (compar
d to f
oat16 a
d bf
oat16).
To g


rat
 off



 mod

s qua
t
z
d us

g MXFP4 data typ
, th
 
as

st approach 
s to us
 AMD Quark's [qua
t
zat
o
 scr
pt](https://quark.docs.amd.com/
at
st/pytorch/
xamp

_quark_torch_
m_ptq.htm
), as a
 
xamp

:
```bash
pytho
 qua
t
z
_quark.py --mod

_d
r Q


/Q


1.5-MoE-A2.7B-Chat \
    --qua
t_sch
m
 
_mxfp4_a_mxfp4 \
    --output_d
r q


_1.5-mo
-a2.7b-mxfp4 \
    --sk
p_
va
uat
o
 \
    --mod

_
xport hf_format \
    --group_s
z
 32
```
Th
 curr

t 

t
grat
o
 supports [a
 comb

at
o
 of FP4, FP6_E3M2, FP6_E2M3](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/mod

_
x
cutor/
ay
rs/qua
t
zat
o
/ut

s/ocp_mx_ut

s.py) us
d for 

th
r 


ghts or act
vat
o
s.
## Us

g Quark Qua
t
z
d 
ay
r

s
 Auto M
x
d Pr
c
s
o
 (AMP) Mod

s
vLLM a
so supports 
oad

g 
ay
r

s
 m
x
d pr
c
s
o
 mod

 qua
t
z
d us

g AMD Quark. Curr

t
y, m
x
d sch
m
 of {MXFP4, FP8} 
s support
d, 
h
r
 FP8 h
r
 d

ot
s for FP8 p
r-t

sor sch
m
. Mor
 m
x
d pr
c
s
o
 sch
m
s ar
 p
a

d to b
 support
d 

 a 

ar futur
, 

c
ud

g
- U
qua
t
z
d L


ar a
d/or MoE 
ay
r(s) as a
 opt
o
 for 
ach 
ay
r, 
.
., m
x
d of {MXFP4, FP8, BF16/FP16}
- MXFP6 qua
t
zat
o
 
xt

s
o
, 
.
., {MXFP4, MXFP6, FP8, BF16/FP16}
A
though o

 ca
 max
m
z
 s
rv

g throughput us

g th
 
o

st pr
c
s
o
 support
d o
 a g
v

 d
v
c
 (
.g. MXFP4 for AMD I
st

ct MI355, FP8 for AMD I
st

ct MI300), th
s
 aggr
ss
v
 sch
m
s ca
 b
 d
tr
m

ta
 to accuracy r
cov
r

g from qua
t
zat
o
 o
 targ
t tasks. M
x
d pr
c
s
o
 a
o
s to str
k
 a ba
a
c
 b
t


 max
m
z

g accuracy a
d throughput.
Th
r
 ar
 t
o st
ps to g


rat
 a
d d
p
oy a m
x
d pr
c
s
o
 mod

 qua
t
z
d 

th AMD Quark, as sho

 b

o
.
### 1. Qua
t
z
 a mod

 us

g m
x
d pr
c
s
o
 

 AMD Quark
F
rst
y, th
 
ay
r

s
 m
x
d-pr
c
s
o
 co
f
gurat
o
 for a g
v

 LLM mod

 
s s
arch
d a
d th

 qua
t
z
d us

g AMD Quark. W
 


 prov
d
 a d
ta


d tutor
a
 

th Quark APIs 
at
r.
As 
xamp

s, 

 prov
d
 som
 r
ady-to-us
 qua
t
z
d m
x
d pr
c
s
o
 mod

 to sho
 th
 usag
 

 vLLM a
d th
 accuracy b


f
ts. Th
y ar
:
- amd/L
ama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8
- amd/M
xtra
-8x7B-I
struct-v0.1-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8
- amd/Q


3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8
### 2. 

f
r

c
 th
 qua
t
z
d m
x
d pr
c
s
o
 mod

 

 vLLM
Mod

s qua
t
z
d 

th AMD Quark us

g m
x
d pr
c
s
o
 ca
 
at
v

y b
 r

oad 

 vLLM, a
d 
.g. 
va
uat
d us

g 
m-
va
uat
o
-har

ss as fo
o
s:
```bash

m_
va
 --mod

 v
m \
    --mod

_args pr
tra


d=amd/L
ama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8,t

sor_para


_s
z
=4,dtyp
=auto,gpu_m
mory_ut


zat
o
=0.8,trust_r
mot
_cod
=Fa
s
 \
    --tasks mm
u \
    --batch_s
z
 auto
```
