# NVIDIA Mod

 Opt
m
z
r
Th
 [NVIDIA Mod

 Opt
m
z
r](https://g
thub.com/NVIDIA/Mod

-Opt
m
z
r) 
s a 

brary d
s
g

d to opt
m
z
 mod

s for 

f
r

c
 

th NVIDIA GPUs. It 

c
ud
s too
s for Post-Tra



g Qua
t
zat
o
 (PTQ) a
d Qua
t
zat
o
 A
ar
 Tra



g (QAT) of Larg
 La
guag
 Mod

s (LLMs), V
s
o
 La
guag
 Mod

s (VLMs), a
d d
ffus
o
 mod

s.
W
 r
comm

d 

sta


g th
 

brary 

th:
```bash
p
p 

sta
 
v
d
a-mod

opt
```
## Support
d Mod

Opt ch
ckpo

t formats
vLLM d
t
cts Mod

Opt ch
ckpo

ts v
a `hf_qua
t_co
f
g.jso
` a
d supports th

fo
o


g `qua
t
zat
o
.qua
t_a
go` va
u
s:
- `FP8`: p
r-t

sor 


ght sca

 (+ opt
o
a
 stat
c act
vat
o
 sca

).
- `FP8_PER_CHANNEL_PER_TOKEN`: p
r-cha


 


ght sca

 a
d dy
am
c p
r-tok

 act
vat
o
 qua
t
zat
o
.
- `FP8_PB_WO` (Mod

Opt may 
m
t `fp8_pb_
o`): b
ock-sca

d FP8 


ght-o

y (typ
ca
y 128×128 b
ocks).
- `NVFP4`: Mod

Opt NVFP4 ch
ckpo

ts (us
 `qua
t
zat
o
="mod

opt_fp4"`).
- `MXFP8`: Mod

Opt MXFP8 ch
ckpo

ts (us
 `qua
t
zat
o
="mod

opt_mxfp8"`).
## Qua
t
z

g Hugg

gFac
 Mod

s 

th PTQ
You ca
 qua
t
z
 Hugg

gFac
 mod

s us

g th
 
xamp

 scr
pts prov
d
d 

 th
 Mod

 Opt
m
z
r r
pos
tory. Th
 pr
mary scr
pt for LLM PTQ 
s typ
ca
y fou
d 

th

 th
 `
xamp

s/
m_ptq` d
r
ctory.
B

o
 
s a
 
xamp

 sho


g ho
 to qua
t
z
 a mod

 us

g mod

opt's PTQ API:
??? cod

    ```pytho

    
mport mod

opt.torch.qua
t
zat
o
 as mtq
    from tra
sform
rs 
mport AutoMod

ForCausa
LM
    # Load th
 mod

 from Hugg

gFac

    mod

 = AutoMod

ForCausa
LM.from_pr
tra


d("
path_or_mod

_
d
")
    # S


ct th
 qua
t
zat
o
 co
f
g, for 
xamp

, FP8
    co
f
g = mtq.FP8_DEFAULT_CFG
    # D
f


 a for
ard 
oop fu
ct
o
 for ca

brat
o

    d
f for
ard_
oop(mod

):
        for data 

 ca

b_s
t:
            mod

(data)
    # PTQ 

th 

-p
ac
 r
p
ac
m

t of qua
t
z
d modu

s
    mod

 = mtq.qua
t
z
(mod

, co
f
g, for
ard_
oop)
    ```
Aft
r th
 mod

 
s qua
t
z
d, you ca
 
xport 
t to a qua
t
z
d ch
ckpo

t us

g th
 
xport API:
```pytho


mport torch
from mod

opt.torch.
xport 
mport 
xport_hf_ch
ckpo

t


th torch.

f
r

c
_mod
():
    
xport_hf_ch
ckpo

t(
        mod

,  # Th
 qua
t
z
d mod

.
        
xport_d
r,  # Th
 d
r
ctory 
h
r
 th
 
xport
d f


s 


 b
 stor
d.
    )
```
Th
 qua
t
z
d ch
ckpo

t ca
 th

 b
 d
p
oy
d 

th vLLM. As a
 
xamp

, th
 fo
o


g cod
 sho
s ho
 to d
p
oy `
v
d
a/L
ama-3.1-8B-I
struct-FP8`, 
h
ch 
s th
 FP8 qua
t
z
d ch
ckpo

t d
r
v
d from `m
ta-
ama/L
ama-3.1-8B-I
struct`, us

g vLLM:
??? cod

    ```pytho

    from v
m 
mport LLM, Samp


gParams
    d
f ma

():
        mod

_
d = "
v
d
a/L
ama-3.1-8B-I
struct-FP8"
        # E
sur
 you sp
c
fy qua
t
zat
o
="mod

opt" 
h

 
oad

g th
 mod

opt ch
ckpo

t
        
m = LLM(mod

=mod

_
d, qua
t
zat
o
="mod

opt", trust_r
mot
_cod
=Tru
)
        samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.9)
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
        outputs = 
m.g


rat
(prompts, samp


g_params)
        for output 

 outputs:
            prompt = output.prompt
            g


rat
d_t
xt = output.outputs[0].t
xt
            pr

t(f"Prompt: {prompt!r}, G


rat
d t
xt: {g


rat
d_t
xt!r}")
    
f __
am
__ == "__ma

__":
        ma

()
    ```
## Ru


g th
 Op

AI-compat
b

 s
rv
r
To s
rv
 a 
oca
 Mod

Opt ch
ckpo

t v
a th
 Op

AI-compat
b

 API:
```bash
v
m s
rv
 
path_to_
xport
d_ch
ckpo

t
 \
  --qua
t
zat
o
 mod

opt \
  --host 0.0.0.0 --port 8000
```
## T
st

g (
oca
 ch
ckpo

ts)
vLLM's Mod

Opt u

t t
sts ar
 gat
d by 
oca
 ch
ckpo

t paths a
d ar
 sk
pp
d
by d
fau
t 

 CI. To ru
 th
 t
sts 
oca
y:
```bash

xport VLLM_TEST_MODELOPT_FP8_PC_PT_MODEL_PATH=
path_to_fp8_pc_pt_ch
ckpo

t


xport VLLM_TEST_MODELOPT_FP8_PB_WO_MODEL_PATH=
path_to_fp8_pb_
o_ch
ckpo

t

pyt
st -q t
sts/qua
t
zat
o
/t
st_mod

opt.py
```
