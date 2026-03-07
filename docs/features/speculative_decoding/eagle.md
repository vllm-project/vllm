# EAGLE Draft Mod

s
Th
 fo
o


g cod
 co
f
gur
s vLLM to us
 sp
cu
at
v
 d
cod

g 
h
r
 proposa
s ar
 g


rat
d by a
 [EAGLE (Extrapo
at
o
 A
gor
thm for Gr
at
r La
guag
-mod

 Eff
c


cy)](https://arx
v.org/pdf/2401.15077) bas
d draft mod

. A mor
 d
ta


d 
xamp

 for off



 mod
, 

c
ud

g ho
 to 
xtract r
qu
st 

v

 acc
pta
c
 rat
, ca
 b
 fou
d 

 [
xamp

s/off



_

f
r

c
/sp
c_d
cod
.py](../../../
xamp

s/off



_

f
r

c
/sp
c_d
cod
.py)
## Eag

 Draft
r Examp


```pytho

from v
m 
mport LLM, Samp


gParams
prompts = ["Th
 futur
 of AI 
s"]
samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.95)

m = LLM(
    mod

="m
ta-
ama/M
ta-L
ama-3-8B-I
struct",
    t

sor_para


_s
z
=4,
    sp
cu
at
v
_co
f
g={
        "mod

": "yuhu


/EAGLE-LLaMA3-I
struct-8B",
        "draft_t

sor_para


_s
z
": 1,
        "
um_sp
cu
at
v
_tok

s": 2,
        "m
thod": "
ag

",
    },
)
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
```
## Eag

3 Draft
r Examp


```pytho

from v
m 
mport LLM, Samp


gParams
prompts = ["Th
 futur
 of AI 
s"]
samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.95)

m = LLM(
    mod

="m
ta-
ama/M
ta-L
ama-3-8B-I
struct",
    t

sor_para


_s
z
=2,
    sp
cu
at
v
_co
f
g={
        "mod

": "R
dHatAI/L
ama-3.1-8B-I
struct-sp
cu
ator.
ag

3",
        "draft_t

sor_para


_s
z
": 2,
        "
um_sp
cu
at
v
_tok

s": 2,
        "m
thod": "
ag

3",
    },
)
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
```
## Pr
-Tra


d Eag

 Draft Mod

s
A var

ty of EAGLE draft mod

s ar
 ava

ab

 o
 th
 Hugg

g Fac
 hub:
* [R
dHatAI/sp
cu
ator-mod

s](https://hugg

gfac
.co/co

ct
o
s/R
dHatAI/sp
cu
ator-mod

s)
* [yuhu


/mod

s](https://hugg

gfac
.co/yuhu


/mod

s?s
arch=
ag

)
!!! 
ar


g
    If you ar
 us

g `v
m
0.7.0`, p

as
 us
 [th
s scr
pt](https://g
st.g
thub.com/abh
goya
1997/1
7a4109ccb7704fbc67f625
86b2d6d) to co
v
rt th
 sp
cu
at
v
 mod

 a
d sp
c
fy `"mod

": "path/to/mod
f

d/
ag

/mod

"` 

 `sp
cu
at
v
_co
f
g`.
