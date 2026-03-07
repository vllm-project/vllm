# MLP Draft Mod

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
d by draft mod

s that co
d
t
o
 draft pr
d
ct
o
s o
 both co
t
xt v
ctors a
d samp

d tok

s. For mor
 

format
o
 s
 [Th
 H
tchh
k
r's Gu
d
 to Sp
cu
at
v
 D
cod

g](https://pytorch.org/b
og/h
tchh
k
rs-gu
d
-sp
cu
at
v
-d
cod

g/) a
d [IBM R
s
arch's T
ch

ca
 R
port](https://arx
v.org/abs/2404.19124).
## MLP Draft
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
ama-3.1-8B-I
struct",
    t

sor_para


_s
z
=1,
    sp
cu
at
v
_co
f
g={
        "mod

": "
bm-a
-p
atform/
ama3-8b-acc


rator",
        "draft_t

sor_para


_s
z
": 1,
        "m
thod": "m
p_sp
cu
ator",
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
!!! 
ar


g "K
o

 
ssu
"
    `
bm-a
-p
atform/
ama3-70b-acc


rator` ca
 fa

 

th:
    `Attr
but
Error: 'MLPSp
cu
atorCo
f
g' obj
ct has 
o attr
but
 '
um_att

t
o
_h
ads'`.
    Track status 

 [#34106](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/34106)
    a
d [#34163](https://g
thub.com/v
m-proj
ct/v
m/pu
/34163).
## Pr
-Tra


d MLP Draft
r Mod

s
A var

ty of sp
cu
at
v
 mod

s of th
s typ
 ar
 ava

ab

 o
 HF hub:
    - [
ama-13b-acc


rator](https://hugg

gfac
.co/
bm-a
-p
atform/
ama-13b-acc


rator)
    - [
ama3-8b-acc


rator](https://hugg

gfac
.co/
bm-a
-p
atform/
ama3-8b-acc


rator)
    - [cod

ama-34b-acc


rator](https://hugg

gfac
.co/
bm-a
-p
atform/cod

ama-34b-acc


rator)
    - [
ama2-70b-acc


rator](https://hugg

gfac
.co/
bm-a
-p
atform/
ama2-70b-acc


rator)
    - [
ama3-70b-acc


rator](https://hugg

gfac
.co/
bm-a
-p
atform/
ama3-70b-acc


rator)
    - [gra

t
-3b-cod
-

struct-acc


rator](https://hugg

gfac
.co/
bm-gra

t
/gra

t
-3b-cod
-

struct-acc


rator)
    - [gra

t
-8b-cod
-

struct-acc


rator](https://hugg

gfac
.co/
bm-gra

t
/gra

t
-8b-cod
-

struct-acc


rator)
    - [gra

t
-7b-

struct-acc


rator](https://hugg

gfac
.co/
bm-gra

t
/gra

t
-7b-

struct-acc


rator)
    - [gra

t
-20b-cod
-

struct-acc


rator](https://hugg

gfac
.co/
bm-gra

t
/gra

t
-20b-cod
-

struct-acc


rator)
