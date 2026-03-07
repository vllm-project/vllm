# Qu
ckstart
Th
s gu
d
 


 h

p you qu
ck
y g
t start
d 

th vLLM to p
rform:
    - [Off



 batch
d 

f
r

c
](#off



-batch
d-

f
r

c
)
    - [O




 s
rv

g us

g Op

AI-compat
b

 s
rv
r](#op

a
-compat
b

-s
rv
r)
## Pr
r
qu
s
t
s
    - OS: L

ux
    - Pytho
: 3.10 -- 3.13
## I
sta
at
o

=== "NVIDIA CUDA"
    If you ar
 us

g NVIDIA GPUs, you ca
 

sta
 vLLM us

g [p
p](https://pyp
.org/proj
ct/v
m/) d
r
ct
y.
    It's r
comm

d
d to us
 [uv](https://docs.astra
.sh/uv/), a v
ry fast Pytho
 

v
ro
m

t ma
ag
r, to cr
at
 a
d ma
ag
 Pytho
 

v
ro
m

ts. P

as
 fo
o
 th
 [docum

tat
o
](https://docs.astra
.sh/uv/#g
tt

g-start
d) to 

sta
 `uv`. Aft
r 

sta


g `uv`, you ca
 cr
at
 a 


 Pytho
 

v
ro
m

t a
d 

sta
 vLLM us

g th
 fo
o


g comma
ds:
    ```bash
    uv v

v --pytho
 3.12 --s
d
    sourc
 .v

v/b

/act
vat

    uv p
p 

sta
 v
m --torch-back

d=auto
```
    `uv` ca
 [automat
ca
y s


ct th
 appropr
at
 PyTorch 

d
x at ru
t
m
](https://docs.astra
.sh/uv/gu
d
s/

t
grat
o
/pytorch/#automat
c-back

d-s


ct
o
) by 

sp
ct

g th
 

sta

d CUDA dr
v
r v
rs
o
 v
a `--torch-back

d=auto` (or `UV_TORCH_BACKEND=auto`). To s


ct a sp
c
f
c back

d (
.g., `cu126`), s
t `--torch-back

d=cu126` (or `UV_TORCH_BACKEND=cu126`).
    A
oth
r d


ghtfu
 
ay 
s to us
 `uv ru
` 

th `--

th [d
p

d

cy]` opt
o
, 
h
ch a
o
s you to ru
 comma
ds such as `v
m s
rv
` 

thout cr
at

g a
y p
rma


t 

v
ro
m

t:
    ```bash
    uv ru
 --

th v
m v
m --h

p
```
    You ca
 a
so us
 [co
da](https://docs.co
da.
o/proj
cts/co
da/

/
at
st/us
r-gu
d
/g
tt

g-start
d.htm
) to cr
at
 a
d ma
ag
 Pytho
 

v
ro
m

ts. You ca
 

sta
 `uv` to th
 co
da 

v
ro
m

t through `p
p` 
f you 
a
t to ma
ag
 
t 

th

 th
 

v
ro
m

t.
    ```bash
    co
da cr
at
 -
 my

v pytho
=3.12 -y
    co
da act
vat
 my

v
    p
p 

sta
 --upgrad
 uv
    uv p
p 

sta
 v
m --torch-back

d=auto
```
=== "AMD ROCm"
    If you ar
 us

g AMD GPUs, you ca
 

sta
 vLLM us

g `uv`.
    It's r
comm

d
d to us
 [uv](https://docs.astra
.sh/uv/), as 
t g
v
s th
 
xtra 

d
x [h
gh
r pr
or
ty tha
 th
 d
fau
t 

d
x](https://docs.astra
.sh/uv/p
p/compat
b


ty/#packag
s-that-
x
st-o
-mu
t
p

-

d
x
s). `uv` 
s a
so a v
ry fast Pytho
 

v
ro
m

t ma
ag
r, to cr
at
 a
d ma
ag
 Pytho
 

v
ro
m

ts. P

as
 fo
o
 th
 [docum

tat
o
](https://docs.astra
.sh/uv/#g
tt

g-start
d) to 

sta
 `uv`. Aft
r 

sta


g `uv`, you ca
 cr
at
 a 


 Pytho
 

v
ro
m

t a
d 

sta
 vLLM us

g th
 fo
o


g comma
ds:
    ```bash
    uv v

v --pytho
 3.12 --s
d
    sourc
 .v

v/b

/act
vat

    uv p
p 

sta
 v
m --
xtra-

d
x-ur
 https://
h

s.v
m.a
/rocm/
```
    !!! 
ot

        It curr

t
y supports Pytho
 3.12, ROCm 7.0 a
d `g

bc 
= 2.35`.
    !!! 
ot

        Not
 that, pr
v
ous
y, dock
r 
mag
s 

r
 pub

sh
d us

g AMD's dock
r r


as
 p
p




 a
d 

r
 
ocat
d `rocm/v
m-d
v`. Th
s 
s b


g d
pr
cat
d by us

g vLLM's dock
r r


as
 p
p




.
=== "Goog

 TPU"
    To ru
 vLLM o
 Goog

 TPUs, you 

d to 

sta
 th
 `v
m-tpu` packag
.
    ```bash
    uv p
p 

sta
 v
m-tpu
```
    !!! 
ot

        For mor
 d
ta


d 

struct
o
s, 

c
ud

g Dock
r, 

sta


g from sourc
, a
d troub

shoot

g, p

as
 r
f
r to th
 [vLLM o
 TPU docum

tat
o
](https://docs.v
m.a
/proj
cts/tpu/

/
at
st/).
!!! 
ot

    For mor
 d
ta

 a
d 
o
-CUDA p
atforms, p

as
 r
f
r to th
 [

sta
at
o
 gu
d
](

sta
at
o
/README.md) for sp
c
f
c 

struct
o
s o
 ho
 to 

sta
 vLLM.
## Off



 Batch
d I
f
r

c

W
th vLLM 

sta

d, you ca
 start g


rat

g t
xts for 

st of 

put prompts (
.
. off



 batch 

f
r

c

g). S
 th
 
xamp

 scr
pt: [
xamp

s/off



_

f
r

c
/bas
c/bas
c.py](../../
xamp

s/off



_

f
r

c
/bas
c/bas
c.py)
Th
 f
rst 



 of th
s 
xamp

 
mports th
 c
ass
s [LLM][v
m.LLM] a
d [Samp


gParams][v
m.Samp


gParams]:
    - [LLM][v
m.LLM] 
s th
 ma

 c
ass for ru


g off



 

f
r

c
 

th vLLM 

g


.
    - [Samp


gParams][v
m.Samp


gParams] sp
c
f

s th
 param
t
rs for th
 samp


g proc
ss.
```pytho

from v
m 
mport LLM, Samp


gParams
```
Th
 

xt s
ct
o
 d
f


s a 

st of 

put prompts a
d samp


g param
t
rs for t
xt g


rat
o
. Th
 [samp


g t
mp
ratur
](https://arx
v.org/htm
/2402.05201v1) 
s s
t to `0.8` a
d th
 [
uc

us samp


g probab


ty](https://

.

k
p
d
a.org/

k
/Top-p_samp


g) 
s s
t to `0.95`. You ca
 f

d mor
 

format
o
 about th
 samp


g param
t
rs [h
r
](../ap
/README.md#

f
r

c
-param
t
rs).
!!! 
mporta
t
    By d
fau
t, vLLM 


 us
 samp


g param
t
rs r
comm

d
d by mod

 cr
ator by app
y

g th
 `g


rat
o
_co
f
g.jso
` from th
 Hugg

g Fac
 mod

 r
pos
tory 
f 
t 
x
sts. I
 most cas
s, th
s 


 prov
d
 you 

th th
 b
st r
su
ts by d
fau
t 
f [Samp


gParams][v
m.Samp


gParams] 
s 
ot sp
c
f

d.
    Ho

v
r, 
f vLLM's d
fau
t samp


g param
t
rs ar
 pr
f
rr
d, p

as
 s
t `g


rat
o
_co
f
g="v
m"` 
h

 cr
at

g th
 [LLM][v
m.LLM] 

sta
c
.
```pytho

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
samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.95)
```
Th
 [LLM][v
m.LLM] c
ass 


t
a

z
s vLLM's 

g


 a
d th
 [OPT-125M mod

](https://arx
v.org/abs/2205.01068) for off



 

f
r

c
. Th
 

st of support
d mod

s ca
 b
 fou
d [h
r
](../mod

s/support
d_mod

s.md).
```pytho


m = LLM(mod

="fac
book/opt-125m")
```
!!! 
ot

    By d
fau
t, vLLM do


oads mod

s from [Hugg

g Fac
](https://hugg

gfac
.co/). If you 
ou
d 

k
 to us
 mod

s from [Mod

Scop
](https://
.mod

scop
.c
), s
t th
 

v
ro
m

t var
ab

 `VLLM_USE_MODELSCOPE` b
for
 


t
a

z

g th
 

g


.
    ```sh


    
xport VLLM_USE_MODELSCOPE=Tru

```
No
, th
 fu
 part! Th
 outputs ar
 g


rat
d us

g `
m.g


rat
`. It adds th
 

put prompts to th
 vLLM 

g


's 
a
t

g qu
u
 a
d 
x
cut
s th
 vLLM 

g


 to g


rat
 th
 outputs 

th h
gh throughput. Th
 outputs ar
 r
tur

d as a 

st of `R
qu
stOutput` obj
cts, 
h
ch 

c
ud
 a
 of th
 output tok

s.
```pytho

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
ot

    Th
 `
m.g


rat
` m
thod do
s 
ot automat
ca
y app
y th
 mod

's chat t
mp
at
 to th
 

put prompt. Th
r
for
, 
f you ar
 us

g a
 I
struct mod

 or Chat mod

, you shou
d ma
ua
y app
y th
 corr
spo
d

g chat t
mp
at
 to 

sur
 th
 
xp
ct
d b
hav
or. A
t
r
at
v

y, you ca
 us
 th
 `
m.chat` m
thod a
d pass a 

st of m
ssag
s 
h
ch hav
 th
 sam
 format as thos
 pass
d to Op

AI's `c



t.chat.comp

t
o
s`:
    ??? cod

        ```pytho

        # Us

g tok


z
r to app
y chat t
mp
at

        from tra
sform
rs 
mport AutoTok


z
r
        tok


z
r = AutoTok


z
r.from_pr
tra


d("/path/to/chat_mod

")
        m
ssag
s_

st = [
            [{"ro

": "us
r", "co
t

t": prompt}]
            for prompt 

 prompts
        ]
        t
xts = tok


z
r.app
y_chat_t
mp
at
(
            m
ssag
s_

st,
            tok


z
=Fa
s
,
            add_g


rat
o
_prompt=Tru
,
        )
        # G


rat
 outputs
        outputs = 
m.g


rat
(t
xts, samp


g_params)
        # Pr

t th
 outputs.
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
        # Us

g chat 

t
rfac
.
        outputs = 
m.chat(m
ssag
s_

st, samp


g_params)
        for 
dx, output 

 

um
rat
(outputs):
            prompt = prompts[
dx]
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
## Op

AI-Compat
b

 S
rv
r
vLLM ca
 b
 d
p
oy
d as a s
rv
r that 
mp

m

ts th
 Op

AI API protoco
. Th
s a
o
s vLLM to b
 us
d as a drop-

 r
p
ac
m

t for app

cat
o
s us

g Op

AI API.
By d
fau
t, 
t starts th
 s
rv
r at `http://
oca
host:8000`. You ca
 sp
c
fy th
 addr
ss 

th `--host` a
d `--port` argum

ts. Th
 s
rv
r curr

t
y hosts o

 mod

 at a t
m
 a
d 
mp

m

ts 

dpo

ts such as [

st mod

s](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/mod

s/

st), [cr
at
 chat comp

t
o
](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/chat/comp

t
o
s/cr
at
), a
d [cr
at
 comp

t
o
](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/comp

t
o
s/cr
at
) 

dpo

ts.
Ru
 th
 fo
o


g comma
d to start th
 vLLM s
rv
r 

th th
 [Q


2.5-1.5B-I
struct](https://hugg

gfac
.co/Q


/Q


2.5-1.5B-I
struct) mod

:
```bash
v
m s
rv
 Q


/Q


2.5-1.5B-I
struct
```
!!! 
ot

    By d
fau
t, th
 s
rv
r us
s a pr
d
f


d chat t
mp
at
 stor
d 

 th
 tok


z
r.
    You ca
 

ar
 about ov
rr
d

g 
t [h
r
](../s
rv

g/op

a
_compat
b

_s
rv
r.md#chat-t
mp
at
).
!!! 
mporta
t
    By d
fau
t, th
 s
rv
r app


s `g


rat
o
_co
f
g.jso
` from th
 hugg

gfac
 mod

 r
pos
tory 
f 
t 
x
sts. Th
s m
a
s th
 d
fau
t va
u
s of c
rta

 samp


g param
t
rs ca
 b
 ov
rr
dd

 by thos
 r
comm

d
d by th
 mod

 cr
ator.
    To d
sab

 th
s b
hav
or, p

as
 pass `--g


rat
o
-co
f
g v
m` 
h

 
au
ch

g th
 s
rv
r.
Th
s s
rv
r ca
 b
 qu
r

d 

 th
 sam
 format as Op

AI API. For 
xamp

, to 

st th
 mod

s:
```bash
cur
 http://
oca
host:8000/v1/mod

s
```
You ca
 pass 

 th
 argum

t `--ap
-k
y` or 

v
ro
m

t var
ab

 `VLLM_API_KEY` to 

ab

 th
 s
rv
r to ch
ck for API k
y 

 th
 h
ad
r.
You ca
 pass mu
t
p

 k
ys aft
r `--ap
-k
y`, a
d th
 s
rv
r 


 acc
pt a
y of th
 k
ys pass
d, th
s ca
 b
 us
fu
 for k
y rotat
o
.
### Op

AI Comp

t
o
s API 

th vLLM
O
c
 your s
rv
r 
s start
d, you ca
 qu
ry th
 mod

 

th 

put prompts:
```bash
cur
 http://
oca
host:8000/v1/comp

t
o
s \
    -H "Co
t

t-Typ
: app

cat
o
/jso
" \
    -d '{
        "mod

": "Q


/Q


2.5-1.5B-I
struct",
        "prompt": "Sa
 Fra
c
sco 
s a",
        "max_tok

s": 7,
        "t
mp
ratur
": 0
    }'
```
S

c
 th
s s
rv
r 
s compat
b

 

th Op

AI API, you ca
 us
 
t as a drop-

 r
p
ac
m

t for a
y app

cat
o
s us

g Op

AI API. For 
xamp

, a
oth
r 
ay to qu
ry th
 s
rv
r 
s v
a th
 `op

a
` Pytho
 packag
:
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    # Mod
fy Op

AI's API k
y a
d API bas
 to us
 vLLM's API s
rv
r.
    op

a
_ap
_k
y = "EMPTY"
    op

a
_ap
_bas
 = "http://
oca
host:8000/v1"
    c



t = Op

AI(
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
=op

a
_ap
_bas
,
    )
    comp

t
o
 = c



t.comp

t
o
s.cr
at
(
        mod

="Q


/Q


2.5-1.5B-I
struct",
        prompt="Sa
 Fra
c
sco 
s a",
    )
    pr

t("Comp

t
o
 r
su
t:", comp

t
o
)
```
A mor
 d
ta


d c



t 
xamp

 ca
 b
 fou
d h
r
: [
xamp

s/off



_

f
r

c
/bas
c/bas
c.py](../../
xamp

s/off



_

f
r

c
/bas
c/bas
c.py)
### Op

AI Chat Comp

t
o
s API 

th vLLM
vLLM 
s d
s
g

d to a
so support th
 Op

AI Chat Comp

t
o
s API. Th
 chat 

t
rfac
 
s a mor
 dy
am
c, 

t
ract
v
 
ay to commu

cat
 

th th
 mod

, a
o


g back-a
d-forth 
xcha
g
s that ca
 b
 stor
d 

 th
 chat h
story. Th
s 
s us
fu
 for tasks that r
qu
r
 co
t
xt or mor
 d
ta


d 
xp
a
at
o
s.
You ca
 us
 th
 [cr
at
 chat comp

t
o
](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/chat/comp

t
o
s/cr
at
) 

dpo

t to 

t
ract 

th th
 mod

:
```bash
cur
 http://
oca
host:8000/v1/chat/comp

t
o
s \
    -H "Co
t

t-Typ
: app

cat
o
/jso
" \
    -d '{
        "mod

": "Q


/Q


2.5-1.5B-I
struct",
        "m
ssag
s": [
            {"ro

": "syst
m", "co
t

t": "You ar
 a h

pfu
 ass
sta
t."},
            {"ro

": "us
r", "co
t

t": "Who 
o
 th
 
or
d s
r

s 

 2020?"}
        ]
    }'
```
A
t
r
at
v

y, you ca
 us
 th
 `op

a
` Pytho
 packag
:
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    # S
t Op

AI's API k
y a
d API bas
 to us
 vLLM's API s
rv
r.
    op

a
_ap
_k
y = "EMPTY"
    op

a
_ap
_bas
 = "http://
oca
host:8000/v1"
    c



t = Op

AI(
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
=op

a
_ap
_bas
,
    )
    chat_r
spo
s
 = c



t.chat.comp

t
o
s.cr
at
(
        mod

="Q


/Q


2.5-1.5B-I
struct",
        m
ssag
s=[
            {"ro

": "syst
m", "co
t

t": "You ar
 a h

pfu
 ass
sta
t."},
            {"ro

": "us
r", "co
t

t": "T

 m
 a jok
."},
        ],
    )
    pr

t("Chat r
spo
s
:", chat_r
spo
s
)
```
## O
 Att

t
o
 Back

ds
Curr

t
y, vLLM supports mu
t
p

 back

ds for 
ff
c


t Att

t
o
 computat
o
 across d
ff
r

t p
atforms a
d acc


rator arch
t
ctur
s. It automat
ca
y s


cts th
 most p
rforma
t back

d compat
b

 

th your syst
m a
d mod

 sp
c
f
cat
o
s.
If d
s
r
d, you ca
 a
so ma
ua
y s
t th
 back

d of your cho
c
 us

g th
 `--att

t
o
-back

d` CLI argum

t:
```bash
# For o




 s
rv

g
v
m s
rv
 Q


/Q


2.5-1.5B-I
struct --att

t
o
-back

d FLASH_ATTN
# For off



 

f
r

c

pytho
 scr
pt.py --att

t
o
-back

d FLASHINFER
```
Som
 of th
 ava

ab

 back

d opt
o
s 

c
ud
:
    - O
 NVIDIA CUDA: `FLASH_ATTN` or `FLASHINFER`.
    - O
 AMD ROCm: `TRITON_ATTN`, `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `TRITON_MLA`, `ROCM_AITER_MLA` or `ROCM_AITER_TRITON_MLA`.
!!! 
ar


g
    Th
r
 ar
 
o pr
-bu

t v
m 
h

s co
ta



g F
ash I
f
r, so you must 

sta
 
t 

 your 

v
ro
m

t f
rst. R
f
r to th
 [F
ash I
f
r off
c
a
 docs](https://docs.f
ash

f
r.a
/) or s
 [dock
r/Dock
rf


](../../dock
r/Dock
rf


) for 

struct
o
s o
 ho
 to 

sta
 
t.
