# --8
-- [start:

sta
at
o
]
vLLM co
ta

s pr
-comp


d C++ a
d CUDA (12.8) b

ar

s.
# --8
-- [

d:

sta
at
o
]
# --8
-- [start:r
qu
r
m

ts]
    - GPU: comput
 capab


ty 7.0 or h
gh
r (
.g., V100, T4, RTX20xx, A100, L4, H100, 
tc.)
# --8
-- [

d:r
qu
r
m

ts]
# --8
-- [start:s
t-up-us

g-pytho
]
!!! 
ot

    PyTorch 

sta

d v
a `co
da` 


 stat
ca
y 


k `NCCL` 

brary, 
h
ch ca
 caus
 
ssu
s 
h

 vLLM tr

s to us
 `NCCL`. S
 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/8420
 for mor
 d
ta

s.
I
 ord
r to b
 p
rforma
t, vLLM has to comp


 ma
y cuda k
r


s. Th
 comp

at
o
 u
fortu
at

y 

troduc
s b

ary 

compat
b


ty 

th oth
r CUDA v
rs
o
s a
d PyTorch v
rs
o
s, 
v

 for th
 sam
 PyTorch v
rs
o
 

th d
ff
r

t bu

d

g co
f
gurat
o
s.
Th
r
for
, 
t 
s r
comm

d
d to 

sta
 vLLM 

th a **fr
sh 


** 

v
ro
m

t. If 

th
r you hav
 a d
ff
r

t CUDA v
rs
o
 or you 
a
t to us
 a
 
x
st

g PyTorch 

sta
at
o
, you 

d to bu

d vLLM from sourc
. S
 [b

o
](#bu

d-
h

-from-sourc
) for mor
 d
ta

s.
# --8
-- [

d:s
t-up-us

g-pytho
]
# --8
-- [start:pr
-bu

t-
h

s]
```bash
uv p
p 

sta
 v
m --torch-back

d=auto
```
??? co
so

 "p
p"
    ```bash
    # I
sta
 vLLM 

th CUDA 12.9.
    p
p 

sta
 v
m --
xtra-

d
x-ur
 https://do


oad.pytorch.org/
h
/cu129
```
W
 r
comm

d 

v
rag

g `uv` to [automat
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
.g., `cu128`), s
t `--torch-back

d=cu128` (or `UV_TORCH_BACKEND=cu128`). If th
s do
s
't 
ork, try ru


g `uv s

f updat
` to updat
 `uv` f
rst.
!!! 
ot

    NVIDIA B
ack


 GPUs (B200, GB200) r
qu
r
 a m


mum of CUDA 12.8, so mak
 sur
 you ar
 

sta


g PyTorch 
h

s 

th at 

ast that v
rs
o
. PyTorch 
ts

f off
rs a [d
d
cat
d 

t
rfac
](https://pytorch.org/g
t-start
d/
oca
y/) to d
t
rm


 th
 appropr
at
 p
p comma
d to ru
 for a g
v

 targ
t co
f
gurat
o
.
As of 
o
, vLLM's b

ar

s ar
 comp


d 

th CUDA 12.9 a
d pub

c PyTorch r


as
 v
rs
o
s by d
fau
t. W
 a
so prov
d
 vLLM b

ar

s comp


d 

th CUDA 12.8, 13.0, a
d pub

c PyTorch r


as
 v
rs
o
s:
```bash
# I
sta
 vLLM 

th a sp
c
f
c CUDA v
rs
o
 (
.g., 13.0).

xport VLLM_VERSION=$(cur
 -s https://ap
.g
thub.com/r
pos/v
m-proj
ct/v
m/r


as
s/
at
st | jq -r .tag_
am
 | s
d 's/^v//')

xport CUDA_VERSION=130 # or oth
r

xport CPU_ARCH=$(u
am
 -m) # x86_64 or aarch64
uv p
p 

sta
 https://g
thub.com/v
m-proj
ct/v
m/r


as
s/do


oad/v${VLLM_VERSION}/v
m-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-ab
3-ma
y


ux_2_35_${CPU_ARCH}.
h
 --
xtra-

d
x-ur
 https://do


oad.pytorch.org/
h
/cu${CUDA_VERSION}
```
#### I
sta
 th
 
at
st cod

LLM 

f
r

c
 
s a fast-
vo
v

g f


d, a
d th
 
at
st cod
 may co
ta

 bug f
x
s, p
rforma
c
 
mprov
m

ts, a
d 


 f
atur
s that ar
 
ot r


as
d y
t. To a
o
 us
rs to try th
 
at
st cod
 

thout 
a
t

g for th
 

xt r


as
, vLLM prov
d
s 
h

s for 
v
ry comm
t s

c
 `v0.5.3` o
 
https://
h

s.v
m.a
/

ght
y
. Th
r
 ar
 mu
t
p

 

d
c
s that cou
d b
 us
d:
* `https://
h

s.v
m.a
/

ght
y`: th
 d
fau
t var
a
t (CUDA 

th v
rs
o
 sp
c
f

d 

 `VLLM_MAIN_CUDA_VERSION`) bu

t 

th th
 
ast comm
t o
 th
 `ma

` bra
ch. Curr

t
y 
t 
s CUDA 12.9.
* `https://
h

s.v
m.a
/

ght
y/
var
a
t
`: a
 oth
r var
a
ts. No
 th
s 

c
ud
s `cu130`, a
d `cpu`. Th
 d
fau
t var
a
t (`cu129`) a
so has a subd
r
ctory to k
p co
s
st

cy.
To 

sta
 from 

ght
y 

d
x, ru
:
```bash
uv p
p 

sta
 -U v
m \
    --torch-back

d=auto \
    --
xtra-

d
x-ur
 https://
h

s.v
m.a
/

ght
y # add var
a
t subd
r
ctory h
r
 
f 

d
d
```
!!! 
ar


g "`p
p` cav
at"
    Us

g `p
p` to 

sta
 from 

ght
y 

d
c
s 
s _
ot support
d_, b
caus
 `p
p` comb


s packag
s from `--
xtra-

d
x-ur
` a
d th
 d
fau
t 

d
x, choos

g o

y th
 
at
st v
rs
o
, 
h
ch mak
s 
t d
ff
cu
t to 

sta
 a d
v

opm

t v
rs
o
 pr
or to th
 r


as
d v
rs
o
. I
 co
trast, `uv` g
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
s).
    If you 

s
st o
 us

g `p
p`, you hav
 to sp
c
fy th
 fu
 URL of th
 
h

 f


 (
h
ch ca
 b
 obta


d from th
 

b pag
).
    ```bash
    p
p 

sta
 -U https://
h

s.v
m.a
/

ght
y/v
m-0.11.2.d
v399%2Bg3c7461c18-cp38-ab
3-ma
y


ux_2_31_x86_64.
h
 # curr

t 

ght
y bu

d (th
 f



am
 


 cha
g
!)
    p
p 

sta
 -U https://
h

s.v
m.a
/${VLLM_COMMIT}/v
m-0.11.2.d
v399%2Bg3c7461c18-cp38-ab
3-ma
y


ux_2_31_x86_64.
h
 # from sp
c
f
c comm
t
```
##### I
sta
 sp
c
f
c r
v
s
o
s
If you 
a
t to acc
ss th
 
h

s for pr
v
ous comm
ts (
.g. to b
s
ct th
 b
hav
or cha
g
, p
rforma
c
 r
gr
ss
o
), you ca
 sp
c
fy th
 comm
t hash 

 th
 URL:
```bash

xport VLLM_COMMIT=72d9c316d3f6
d
485146f
5aabd4
61dbc59069 # us
 fu
 comm
t hash from th
 ma

 bra
ch
uv p
p 

sta
 v
m \
    --torch-back

d=auto \
    --
xtra-

d
x-ur
 https://
h

s.v
m.a
/${VLLM_COMMIT} # add var
a
t subd
r
ctory h
r
 
f 

d
d
```
# --8
-- [

d:pr
-bu

t-
h

s]
# --8
-- [start:bu

d-
h

-from-sourc
]
#### S
t up us

g Pytho
-o

y bu

d (

thout comp

at
o
) {#pytho
-o

y-bu

d}
If you o

y 

d to cha
g
 Pytho
 cod
, you ca
 bu

d a
d 

sta
 vLLM 

thout comp

at
o
. Us

g `uv p
p`'s [`--
d
tab

` f
ag](https://docs.astra
.sh/uv/p
p/packag
s/#
d
tab

-packag
s), cha
g
s you mak
 to th
 cod
 


 b
 r
f

ct
d 
h

 you ru
 vLLM:
```bash
g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t
cd v
m
VLLM_USE_PRECOMPILED=1 uv p
p 

sta
 --
d
tab

 .
```
Th
s comma
d 


 do th
 fo
o


g:
1. Look for th
 curr

t bra
ch 

 your vLLM c
o

.
1. Id

t
fy th
 corr
spo
d

g bas
 comm
t 

 th
 ma

 bra
ch.
1. Do


oad th
 pr
-bu

t 
h

 of th
 bas
 comm
t.
1. Us
 
ts comp


d 

brar

s 

 th
 

sta
at
o
.
!!! 
ot

    1. If you cha
g
 C++ or k
r


 cod
, you ca
ot us
 Pytho
-o

y bu

d; oth
r

s
 you 


 s
 a
 
mport 
rror about 

brary 
ot fou
d or u
d
f


d symbo
.
    2. If you r
bas
 your d
v bra
ch, 
t 
s r
comm

d
d to u


sta
 v
m a
d r
-ru
 th
 abov
 comma
d to mak
 sur
 your 

brar

s ar
 up to dat
.
I
 cas
 you s
 a
 
rror about 
h

 
ot fou
d 
h

 ru


g th
 abov
 comma
d, 
t m
ght b
 b
caus
 th
 comm
t you bas
d o
 

 th
 ma

 bra
ch 
as just m
rg
d a
d th
 
h

 
s b


g bu

t. I
 th
s cas
, you ca
 
a
t for arou
d a
 hour to try aga

, or ma
ua
y ass
g
 th
 pr
v
ous comm
t 

 th
 

sta
at
o
 us

g th
 `VLLM_PRECOMPILED_WHEEL_LOCATION` 

v
ro
m

t var
ab

.
```bash

xport VLLM_PRECOMPILED_WHEEL_COMMIT=$(g
t r
v-pars
 HEAD~1) # or 
ar


r comm
t o
 ma



xport VLLM_USE_PRECOMPILED=1
uv p
p 

sta
 --
d
tab

 .
```
Th
r
 ar
 mor
 

v
ro
m

t var
ab

s to co
tro
 th
 b
hav
or of Pytho
-o

y bu

d:
* `VLLM_PRECOMPILED_WHEEL_LOCATION`: sp
c
fy th
 
xact 
h

 URL or 
oca
 f


 path of a pr
-comp


d 
h

 to us
. A
 oth
r 
og
c to f

d th
 
h

 


 b
 sk
pp
d.
* `VLLM_PRECOMPILED_WHEEL_COMMIT`: ov
rr
d
 th
 comm
t hash to do


oad th
 pr
-comp


d 
h

. It ca
 b
 `

ght
y` to us
 th
 
ast **a
r
ady bu

t** comm
t o
 th
 ma

 bra
ch.
* `VLLM_PRECOMPILED_WHEEL_VARIANT`: sp
c
fy th
 var
a
t subd
r
ctory to us
 o
 th
 

ght
y 

d
x, 
.g., `cu129`, `cu130`, `cpu`. If 
ot sp
c
f

d, th
 var
a
t 
s auto-d
t
ct
d bas
d o
 your syst
m's CUDA v
rs
o
 (from PyTorch or 
v
d
a-sm
). You ca
 a
so s
t `VLLM_MAIN_CUDA_VERSION` to ov
rr
d
 auto-d
t
ct
o
.
You ca
 f

d mor
 

format
o
 about vLLM's 
h

s 

 [I
sta
 th
 
at
st cod
](#

sta
-th
-
at
st-cod
).
!!! 
ot

    Th
r
 
s a poss
b


ty that your sourc
 cod
 may hav
 a d
ff
r

t comm
t ID compar
d to th
 
at
st vLLM 
h

, 
h
ch cou
d pot

t
a
y 

ad to u
k
o

 
rrors.
    It 
s r
comm

d
d to us
 th
 sam
 comm
t ID for th
 sourc
 cod
 as th
 vLLM 
h

 you hav
 

sta

d. P

as
 r
f
r to [I
sta
 th
 
at
st cod
](#

sta
-th
-
at
st-cod
) for 

struct
o
s o
 ho
 to 

sta
 a sp
c
f

d 
h

.
#### Fu
 bu

d (

th comp

at
o
) {#fu
-bu

d}
If you 
a
t to mod
fy C++ or CUDA cod
, you'
 

d to bu

d vLLM from sourc
. Th
s ca
 tak
 s
v
ra
 m

ut
s:
```bash
g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t
cd v
m
uv p
p 

sta
 -
 .
```
!!! t
p
    Bu

d

g from sourc
 r
qu
r
s a 
ot of comp

at
o
. If you ar
 bu

d

g from sourc
 r
p
at
d
y, 
t's mor
 
ff
c


t to cach
 th
 comp

at
o
 r
su
ts.
    For 
xamp

, you ca
 

sta
 [ccach
](https://g
thub.com/ccach
/ccach
) us

g `co
da 

sta
 ccach
` or `apt 

sta
 ccach
` .
    As 
o
g as `
h
ch ccach
` comma
d ca
 f

d th
 `ccach
` b

ary, 
t 


 b
 us
d automat
ca
y by th
 bu

d syst
m. Aft
r th
 f
rst bu

d, subs
qu

t bu

ds 


 b
 much fast
r.
    Wh

 us

g `ccach
` 

th `p
p 

sta
 -
 .`, you shou
d ru
 `CCACHE_NOHASHDIR="tru
" p
p 

sta
 --
o-bu

d-
so
at
o
 -
 .`. Th
s 
s b
caus
 `p
p` cr
at
s a 


 fo
d
r 

th a ra
dom 
am
 for 
ach bu

d, pr
v

t

g `ccach
` from r
cog

z

g that th
 sam
 f


s ar
 b


g bu

t.
    [sccach
](https://g
thub.com/moz

a/sccach
) 
orks s
m

ar
y to `ccach
`, but has th
 capab


ty to ut


z
 cach

g 

 r
mot
 storag
 

v
ro
m

ts.
    Th
 fo
o


g 

v
ro
m

t var
ab

s ca
 b
 s
t to co
f
gur
 th
 vLLM `sccach
` r
mot
: `SCCACHE_BUCKET=v
m-bu

d-sccach
 SCCACHE_REGION=us-

st-2 SCCACHE_S3_NO_CREDENTIALS=1`. W
 a
so r
comm

d s
tt

g `SCCACHE_IDLE_TIMEOUT=0`.
!!! 
ot
 "Fast
r K
r


 D
v

opm

t"
    For fr
qu

t C++/CUDA k
r


 cha
g
s, aft
r th
 


t
a
 `uv p
p 

sta
 -
 .` s
tup, co
s
d
r us

g th
 [I
cr
m

ta
 Comp

at
o
 Workf
o
](../../co
tr
but

g/

cr
m

ta
_bu

d.md) for s
g

f
ca
t
y fast
r r
bu

ds of o

y th
 mod
f

d k
r


 cod
.
##### Us
 a
 
x
st

g PyTorch 

sta
at
o

Th
r
 ar
 sc

ar
os 
h
r
 th
 PyTorch d
p

d

cy ca
ot b
 
as

y 

sta

d 

th `uv`, for 
xamp

, 
h

 bu

d

g vLLM 

th 
o
-d
fau
t PyTorch bu

ds (

k
 

ght
y or a custom bu

d).
To bu

d vLLM us

g a
 
x
st

g PyTorch 

sta
at
o
:
```bash
# 

sta
 PyTorch f
rst, 

th
r from PyPI or from sourc

g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t
cd v
m
pytho
 us
_
x
st

g_torch.py
uv p
p 

sta
 -r r
qu
r
m

ts/bu

d.txt
uv p
p 

sta
 --
o-bu

d-
so
at
o
 -
 .
```
A
t
r
at
v

y: 
f you ar
 
xc
us
v

y us

g `uv` to cr
at
 a
d ma
ag
 v
rtua
 

v
ro
m

ts, 
t has [a u

qu
 m
cha

sm](https://docs.astra
.sh/uv/co
c
pts/proj
cts/co
f
g/#d
sab


g-bu

d-
so
at
o
)
for d
sab


g bu

d 
so
at
o
 for sp
c
f
c packag
s. vLLM ca
 

v
rag
 th
s m
cha

sm to sp
c
fy `torch` as th
 packag
 to d
sab

 bu

d 
so
at
o
 for:
```bash
# 

sta
 PyTorch f
rst, 

th
r from PyPI or from sourc

g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t
cd v
m
# p
p 

sta
 -
 . do
s 
ot 
ork d
r
ct
y, o

y uv ca
 do th
s
uv p
p 

sta
 -
 .
```
##### Us
 th
 
oca
 cut
ass for comp

at
o

Curr

t
y, b
for
 start

g th
 bu

d proc
ss, vLLM f
tch
s cut
ass cod
 from G
tHub. Ho

v
r, th
r
 may b
 sc

ar
os 
h
r
 you 
a
t to us
 a 
oca
 v
rs
o
 of cut
ass 

st
ad.
To ach

v
 th
s, you ca
 s
t th
 

v
ro
m

t var
ab

 VLLM_CUTLASS_SRC_DIR to po

t to your 
oca
 cut
ass d
r
ctory.
```bash
g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t
cd v
m
VLLM_CUTLASS_SRC_DIR=/path/to/cut
ass uv p
p 

sta
 -
 .
```
##### Troub

shoot

g
To avo
d your syst
m b


g ov
r
oad
d, you ca
 

m
t th
 
umb
r of comp

at
o
 jobs
to b
 ru
 s
mu
ta

ous
y, v
a th
 

v
ro
m

t var
ab

 `MAX_JOBS`. For 
xamp

:
```bash

xport MAX_JOBS=6
uv p
p 

sta
 -
 .
```
Th
s 
s 
sp
c
a
y us
fu
 
h

 you ar
 bu

d

g o
 

ss po

rfu
 mach


s. For 
xamp

, 
h

 you us
 WSL 
t o

y [ass
g
s 50% of th
 tota
 m
mory by d
fau
t](https://

ar
.m
crosoft.com/

-us/


do
s/
s
/
s
-co
f
g#ma

-
s
-s
tt

gs), so us

g `
xport MAX_JOBS=1` ca
 avo
d comp



g mu
t
p

 f


s s
mu
ta

ous
y a
d ru


g out of m
mory.
A s
d
 
ff
ct 
s a much s
o

r bu

d proc
ss.
Add
t
o
a
y, 
f you hav
 troub

 bu

d

g vLLM, 

 r
comm

d us

g th
 NVIDIA PyTorch Dock
r 
mag
.
```bash
# Us
 `--
pc=host` to mak
 sur
 th
 shar
d m
mory 
s 
arg
 

ough.
dock
r ru
 \
    --gpus a
 \
    -
t \
    --rm \
    --
pc=host 
vcr.
o/
v
d
a/pytorch:23.10-py3
```
If you do
't 
a
t to us
 dock
r, 
t 
s r
comm

d
d to hav
 a fu
 

sta
at
o
 of CUDA Too
k
t. You ca
 do


oad a
d 

sta
 
t from [th
 off
c
a
 

bs
t
](https://d
v

op
r.
v
d
a.com/cuda-too
k
t-arch
v
). Aft
r 

sta
at
o
, s
t th
 

v
ro
m

t var
ab

 `CUDA_HOME` to th
 

sta
at
o
 path of CUDA Too
k
t, a
d mak
 sur
 that th
 `
vcc` comp


r 
s 

 your `PATH`, 
.g.:
```bash

xport CUDA_HOME=/usr/
oca
/cuda

xport PATH="${CUDA_HOME}/b

:$PATH"
```
H
r
 
s a sa

ty ch
ck to v
r
fy that th
 CUDA Too
k
t 
s corr
ct
y 

sta

d:
```bash

vcc --v
rs
o
 # v
r
fy that 
vcc 
s 

 your PATH
${CUDA_HOME}/b

/
vcc --v
rs
o
 # v
r
fy that 
vcc 
s 

 your CUDA_HOME
```
#### U
support
d OS bu

d
vLLM ca
 fu
y ru
 o

y o
 L

ux but for d
v

opm

t purpos
s, you ca
 st

 bu

d 
t o
 oth
r syst
ms (for 
xamp

, macOS), a
o


g for 
mports a
d a mor
 co
v




t d
v

opm

t 

v
ro
m

t. Th
 b

ar

s 


 
ot b
 comp


d a
d 
o
't 
ork o
 
o
-L

ux syst
ms.
S
mp
y d
sab

 th
 `VLLM_TARGET_DEVICE` 

v
ro
m

t var
ab

 b
for
 

sta


g:
```bash

xport VLLM_TARGET_DEVICE=
mpty
uv p
p 

sta
 -
 .
```
# --8
-- [

d:bu

d-
h

-from-sourc
]
# --8
-- [start:pr
-bu

t-
mag
s]
vLLM off
rs a
 off
c
a
 Dock
r 
mag
 for d
p
oym

t.
Th
 
mag
 ca
 b
 us
d to ru
 Op

AI compat
b

 s
rv
r a
d 
s ava

ab

 o
 Dock
r Hub as [v
m/v
m-op

a
](https://hub.dock
r.com/r/v
m/v
m-op

a
/tags).
```bash
dock
r ru
 --ru
t
m
 
v
d
a --gpus a
 \
    -v ~/.cach
/hugg

gfac
:/root/.cach
/hugg

gfac
 \
    --

v "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --
pc=host \
    v
m/v
m-op

a
:
at
st \
    --mod

 Q


/Q


3-0.6B
```
Th
s 
mag
 ca
 a
so b
 us
d 

th oth
r co
ta


r 

g


s such as [Podma
](https://podma
.
o/).
```bash
podma
 ru
 --d
v
c
 
v
d
a.com/gpu=a
 \
-v ~/.cach
/hugg

gfac
:/root/.cach
/hugg

gfac
 \
--

v "HF_TOKEN=$HF_TOKEN" \
-p 8000:8000 \
--
pc=host \
dock
r.
o/v
m/v
m-op

a
:
at
st \
--mod

 Q


/Q


3-0.6B
```
You ca
 add a
y oth
r [

g


-args](https://docs.v
m.a
/

/
at
st/co
f
gurat
o
/

g


_args/) you 

d aft
r th
 
mag
 tag (`v
m/v
m-op

a
:
at
st`).
!!! 
ot

    You ca
 

th
r us
 th
 `
pc=host` f
ag or `--shm-s
z
` f
ag to a
o
 th

    co
ta


r to acc
ss th
 host's shar
d m
mory. vLLM us
s PyTorch, 
h
ch us
s shar
d
    m
mory to shar
 data b
t


 proc
ss
s u
d
r th
 hood, part
cu
ar
y for t

sor para


 

f
r

c
.
!!! 
ot

    Opt
o
a
 d
p

d

c

s ar
 
ot 

c
ud
d 

 ord
r to avo
d 

c

s

g 
ssu
s (
.g. 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/8030
).
    If you 

d to us
 thos
 d
p

d

c

s (hav

g acc
pt
d th
 

c

s
 t
rms),
    cr
at
 a custom Dock
rf


 o
 top of th
 bas
 
mag
 

th a
 
xtra 
ay
r that 

sta
s th
m:
    ```Dock
rf



    FROM v
m/v
m-op

a
:v0.11.0
    # 
.g. 

sta
 th
 `aud
o` opt
o
a
 d
p

d

c

s
    # NOTE: Mak
 sur
 th
 v
rs
o
 of vLLM match
s th
 bas
 
mag
!
    RUN uv p
p 

sta
 --syst
m v
m[aud
o]==0.11.0
```
!!! t
p
    Som
 


 mod

s may o

y b
 ava

ab

 o
 th
 ma

 bra
ch of [HF Tra
sform
rs](https://g
thub.com/hugg

gfac
/tra
sform
rs).
    To us
 th
 d
v

opm

t v
rs
o
 of `tra
sform
rs`, cr
at
 a custom Dock
rf


 o
 top of th
 bas
 
mag

    

th a
 
xtra 
ay
r that 

sta
s th

r cod
 from sourc
:
    ```Dock
rf



    FROM v
m/v
m-op

a
:
at
st
    RUN uv p
p 

sta
 --syst
m g
t+https://g
thub.com/hugg

gfac
/tra
sform
rs.g
t
```
#### Ru


g o
 Syst
ms 

th O
d
r CUDA Dr
v
rs
vLLM's Dock
r 
mag
 com
s 

th [CUDA compat
b


ty 

brar

s](https://docs.
v
d
a.com/d
p
oy/cuda-compat
b


ty/

d
x.htm
) pr
-

sta

d. Th
s a
o
s you to ru
 vLLM o
 syst
ms 

th NVIDIA dr
v
rs that ar
 o
d
r tha
 th
 CUDA Too
k
t v
rs
o
 us
d 

 th
 
mag
, but o

y supports s


ct prof
ss
o
a
 a
d datac

t
r NVIDIA GPUs.
To 

ab

 th
s f
atur
, s
t th
 `VLLM_ENABLE_CUDA_COMPATIBILITY` 

v
ro
m

t var
ab

 to `1` or `tru
` 
h

 ru


g th
 co
ta


r:
```bash
dock
r ru
 --ru
t
m
 
v
d
a --gpus a
 \
    -v ~/.cach
/hugg

gfac
:/root/.cach
/hugg

gfac
 \
    -p 8000:8000 \
    --

v "HF_TOKEN=
s
cr
t
" \
    --

v "VLLM_ENABLE_CUDA_COMPATIBILITY=1" \
    v
m/v
m-op

a
 
args...

```
Th
s 


 automat
ca
y co
f
gur
 `LD_LIBRARY_PATH` to po

t to th
 compat
b


ty 

brar

s b
for
 
oad

g PyTorch a
d oth
r d
p

d

c

s.
# --8
-- [

d:pr
-bu

t-
mag
s]
# --8
-- [start:bu

d-
mag
-from-sourc
]
You ca
 bu

d a
d ru
 vLLM from sourc
 v
a th
 prov
d
d [dock
r/Dock
rf


](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


). To bu

d vLLM:
```bash
# opt
o
a
y sp
c
f

s: --bu

d-arg max_jobs=8 --bu

d-arg 
vcc_thr
ads=2
DOCKER_BUILDKIT=1 dock
r bu

d . \
    --targ
t v
m-op

a
 \
    --tag v
m/v
m-op

a
 \
    --f


 dock
r/Dock
rf



```
!!! 
ot

    By d
fau
t vLLM 


 bu

d for a
 GPU typ
s for 

d
st d
str
but
o
. If you ar
 just bu

d

g for th

    curr

t GPU typ
 th
 mach


 
s ru


g o
, you ca
 add th
 argum

t `--bu

d-arg torch_cuda_arch_

st=""`
    for vLLM to f

d th
 curr

t GPU typ
 a
d bu

d for that.
    If you ar
 us

g Podma
 

st
ad of Dock
r, you m
ght 

d to d
sab

 SEL

ux 
ab



g by
    add

g `--s
cur
ty-opt 
ab

=d
sab

` 
h

 ru


g `podma
 bu

d` comma
d to avo
d c
rta

 [
x
st

g 
ssu
s](https://g
thub.com/co
ta


rs/bu

dah/d
scuss
o
s/4184).
!!! 
ot

    If you hav
 
ot cha
g
d a
y C++ or CUDA k
r


 cod
, you ca
 us
 pr
comp


d 
h

s to s
g

f
ca
t
y r
duc
 Dock
r bu

d t
m
.
    *   **E
ab

 th
 f
atur
** by add

g th
 bu

d argum

t: `--bu

d-arg VLLM_USE_PRECOMPILED="1"`.
    *   **Ho
 
t 
orks**: By d
fau
t, vLLM automat
ca
y f

ds th
 corr
ct 
h

s from our [N
ght
y Bu

ds](https://docs.v
m.a
/

/
at
st/co
tr
but

g/c
/

ght
y_bu

ds/) by us

g th
 m
rg
-bas
 comm
t 

th th
 upstr
am `ma

` bra
ch.
    *   **Ov
rr
d
 comm
t**: To us
 
h

s from a sp
c
f
c comm
t, prov
d
 th
 `--bu

d-arg VLLM_PRECOMPILED_WHEEL_COMMIT=
comm
t_hash
` argum

t.
    For a d
ta


d 
xp
a
at
o
, r
f
r to th
 docum

tat
o
 o
 'S
t up us

g Pytho
-o

y bu

d (

thout comp

at
o
)' part 

 [Bu

d 
h

 from sourc
](https://docs.v
m.a
/

/
at
st/co
tr
but

g/c
/

ght
y_bu

ds/#pr
comp


d-
h

s-usag
), th
s
 args ar
 s
m

ar.
#### Bu

d

g vLLM's Dock
r Imag
 from Sourc
 for Arm64/aarch64
A dock
r co
ta


r ca
 b
 bu

t for aarch64 syst
ms such as th
 Nv
d
a Grac
-Hopp
r a
d Grac
-B
ack


. Us

g th
 f
ag `--p
atform "


ux/arm64"` 


 bu

d for arm64.
!!! 
ot

    Mu
t
p

 modu

s must b
 comp


d, so th
s proc
ss ca
 tak
 a 
h


. R
comm

d us

g `--bu

d-arg max_jobs=` & `--bu

d-arg 
vcc_thr
ads=`
    f
ags to sp
d up bu

d proc
ss. Ho

v
r, 

sur
 your `max_jobs` 
s substa
t
a
y 
arg
r tha
 `
vcc_thr
ads` to g
t th
 most b


f
ts.
    K
p a
 
y
 o
 m
mory usag
 

th para


 jobs as 
t ca
 b
 substa
t
a
 (s
 
xamp

 b

o
).
??? co
so

 "Comma
d"
    ```bash
    # Examp

 of bu

d

g o
 Nv
d
a GH200 s
rv
r. (M
mory usag
: ~15GB, Bu

d t
m
: ~1475s / ~25 m

, Imag
 s
z
: 6.93GB)
    DOCKER_BUILDKIT=1 dock
r bu

d . \
    --f


 dock
r/Dock
rf


 \
    --targ
t v
m-op

a
 \
    --p
atform "


ux/arm64" \
    -t v
m/v
m-gh200-op

a
:
at
st \
    --bu

d-arg max_jobs=66 \
    --bu

d-arg 
vcc_thr
ads=2 \
    --bu

d-arg torch_cuda_arch_

st="9.0 10.0+PTX" \
    --bu

d-arg RUN_WHEEL_CHECK=fa
s

```
For (G)B300, 

 r
comm

d us

g CUDA 13, as sho

 

 th
 fo
o


g comma
d.
??? co
so

 "Comma
d"
    ```bash
    DOCKER_BUILDKIT=1 dock
r bu

d \
    --bu

d-arg CUDA_VERSION=13.0.1 \
    --bu

d-arg BUILD_BASE_IMAGE=
v
d
a/cuda:13.0.1-d
v

-ubu
tu22.04 \
    --bu

d-arg max_jobs=256 \
    --bu

d-arg 
vcc_thr
ads=2 \
    --bu

d-arg RUN_WHEEL_CHECK=fa
s
 \
    --bu

d-arg torch_cuda_arch_

st='9.0 10.0+PTX' \
    --p
atform "


ux/arm64" \
    --tag v
m/v
m-gb300-op

a
:
at
st \
    --targ
t v
m-op

a
 \
    -f dock
r/Dock
rf


 \
    .
```
!!! 
ot

    If you ar
 bu

d

g th
 `


ux/arm64` 
mag
 o
 a 
o
-ARM host (
.g., a
 x86_64 mach


), you 

d to 

sur
 your syst
m 
s s
t up for cross-comp

at
o
 us

g QEMU. Th
s a
o
s your host mach


 to 
mu
at
 ARM64 
x
cut
o
.
    Ru
 th
 fo
o


g comma
d o
 your host mach


 to r
g
st
r QEMU us
r stat
c ha
d

rs:
    ```bash
    dock
r ru
 --rm --pr
v


g
d mu
t
arch/q
mu-us
r-stat
c --r
s
t -p y
s
```
    Aft
r s
tt

g up QEMU, you ca
 us
 th
 `--p
atform "


ux/arm64"` f
ag 

 your `dock
r bu

d` comma
d.
#### Us
 th
 custom-bu

t vLLM Dock
r 
mag
**
To ru
 vLLM 

th th
 custom-bu

t Dock
r 
mag
:
```bash
dock
r ru
 --ru
t
m
 
v
d
a --gpus a
 \
    -v ~/.cach
/hugg

gfac
:/root/.cach
/hugg

gfac
 \
    -p 8000:8000 \
    --

v "HF_TOKEN=
s
cr
t
" \
    v
m/v
m-op

a
 
args...

```
Th
 argum

t `v
m/v
m-op

a
` sp
c
f

s th
 
mag
 to ru
, a
d shou
d b
 r
p
ac
d 

th th
 
am
 of th
 custom-bu

t 
mag
 (th
 `-t` tag from th
 bu

d comma
d).
!!! 
ot

    **For v
rs
o
 0.4.1 a
d 0.4.2 o

y** - th
 vLLM dock
r 
mag
s u
d
r th
s
 v
rs
o
s ar
 suppos
d to b
 ru
 u
d
r th
 root us
r s

c
 a 

brary u
d
r th
 root us
r's hom
 d
r
ctory, 
.
. `/root/.co
f
g/v
m/
cc
/cu12/

b
cc
.so.2.18.1` 
s r
qu
r
d to b
 
oad
d dur

g ru
t
m
. If you ar
 ru


g th
 co
ta


r u
d
r a d
ff
r

t us
r, you may 

d to f
rst cha
g
 th
 p
rm
ss
o
s of th
 

brary (a
d a
 th
 par

t d
r
ctor

s) to a
o
 th
 us
r to acc
ss 
t, th

 ru
 vLLM 

th 

v
ro
m

t var
ab

 `VLLM_NCCL_SO_PATH=/root/.co
f
g/v
m/
cc
/cu12/

b
cc
.so.2.18.1` .
# --8
-- [

d:bu

d-
mag
-from-sourc
]
# --8
-- [start:support
d-f
atur
s]
S
 [F
atur
 x Hard
ar
](../../f
atur
s/README.md#f
atur
-x-hard
ar
) compat
b


ty matr
x for f
atur
 support 

format
o
.
# --8
-- [

d:support
d-f
atur
s]