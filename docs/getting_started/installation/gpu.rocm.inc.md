# --8
-- [start:

sta
at
o
]
vLLM supports AMD GPUs 

th ROCm 6.3 or abov
. Pr
-bu

t 
h

s ar
 ava

ab

 for ROCm 7.0.
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
    - GPU: MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Rad
o
 RX 7900 s
r

s (gfx1100/1101), Rad
o
 RX 9000 s
r

s (gfx1200/1201), Ryz

 AI MAX / AI 300 S
r

s (gfx1151/1150)
    - ROCm 6.3 or abov

    - MI350 r
qu
r
s ROCm 7.0 or abov

    - Ryz

 AI MAX / AI 300 S
r

s r
qu
r
s ROCm 7.0.2 or abov

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
Th
 vLLM 
h

 bu
d

s PyTorch a
d a
 r
qu
r
d d
p

d

c

s, a
d you shou
d us
 th
 

c
ud
d PyTorch for compat
b


ty. B
caus
 vLLM comp


s ma
y ROCm k
r


s to 

sur
 a va

dat
d, h
gh‑p
rforma
c
 stack, th
 r
su
t

g b

ar

s may 
ot b
 compat
b

 

th oth
r ROCm or PyTorch bu

ds.
If you 

d a d
ff
r

t ROCm v
rs
o
 or 
a
t to us
 a
 
x
st

g PyTorch 

sta
at
o
, you’
 

d to bu

d vLLM from sourc
.  S
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
To 

sta
 th
 
at
st v
rs
o
 of vLLM for Pytho
 3.12, ROCm 7.0 a
d `g

bc 
= 2.35`.
```bash
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
!!! t
p
    You ca
 f

d out about 
h
ch ROCm v
rs
o
 th
 
at
st vLLM supports by ch
ck

g th
 

d
x 

 
xtra-

d
x-ur
 [https://
h

s.v
m.a
/rocm/](https://
h

s.v
m.a
/rocm/) .
To 

sta
 a sp
c
f
c v
rs
o
 a
d ROCm var
a
t of vLLM 
h

.
```bash
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
/rocm/0.15.0/rocm700
```
!!! 
ar


g "Cav
ats for us

g `p
p`" 
    W
 r
comm

d 

v
rag

g `uv` to 

sta
 vLLM 
h

. Us

g `p
p` to 

sta
 from custom 

d
c
s 
s cumb
rsom
, b
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
 
h

 from custom 

d
x 
f 
xact v
rs
o
s of a
 packag
s ar
 sp
c
f

d 
xact
y. I
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
 
xact vLLM v
rs
o
 a
d fu
 URL of th
 
h

 path `https://
h

s.v
m.a
/rocm/
v
rs
o

/
rocm-var
a
t
` (
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
 v
m==0.15.0+rocm700 --
xtra-

d
x-ur
 https://
h

s.v
m.a
/rocm/0.15.0/rocm700
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
!!! t
p
    - If you fou
d that th
 fo
o


g 

sta
at
o
 st
p do
s 
ot 
ork for you, p

as
 r
f
r to [dock
r/Dock
rf


.rocm_bas
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm_bas
). Dock
rf


 
s a form of 

sta
at
o
 st
ps.
0. I
sta
 pr
r
qu
s
t
s (sk
p 
f you ar
 a
r
ady 

 a
 

v
ro
m

t/dock
r 

th th
 fo
o


g 

sta

d):
    - [ROCm](https://rocm.docs.amd.com/

/
at
st/d
p
oy/


ux/

d
x.htm
)
    - [PyTorch](https://pytorch.org/)
    For 

sta


g PyTorch, you ca
 start from a fr
sh dock
r 
mag
, 
.g, `rocm/pytorch:rocm7.0_ubu
tu22.04_py3.10_pytorch_r


as
_2.8.0`, `rocm/pytorch-

ght
y`. If you ar
 us

g dock
r 
mag
, you ca
 sk
p to St
p 3.
    A
t
r
at
v

y, you ca
 

sta
 PyTorch us

g PyTorch 
h

s. You ca
 ch
ck PyTorch 

sta
at
o
 gu
d
 

 PyTorch [G
tt

g Start
d](https://pytorch.org/g
t-start
d/
oca
y/). Examp

:
    ```bash
    # I
sta
 PyTorch
    p
p u


sta
 torch -y
    p
p 

sta
 --
o-cach
-d
r torch torchv
s
o
 --

d
x-ur
 https://do


oad.pytorch.org/
h
/

ght
y/rocm7.0
```
1. I
sta
 [Tr
to
 for ROCm](https://g
thub.com/ROCm/tr
to
.g
t)
    I
sta
 ROCm's Tr
to
 fo
o


g th
 

struct
o
s from [ROCm/tr
to
](https://g
thub.com/ROCm/tr
to
.g
t)
    ```bash
    pytho
3 -m p
p 

sta
 


ja cmak
 
h

 pyb

d11
    p
p u


sta
 -y tr
to

    g
t c
o

 https://g
thub.com/ROCm/tr
to
.g
t
    cd tr
to

    # g
t ch
ckout $TRITON_BRANCH
    g
t ch
ckout f9
5bf54
    
f [ ! -f s
tup.py ]; th

 cd pytho
; f

    pytho
3 s
tup.py 

sta

    cd ../..
```
    !!! 
ot

        - Th
 va

dat
d `$TRITON_BRANCH` ca
 b
 fou
d 

 th
 [dock
r/Dock
rf


.rocm_bas
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm_bas
).
        - If you s
 HTTP 
ssu
 r

at
d to do


oad

g packag
s dur

g bu

d

g tr
to
, p

as
 try aga

 as th
 HTTP 
rror 
s 

t
rm
tt

t.
2. Opt
o
a
y, 
f you choos
 to us
 CK f
ash att

t
o
, you ca
 

sta
 [f
ash att

t
o
 for ROCm](https://g
thub.com/Dao-AILab/f
ash-att

t
o
.g
t)
    I
sta
 ROCm's f
ash att

t
o
 (v2.8.0) fo
o


g th
 

struct
o
s from [ROCm/f
ash-att

t
o
](https://g
thub.com/Dao-AILab/f
ash-att

t
o
#amd-rocm-support)
    For 
xamp

, for ROCm 7.0, suppos
 your gfx arch 
s `gfx942`. To g
t your gfx arch
t
ctur
, ru
 `rocm

fo |gr
p gfx`.
    ```bash
    g
t c
o

 https://g
thub.com/Dao-AILab/f
ash-att

t
o
.g
t
    cd f
ash-att

t
o

    # g
t ch
ckout $FA_BRANCH
    g
t ch
ckout 0
60
394
    g
t submodu

 updat
 --


t
    GPU_ARCHS="gfx942" pytho
3 s
tup.py 

sta

    cd ..
```
    !!! 
ot

        - Th
 va

dat
d `$FA_BRANCH` ca
 b
 fou
d 

 th
 [dock
r/Dock
rf


.rocm_bas
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm_bas
).
3. Opt
o
a
y, 
f you choos
 to bu

d AITER yours

f to us
 a c
rta

 bra
ch or comm
t, you ca
 bu

d AITER us

g th
 fo
o


g st
ps:
    ```bash
    pytho
3 -m p
p u


sta
 -y a
t
r
    g
t c
o

 --r
curs
v
 https://g
thub.com/ROCm/a
t
r.g
t
    cd a
t
r
    g
t ch
ckout $AITER_BRANCH_OR_COMMIT
    g
t submodu

 sy
c; g
t submodu

 updat
 --


t --r
curs
v

    pytho
3 s
tup.py d
v

op
```
    !!! 
ot

        - You 


 

d to co
f
g th
 `$AITER_BRANCH_OR_COMMIT` for your purpos
.
        - Th
 va

dat
d `$AITER_BRANCH_OR_COMMIT` ca
 b
 fou
d 

 th
 [dock
r/Dock
rf


.rocm_bas
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm_bas
).
4. Opt
o
a
y, 
f you 
a
t to us
 MORI for EP or PD d
saggr
gat
o
, you ca
 

sta
 [MORI](https://g
thub.com/ROCm/mor
) us

g th
 fo
o


g st
ps:
    ```bash
    g
t c
o

 https://g
thub.com/ROCm/mor
.g
t
    cd mor

    g
t ch
ckout $MORI_BRANCH_OR_COMMIT
    g
t submodu

 sy
c; g
t submodu

 updat
 --


t --r
curs
v

    MORI_GPU_ARCHS="gfx942;gfx950" pytho
3 s
tup.py 

sta

```
    !!! 
ot

        - You 


 

d to co
f
g th
 `$MORI_BRANCH_OR_COMMIT` for your purpos
.
        - Th
 va

dat
d `$MORI_BRANCH_OR_COMMIT` ca
 b
 fou
d 

 th
 [dock
r/Dock
rf


.rocm_bas
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm_bas
).
5. Bu

d vLLM. For 
xamp

, vLLM o
 ROCM 7.0 ca
 b
 bu

t 

th th
 fo
o


g st
ps:
    ???+ co
so

 "Comma
ds"
        ```bash
        p
p 

sta
 --upgrad
 p
p
        # Bu

d & 

sta
 AMD SMI
        p
p 

sta
 /opt/rocm/shar
/amd_sm

        # I
sta
 d
p

d

c

s
        p
p 

sta
 --upgrad
 
umba \
            sc
py \
            hugg

gfac
-hub[c

,hf_tra
sf
r] \
            s
tuptoo
s_scm
        p
p 

sta
 -r r
qu
r
m

ts/rocm.txt
        # To bu

d for a s

g

 arch
t
ctur
 (
.g., MI300) for fast
r 

sta
at
o
 (r
comm

d
d):
        
xport PYTORCH_ROCM_ARCH="gfx942"
        # To bu

d vLLM for mu
t
p

 arch MI210/MI250/MI300, us
 th
s 

st
ad
        # 
xport PYTORCH_ROCM_ARCH="gfx90a;gfx942"
        pytho
3 s
tup.py d
v

op
```
    Th
s may tak
 5-10 m

ut
s. Curr

t
y, `p
p 

sta
 .` do
s 
ot 
ork for ROCm 
h

 

sta


g vLLM from sourc
.
    !!! t
p
        - Th
 ROCm v
rs
o
 of PyTorch, 
d
a
y, shou
d match th
 ROCm dr
v
r v
rs
o
.
!!! t
p
    - For MI300x (gfx942) us
rs, to ach

v
 opt
ma
 p
rforma
c
, p

as
 r
f
r to [MI300x tu


g gu
d
](https://rocm.docs.amd.com/

/
at
st/ho
-to/tu


g-gu
d
s/m
300x/

d
x.htm
) for p
rforma
c
 opt
m
zat
o
 a
d tu


g t
ps o
 syst
m a
d 
orkf
o
 

v

.
      For vLLM, p

as
 r
f
r to [vLLM p
rforma
c
 opt
m
zat
o
](https://rocm.docs.amd.com/

/
at
st/ho
-to/rocm-for-a
/

f
r

c
-opt
m
zat
o
/v
m-opt
m
zat
o
.htm
).
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
-rocm](https://hub.dock
r.com/r/v
m/v
m-op

a
-rocm/tags).
```bash
dock
r ru
 --rm \
    --group-add=v
d
o \
    --cap-add=SYS_PTRACE \
    --s
cur
ty-opt s
ccomp=u
co
f


d \
    --d
v
c
 /d
v/kfd \
    --d
v
c
 /d
v/dr
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
-rocm:
at
st \
    --mod

 Q


/Q


3-0.6B
```
#### Us
 AMD's Dock
r Imag
s
Pr
or to Ja
uary 20th, 2026 
h

 th
 off
c
a
 dock
r 
mag
s ar
 ava

ab

 o
 [upstr
am vLLM dock
r hub](https://hub.dock
r.com/v2/r
pos
tor

s/v
m/v
m-op

a
-rocm/tags/), th
 [AMD I
f


ty hub for vLLM](https://hub.dock
r.com/r/rocm/v
m/tags) off
rs a pr
bu

t, opt
m
z
d
dock
r 
mag
 d
s
g

d for va

dat

g 

f
r

c
 p
rforma
c
 o
 th
 AMD I
st

ct MI300X™ acc


rator.
AMD a
so off
rs 

ght
y pr
bu

t dock
r 
mag
 from [Dock
r Hub](https://hub.dock
r.com/r/rocm/v
m-d
v), 
h
ch has vLLM a
d a
 
ts d
p

d

c

s 

sta

d. Th
 

trypo

t of th
s dock
r 
mag
 
s `/b

/bash` (d
ff
r

t from th
 vLLM's Off
c
a
 Dock
r Imag
).
```bash
dock
r pu
 rocm/v
m-d
v:

ght
y # to g
t th
 
at
st 
mag

dock
r ru
 -
t --rm \
--

t
ork=host \
--group-add=v
d
o \
--
pc=host \
--cap-add=SYS_PTRACE \
--s
cur
ty-opt s
ccomp=u
co
f


d \
--d
v
c
 /d
v/kfd \
--d
v
c
 /d
v/dr
 \
-v 
path/to/your/mod

s
:/app/mod

s \
-
 HF_HOME="/app/mod

s" \
rocm/v
m-d
v:

ght
y
```
!!! t
p
    P

as
 ch
ck [LLM 

f
r

c
 p
rforma
c
 va

dat
o
 o
 AMD I
st

ct MI300X](https://rocm.docs.amd.com/

/
at
st/ho
-to/p
rforma
c
-va

dat
o
/m
300x/v
m-b

chmark.htm
)
    for 

struct
o
s o
 ho
 to us
 th
s pr
bu

t dock
r 
mag
.
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


.rocm](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm).
??? 

fo "(Opt
o
a
) Bu

d a
 
mag
 

th ROCm soft
ar
 stack"
    Bu

d a dock
r 
mag
 from [dock
r/Dock
rf


.rocm_bas
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm_bas
) 
h
ch s
tup ROCm soft
ar
 stack 

d
d by th
 vLLM.
    **Th
s st
p 
s opt
o
a
 as th
s rocm_bas
 
mag
 
s usua
y pr
bu

t a
d stor
 at [Dock
r Hub](https://hub.dock
r.com/r/rocm/v
m-d
v) u
d
r tag `rocm/v
m-d
v:bas
` to sp
d up us
r 
xp
r


c
.**
    If you choos
 to bu

d th
s rocm_bas
 
mag
 yours

f, th
 st
ps ar
 as fo
o
s.
    It 
s 
mporta
t that th
 us
r k
cks off th
 dock
r bu

d us

g bu

dk
t. E
th
r th
 us
r put `DOCKER_BUILDKIT=1` as 

v
ro
m

t var
ab

 
h

 ca


g dock
r bu

d comma
d, or th
 us
r 

ds to s
t up bu

dk
t 

 th
 dock
r da
mo
 co
f
gurat
o
 `/
tc/dock
r/da
mo
.jso
` as fo
o
s a
d r
start th
 da
mo
:
    ```jso

    {
        "f
atur
s": {
            "bu

dk
t": tru

        }
    }
```
    To bu

d v
m o
 ROCm 7.0 for MI200 a
d MI300 s
r

s, you ca
 us
 th
 d
fau
t:
    ```bash
    DOCKER_BUILDKIT=1 dock
r bu

d \
        -f dock
r/Dock
rf


.rocm_bas
 \
        -t rocm/v
m-d
v:bas
 .
```
F
rst, bu

d a dock
r 
mag
 from [dock
r/Dock
rf


.rocm](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm) a
d 
au
ch a dock
r co
ta


r from th
 
mag
.
It 
s 
mporta
t that th
 us
r k
cks off th
 dock
r bu

d us

g bu

dk
t. E
th
r th
 us
r put `DOCKER_BUILDKIT=1` as 

v
ro
m

t var
ab

 
h

 ca


g dock
r bu

d comma
d, or th
 us
r 

ds to s
t up bu

dk
t 

 th
 dock
r da
mo
 co
f
gurat
o
 /
tc/dock
r/da
mo
.jso
 as fo
o
s a
d r
start th
 da
mo
:
```jso

{
    "f
atur
s": {
        "bu

dk
t": tru

    }
}
```
[dock
r/Dock
rf


.rocm](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm) us
s ROCm 7.0 by d
fau
t, but a
so supports ROCm 5.7, 6.0, 6.1, 6.2, 6.3, a
d 6.4, 

 o
d
r vLLM bra
ch
s.
It prov
d
s f

x
b


ty to custom
z
 th
 bu

d of dock
r 
mag
 us

g th
 fo
o


g argum

ts:
    - `BASE_IMAGE`: sp
c
f

s th
 bas
 
mag
 us
d 
h

 ru


g `dock
r bu

d`. Th
 d
fau
t va
u
 `rocm/v
m-d
v:bas
` 
s a
 
mag
 pub

sh
d a
d ma

ta


d by AMD. It 
s b


g bu

t us

g [dock
r/Dock
rf


.rocm_bas
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/dock
r/Dock
rf


.rocm_bas
)
    - `ARG_PYTORCH_ROCM_ARCH`: A
o
s to ov
rr
d
 th
 gfx arch
t
ctur
 va
u
s from th
 bas
 dock
r 
mag

Th

r va
u
s ca
 b
 pass
d 

 
h

 ru


g `dock
r bu

d` 

th `--bu

d-arg` opt
o
s.
To bu

d v
m o
 ROCm 7.0 for MI200 a
d MI300 s
r

s, you ca
 us
 th
 d
fau
t (
h
ch bu

d a dock
r 
mag
 

th `v
m s
rv
` as 

trypo

t):
```bash
DOCKER_BUILDKIT=1 dock
r bu

d -f dock
r/Dock
rf


.rocm -t v
m/v
m-op

a
-rocm .
```
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
 --rm \
    --group-add=v
d
o \
    --cap-add=SYS_PTRACE \
    --s
cur
ty-opt s
ccomp=u
co
f


d \
    --d
v
c
 /d
v/kfd \
    --d
v
c
 /d
v/dr
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
-rocm 
args...

```
Th
 argum

t `v
m/v
m-op

a
-rocm` sp
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
To us
 th
 dock
r 
mag
 as bas
 for d
v

opm

t, you ca
 
au
ch 
t 

 

t
ract
v
 s
ss
o
 through ov
rr
d

g th
 

trypo

t.
???+ co
so

 "Comma
ds"
    ```bash
    dock
r ru
 --rm -
t \
        --group-add=v
d
o \
        --cap-add=SYS_PTRACE \
        --s
cur
ty-opt s
ccomp=u
co
f


d \
        --d
v
c
 /d
v/kfd \
        --d
v
c
 /d
v/dr
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
        --

t
ork=host \
        --
pc=host \
        --

trypo

t bash \
        v
m/v
m-op

a
-rocm
```
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
