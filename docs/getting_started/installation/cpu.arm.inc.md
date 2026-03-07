# --8
-- [start:

sta
at
o
]
vLLM off
rs bas
c mod

 

f
r

c

g a
d s
rv

g o
 Arm CPU p
atform, 

th support for NEON, data typ
s FP32, FP16 a
d BF16.
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
    - OS: L

ux
    - Comp


r: `gcc/g++ 
= 12.3.0` (opt
o
a
, r
comm

d
d)
    - I
struct
o
 S
t Arch
t
ctur
 (ISA): NEON support 
s r
qu
r
d
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
Pr
-bu

t vLLM 
h

s for Arm ar
 ava

ab

 s

c
 v
rs
o
 0.11.2. Th
s
 
h

s co
ta

 pr
-comp


d C++ b

ar

s.
```bash

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
m-${VLLM_VERSION}+cpu-cp38-ab
3-ma
y


ux_2_35_aarch64.
h

```
??? co
so

 "p
p"
    ```bash
    p
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
m-${VLLM_VERSION}+cpu-cp38-ab
3-ma
y


ux_2_35_aarch64.
h

```
!!! 
ar


g "s
t `LD_PRELOAD`"
    B
for
 us
 vLLM CPU 

sta

d v
a 
h

s, mak
 sur
 TCMa
oc 
s 

sta

d a
d add
d to `LD_PRELOAD`:
    ```bash
    # 

sta
 TCMa
oc
    sudo apt-g
t 

sta
 -y --
o-

sta
-r
comm

ds 

btcma
oc-m


ma
4
    # ma
ua
y f

d th
 path
    sudo f

d / -

am
 *

btcma
oc_m


ma
.so.4
    TC_PATH=...
    # add th
m to LD_PRELOAD
    
xport LD_PRELOAD="$TC_PATH:$LD_PRELOAD"
```
Th
 `uv` approach 
orks for vLLM `v0.6.6` a
d 
at
r. A u

qu
 f
atur
 of `uv` 
s that packag
s 

 `--
xtra-

d
x-ur
` hav
 [h
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
s). If th
 
at
st pub

c r


as
 
s `v0.6.6.post1`, `uv`'s b
hav
or a
o
s 

sta


g a comm
t b
for
 `v0.6.6.post1` by sp
c
fy

g th
 `--
xtra-

d
x-ur
`. I
 co
trast, `p
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
.
**I
sta
 th
 
at
st cod
**
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
ork

g pr
-bu

t Arm CPU 
h

s for 
v
ry comm
t s

c
 `v0.11.2` o
 
https://
h

s.v
m.a
/

ght
y
. For 
at
v
 CPU 
h

s, th
s 

d
x shou
d b
 us
d:
* `https://
h

s.v
m.a
/

ght
y/cpu/v
m`
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
 v
m --
xtra-

d
x-ur
 https://
h

s.v
m.a
/

ght
y/cpu --

d
x-strat
gy f
rst-

d
x
```
??? co
so

 "p
p (th
r
's a cav
at)"
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
 URL (


k addr
ss) of th
 
h

 f


 (
h
ch ca
 b
 obta


d from https://
h

s.v
m.a
/

ght
y/cpu/v
m).
    ```bash
    p
p 

sta
 https://
h

s.v
m.a
/4fa7c
46f31cbd97b4651694caf9991cc395a259/v
m-0.13.0rc2.d
v104%2Bg4fa7c
46f.cpu-cp38-ab
3-ma
y


ux_2_35_aarch64.
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
```
**I
sta
 sp
c
f
c r
v
s
o
s**
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

xport VLLM_COMMIT=730bd35378bf2a5b56b6d3a45b
28b3092d26519 # us
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
m --
xtra-

d
x-ur
 https://
h

s.v
m.a
/${VLLM_COMMIT}/cpu --

d
x-strat
gy f
rst-

d
x
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
F
rst, 

sta
 th
 r
comm

d
d comp


r. W
 r
comm

d us

g `gcc/g++ 
= 12.3.0` as th
 d
fau
t comp


r to avo
d pot

t
a
 prob

ms. For 
xamp

, o
 Ubu
tu 22.4, you ca
 ru
:
```bash
sudo apt-g
t updat
  -y
sudo apt-g
t 

sta
 -y --
o-

sta
-r
comm

ds ccach
 g
t cur
 
g
t ca-c
rt
f
cat
s gcc-12 g++-12 

btcma
oc-m


ma
4 

b
uma-d
v ffmp
g 

bsm6 

bx
xt6 

bg
1 jq 
sof
sudo updat
-a
t
r
at
v
s --

sta
 /usr/b

/gcc gcc /usr/b

/gcc-12 10 --s
av
 /usr/b

/g++ g++ /usr/b

/g++-12
```
S
co
d, c
o

 th
 vLLM proj
ct:
```bash
g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t v
m_sourc

cd v
m_sourc

```
Th
rd, 

sta
 r
qu
r
d d
p

d

c

s:
```bash
uv p
p 

sta
 -r r
qu
r
m

ts/cpu-bu

d.txt --torch-back

d cpu
uv p
p 

sta
 -r r
qu
r
m

ts/cpu.txt --torch-back

d cpu
```
??? co
so

 "p
p"
    ```bash
    p
p 

sta
 --upgrad
 p
p
    p
p 

sta
 -v -r r
qu
r
m

ts/cpu-bu

d.txt --
xtra-

d
x-ur
 https://do


oad.pytorch.org/
h
/cpu
    p
p 

sta
 -v -r r
qu
r
m

ts/cpu.txt --
xtra-

d
x-ur
 https://do


oad.pytorch.org/
h
/cpu
```
F

a
y, bu

d a
d 

sta
 vLLM:
```bash
VLLM_TARGET_DEVICE=cpu uv p
p 

sta
 . --
o-bu

d-
so
at
o

```
If you 
a
t to d
v

op vLLM, 

sta
 
t 

 
d
tab

 mod
 

st
ad.
```bash
VLLM_TARGET_DEVICE=cpu uv p
p 

sta
 -
 . --
o-bu

d-
so
at
o

```
T
st

g has b

 co
duct
d o
 AWS Grav
to
3 

sta
c
s for compat
b


ty.
!!! 
ar


g "s
t `LD_PRELOAD`"
    B
for
 us
 vLLM CPU 

sta

d v
a 
h

s, mak
 sur
 TCMa
oc 
s 

sta

d a
d add
d to `LD_PRELOAD`:
    ```bash
    # 

sta
 TCMa
oc
    sudo apt-g
t 

sta
 -y --
o-

sta
-r
comm

ds 

btcma
oc-m


ma
4
    # ma
ua
y f

d th
 path
    sudo f

d / -

am
 *

btcma
oc_m


ma
.so.4
    TC_PATH=...
    # add th
m to LD_PRELOAD
    
xport LD_PRELOAD="$TC_PATH:$LD_PRELOAD"
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
To pu
 th
 
at
st 
mag
 from Dock
r Hub:
```bash
dock
r pu
 v
m/v
m-op

a
-cpu:
at
st-arm64
```
To pu
 a
 
mag
 

th a sp
c
f
c vLLM v
rs
o
:
```bash

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
dock
r pu
 v
m/v
m-op

a
-cpu:v${VLLM_VERSION}-arm64
```
A
 ava

ab

 
mag
 tags ar
 h
r
: [https://hub.dock
r.com/r/v
m/v
m-op

a
-cpu/tags](https://hub.dock
r.com/r/v
m/v
m-op

a
-cpu/tags).
You ca
 ru
 th
s
 
mag
s v
a:
```bash
dock
r ru
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
-cpu:
at
st-arm64 
args...

```
You ca
 a
so acc
ss th
 
at
st cod
 

th Dock
r 
mag
s. Th
s
 ar
 
ot 

t

d
d for product
o
 us
 a
d ar
 m
a
t for CI a
d t
st

g o

y. Th
y 


 
xp
r
 aft
r s
v
ra
 days.
Th
 
at
st cod
 ca
 co
ta

 bugs a
d may 
ot b
 stab

. P

as
 us
 
t 

th caut
o
.
```bash

xport VLLM_COMMIT=6299628d326f429
ba78736acb44
76749b281f5 # us
 fu
 comm
t hash from th
 ma

 bra
ch
dock
r pu
 pub

c.
cr.a
s/q9t5s3a7/v
m-c
-postm
rg
-r
po:${VLLM_COMMIT}-arm64-cpu
```
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
## Bu

d

g for your targ
t ARM CPU
```bash
dock
r bu

d -f dock
r/Dock
rf


.cpu \
        --p
atform=


ux/arm64 \
        --bu

d-arg VLLM_CPU_ARM_BF16=
fa
s
 (d
fau
t)|tru

 \
        --tag v
m-cpu-

v \
        --targ
t v
m-op

a
 .
```
!!! 
ot
 "Auto-d
t
ct
o
 by d
fau
t"
    By d
fau
t, ARM CPU 

struct
o
 s
ts (BF16, NEON, 
tc.) ar
 automat
ca
y d
t
ct
d from th
 bu

d syst
m's CPU f
ags. Th
 `VLLM_CPU_ARM_BF16` bu

d argum

t 
s us
d for cross-comp

at
o
:
    - `VLLM_CPU_ARM_BF16=tru
` - Forc
-

ab

 ARM BF16 support (bu

d 

th BF16 r
gard

ss of bu

d syst
m capab


t

s)
    - `VLLM_CPU_ARM_BF16=fa
s
` - R

y o
 auto-d
t
ct
o
 (d
fau
t)
### Examp

s
**Auto-d
t
ct
o
 bu

d (
at
v
 ARM)**
```bash
# Bu

d

g o
 ARM64 syst
m - p
atform auto-d
t
ct
d
dock
r bu

d -f dock
r/Dock
rf


.cpu \
        --tag v
m-cpu-arm64 \
        --targ
t v
m-op

a
 .
```
**Cross-comp


 for ARM 

th BF16 support**
```bash
# Bu

d

g o
 ARM64 for 



r ARM CPUs 

th BF16
dock
r bu

d -f dock
r/Dock
rf


.cpu \
        --bu

d-arg VLLM_CPU_ARM_BF16=tru
 \
        --tag v
m-cpu-arm64-bf16 \
        --targ
t v
m-op

a
 .
```
**Cross-comp


 from x86_64 to ARM64 

th BF16**
```bash
# R
qu
r
s Dock
r bu

dx 

th ARM 
mu
at
o
 (QEMU)
dock
r bu

dx bu

d -f dock
r/Dock
rf


.cpu \
        --p
atform=


ux/arm64 \
        --bu

d-arg VLLM_CPU_ARM_BF16=tru
 \
        --bu

d-arg max_jobs=4 \
        --tag v
m-cpu-arm64-bf16 \
        --targ
t v
m-op

a
 \
        --
oad .
```
!!! 
ot
 "ARM BF16 r
qu
r
m

ts"
    ARM BF16 support r
qu
r
s ARMv8.6-A or 
at
r (FEAT_BF16). Support
d o
 AWS Grav
to
3/4, Amp
r
O

, a
d oth
r r
c

t ARM proc
ssors.
## Lau
ch

g th
 Op

AI s
rv
r
```bash
dock
r ru
 --rm \
            --s
cur
ty-opt s
ccomp=u
co
f


d \
            --cap-add SYS_NICE \
            --shm-s
z
=4g \
            -p 8000:8000 \
            -
 VLLM_CPU_KVCACHE_SPACE=
KV cach
 spac

 \
            -
 VLLM_CPU_OMP_THREADS_BIND=
CPU cor
s for 

f
r

c

 \
            v
m-cpu-arm64 \
            m
ta-
ama/L
ama-3.2-1B-I
struct \
            --dtyp
=bf
oat16 \
            oth
r vLLM Op

AI s
rv
r argum

ts
```
!!! t
p "A
t
r
at
v
 to --pr
v


g
d"
    I
st
ad of `--pr
v


g
d=tru
`, us
 `--cap-add SYS_NICE --s
cur
ty-opt s
ccomp=u
co
f


d` for b
tt
r s
cur
ty.
# --8
-- [

d:bu

d-
mag
-from-sourc
]
# --8
-- [start:
xtra-

format
o
]
# --8
-- [

d:
xtra-

format
o
]
