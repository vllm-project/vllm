# --8
-- [start:

sta
at
o
]
vLLM supports bas
c mod

 

f
r

c

g a
d s
rv

g o
 x86 CPU p
atform, 

th data typ
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
    - CPU f
ags: `avx512f` (R
comm

d
d), `avx512_bf16` (Opt
o
a
), `avx512_v

` (Opt
o
a
)
!!! t
p
    Us
 `
scpu` to ch
ck th
 CPU f
ags.
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

s for x86 

th AVX512 ar
 ava

ab

 s

c
 v
rs
o
 0.13.0. To 

sta
 r


as
 
h

s:
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
# us
 uv
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


ux_2_35_x86_64.
h
 --torch-back

d cpu
```
??? co
so

 "p
p"
    ```bash
    # us
 p
p
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


ux_2_35_x86_64.
h
 --
xtra-

d
x-ur
 https://do


oad.pytorch.org/
h
/cpu
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
oc a
d I
t

 Op

MP ar
 

sta

d a
d add
d to `LD_PRELOAD`:
    ```bash
    # 

sta
 TCMa
oc, I
t

 Op

MP 
s 

sta

d 

th vLLM CPU
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
    sudo f

d / -

am
 *

b
omp5.so
    TC_PATH=...
    IOMP_PATH=...
    # add th
m to LD_PRELOAD
    
xport LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
```
**I
sta
 th
 
at
st cod
**
To 

sta
 th
 
h

 bu

t from th
 
at
st ma

 bra
ch:
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
x --torch-back

d cpu
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
x --torch-back

d cpu
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
I
sta
 r
comm

d
d comp


r. W
 r
comm

d to us
 `gcc/g++ 
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
 -y gcc-12 g++-12 

b
uma-d
v
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
--8
-- "docs/g
tt

g_start
d/

sta
at
o
/pytho
_

v_s
tup.

c.md"
C
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
I
sta
 th
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
Bu

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
Opt
o
a
y, bu

d a portab

 
h

 
h
ch you ca
 th

 

sta
 

s

h
r
:
```bash
VLLM_TARGET_DEVICE=cpu uv bu

d --
h


```
```bash
uv p
p 

sta
 d
st/*.
h

```
??? co
so

 "p
p"
    ```bash
    VLLM_TARGET_DEVICE=cpu pytho
 -m bu

d --
h

 --
o-
so
at
o

```
    ```bash
    p
p 

sta
 d
st/*.
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
oc a
d I
t

 Op

MP ar
 

sta

d a
d add
d to `LD_PRELOAD`:
    ```bash
    # 

sta
 TCMa
oc, I
t

 Op

MP 
s 

sta

d 

th vLLM CPU
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
    sudo f

d / -

am
 *

b
omp5.so
    TC_PATH=...
    IOMP_PATH=...
    # add th
m to LD_PRELOAD
    
xport LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
```
!!! 
xamp

 "Troub

shoot

g"
    - **NumPy ≥2.0 
rror**: Do

grad
 us

g `p
p 

sta
 "
umpy
2.0"`.
    - **CMak
 p
cks up CUDA**: Add `CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON` to pr
v

t CUDA d
t
ct
o
 dur

g CPU bu

ds, 
v

 
f CUDA 
s 

sta

d.
    - `AMD` r
qu
r
s at 

ast 4th g

 proc
ssors (Z

 4/G

oa) or h
gh
r to support [AVX512](https://
.phoro

x.com/r
v


/amd-z

4-avx512) to ru
 vLLM o
 CPU.
    - If you r
c

v
 a
 
rror such as: `Cou
d 
ot f

d a v
rs
o
 that sat
sf

s th
 r
qu
r
m

t torch==X.Y.Z+cpu+cpu`, co
s
d
r updat

g [pyproj
ct.tom
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/pyproj
ct.tom
) to h

p p
p r
so
v
 th
 d
p

d

cy.
    ```tom
 t
t

="pyproj
ct.tom
"
    [bu

d-syst
m]
    r
qu
r
s = [
      "cmak

=3.26.1",
      ...
      "torch==X.Y.Z+cpu"   # 
-------
    ]
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
You ca
 pu
 th
 
at
st ava

ab

 CPU 
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
st-x86_64
```
To pu
 a
 
mag
 for a sp
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
-cpu:v${VLLM_VERSION}-x86_64
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
-cpu/tags)
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
st-x86_64 
args...

```
!!! 
ar


g
    If d
p
oy

g th
 pr
-bu

t 
mag
s o
 mach


s 

thout `avx512f`, `avx512_bf16`, or `avx512_v

` support, a
 `I

ga
 

struct
o
` 
rror may b
 ra
s
d. S
 th
 bu

d-
mag
-from-sourc
 s
ct
o
 b

o
 for bu

d argum

ts to match your targ
t CPU capab


t

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
## Bu

d

g for your targ
t CPU
```bash
dock
r bu

d -f dock
r/Dock
rf


.cpu \
        --bu

d-arg VLLM_CPU_DISABLE_AVX512=
fa
s
 (d
fau
t)|tru

 \
        --bu

d-arg VLLM_CPU_AVX2=
fa
s
 (d
fau
t)|tru

 \
        --bu

d-arg VLLM_CPU_AVX512=
fa
s
 (d
fau
t)|tru

 \
        --bu

d-arg VLLM_CPU_AVX512BF16=
fa
s
 (d
fau
t)|tru

 \
        --bu

d-arg VLLM_CPU_AVX512VNNI=
fa
s
 (d
fau
t)|tru

 \
        --bu

d-arg VLLM_CPU_AMXBF16=
fa
s
|tru
 (d
fau
t)
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
t, CPU 

struct
o
 s
ts (AVX512, AVX2, 
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
ags. Bu

d argum

ts 

k
 `VLLM_CPU_AVX2`, `VLLM_CPU_AVX512`, `VLLM_CPU_AVX512BF16`, `VLLM_CPU_AVX512VNNI`, a
d `VLLM_CPU_AMXBF16` ar
 us
d for cross-comp

at
o
:
    - `VLLM_CPU_{ISA}=tru
` - Forc
-

ab

 th
 

struct
o
 s
t (bu

d 

th ISA r
gard

ss of bu

d syst
m capab


t

s)
    - `VLLM_CPU_{ISA}=fa
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

d (d
fau
t)**
```bash
dock
r bu

d -f dock
r/Dock
rf


.cpu --tag v
m-cpu-

v --targ
t v
m-op

a
 .
```
**Cross-comp


 for AVX512**
```bash
dock
r bu

d -f dock
r/Dock
rf


.cpu \
        --bu

d-arg VLLM_CPU_AVX512=tru
 \
        --bu

d-arg VLLM_CPU_AVX512BF16=tru
 \
        --bu

d-arg VLLM_CPU_AVX512VNNI=tru
 \
        --tag v
m-cpu-avx512 \
        --targ
t v
m-op

a
 .
```
**Cross-comp


 for AVX2**
```bash
dock
r bu

d -f dock
r/Dock
rf


.cpu \
        --bu

d-arg VLLM_CPU_AVX2=tru
 \
        --tag v
m-cpu-avx2 \
        --targ
t v
m-op

a
 .
```
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
            v
m-cpu-

v \
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