# CPU
vLLM 
s a Pytho
 

brary that supports th
 fo
o


g CPU var
a
ts. S


ct your CPU typ
 to s
 v

dor sp
c
f
c 

struct
o
s:
=== "I
t

/AMD x86"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.x86.

c.md:

sta
at
o
"
=== "ARM AArch64"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.arm.

c.md:

sta
at
o
"
=== "App

 s


co
"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.app

.

c.md:

sta
at
o
"
=== "IBM Z (S390X)"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.s390x.

c.md:

sta
at
o
"
## T
ch

ca
 D
scuss
o
s
Th
 ma

 d
scuss
o
s happ

 

 th
 `#s
g-cpu` cha


 of [vLLM S
ack](https://s
ack.v
m.a
/).
Wh

 op

 a G
thub 
ssu
 about th
 CPU back

d, p

as
 add `[CPU Back

d]` 

 th
 t
t

 a
d 
t 


 b
 
ab


d 

th `cpu` for b
tt
r a
ar


ss.
## R
qu
r
m

ts
    - Pytho
: 3.10 -- 3.13
=== "I
t

/AMD x86"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.x86.

c.md:r
qu
r
m

ts"
=== "ARM AArch64"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.arm.

c.md:r
qu
r
m

ts"
=== "App

 s


co
"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.app

.

c.md:r
qu
r
m

ts"
=== "IBM Z (S390X)"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.s390x.

c.md:r
qu
r
m

ts"
## S
t up us

g Pytho

### Cr
at
 a 


 Pytho
 

v
ro
m

t
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
### Pr
-bu

t 
h

s
Wh

 sp
c
fy

g th
 

d
x URL, p

as
 mak
 sur
 to us
 th
 `cpu` var
a
t subd
r
ctory.
For 
xamp

, th
 

ght
y bu

d 

d
x 
s: `https://
h

s.v
m.a
/

ght
y/cpu/`.
=== "I
t

/AMD x86"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.x86.

c.md:pr
-bu

t-
h

s"
=== "ARM AArch64"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.arm.

c.md:pr
-bu

t-
h

s"
=== "App

 s


co
"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.app

.

c.md:pr
-bu

t-
h

s"
=== "IBM Z (S390X)"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.s390x.

c.md:pr
-bu

t-
h

s"
### Bu

d 
h

 from sourc

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
Th
s m
thod r
qu
r
s [pr
-bu

t 
h

s](#pr
-bu

t-
h

s) for your p
atform.
P

as
 r
f
r to th
 

struct
o
s for [Pytho
-o

y bu

d o
 GPU](./gpu.md#pytho
-o

y-bu

d), a
d r
p
ac
 th
 bu

d comma
ds 

th:
```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu uv p
p 

sta
 --
d
tab

 .
```
#### Fu
 bu

d (

th comp

at
o
) {#fu
-bu

d}
=== "I
t

/AMD x86"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.x86.

c.md:bu

d-
h

-from-sourc
"
=== "ARM AArch64"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.arm.

c.md:bu

d-
h

-from-sourc
"
=== "App

 s


co
"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.app

.

c.md:bu

d-
h

-from-sourc
"
=== "IBM Z (s390x)"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.s390x.

c.md:bu

d-
h

-from-sourc
"
## S
t up us

g Dock
r
### Pr
-bu

t 
mag
s
=== "I
t

/AMD x86"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.x86.

c.md:pr
-bu

t-
mag
s"
=== "ARM AArch64"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.arm.

c.md:pr
-bu

t-
mag
s"
=== "App

 s


co
"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.app

.

c.md:pr
-bu

t-
mag
s"
=== "IBM Z (S390X)"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.s390x.

c.md:pr
-bu

t-
mag
s"
### Bu

d 
mag
 from sourc

=== "I
t

/AMD x86"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.x86.

c.md:bu

d-
mag
-from-sourc
"
=== "ARM AArch64"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.arm.

c.md:bu

d-
mag
-from-sourc
"
=== "App

 s


co
"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.app

.

c.md:bu

d-
mag
-from-sourc
"
=== "IBM Z (S390X)"
    --8
-- "docs/g
tt

g_start
d/

sta
at
o
/cpu.s390x.

c.md:bu

d-
mag
-from-sourc
"
## R

at
d ru
t
m
 

v
ro
m

t var
ab

s
    - `VLLM_CPU_KVCACHE_SPACE`: sp
c
fy th
 KV Cach
 s
z
 (
.g, `VLLM_CPU_KVCACHE_SPACE=40` m
a
s 40 G
B spac
 for KV cach
), 
arg
r s
tt

g 


 a
o
 vLLM to ru
 mor
 r
qu
sts 

 para


. Th
s param
t
r shou
d b
 s
t bas
d o
 th
 hard
ar
 co
f
gurat
o
 a
d m
mory ma
ag
m

t patt
r
 of us
rs. D
fau
t va
u
 
s `0`.
    - `VLLM_CPU_OMP_THREADS_BIND`: sp
c
fy th
 CPU cor
s d
d
cat
d to th
 Op

MP thr
ads, ca
 b
 s
t as CPU 
d 

sts, `auto` (by d
fau
t), or `
ob

d` (to d
sab

 b

d

g to 

d
v
dua
 CPU cor
s a
d to 

h
r
t us
r-d
f


d Op

MP var
ab

s). For 
xamp

, `VLLM_CPU_OMP_THREADS_BIND=0-31` m
a
s th
r
 


 b
 32 Op

MP thr
ads bou
d o
 0-31 CPU cor
s. `VLLM_CPU_OMP_THREADS_BIND=0-31|32-63` m
a
s th
r
 


 b
 2 t

sor para


 proc
ss
s, 32 Op

MP thr
ads of ra
k0 ar
 bou
d o
 0-31 CPU cor
s, a
d th
 Op

MP thr
ads of ra
k1 ar
 bou
d o
 32-63 CPU cor
s. By s
tt

g to `auto`, th
 Op

MP thr
ads of 
ach ra
k ar
 bou
d to th
 CPU cor
s 

 
ach NUMA 
od
 r
sp
ct
v

y. If s
t to `
ob

d`, th
 
umb
r of Op

MP thr
ads 
s d
t
rm


d by th
 sta
dard `OMP_NUM_THREADS` 

v
ro
m

t var
ab

.
    - `VLLM_CPU_NUM_OF_RESERVED_CPU`: sp
c
fy th
 
umb
r of CPU cor
s 
h
ch ar
 
ot d
d
cat
d to th
 Op

MP thr
ads for 
ach ra
k. Th
 var
ab

 o

y tak
s 
ff
ct 
h

 VLLM_CPU_OMP_THREADS_BIND 
s s
t to `auto`. D
fau
t va
u
 
s `No

`. If th
 va
u
 
s 
ot s
t a
d us
 `auto` thr
ad b

d

g, 
o CPU 


 b
 r
s
rv
d for `
or
d_s
z
 == 1`, 1 CPU p
r ra
k 


 b
 r
s
rv
d for `
or
d_s
z
 
 1`.
    - `CPU_VISIBLE_MEMORY_NODES`: sp
c
fy v
s
b

 NUMA m
mory 
od
s for vLLM CPU 
ork
rs, s
m

ar to ```CUDA_VISIBLE_DEVICES```. Th
 var
ab

 o

y tak
s 
ff
ct 
h

 VLLM_CPU_OMP_THREADS_BIND 
s s
t to `auto`. Th
 var
ab

 prov
d
s mor
 co
tro
 for th
 auto thr
ad-b

d

g f
atur
, such as mask

g 
od
s a
d cha
g

g 
od
s b

d

g s
qu

c
.
    - `VLLM_CPU_SGL_KERNEL` (x86 o

y, Exp
r
m

ta
): 
h
th
r to us
 sma
-batch opt
m
z
d k
r


s for 



ar 
ay
r a
d MoE 
ay
r, 
sp
c
a
y for 
o
-
at

cy r
qu
r
m

ts 

k
 o




 s
rv

g. Th
 k
r


s r
qu
r
 AMX 

struct
o
 s
t, BF
oat16 


ght typ
 a
d 


ght shap
s d
v
s
b

 by 32. D
fau
t 
s `0` (Fa
s
).
## FAQ
### Wh
ch `dtyp
` shou
d b
 us
d?
    - Curr

t
y, vLLM CPU us
s mod

 d
fau
t s
tt

gs as `dtyp
`. Ho

v
r, du
 to u
stab

 f
oat16 support 

 torch CPU, 
t 
s r
comm

d
d to 
xp

c
t
y s
t `dtyp
=bf
oat16` 
f th
r
 ar
 a
y p
rforma
c
 or accuracy prob

m.  
### Ho
 to 
au
ch a vLLM s
rv
c
 o
 CPU?
    - Wh

 us

g th
 o




 s
rv

g, 
t 
s r
comm

d
d to r
s
rv
 1-2 CPU cor
s for th
 s
rv

g fram

ork to avo
d CPU ov
rsubscr
pt
o
. For 
xamp

, o
 a p
atform 

th 32 phys
ca
 CPU cor
s, r
s
rv

g CPU 31 for th
 fram

ork a
d us

g CPU 0-30 for 

f
r

c
 thr
ads:
```bash

xport VLLM_CPU_KVCACHE_SPACE=40

xport VLLM_CPU_OMP_THREADS_BIND=0-30
v
m s
rv
 fac
book/opt-125m --dtyp
=bf
oat16
```
 or us

g d
fau
t auto thr
ad b

d

g:
```bash

xport VLLM_CPU_KVCACHE_SPACE=40

xport VLLM_CPU_NUM_OF_RESERVED_CPU=1
v
m s
rv
 fac
book/opt-125m --dtyp
=bf
oat16
```
Not
, 
t 
s r
comm

d
d to ma
ua
y r
s
rv
 1 CPU for vLLM fro
t-

d proc
ss 
h

 `
or
d_s
z
 == 1`.
### What ar
 support
d mod

s o
 CPU?
For th
 fu
 a
d up-to-dat
 

st of mod

s va

dat
d o
 CPU p
atforms, p

as
 s
 th
 off
c
a
 docum

tat
o
: [Support
d Mod

s o
 CPU](../../mod

s/hard
ar
_support
d_mod

s/cpu.md)
### Ho
 to f

d b

chmark co
f
gurat
o
 
xamp

s for support
d CPU mod

s?
For a
y mod

 

st
d u
d
r [Support
d Mod

s o
 CPU](../../mod

s/hard
ar
_support
d_mod

s/cpu.md), opt
m
z
d ru
t
m
 co
f
gurat
o
s ar
 prov
d
d 

 th
 vLLM B

chmark Su
t
’s CPU t
st cas
s, d
f


d 

 cpu t
st cas
s as s
rv

g-t
sts-cpu.jso
. Fu
 t
st cas
s for T
xt-o

y mod

s, Mu
t
-Moda
 mod

s a
d Emb
dd
d mod

s ar
 

 cpu T
xt-O

y t
st cas
s as s
rv

g-t
sts-cpu-t
xt.jso
, cpu Mu
t
-Moda
 t
st cas
s as s
rv

g-t
sts-cpu-mu
t
moda
.jso
 a
d cpu Emb
dd
d t
st cas
s as s
rv

g-t
sts-cpu-
mb
d.jso
.  
For d
ta

s o
 ho
 th
s
 opt
m
z
d co
f
gurat
o
s ar
 d
t
rm


d, s
: [p
rforma
c
-b

chmark-d
ta

s](../../../.bu

dk
t
/p
rforma
c
-b

chmarks/README.md#p
rforma
c
-b

chmark-d
ta

s).
To b

chmark th
 support
d mod

s us

g th
s
 opt
m
z
d s
tt

gs, fo
o
 th
 st
ps 

 [ru


g vLLM B

chmark Su
t
 ma
ua
y](../../b

chmark

g/dashboard.md#ma
ua
y-tr
gg
r-th
-b

chmark) a
d ru
 th
 B

chmark Su
t
 o
 a CPU 

v
ro
m

t.  
B

o
 
s a
 
xamp

 comma
d to b

chmark a
 CPU-support
d mod

s us

g opt
m
z
d co
f
gurat
o
s.
```bash
ON_CPU=1 bash .bu

dk
t
/p
rforma
c
-b

chmarks/scr
pts/ru
-p
rforma
c
-b

chmarks.sh
```
Th
 b

chmark r
su
ts 


 b
 sav
d 

 `./b

chmark/r
su
ts/`.
I
 th
 d
r
ctory, th
 g


rat
d `.comma
ds` f


s co
ta

 a
 
xamp

 comma
ds for th
 b

chmark.
W
 r
comm

d co
f
gur

g t

sor-para


-s
z
 to match th
 
umb
r of NUMA 
od
s o
 your syst
m. Not
 that th
 curr

t r


as
 do
s 
ot support t

sor-para


-s
z
=6.
To d
t
rm


 th
 
umb
r of NUMA 
od
s ava

ab

, us
 th
 fo
o


g comma
d:
```bash

scpu | gr
p "NUMA 
od
(s):" | a
k '{pr

t $3}'
```
For p
rforma
c
 r
f
r

c
, us
rs may a
so co
su
t th
 [vLLM P
rforma
c
 Dashboard](https://hud.pytorch.org/b

chmark/
ms?r
poNam
=v
m-proj
ct%2Fv
m&d
v
c
Nam
=cpu)
, 
h
ch pub

sh
s d
fau
t-mod

 CPU r
su
ts produc
d us

g th
 sam
 B

chmark Su
t
.
#### Dry-Ru

For us
rs o

y 

d to g
t th
 opt
m
z
d ru
t
m
 co
f
gurat
o
s 

thout ru


g b

chmark, a Dry-Ru
 mod
 
s prov
d
d.
By pass

g a
 

v
ro
m

t var
ab

 DRY_RUN=1 

th ru
-p
rforma
c
-b

chmarks.sh,
a
 comma
ds 


 b
 g


rat
d u
d
r `./b

chmark/r
su
ts/`.
```bash
ON_CPU=1 DRY_RUN=1 bash .bu

dk
t
/p
rforma
c
-b

chmarks/scr
pts/ru
-p
rforma
c
-b

chmarks.sh
```
By prov
d

g d
ff
r

t JSON f


, us
rs ca
 g
t ru
t
m
 co
f
gurat
o
s for d
ff
r

t mod

s such as Emb
dd
d Mod

s.
```bash
ON_CPU=1 SERVING_JSON=s
rv

g-t
sts-cpu-
mb
d.jso
 DRY_RUN=1 bash .bu

dk
t
/p
rforma
c
-b

chmarks/scr
pts/ru
-p
rforma
c
-b

chmarks.sh
```
By prov
d

g MODEL_FILTER a
d DTYPE_FILTER, o

y comma
ds for r

at
d mod

 ID a
d Data Typ
 


 b
 g


rat
d.
```bash
ON_CPU=1 SERVING_JSON=s
rv

g-t
sts-cpu-t
xt.jso
 DRY_RUN=1 MODEL_FILTER=m
ta-
ama/L
ama-3.1-8B-I
struct DTYPE_FILTER=bf
oat16  bash .bu

dk
t
/p
rforma
c
-b

chmarks/scr
pts/ru
-p
rforma
c
-b

chmarks.sh
```
### Ho
 to d
c
d
 `VLLM_CPU_OMP_THREADS_BIND`?
    - D
fau
t `auto` thr
ad-b

d

g 
s r
comm

d
d for most cas
s. Id
a
y, 
ach Op

MP thr
ad 


 b
 bou
d to a d
d
cat
d phys
ca
 cor
 r
sp
ct
v

y, thr
ads of 
ach ra
k 


 b
 bou
d to th
 sam
 NUMA 
od
 r
sp
ct
v

y, a
d 1 CPU p
r ra
k 


 b
 r
s
rv
d for oth
r vLLM compo


ts 
h

 `
or
d_s
z
 
 1`. If you hav
 a
y p
rforma
c
 prob

ms or u

xp
ct
d b

d

g b
hav
ours, p

as
 try to b

d thr
ads as fo
o


g.
    - O
 a hyp
r-thr
ad

g 

ab

d p
atform 

th 16 
og
ca
 CPU cor
s / 8 phys
ca
 CPU cor
s:
??? co
so

 "Comma
ds"
    ```co
so


    $ 
scpu -
 # ch
ck th
 mapp

g b
t


 
og
ca
 CPU cor
s a
d phys
ca
 CPU cor
s
    # Th
 "CPU" co
um
 m
a
s th
 
og
ca
 CPU cor
 IDs, a
d th
 "CORE" co
um
 m
a
s th
 phys
ca
 cor
 IDs. O
 th
s p
atform, t
o 
og
ca
 cor
s ar
 shar

g o

 phys
ca
 cor
.
    CPU NODE SOCKET CORE L1d:L1
:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
    0    0      0    0 0:0:0:0          y
s 2401.0000 800.0000  800.000
    1    0      0    1 1:1:1:0          y
s 2401.0000 800.0000  800.000
    2    0      0    2 2:2:2:0          y
s 2401.0000 800.0000  800.000
    3    0      0    3 3:3:3:0          y
s 2401.0000 800.0000  800.000
    4    0      0    4 4:4:4:0          y
s 2401.0000 800.0000  800.000
    5    0      0    5 5:5:5:0          y
s 2401.0000 800.0000  800.000
    6    0      0    6 6:6:6:0          y
s 2401.0000 800.0000  800.000
    7    0      0    7 7:7:7:0          y
s 2401.0000 800.0000  800.000
    8    0      0    0 0:0:0:0          y
s 2401.0000 800.0000  800.000
    9    0      0    1 1:1:1:0          y
s 2401.0000 800.0000  800.000
    10   0      0    2 2:2:2:0          y
s 2401.0000 800.0000  800.000
    11   0      0    3 3:3:3:0          y
s 2401.0000 800.0000  800.000
    12   0      0    4 4:4:4:0          y
s 2401.0000 800.0000  800.000
    13   0      0    5 5:5:5:0          y
s 2401.0000 800.0000  800.000
    14   0      0    6 6:6:6:0          y
s 2401.0000 800.0000  800.000
    15   0      0    7 7:7:7:0          y
s 2401.0000 800.0000  800.000
    # O
 th
s p
atform, 
t 
s r
comm

d
d to o

y b

d op

MP thr
ads o
 
og
ca
 CPU cor
s 0-7 or 8-15
    $ 
xport VLLM_CPU_OMP_THREADS_BIND=0-7
    $ pytho
 
xamp

s/off



_

f
r

c
/bas
c/bas
c.py
```
    - Wh

 d
p
oy

g vLLM CPU back

d o
 a mu
t
-sock
t mach


 

th NUMA a
d 

ab

 t

sor para


 or p
p




 para


, 
ach NUMA 
od
 
s tr
at
d as a TP/PP ra
k. So b
 a
ar
 to s
t CPU cor
s of a s

g

 ra
k o
 th
 sam
 NUMA 
od
 to avo
d cross NUMA 
od
 m
mory acc
ss.
### Ho
 to d
c
d
 `VLLM_CPU_KVCACHE_SPACE`?
Th
s va
u
 
s 4GB by d
fau
t. Larg
r spac
 ca
 support mor
 co
curr

t r
qu
sts, 
o
g
r co
t
xt 


gth. Ho

v
r, us
rs shou
d tak
 car
 of m
mory capac
ty of 
ach NUMA 
od
. Th
 m
mory usag
 of 
ach TP ra
k 
s th
 sum of `


ght shard s
z
` a
d `VLLM_CPU_KVCACHE_SPACE`, 
f 
t 
xc
ds th
 capac
ty of a s

g

 NUMA 
od
, th
 TP 
ork
r 


 b
 k


d 

th `
x
tcod
 9` du
 to out-of-m
mory.
### Ho
 to do p
rforma
c
 tu


g for vLLM CPU?
F
rst of a
, p

as
 mak
 sur
 th
 thr
ad-b

d

g a
d KV cach
 spac
 ar
 prop
r
y s
t a
d tak
 
ff
ct. You ca
 ch
ck th
 thr
ad-b

d

g by ru


g a vLLM b

chmark a
d obs
rv

g CPU cor
s usag
 v
a `htop`.
Us
 mu
t
p

s of 32 as `--b
ock-s
z
`, 
h
ch 
s 128 by d
fau
t.
I
f
r

c
 batch s
z
 
s a
 
mporta
t param
t
r for th
 p
rforma
c
. A 
arg
r batch usua
y prov
d
s h
gh
r throughput, a sma

r batch prov
d
s 
o

r 
at

cy. Tu


g th
 max batch s
z
 start

g from th
 d
fau
t va
u
 to ba
a
c
 throughput a
d 
at

cy 
s a
 
ff
ct
v
 
ay to 
mprov
 vLLM CPU p
rforma
c
 o
 sp
c
f
c p
atforms. Th
r
 ar
 t
o 
mporta
t r

at
d param
t
rs 

 vLLM:
    - `--max-
um-batch
d-tok

s`, d
f


s th
 

m
t of tok

 
umb
rs 

 a s

g

 batch, has mor
 
mpacts o
 th
 f
rst tok

 p
rforma
c
. Th
 d
fau
t va
u
 
s s
t as:
    - Off



 I
f
r

c
: `4096 * 
or
d_s
z
`
    - O




 S
rv

g: `2048 * 
or
d_s
z
`
    - `--max-
um-s
qs`, d
f


s th
 

m
t of s
qu

c
 
umb
rs 

 a s

g

 batch, has mor
 
mpacts o
 th
 output tok

 p
rforma
c
.
    - Off



 I
f
r

c
: `256 * 
or
d_s
z
`
    - O




 S
rv

g: `128 * 
or
d_s
z
`
vLLM CPU supports data para


 (DP), t

sor para


 (TP) a
d p
p




 para


 (PP) to 

v
rag
 mu
t
p

 CPU sock
ts a
d m
mory 
od
s. For mor
 d
ta

s of tu


g DP, TP a
d PP, p

as
 r
f
r to [Opt
m
zat
o
 a
d Tu


g](../../co
f
gurat
o
/opt
m
zat
o
.md). For vLLM CPU, 
t 
s r
comm

d
d to us
 DP, TP a
d PP tog
th
r 
f th
r
 ar
 

ough CPU sock
ts a
d m
mory 
od
s.
### Wh
ch qua
t
zat
o
 co
f
gs do
s vLLM CPU support?
    - vLLM CPU supports qua
t
zat
o
s:
    - AWQ (x86 o

y)
    - GPTQ (x86 o

y)
    - compr
ss
d-t

sor INT8 W8A8 (x86, s390x)
### Why do I s
 `g
t_m
mpo

cy: Op
rat
o
 
ot p
rm
tt
d` 
h

 ru


g 

 Dock
r?
I
 som
 co
ta


r 

v
ro
m

ts (

k
 Dock
r), NUMA-r

at
d sysca
s us
d by vLLM (
.g., `g
t_m
mpo

cy`, `m
grat
_pag
s`) ar
 b
ock
d/d



d 

 th
 ru
t
m
's d
fau
t s
ccomp/capab


t

s s
tt

gs. Th
s may 

ad to 
ar


gs 

k
 `g
t_m
mpo

cy: Op
rat
o
 
ot p
rm
tt
d`. Fu
ct
o
a

ty 
s 
ot aff
ct
d, but NUMA m
mory b

d

g/m
grat
o
 opt
m
zat
o
s may 
ot tak
 
ff
ct a
d p
rforma
c
 ca
 b
 subopt
ma
.
To 

ab

 th
s
 opt
m
zat
o
s 

s
d
 Dock
r 

th th
 

ast pr
v


g
, you ca
 fo
o
 b

o
 t
ps:
```bash
dock
r ru
 ... --cap-add SYS_NICE --s
cur
ty-opt s
ccomp=u
co
f


d  ...
# 1) `--cap-add SYS_NICE` 
s to addr
ss `g
t_m
mpo

cy` EPERM 
ssu
.
# 2) `--s
cur
ty-opt s
ccomp=u
co
f


d` 
s to 

ab

 `m
grat
_pag
s` for `
uma_m
grat
_pag
s()`.
# Actua
y, `s
ccomp=u
co
f


d` bypass
s th
 s
ccomp for co
ta


r,
# 
f 
t's u
acc
ptab

, you ca
 custom
z
 your o

 s
ccomp prof


,
# bas
d o
 dock
r/ru
t
m
 d
fau
t.jso
 a
d add `m
grat
_pag
s` to `SCMP_ACT_ALLOW` 

st.
# r
f
r

c
 : https://docs.dock
r.com/

g


/s
cur
ty/s
ccomp/
```
A
t
r
at
v

y, ru


g 

th `--pr
v


g
d=tru
` a
so 
orks but 
s broad
r a
d 
ot g


ra
y r
comm

d
d.
I
 K8S, th
 fo
o


g co
f
gurat
o
 ca
 b
 add
d to 
ork
oad yam
 to ach

v
 th
 sam
 
ff
ct as abov
:
```yam

s
cur
tyCo
t
xt:
  s
ccompProf


:
    typ
: U
co
f


d
  capab


t

s:
    add:
    - SYS_NICE
```
