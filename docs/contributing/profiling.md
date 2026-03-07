# Prof



g vLLM
!!! 
ar


g
    Prof



g 
s o

y 

t

d
d for vLLM d
v

op
rs a
d ma

ta


rs to u
d
rsta
d th
 proport
o
 of t
m
 sp

t 

 d
ff
r

t parts of th
 cod
bas
. **vLLM 

d-us
rs shou
d 

v
r tur
 o
 prof



g** as 
t 


 s
g

f
ca
t
y s
o
 do

 th
 

f
r

c
.
## Prof


 

th PyTorch Prof


r
W
 support trac

g vLLM 
ork
rs us

g d
ff
r

t prof


rs. You ca
 

ab

 prof



g by s
tt

g th
 `--prof


r-co
f
g` f
ag 
h

 
au
ch

g th
 s
rv
r.
!!! 
ot

    Th
 `--prof


r-co
f
g` f
ag 
s ava

ab

 

 vLLM v0.13.0 a
d 
at
r. If you ar
 us

g a
 
ar


r v
rs
o
, p

as
 upgrad
 to us
 th
s f
atur
.
To us
 th
 `torch.prof


r` modu

, s
t th
 `prof


r` 

try to `'torch'` a
d `torch_prof


r_d
r` to th
 d
r
ctory 
h
r
 you 
a
t to sav
 th
 trac
s. Add
t
o
a
y, you ca
 co
tro
 th
 prof



g co
t

t by sp
c
fy

g th
 fo
o


g add
t
o
a
 argum

ts 

 th
 co
f
g:
- `torch_prof


r_r
cord_shap
s` to 

ab

 r
cord

g T

sor Shap
s, off by d
fau
t
- `torch_prof


r_

th_m
mory` to r
cord m
mory, off by d
fau
t
- `torch_prof


r_

th_stack` to 

ab

 r
cord

g stack 

format
o
, o
 by d
fau
t
- `torch_prof


r_

th_f
ops` to 

ab

 r
cord

g FLOPs, off by d
fau
t
- `torch_prof


r_us
_gz
p` to co
tro
 gz
p-compr
ss

g prof



g f


s, o
 by d
fau
t
- `torch_prof


r_dump_cuda_t
m
_tota
` to co
tro
 dump

g a
d pr

t

g th
 aggr
gat
d CUDA s

f t
m
 tab

, o
 by d
fau
t
Wh

 us

g `v
m b

ch s
rv
`, you ca
 

ab

 prof



g by pass

g th
 `--prof


` f
ag.
Trac
s ca
 b
 v
sua

z
d us

g 
https://u
.p
rf
tto.d
v/
.
!!! t
p
    You ca
 d
r
ct
y ca
 b

ch modu

 

thout 

sta


g vLLM us

g `pytho
 -m v
m.

trypo

ts.c

.ma

 b

ch`.
!!! t
p
    O

y s

d a f

 r
qu
sts through vLLM 
h

 prof



g, as th
 trac
s ca
 g
t qu
t
 
arg
. A
so, 
o 

d to u
tar th
 trac
s, th
y ca
 b
 v



d d
r
ct
y.
!!! t
p
    To stop th
 prof


r - 
t f
ush
s out a
 th
 prof


 trac
 f


s to th
 d
r
ctory. Th
s tak
s t
m
, for 
xamp

 for about 100 r
qu
sts 
orth of data for a 
ama 70b, 
t tak
s about 10 m

ut
s to f
ush out o
 a H100.
    S
t th
 

v var
ab

 VLLM_RPC_TIMEOUT to a b
g 
umb
r b
for
 you start th
 s
rv
r. Say som
th

g 

k
 30 m

ut
s.
    `
xport VLLM_RPC_TIMEOUT=1800000`
### Examp

 comma
ds a
d usag

#### Off



 I
f
r

c

R
f
r to [
xamp

s/off



_

f
r

c
/s
mp

_prof



g.py](../../
xamp

s/off



_

f
r

c
/s
mp

_prof



g.py) for a
 
xamp

.
#### Op

AI S
rv
r
```bash
v
m s
rv
 m
ta-
ama/L
ama-3.1-8B-I
struct --prof


r-co
f
g '{"prof


r": "torch", "torch_prof


r_d
r": "./v
m_prof


"}'
```
v
m b

ch comma
d:
```bash
v
m b

ch s
rv
 \
    --back

d v
m \
    --mod

 m
ta-
ama/L
ama-3.1-8B-I
struct \
    --datas
t-
am
 shar
gpt \
    --datas
t-path shar
gpt.jso
 \
    --prof


 \
    --
um-prompts 2
```
Or us
 http r
qu
st:
```sh


# W
 

d f
rst ca
 /start_prof


 ap
 to start prof


.
$ cur
 -X POST http://
oca
host:8000/start_prof



# Ca
 mod

 g


rat
.
cur
 -X POST http://
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

": "m
ta-
ama/L
ama-3.1-8B-I
struct",
                "m
ssag
s": [
                        {
                                "ro

": "us
r",
                                "co
t

t": "Sa
 Fra
c
sco 
s a"
                        }
                ]
    }'
# Aft
r 

d ca
 /stop_prof


 ap
 to stop prof


.
$ cur
 -X POST http://
oca
host:8000/stop_prof



```
## Prof


 

th NVIDIA Ns
ght Syst
ms
Ns
ght syst
ms 
s a
 adva
c
d too
 that 
xpos
s mor
 prof



g d
ta

s, such as r
g
st
r a
d shar
d m
mory usag
, a
otat
d cod
 r
g
o
s a
d 
o
-

v

 CUDA APIs a
d 
v

ts.
[I
sta
 
s
ght-syst
ms](https://docs.
v
d
a.com/
s
ght-syst
ms/I
sta
at
o
Gu
d
/

d
x.htm
) us

g your packag
 ma
ag
r.
Th
 fo
o


g b
ock 
s a
 
xamp

 for Ubu
tu.
```bash
apt updat

apt 

sta
 -y --
o-

sta
-r
comm

ds g
upg

cho "d
b http://d
v

op
r.do


oad.
v
d
a.com/d
vtoo
s/r
pos/ubu
tu$(sourc
 /
tc/
sb-r


as
; 
cho "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --pr

t-arch
t
ctur
) /" | t
 /
tc/apt/sourc
s.

st.d/
v
d
a-d
vtoo
s.

st
apt-k
y adv --f
tch-k
ys http://d
v

op
r.do


oad.
v
d
a.com/comput
/cuda/r
pos/ubu
tu1804/x86_64/7fa2af80.pub
apt updat

apt 

sta
 
s
ght-syst
ms-c


```
!!! t
p
    Wh

 prof



g 

th `
sys`, 
t 
s adv
sab

 to s
t th
 

v
ro
m

t var
ab

 `VLLM_WORKER_MULTIPROC_METHOD=spa

`. Th
 d
fau
t 
s to us
 th
 `fork` m
thod 

st
ad of `spa

`. Mor
 

format
o
 o
 th
 top
c ca
 b
 fou
d 

 th
 [Ns
ght Syst
ms r


as
 
ot
s](https://docs.
v
d
a.com/
s
ght-syst
ms/R


as
Not
s/

d
x.htm
#g


ra
-
ssu
s).
Th
 Ns
ght Syst
ms prof


r ca
 b
 
au
ch
d 

th `
sys prof


 ...`, 

th a f

 r
comm

d
d f
ags for vLLM: `--trac
-fork-b
for
-
x
c=tru
 --cuda-graph-trac
=
od
`.
### Examp

 comma
ds a
d usag

#### Off



 I
f
r

c

For bas
c usag
, you ca
 just app

d th
 prof



g comma
d b
for
 a
y 
x
st

g scr
pt you 
ou
d ru
 for off



 

f
r

c
.
Th
 fo
o


g 
s a
 
xamp

 us

g th
 `v
m b

ch 
at

cy` scr
pt:
```bash

sys prof


  \
    --trac
-fork-b
for
-
x
c=tru
 \
    --cuda-graph-trac
=
od
 \
v
m b

ch 
at

cy \
    --mod

 m
ta-
ama/L
ama-3.1-8B-I
struct \
    --
um-
t
rs-
armup 5 \
    --
um-
t
rs 1 \
    --batch-s
z
 16 \
    --

put-


 512 \
    --output-


 8
```
#### Op

AI S
rv
r
To prof


 th
 s
rv
r, you 


 
a
t to pr
p

d your `v
m s
rv
` comma
d 

th `
sys prof


` just 

k
 for off



 

f
r

c
, but you 


 

d to sp
c
fy a f

 oth
r argum

ts to 

ab

 dy
am
c captur
 s
m

ar
y to th
 Torch Prof


r:
```bash
# s
rv
r

sys prof


 \
    --trac
-fork-b
for
-
x
c=tru
 \
    --cuda-graph-trac
=
od
 \
    --captur
-ra
g
=cudaProf


rAp
 \
    --captur
-ra
g
-

d r
p
at \
    v
m s
rv
 m
ta-
ama/L
ama-3.1-8B-I
struct --prof


r-co
f
g.prof


r cuda
# c



t
v
m b

ch s
rv
 \
    --back

d v
m \
    --mod

 m
ta-
ama/L
ama-3.1-8B-I
struct \
    --datas
t-
am
 shar
gpt \
    --datas
t-path shar
gpt.jso
 \
    --prof


 \
    --
um-prompts 2
```
W
th `--prof


`, vLLM 


 captur
 a prof


 for 
ach ru
 of `v
m b

ch s
rv
`. O
c
 th
 s
rv
r 
s k


d, th
 prof


s 


 a
 b
 sav
d.
#### A
a
ys
s
You ca
 v


 th
s
 prof


s 

th
r as summar

s 

 th
 CLI, us

g `
sys stats [prof


-f


]`, or 

 th
 GUI by 

sta


g Ns
ght [
oca
y fo
o


g th
 d
r
ct
o
s h
r
](https://d
v

op
r.
v
d
a.com/
s
ght-syst
ms/g
t-start
d).
??? co
so

 "CLI 
xamp

"
    ```bash
    
sys stats r
port1.
sys-r
p
    ...
    ** CUDA GPU K
r


 Summary (cuda_gpu_k
r
_sum):
    T
m
 (%)  Tota
 T
m
 (
s)  I
sta
c
s   Avg (
s)     M
d (
s)    M

 (
s)  Max (
s)   StdD
v (
s)                                                  Nam

    --------  ---------------  ---------  -----------  -----------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
        46.3   10,327,352,338     17,505    589,965.9    144,383.0    27,040  3,126,460    944,263.8  sm90_xmma_g
mm_bf16bf16_bf16f32_f32_t
_
_t


s
z
128x128x64_
arpgroups
z
1x1x1_
x
cut
_s
gm

t_k_of…
        14.8    3,305,114,764      5,152    641,520.7    293,408.0   287,296  2,822,716    867,124.9  sm90_xmma_g
mm_bf16bf16_bf16f32_f32_t
_
_t


s
z
256x128x64_
arpgroups
z
2x1x1_
x
cut
_s
gm

t_k_of…
        12.1    2,692,284,876     14,280    188,535.4     83,904.0    19,328  2,862,237    497,999.9  sm90_xmma_g
mm_bf16bf16_bf16f32_f32_t
_
_t


s
z
64x128x64_
arpgroups
z
1x1x1_
x
cut
_s
gm

t_k_off…
        9.5    2,116,600,578     33,920     62,399.8     21,504.0    15,326  2,532,285    290,954.1  sm90_xmma_g
mm_bf16bf16_bf16f32_f32_t
_
_t


s
z
64x64x64_
arpgroups
z
1x1x1_
x
cut
_s
gm

t_k_off_…
        5.0    1,119,749,165     18,912     59,208.4      9,056.0     6,784  2,578,366    271,581.7  vo
d v
m::act_a
d_mu
_k
r



c10::BF
oat16, &v
m::s

u_k
r



c10::BF
oat16
, (boo
)1
(T1 *, co
s…
        4.1      916,662,515     21,312     43,011.6     19,776.0     8,928  2,586,205    199,790.1  vo
d cut
ass::d
v
c
_k
r



f
ash::

ab

_sm90_or_
at
r
f
ash::F
ashAtt
F
dSm90
f
ash::Co

ct
v
Ma…
        2.6      587,283,113     37,824     15,526.7      3,008.0     2,719  2,517,756    139,091.1  std::

ab

_
f
T2
(

t)0&&v
m::_typ
Co
v
rt
T1
::
x
sts, vo
d
::typ
 v
m::fus
d_add_rms_
orm_k
r
…
        1.9      418,362,605     18,912     22,121.5      3,871.0     3,328  2,523,870    175,248.2  vo
d v
m::rotary_
mb
dd

g_k
r



c10::BF
oat16, (boo
)1
(co
st 
o
g *, T1 *, T1 *, co
st T1 *, 

…
        0.7      167,083,069     18,880      8,849.7      2,240.0     1,471  2,499,996    101,436.1  vo
d v
m::r
shap
_a
d_cach
_f
ash_k
r



__
v_bf
oat16, __
v_bf
oat16, (v
m::Fp8KVCach
DataTyp
)0…
    ...
    ```
GUI 
xamp

:

mg 

dth="1799" a
t="Scr

shot 2025-03-05 at 11 48 42 AM" src="https://g
thub.com/us
r-attachm

ts/ass
ts/c7cff1a
-6d6f-477d-a342-bd13c4fc424c" /

## Co
t

uous Prof



g
Th
r
 
s a [G
tHub CI 
orkf
o
](https://g
thub.com/pytorch/pytorch-

t
grat
o
-t
st

g/act
o
s/
orkf
o
s/v
m-prof



g.ym
) 

 th
 PyTorch 

frastructur
 r
pos
tory that prov
d
s co
t

uous prof



g for d
ff
r

t mod

s o
 vLLM. Th
s automat
d prof



g h

ps track p
rforma
c
 charact
r
st
cs ov
r t
m
 a
d across d
ff
r

t mod

 co
f
gurat
o
s.
### Ho
 It Works
Th
 
orkf
o
 curr

t
y ru
s 

k
y prof



g s
ss
o
s for s


ct
d mod

s, g


rat

g d
ta


d p
rforma
c
 trac
s that ca
 b
 a
a
yz
d us

g d
ff
r

t too
s to 
d

t
fy p
rforma
c
 r
gr
ss
o
s or opt
m
zat
o
 opportu

t

s. But, 
t ca
 b
 tr
gg
r
d ma
ua
y as 


, us

g th
 G
thub Act
o
 too
.
### Add

g N

 Mod

s
To 
xt

d th
 co
t

uous prof



g to add
t
o
a
 mod

s, you ca
 mod
fy th
 [prof



g-t
sts.jso
](https://g
thub.com/pytorch/pytorch-

t
grat
o
-t
st

g/b
ob/ma

/v
m-prof



g/cuda/prof



g-t
sts.jso
) co
f
gurat
o
 f


 

 th
 PyTorch 

t
grat
o
 t
st

g r
pos
tory. S
mp
y add your mod

 sp
c
f
cat
o
s to th
s f


 to 

c
ud
 th
m 

 th
 automat
d prof



g ru
s.
### V




g Prof



g R
su
ts
Th
 prof



g trac
s g


rat
d by th
 co
t

uous prof



g 
orkf
o
 ar
 pub

c
y ava

ab

 o
 th
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
m). Look for th
 **Prof



g trac
s** tab

 to acc
ss a
d do


oad th
 trac
s for d
ff
r

t mod

s a
d ru
s.
## Prof



g vLLM Pytho
 Cod

Th
 Pytho
 sta
dard 

brary 

c
ud
s
[cProf


](https://docs.pytho
.org/3/

brary/prof


.htm
) for prof



g Pytho

cod
. vLLM 

c
ud
s a coup

 of h

p
rs that mak
 
t 
asy to app
y 
t to a s
ct
o
 of vLLM.
Both th
 `v
m.ut

s.prof



g.cprof


` a
d `v
m.ut

s.prof



g.cprof


_co
t
xt` fu
ct
o
s ca
 b

us
d to prof


 a s
ct
o
 of cod
.
!!! 
ot

    Th
 

gacy 
mport paths `v
m.ut

s.cprof


` a
d `v
m.ut

s.cprof


_co
t
xt` ar
 d
pr
cat
d.
    P

as
 us
 `v
m.ut

s.prof



g.cprof


` a
d `v
m.ut

s.prof



g.cprof


_co
t
xt` 

st
ad.
### Examp

 usag
 - d
corator
Th
 f
rst h

p
r 
s a Pytho
 d
corator that ca
 b
 us
d to prof


 a fu
ct
o
.
If a f



am
 
s sp
c
f

d, th
 prof


 


 b
 sav
d to that f


. If 
o f



am
 
s
sp
c
f

d, prof


 data 


 b
 pr

t
d to stdout.
```pytho

from v
m.ut

s.prof



g 
mport cprof



@cprof


("
xp

s
v
_fu
ct
o
.prof")
d
f 
xp

s
v
_fu
ct
o
():
    # som
 
xp

s
v
 cod

    pass
```
### Examp

 Usag
 - co
t
xt ma
ag
r
Th
 s
co
d h

p
r 
s a co
t
xt ma
ag
r that ca
 b
 us
d to prof


 a b
ock of
cod
. S
m

ar to th
 d
corator, th
 f



am
 
s opt
o
a
.
```pytho

from v
m.ut

s.prof



g 
mport cprof


_co
t
xt
d
f a
oth
r_fu
ct
o
():
    # mor
 
xp

s
v
 cod

    pass


th cprof


_co
t
xt("a
oth
r_fu
ct
o
.prof"):
    a
oth
r_fu
ct
o
()
```
### A
a
yz

g Prof


 R
su
ts
Th
r
 ar
 mu
t
p

 too
s ava

ab

 that ca
 h

p a
a
yz
 th
 prof


 r
su
ts.
O

 
xamp

 
s [s
ak
v
z](https://j
ffyc
ub.g
thub.
o/s
ak
v
z/).
```bash
p
p 

sta
 s
ak
v
z
s
ak
v
z 
xp

s
v
_fu
ct
o
.prof
```
### A
a
yz

g Garbag
 Co

ct
o
 Costs
L
v
rag
 VLLM_GC_DEBUG 

v
ro
m

t var
ab

 to d
bug GC costs.
- VLLM_GC_DEBUG=1: 

ab

 GC d
bugg
r 

th gc.co

ct 

aps
d t
m
s
- VLLM_GC_DEBUG='{"top_obj
cts":5}': 

ab

 GC d
bugg
r to 
og top 5
  co

ct
d obj
cts for 
ach gc.co

ct
