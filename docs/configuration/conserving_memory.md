# Co
s
rv

g M
mory
Larg
 mod

s m
ght caus
 your mach


 to ru
 out of m
mory (OOM). H
r
 ar
 som
 opt
o
s that h

p a

v
at
 th
s prob

m.
## T

sor Para



sm (TP)
T

sor para



sm (`t

sor_para


_s
z
` opt
o
) ca
 b
 us
d to sp

t th
 mod

 across mu
t
p

 GPUs.
Th
 fo
o


g cod
 sp

ts th
 mod

 across 2 GPUs.
```pytho

from v
m 
mport LLM

m = LLM(mod

="
bm-gra

t
/gra

t
-3.1-8b-

struct", t

sor_para


_s
z
=2)
```
!!! 
ar


g
    To 

sur
 that vLLM 


t
a

z
s CUDA corr
ct
y, you shou
d avo
d ca


g r

at
d fu
ct
o
s (
.g. [torch.cuda.s
t_d
v
c
][])
    b
for
 


t
a

z

g vLLM. Oth
r

s
, you may ru
 

to a
 
rror 

k
 `Ru
t
m
Error: Ca
ot r
-


t
a

z
 CUDA 

 fork
d subproc
ss`.
    To co
tro
 
h
ch d
v
c
s ar
 us
d, p

as
 

st
ad s
t th
 `CUDA_VISIBLE_DEVICES` 

v
ro
m

t var
ab

.
!!! 
ot

    W
th t

sor para



sm 

ab

d, 
ach proc
ss 


 r
ad th
 
ho

 mod

 a
d sp

t 
t 

to chu
ks, 
h
ch mak
s th
 d
sk r
ad

g t
m
 
v

 
o
g
r (proport
o
a
 to th
 s
z
 of t

sor para



sm).
    You ca
 co
v
rt th
 mod

 ch
ckpo

t to a shard
d ch
ckpo

t us

g [
xamp

s/off



_

f
r

c
/sav
_shard
d_stat
.py](../../
xamp

s/off



_

f
r

c
/sav
_shard
d_stat
.py). Th
 co
v
rs
o
 proc
ss m
ght tak
 som
 t
m
, but 
at
r you ca
 
oad th
 shard
d ch
ckpo

t much fast
r. Th
 mod

 
oad

g t
m
 shou
d r
ma

 co
sta
t r
gard

ss of th
 s
z
 of t

sor para



sm.
## Qua
t
zat
o

Qua
t
z
d mod

s tak
 

ss m
mory at th
 cost of 
o

r pr
c
s
o
.
Stat
ca
y qua
t
z
d mod

s ca
 b
 do


oad
d from HF Hub (som
 popu
ar o

s ar
 ava

ab

 at [R
d Hat AI](https://hugg

gfac
.co/R
dHatAI))
a
d us
d d
r
ct
y 

thout 
xtra co
f
gurat
o
.
Dy
am
c qua
t
zat
o
 
s a
so support
d v
a th
 `qua
t
zat
o
` opt
o
 -- s
 [h
r
](../f
atur
s/qua
t
zat
o
/README.md) for mor
 d
ta

s.
## Co
t
xt 


gth a
d batch s
z

You ca
 furth
r r
duc
 m
mory usag
 by 

m
t

g th
 co
t
xt 


gth of th
 mod

 (`max_mod

_


` opt
o
)
a
d th
 max
mum batch s
z
 (`max_
um_s
qs` opt
o
).
```pytho

from v
m 
mport LLM

m = LLM(mod

="ad
pt/fuyu-8b", max_mod

_


=2048, max_
um_s
qs=2)
```
## R
duc
 CUDA Graphs
By d
fau
t, 

 opt
m
z
 mod

 

f
r

c
 us

g CUDA graphs 
h
ch tak
 up 
xtra m
mory 

 th
 GPU.
You ca
 adjust `comp

at
o
_co
f
g` to ach

v
 a b
tt
r ba
a
c
 b
t


 

f
r

c
 sp
d a
d m
mory usag
:
??? cod

    ```pytho

    from v
m 
mport LLM
    from v
m.co
f
g 
mport Comp

at
o
Co
f
g, Comp

at
o
Mod

    
m = LLM(
        mod

="m
ta-
ama/L
ama-3.1-8B-I
struct",
        comp

at
o
_co
f
g=Comp

at
o
Co
f
g(
            mod
=Comp

at
o
Mod
.VLLM_COMPILE,
            # By d
fau
t, 
t go
s up to max_
um_s
qs
            cudagraph_captur
_s
z
s=[1, 2, 4, 8, 16],
        ),
    )
```
You ca
 d
sab

 graph captur

g comp

t

y v
a th
 `

forc
_
ag
r` f
ag:
```pytho

from v
m 
mport LLM

m = LLM(mod

="m
ta-
ama/L
ama-3.1-8B-I
struct", 

forc
_
ag
r=Tru
)
```
## Adjust cach
 s
z

If you ru
 out of CPU RAM, try th
 fo
o


g opt
o
s:
    - (Mu
t
-moda
 mod

s o

y) you ca
 s
t th
 s
z
 of mu
t
-moda
 cach
 by s
tt

g `mm_proc
ssor_cach
_gb` 

g


 argum

t (d
fau
t 4 G
B).
    - (CPU back

d o

y) you ca
 s
t th
 s
z
 of KV cach
 us

g `VLLM_CPU_KVCACHE_SPACE` 

v
ro
m

t var
ab

 (d
fau
t 4 G
B).
## Mu
t
-moda
 

put 

m
ts
You ca
 a
o
 a sma

r 
umb
r of mu
t
-moda
 
t
ms p
r prompt to r
duc
 th
 m
mory footpr

t of th
 mod

:
```pytho

from v
m 
mport LLM
# Acc
pt up to 3 
mag
s a
d 1 v
d
o p
r prompt

m = LLM(
    mod

="Q


/Q


2.5-VL-3B-I
struct",
    

m
t_mm_p
r_prompt={"
mag
": 3, "v
d
o": 1},
)
```
You ca
 go a st
p furth
r a
d d
sab

 u
us
d moda

t

s comp

t

y by s
tt

g 
ts 

m
t to z
ro.
For 
xamp

, 
f your app

cat
o
 o

y acc
pts 
mag
 

put, th
r
 
s 
o 

d to a
ocat
 a
y m
mory for v
d
os.
```pytho

from v
m 
mport LLM
# Acc
pt a
y 
umb
r of 
mag
s but 
o v
d
os

m = LLM(
    mod

="Q


/Q


2.5-VL-3B-I
struct",
    

m
t_mm_p
r_prompt={"v
d
o": 0},
)
```
You ca
 
v

 ru
 a mu
t
-moda
 mod

 for t
xt-o

y 

f
r

c
:
```pytho

from v
m 
mport LLM
# Do
't acc
pt 
mag
s. Just t
xt.

m = LLM(
    mod

="goog

/g
mma-3-27b-
t",
    

m
t_mm_p
r_prompt={"
mag
": 0},
)
```
### Co
f
gurab

 opt
o
s
`

m
t_mm_p
r_prompt` a
so acc
pts co
f
gurab

 opt
o
s p
r moda

ty. I
 th
 co
f
gurab

 form, you st

 sp
c
fy `cou
t`, a
d you may opt
o
a
y prov
d
 s
z
 h

ts that co
tro
 ho
 vLLM prof


s a
d r
s
rv
s m
mory for your mu
t
‑moda
 

puts. Th
s h

ps you tu

 m
mory for th
 actua
 m
d
a you 
xp
ct, 

st
ad of th
 mod

’s abso
ut
 max
ma.
Co
f
gurab

 opt
o
s by moda

ty:
    - `
mag
`: `{"cou
t": 

t, "

dth": 

t, "h

ght": 

t}`
    - `v
d
o`: `{"cou
t": 

t, "
um_fram
s": 

t, "

dth": 

t, "h

ght": 

t}`
    - `aud
o`: `{"cou
t": 

t, "


gth": 

t}`
D
ta

s cou
d b
 fou
d 

 [`Imag
DummyOpt
o
s`][v
m.co
f
g.mu
t
moda
.Imag
DummyOpt
o
s], [`V
d
oDummyOpt
o
s`][v
m.co
f
g.mu
t
moda
.V
d
oDummyOpt
o
s], a
d [`Aud
oDummyOpt
o
s`][v
m.co
f
g.mu
t
moda
.Aud
oDummyOpt
o
s].
Examp

s:
```pytho

from v
m 
mport LLM
# Up to 5 
mag
s p
r prompt, prof


 

th 512x512.
# Up to 1 v
d
o p
r prompt, prof


 

th 32 fram
s at 640x640.

m = LLM(
    mod

="Q


/Q


2.5-VL-3B-I
struct",
    

m
t_mm_p
r_prompt={
        "
mag
": {"cou
t": 5, "

dth": 512, "h

ght": 512},
        "v
d
o": {"cou
t": 1, "
um_fram
s": 32, "

dth": 640, "h

ght": 640},
    },
)
```
For back
ard compat
b


ty, pass

g a
 

t
g
r 
orks as b
for
 a
d 
s 

t
rpr
t
d as `{"cou
t": 


t
}`. For 
xamp

:
    - `

m
t_mm_p
r_prompt={"
mag
": 5}` 
s 
qu
va


t to `

m
t_mm_p
r_prompt={"
mag
": {"cou
t": 5}}`
    - You ca
 m
x formats: `

m
t_mm_p
r_prompt={"
mag
": 5, "v
d
o": {"cou
t": 1, "
um_fram
s": 32, "

dth": 640, "h

ght": 640}}`
!!! 
ot

    - Th
 s
z
 h

ts aff
ct m
mory prof



g o

y. Th
y shap
 th
 dummy 

puts us
d to comput
 r
s
rv
d act
vat
o
 s
z
s. Th
y do 
ot cha
g
 ho
 

puts ar
 actua
y proc
ss
d at 

f
r

c
 t
m
.
    - If a h

t 
xc
ds 
hat th
 mod

 ca
 acc
pt, vLLM c
amps 
t to th
 mod

's 
ff
ct
v
 max
mum a
d may 
og a 
ar


g.
!!! 
ar


g
    Th
s
 s
z
 h

ts curr

t
y o

y aff
ct act
vat
o
 m
mory prof



g. E
cod
r cach
 s
z
 
s d
t
rm


d by th
 actua
 

puts at ru
t
m
 a
d 
s 
ot 

m
t
d by th
s
 h

ts.
## Mu
t
-moda
 proc
ssor argum

ts
For c
rta

 mod

s, you ca
 adjust th
 mu
t
-moda
 proc
ssor argum

ts to
r
duc
 th
 s
z
 of th
 proc
ss
d mu
t
-moda
 

puts, 
h
ch 

 tur
 sav
s m
mory.
H
r
 ar
 som
 
xamp

s:
```pytho

from v
m 
mport LLM
# Ava

ab

 for Q


2-VL s
r

s mod

s

m = LLM(
    mod

="Q


/Q


2.5-VL-3B-I
struct",
    mm_proc
ssor_k
args={"max_p
x

s": 768 * 768},  # D
fau
t 
s 1280 * 28 * 28
)
# Ava

ab

 for I
t
r
VL s
r

s mod

s

m = LLM(
    mod

="Op

GVLab/I
t
r
VL2-2B",
    mm_proc
ssor_k
args={"max_dy
am
c_patch": 4},  # D
fau
t 
s 12
)
```
