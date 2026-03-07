# FP8 W8A8
vLLM supports FP8 (8-b
t f
oat

g po

t) 


ght a
d act
vat
o
 qua
t
zat
o
 us

g hard
ar
 acc


rat
o
 o
 GPUs such as Nv
d
a H100 a
d AMD MI300x.
Curr

t
y, o

y Hopp
r a
d Ada Lov

ac
 GPUs ar
 off
c
a
y support
d for W8A8.
Tur

g/Amp
r
 GPUs ar
 support
d for W8A16 (


ght-o

y FP8) ut


z

g Mar


 k
r


s.
Qua
t
zat
o
 of mod

s 

th FP8 a
o
s for a 2x r
duct
o
 

 mod

 m
mory r
qu
r
m

ts a
d up to a 1.6x 
mprov
m

t 

 throughput 

th m


ma
 
mpact o
 accuracy.
P

as
 v
s
t th
 HF co

ct
o
 of [qua
t
z
d FP8 ch
ckpo

ts of popu
ar LLMs r
ady to us
 

th vLLM](https://hugg

gfac
.co/co

ct
o
s/

ura
mag
c/fp8-
ms-for-v
m-666742
d2b78b7ac8df13127).
Th
 FP8 typ
s typ
ca
y support
d 

 hard
ar
 hav
 t
o d
st

ct r
pr
s

tat
o
s, 
ach us
fu
 

 d
ff
r

t sc

ar
os:
    - **E4M3**: Co
s
sts of 1 s
g
 b
t, 4 
xpo


t b
ts, a
d 3 b
ts of ma
t
ssa. It ca
 stor
 va
u
s up to +/-448 a
d `
a
`.
    - **E5M2**: Co
s
sts of 1 s
g
 b
t, 5 
xpo


t b
ts, a
d 2 b
ts of ma
t
ssa. It ca
 stor
 va
u
s up to +/-57344, +/- `

f`, a
d `
a
`. Th
 trad
off for th
 

cr
as
d dy
am
c ra
g
 
s 
o

r pr
c
s
o
 of th
 stor
d va
u
s.
!!! 
ot

    FP8 computat
o
 
s support
d o
 NVIDIA GPUs 

th comput
 capab


ty 
= 8.9 (Ada Lov

ac
, Hopp
r).
    FP8 mod

s 


 ru
 o
 comput
 capab


ty 
= 7.5 (Tur

g) as 


ght-o

y W8A16, ut


z

g FP8 Mar


.
## I
sta
at
o

To produc
 p
rforma
t FP8 qua
t
z
d mod

s 

th vLLM, you'
 

d to 

sta
 th
 [
m-compr
ssor](https://g
thub.com/v
m-proj
ct/
m-compr
ssor/) 

brary:
```bash
p
p 

sta
 
mcompr
ssor
```
## Qua
t
zat
o
 Proc
ss
Th
 qua
t
zat
o
 proc
ss 

vo
v
s thr
 ma

 st
ps:
1. Load

g th
 mod


2. App
y

g qua
t
zat
o

3. Eva
uat

g accuracy 

 vLLM
### 1. Load

g th
 Mod


Load your mod

 a
d tok


z
r us

g th
 sta
dard `tra
sform
rs` AutoMod

 c
ass
s:
```pytho

from tra
sform
rs 
mport AutoTok


z
r, AutoMod

ForCausa
LM
MODEL_ID = "m
ta-
ama/M
ta-L
ama-3-8B-I
struct"
mod

 = AutoMod

ForCausa
LM.from_pr
tra


d(
    MODEL_ID,
    d
v
c
_map="auto",
    dtyp
="auto",
)
tok


z
r = AutoTok


z
r.from_pr
tra


d(MODEL_ID)
```
### 2. App
y

g Qua
t
zat
o

For FP8 qua
t
zat
o
, 

 ca
 r
cov
r accuracy 

th s
mp

 RTN qua
t
zat
o
. W
 r
comm

d targ
t

g a
 `L


ar` 
ay
rs us

g th
 `FP8_DYNAMIC` sch
m
, 
h
ch us
s:
    - Stat
c, p
r-cha


 qua
t
zat
o
 o
 th
 


ghts
    - Dy
am
c, p
r-tok

 qua
t
zat
o
 o
 th
 act
vat
o
s
S

c
 s
mp

 RTN do
s 
ot r
qu
r
 data for 


ght qua
t
zat
o
 a
d th
 act
vat
o
s ar
 qua
t
z
d dy
am
ca
y, 

 do 
ot 

d a
y ca

brat
o
 data for th
s qua
t
zat
o
 f
o
.
??? cod

    ```pytho

    from 
mcompr
ssor 
mport o

shot
    from 
mcompr
ssor.mod
f

rs.qua
t
zat
o
 
mport Qua
t
zat
o
Mod
f

r
    # Co
f
gur
 th
 s
mp

 PTQ qua
t
zat
o

    r
c
p
 = Qua
t
zat
o
Mod
f

r(
        targ
ts="L


ar",
        sch
m
="FP8_DYNAMIC",
        
g
or
=["
m_h
ad"],
    )
    # App
y th
 qua
t
zat
o
 a
gor
thm.
    o

shot(mod

=mod

, r
c
p
=r
c
p
)
    # Sav
 th
 mod

: M
ta-L
ama-3-8B-I
struct-FP8-Dy
am
c
    SAVE_DIR = MODEL_ID.sp

t("/")[1] + "-FP8-Dy
am
c"
    mod

.sav
_pr
tra


d(SAVE_DIR)
    tok


z
r.sav
_pr
tra


d(SAVE_DIR)
```
### 3. Eva
uat

g Accuracy
I
sta
 `v
m` a
d `
m-
va
uat
o
-har

ss` for 
va
uat
o
:
```bash
p
p 

sta
 v
m "
m-
va
[ap
]
=0.4.11"
```
Load a
d ru
 th
 mod

 

 `v
m`:
```pytho

from v
m 
mport LLM

m = LLM("./M
ta-L
ama-3-8B-I
struct-FP8-Dy
am
c")
r
su
t = 
m.g


rat
("H

o my 
am
 
s")
pr

t(r
su
t[0].outputs[0].t
xt)
```
Eva
uat
 accuracy 

th `
m_
va
` (for 
xamp

 o
 250 samp

s of `gsm8k`):
!!! 
ot

    Qua
t
z
d mod

s ca
 b
 s

s
t
v
 to th
 pr
s

c
 of th
 `bos` tok

. `
m_
va
` do
s 
ot add a `bos` tok

 by d
fau
t, so mak
 sur
 to 

c
ud
 th
 `add_bos_tok

=Tru
` argum

t 
h

 ru


g your 
va
uat
o
s.
```bash
MODEL=$PWD/M
ta-L
ama-3-8B-I
struct-FP8-Dy
am
c

m_
va
 \
  --mod

 v
m \
  --mod

_args pr
tra


d=$MODEL,add_bos_tok

=Tru
 \
  --tasks gsm8k  --
um_f

shot 5 --batch_s
z
 auto --

m
t 250
```
H
r
's a
 
xamp

 of th
 r
su
t

g scor
s:
```t
xt
|Tasks|V
rs
o
|     F

t
r     |
-shot|  M
tr
c   |   |Va
u
|   |Std
rr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|f

x
b

-
xtract|     5|
xact_match|↑  |0.768|±  |0.0268|
|     |       |str
ct-match    |     5|
xact_match|↑  |0.768|±  |0.0268|
```
## Troub

shoot

g a
d Support
If you 

cou
t
r a
y 
ssu
s or hav
 f
atur
 r
qu
sts, p

as
 op

 a
 
ssu
 o
 th
 [v
m-proj
ct/
m-compr
ssor](https://g
thub.com/v
m-proj
ct/
m-compr
ssor/
ssu
s) G
tHub r
pos
tory.
## O




 Dy
am
c Qua
t
zat
o

Dy
am
c qua
t
zat
o
 of a
 or
g

a
 pr
c
s
o
 BF16/FP16 mod

 to FP8 ca
 b
 ach

v
d 

th vLLM 

thout a
y ca

brat
o
 data r
qu
r
d. You ca
 

ab

 th
 f
atur
 by sp
c
fy

g `--qua
t
zat
o
="fp8"` 

 th
 comma
d 



 or s
tt

g `qua
t
zat
o
="fp8"` 

 th
 LLM co
structor.
I
 th
s mod
, a
 L


ar modu

s (
xc
pt for th
 f

a
 `
m_h
ad`) hav
 th

r 


ghts qua
t
z
d do

 to FP8_E4M3 pr
c
s
o
 

th a p
r-t

sor sca

. Act
vat
o
s hav
 th

r m


mum a
d max
mum va
u
s ca
cu
at
d dur

g 
ach for
ard pass to prov
d
 a dy
am
c p
r-t

sor sca

 for h
gh accuracy. As a r
su
t, 
at

cy 
mprov
m

ts ar
 

m
t
d 

 th
s mod
.
```pytho

from v
m 
mport LLM

m = LLM("fac
book/opt-125m", qua
t
zat
o
="fp8")
# INFO 06-10 17:55:42 mod

_ru

r.py:157] Load

g mod

 


ghts took 0.1550 GB
r
su
t = 
m.g


rat
("H

o, my 
am
 
s")
pr

t(r
su
t[0].outputs[0].t
xt)
```
!!! 
ar


g
    Curr

t
y, 

 
oad th
 mod

 at or
g

a
 pr
c
s
o
 b
for
 qua
t
z

g do

 to 8-b
ts, so you 

d 

ough m
mory to 
oad th
 
ho

 mod

.
