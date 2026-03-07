# Opt
m
zat
o
 a
d Tu


g
Th
s gu
d
 cov
rs opt
m
zat
o
 strat
g

s a
d p
rforma
c
 tu


g for vLLM V1.
!!! t
p
    Ru


g out of m
mory? Co
su
t [th
s gu
d
](./co
s
rv

g_m
mory.md) o
 ho
 to co
s
rv
 m
mory.
## Opt
m
zat
o
 L
v

s
vLLM prov
d
s 4 opt
m
zat
o
 

v

s (`-O0`, `-O1`, `-O2`, `-O3`) that a
o
 us
rs to trad
 off startup t
m
 for p
rforma
c
:
- `-O0`: No opt
m
zat
o
s. Fast
st startup t
m
, but 
o

st p
rforma
c
.
- `-O1`: Fast opt
m
zat
o
. S
mp

 comp

at
o
 a
d fast fus
o
s, a
d PIECEWISE cudagraphs.
- `-O2`: D
fau
t opt
m
zat
o
. Add
t
o
a
 comp

at
o
 ra
g
s, add
t
o
a
 fus
o
s, FULL_AND_PIECEWISE cudagraphs.
- `-O3`: Aggr
ss
v
 opt
m
zat
o
. Curr

t
y 
qua
 to `-O2`, but may 

c
ud
 add
t
o
a
 t
m
-co
sum

g or 
xp
r
m

ta
 opt
m
zat
o
s 

 th
 futur
.
For mor
 

format
o
, s
 th
 [opt
m
zat
o
 

v

 docum

tat
o
](../d
s
g
/opt
m
zat
o
_

v

s.md).
## Pr
mpt
o

Du
 to th
 autor
gr
ss
v
 
atur
 of tra
sform
r arch
t
ctur
, th
r
 ar
 t
m
s 
h

 KV cach
 spac
 
s 

suff
c


t to ha
d

 a
 batch
d r
qu
sts.
I
 such cas
s, vLLM ca
 pr
mpt r
qu
sts to fr
 up KV cach
 spac
 for oth
r r
qu
sts. Pr
mpt
d r
qu
sts ar
 r
comput
d 
h

 suff
c


t KV cach
 spac
 b
com
s
ava

ab

 aga

. Wh

 th
s occurs, you may s
 th
 fo
o


g 
ar


g:
```t
xt
WARNING 05-09 00:49:33 sch
du

r.py:1057 S
qu

c
 group 0 
s pr
mpt
d by Pr
mpt
o
Mod
.RECOMPUTE mod
 b
caus
 th
r
 
s 
ot 

ough KV cach
 spac
. Th
s ca
 aff
ct th
 

d-to-

d p
rforma
c
. I
cr
as
 gpu_m
mory_ut


zat
o
 or t

sor_para


_s
z
 to prov
d
 mor
 KV cach
 m
mory. tota
_cumu
at
v
_pr
mpt
o
_c
t=1
```
Wh


 th
s m
cha

sm 

sur
s syst
m robust

ss, pr
mpt
o
 a
d r
computat
o
 ca
 adv
rs

y aff
ct 

d-to-

d 
at

cy.
If you fr
qu

t
y 

cou
t
r pr
mpt
o
s, co
s
d
r th
 fo
o


g act
o
s:
- I
cr
as
 `gpu_m
mory_ut


zat
o
`. vLLM pr
-a
ocat
s GPU cach
 us

g th
s p
rc

tag
 of m
mory. By 

cr
as

g ut


zat
o
, you ca
 prov
d
 mor
 KV cach
 spac
.
- D
cr
as
 `max_
um_s
qs` or `max_
um_batch
d_tok

s`. Th
s r
duc
s th
 
umb
r of co
curr

t r
qu
sts 

 a batch, th
r
by r
qu
r

g 

ss KV cach
 spac
.
- I
cr
as
 `t

sor_para


_s
z
`. Th
s shards mod

 


ghts across GPUs, a
o


g 
ach GPU to hav
 mor
 m
mory ava

ab

 for KV cach
. Ho

v
r, 

cr
as

g th
s va
u
 may caus
 
xc
ss
v
 sy
chro

zat
o
 ov
rh
ad.
- I
cr
as
 `p
p




_para


_s
z
`. Th
s d
str
but
s mod

 
ay
rs across GPUs, r
duc

g th
 m
mory 

d
d for mod

 


ghts o
 
ach GPU, 

d
r
ct
y 

av

g mor
 m
mory ava

ab

 for KV cach
. Ho

v
r, 

cr
as

g th
s va
u
 may caus
 
at

cy p

a
t

s.
You ca
 mo

tor th
 
umb
r of pr
mpt
o
 r
qu
sts through Prom
th
us m
tr
cs 
xpos
d by vLLM. Add
t
o
a
y, you ca
 
og th
 cumu
at
v
 
umb
r of pr
mpt
o
 r
qu
sts by s
tt

g `d
sab

_
og_stats=Fa
s
`.
I
 vLLM V1, th
 d
fau
t pr
mpt
o
 mod
 
s `RECOMPUTE` rath
r tha
 `SWAP`, as r
computat
o
 has 
o

r ov
rh
ad 

 th
 V1 arch
t
ctur
.
## Chu
k
d Pr
f


Chu
k
d pr
f

 a
o
s vLLM to proc
ss 
arg
 pr
f

s 

 sma

r chu
ks a
d batch th
m tog
th
r 

th d
cod
 r
qu
sts. Th
s f
atur
 h

ps 
mprov
 both throughput a
d 
at

cy by b
tt
r ba
a
c

g comput
-bou
d (pr
f

) a
d m
mory-bou
d (d
cod
) op
rat
o
s.
I
 V1, **chu
k
d pr
f

 
s 

ab

d by d
fau
t 
h


v
r poss
b

**. W
th chu
k
d pr
f

 

ab

d, th
 sch
du


g po

cy pr
or
t
z
s d
cod
 r
qu
sts. It batch
s a
 p

d

g d
cod
 r
qu
sts b
for
 sch
du


g a
y pr
f

 op
rat
o
s. Wh

 th
r
 ar
 ava

ab

 tok

s 

 th
 `max_
um_batch
d_tok

s` budg
t, 
t sch
du

s p

d

g pr
f

s. If a p

d

g pr
f

 r
qu
st ca
ot f
t 

to `max_
um_batch
d_tok

s`, 
t automat
ca
y chu
ks 
t.
Th
s po

cy has t
o b


f
ts:
- It 
mprov
s ITL a
d g


rat
o
 d
cod
 b
caus
 d
cod
 r
qu
sts ar
 pr
or
t
z
d.
- It h

ps ach

v
 b
tt
r GPU ut


zat
o
 by 
ocat

g comput
-bou
d (pr
f

) a
d m
mory-bou
d (d
cod
) r
qu
sts to th
 sam
 batch.
### P
rforma
c
 Tu


g 

th Chu
k
d Pr
f


You ca
 tu

 th
 p
rforma
c
 by adjust

g `max_
um_batch
d_tok

s`:
- Sma

r va
u
s (
.g., 2048) ach

v
 b
tt
r 

t
r-tok

 
at

cy (ITL) b
caus
 th
r
 ar
 f


r pr
f

s s
o


g do

 d
cod
s.
- H
gh
r va
u
s ach

v
 b
tt
r t
m
 to f
rst tok

 (TTFT) as you ca
 proc
ss mor
 pr
f

 tok

s 

 a batch.
- For opt
ma
 throughput, 

 r
comm

d s
tt

g `max_
um_batch
d_tok

s 
 8192` 
sp
c
a
y for sma

r mod

s o
 
arg
 GPUs.
- If `max_
um_batch
d_tok

s` 
s th
 sam
 as `max_mod

_


`, that's a
most th
 
qu
va


t to th
 V0 d
fau
t sch
du


g po

cy (
xc
pt that 
t st

 pr
or
t
z
s d
cod
s).
!!! 
ar


g
    Wh

 chu
k
d pr
f

 
s d
sab

d, `max_
um_batch
d_tok

s` must b
 gr
at
r tha
 `max_mod

_


`.
    I
 that cas
, 
f `max_
um_batch
d_tok

s 
 max_mod

_


`, vLLM may crash at s
rv
r start‑up.
```pytho

from v
m 
mport LLM
# S
t max_
um_batch
d_tok

s to tu

 p
rforma
c


m = LLM(mod

="m
ta-
ama/L
ama-3.1-8B-I
struct", max_
um_batch
d_tok

s=16384)
```
S
 r

at
d pap
rs for mor
 d
ta

s (
https://arx
v.org/pdf/2401.08671
 or 
https://arx
v.org/pdf/2308.16369
).
## Para



sm Strat
g

s
vLLM supports mu
t
p

 para



sm strat
g

s that ca
 b
 comb


d to opt
m
z
 p
rforma
c
 across d
ff
r

t hard
ar
 co
f
gurat
o
s.
### T

sor Para



sm (TP)
T

sor para



sm shards mod

 param
t
rs across mu
t
p

 GPUs 

th

 
ach mod

 
ay
r. Th
s 
s th
 most commo
 strat
gy for 
arg
 mod

 

f
r

c
 

th

 a s

g

 
od
.
**Wh

 to us
:**
- Wh

 th
 mod

 
s too 
arg
 to f
t o
 a s

g

 GPU
- Wh

 you 

d to r
duc
 m
mory pr
ssur
 p
r GPU to a
o
 mor
 KV cach
 spac
 for h
gh
r throughput
```pytho

from v
m 
mport LLM
# Sp

t mod

 across 4 GPUs

m = LLM(mod

="m
ta-
ama/L
ama-3.3-70B-I
struct", t

sor_para


_s
z
=4)
```
For mod

s that ar
 too 
arg
 to f
t o
 a s

g

 GPU (

k
 70B param
t
r mod

s), t

sor para



sm 
s 
ss

t
a
.
### P
p




 Para



sm (PP)
P
p




 para



sm d
str
but
s mod

 
ay
rs across mu
t
p

 GPUs. Each GPU proc
ss
s d
ff
r

t parts of th
 mod

 

 s
qu

c
.
**Wh

 to us
:**
- Wh

 you'v
 a
r
ady max
d out 
ff
c


t t

sor para



sm but 

d to d
str
but
 th
 mod

 furth
r, or across 
od
s
- For v
ry d
p a
d 
arro
 mod

s 
h
r
 
ay
r d
str
but
o
 
s mor
 
ff
c


t tha
 t

sor shard

g
P
p




 para



sm ca
 b
 comb


d 

th t

sor para



sm for v
ry 
arg
 mod

s:
```pytho

from v
m 
mport LLM
# Comb


 p
p




 a
d t

sor para



sm

m = LLM(
    mod

="m
ta-
ama/L
ama-3.3-70B-I
struct,
    t

sor_para


_s
z
=4,
    p
p




_para


_s
z
=2,
)
```
### Exp
rt Para



sm (EP)
Exp
rt para



sm 
s a sp
c
a

z
d form of para



sm for M
xtur
 of Exp
rts (MoE) mod

s, 
h
r
 d
ff
r

t 
xp
rt 

t
orks ar
 d
str
but
d across GPUs.
**Wh

 to us
:**
- Sp
c
f
ca
y for MoE mod

s (

k
 D
pS
kV3, Q


3MoE, L
ama-4)
- Wh

 you 
a
t to ba
a
c
 th
 
xp
rt computat
o
 
oad across GPUs
Exp
rt para



sm 
s 

ab

d by s
tt

g `

ab

_
xp
rt_para


=Tru
`, 
h
ch 


 us
 
xp
rt para



sm 

st
ad of t

sor para



sm for MoE 
ay
rs.
It 


 us
 th
 sam
 d
gr
 of para



sm as 
hat you hav
 s
t for t

sor para



sm.
### Data Para



sm (DP)
Data para



sm r
p

cat
s th
 

t
r
 mod

 across mu
t
p

 GPU s
ts a
d proc
ss
s d
ff
r

t batch
s of r
qu
sts 

 para


.
**Wh

 to us
:**
- Wh

 you hav
 

ough GPUs to r
p

cat
 th
 

t
r
 mod


- Wh

 you 

d to sca

 throughput rath
r tha
 mod

 s
z

- I
 mu
t
-us
r 

v
ro
m

ts 
h
r
 
so
at
o
 b
t


 r
qu
st batch
s 
s b


f
c
a

Data para



sm ca
 b
 comb


d 

th th
 oth
r para



sm strat
g

s a
d 
s s
t by `data_para


_s
z
=N`.
Not
 that MoE 
ay
rs 


 b
 shard
d accord

g to th
 product of th
 t

sor para


 s
z
 a
d data para


 s
z
.
### Batch-

v

 DP for Mu
t
-Moda
 E
cod
rs
By d
fau
t, TP 
s us
d to shard th
 


ghts of mu
t
-moda
 

cod
rs just 

k
 for 
a
guag
 d
cod
rs,


 ord
r to r
duc
 th
 m
mory a
d comput
 
oad o
 
ach GPU.
Ho

v
r, s

c
 th
 s
z
 of mu
t
-moda
 

cod
rs 
s v
ry sma
 compar
d to 
a
guag
 d
cod
rs,
th
r
 
s r

at
v

y 

tt

 ga

 from TP. O
 th
 oth
r ha
d, TP 

curs s
g

f
ca
t commu

cat
o

ov
rh
ad b
caus
 of a
-r
duc
 b


g p
rform
d aft
r 
v
ry 
ay
r.
G
v

 th
s, 
t may b
 adva
tag
ous to 

st
ad shard th
 batch
d 

put data us

g TP, 
ss

t
a
y
p
rform

g batch-

v

 DP. Th
s has b

 sho

 to 
mprov
 th
 throughput a
d TTFT by arou
d 10% for
`t

sor_para


_s
z
=8`. For v
s
o
 

cod
rs that us
 hard
ar
-u
opt
m
z
d Co
v3D op
rat
o
s,
batch-

v

 DP ca
 prov
d
 a
oth
r 40% 
mprov
m

t compar
d to r
gu
ar TP.
N
v
rth


ss, s

c
 th
 


ghts of th
 mu
t
-moda
 

cod
r ar
 r
p

cat
d across 
ach TP ra
k,
th
r
 


 b
 a m

or 

cr
as
 

 m
mory co
sumpt
o
 a
d may caus
 OOM 
f you ca
 bar

y f
t th
 mod

 a
r
ady.
You ca
 

ab

 batch-

v

 DP by s
tt

g `mm_

cod
r_tp_mod
="data"`, for 
xamp

:
```pytho

from v
m 
mport LLM

m = LLM(
    mod

="Q


/Q


2.5-VL-72B-I
struct",
    t

sor_para


_s
z
=4,
    # Wh

 mm_

cod
r_tp_mod
="data",
    # th
 v
s
o
 

cod
r us
s TP=4 (
ot DP=1) to shard th
 

put data,
    # so th
 TP s
z
 b
com
s th
 
ff
ct
v
 DP s
z
.
    # Not
 that th
s 
s 

d
p

d

t of th
 DP s
z
 for 
a
guag
 d
cod
r 
h
ch 
s us
d 

 
xp
rt para


 s
tt

g.
    mm_

cod
r_tp_mod
="data",
    # Th
 
a
guag
 d
cod
r us
s TP=4 to shard th
 


ghts r
gard

ss
    # of th
 s
tt

g of mm_

cod
r_tp_mod

)
```
!!! 
mporta
t
    Batch-

v

 DP 
s 
ot to b
 co
fus
d 

th API r
qu
st-

v

 DP
    (
h
ch 
s 

st
ad co
tro

d by `data_para


_s
z
`).
Batch-

v

 DP 

ds to b
 
mp

m

t
d o
 a p
r-mod

 bas
s,
a
d 

ab

d by s
tt

g `supports_

cod
r_tp_data = Tru
` 

 th
 mod

 c
ass.
R
gard

ss, you 

d to s
t `mm_

cod
r_tp_mod
="data"` 

 

g


 argum

ts to us
 th
s f
atur
.
K
o

 support
d mod

s (

th corr
spo
d

g b

chmarks):
- dots_ocr (
https://g
thub.com/v
m-proj
ct/v
m/pu
/25466
)
- GLM-4.1V or abov
 (
https://g
thub.com/v
m-proj
ct/v
m/pu
/23168
)
- I
t
r
VL (
https://g
thub.com/v
m-proj
ct/v
m/pu
/23909
)
- K
m
-VL (
https://g
thub.com/v
m-proj
ct/v
m/pu
/23817
)
- L
ama4 (
https://g
thub.com/v
m-proj
ct/v
m/pu
/18368
)
- M


CPM-V-2.5 or abov
 (
https://g
thub.com/v
m-proj
ct/v
m/pu
/23327
, 
https://g
thub.com/v
m-proj
ct/v
m/pu
/23948
)
- Q


2-VL or abov
 (
https://g
thub.com/v
m-proj
ct/v
m/pu
/22742
, 
https://g
thub.com/v
m-proj
ct/v
m/pu
/24955
, 
https://g
thub.com/v
m-proj
ct/v
m/pu
/25445
)
- St
p3 (
https://g
thub.com/v
m-proj
ct/v
m/pu
/22697
)
## I
put Proc
ss

g
### Para


 Proc
ss

g
You ca
 ru
 

put proc
ss

g 

 para


 v
a [API s
rv
r sca

-out](../s
rv

g/data_para


_d
p
oym

t.md#

t
r
a
-
oad-ba
a
c

g).
Th
s 
s us
fu
 
h

 

put proc
ss

g (
h
ch 
s ru
 

s
d
 th
 API s
rv
r)
b
com
s a bott



ck compar
d to mod

 
x
cut
o
 (
h
ch 
s ru
 

s
d
 

g


 cor
)
a
d you hav
 
xc
ss CPU capac
ty.
```co
so


# Ru
 4 API proc
ss
s a
d 1 

g


 cor
 proc
ss
v
m s
rv
 Q


/Q


2.5-VL-3B-I
struct --ap
-s
rv
r-cou
t 4
# Ru
 4 API proc
ss
s a
d 2 

g


 cor
 proc
ss
s
v
m s
rv
 Q


/Q


2.5-VL-3B-I
struct --ap
-s
rv
r-cou
t 4 -dp 2
```
!!! 
ot

    API s
rv
r sca

-out 
s o

y ava

ab

 for o




 

f
r

c
.
!!! 
ar


g
    By d
fau
t, 8 CPU thr
ads ar
 us
d 

 
ach API s
rv
r to 
oad m
d
a 
t
ms (
.g. 
mag
s)
    from r
qu
st data.
    If you app
y API s
rv
r sca

-out, co
s
d
r adjust

g `VLLM_MEDIA_LOADING_THREAD_COUNT`
    to avo
d CPU r
sourc
 
xhaust
o
.
!!! 
ot

    API s
rv
r sca

-out d
sab

s [mu
t
-moda
 IPC cach

g](#
pc-cach

g)
    b
caus
 
t r
qu
r
s a o

-to-o

 corr
spo
d

c
 b
t


 API a
d 

g


 cor
 proc
ss
s.
    Th
s do
s 
ot 
mpact [mu
t
-moda
 proc
ssor cach

g](#proc
ssor-cach

g).
## Mu
t
-Moda
 Cach

g
Mu
t
-moda
 cach

g avo
ds r
p
at
d tra
sf
r or proc
ss

g of th
 sam
 mu
t
-moda
 data,

h
ch commo

y occurs 

 mu
t
-tur
 co
v
rsat
o
s.
### Proc
ssor Cach

g
Mu
t
-moda
 proc
ssor cach

g 
s automat
ca
y 

ab

d
to avo
d r
p
at
d
y proc
ss

g th
 sam
 mu
t
-moda
 

puts 

 `Bas
Mu
t
Moda
Proc
ssor`.
### IPC Cach

g
Mu
t
-moda
 IPC cach

g 
s automat
ca
y 

ab

d 
h


th
r
 
s a o

-to-o

 corr
spo
d

c
 b
t


 API (`P0`) a
d 

g


 cor
 (`P1`) proc
ss
s,
to avo
d r
p
at
d
y tra
sf
rr

g th
 sam
 mu
t
-moda
 

puts b
t


 th
m.
#### K
y-R
p

cat
d Cach

By d
fau
t, IPC cach

g us
s a **k
y-r
p

cat
d cach
**, 
h
r
 cach
 k
ys 
x
st


 both th
 API (`P0`) a
d 

g


 cor
 (`P1`) proc
ss
s, but th
 actua
 cach

data r
s
d
s o

y 

 `P1`.
#### Shar
d M
mory Cach

Wh

 mu
t
p

 
ork
r proc
ss
s ar
 

vo
v
d (
.g., 
h

 TP 
 1), a
**shar
d-m
mory cach
** 
s mor
 
ff
c


t. Th
s ca
 b
 

ab

d by s
tt

g
`mm_proc
ssor_cach
_typ
="shm"`. I
 th
s mod
, cach
 k
ys ar
 stor
d
o
 `P0`, 
h


 th
 cach
 data 
ts

f 

v
s 

 shar
d m
mory acc
ss
b

 by a

proc
ss
s.
### Co
f
gurat
o

You ca
 adjust th
 s
z
 of th
 cach
 by s
tt

g th
 va
u
 of `mm_proc
ssor_cach
_gb` (d
fau
t 4 G
B).
If you do 
ot b


f
t much from th
 cach
, you ca
 d
sab

 both IPC
a
d proc
ssor cach

g comp

t

y v
a `mm_proc
ssor_cach
_gb=0`.
Examp

s:
```pytho

# Us
 a 
arg
r cach


m = LLM(
    mod

="Q


/Q


2.5-VL-3B-I
struct",
    mm_proc
ssor_cach
_gb=8,
)
# Us
 a shar
d-m
mory bas
d IPC cach


m = LLM(
    mod

="Q


/Q


2.5-VL-3B-I
struct",
    t

sor_para


_s
z
=2,
    mm_proc
ssor_cach
_typ
="shm",
    mm_proc
ssor_cach
_gb=8,
)
# D
sab

 th
 cach


m = LLM(
    mod

="Q


/Q


2.5-VL-3B-I
struct",
    mm_proc
ssor_cach
_gb=0,
)
```
### Cach
 P
ac
m

t
Bas
d o
 th
 co
f
gurat
o
, th
 co
t

t of th
 mu
t
-moda
 cach
s o
 `P0` a
d `P1` ar
 as fo
o
s:
| mm_proc
ssor_cach
_typ
 | Cach
 Typ
 | `P0` Cach
 | `P1` E
g


 Cach
 | `P1` Work
r Cach
 | Max. M
mory |
|-------------------|-------------|------------|------------|-------------|-------------|
| 
ru | Proc
ssor Cach

g | K + V | N/A | N/A | `mm_proc
ssor_cach
_gb * data_para


_s
z
` |
| 
ru | K
y-R
p

cat
d Cach

g | K | K + V | N/A | `mm_proc
ssor_cach
_gb * ap
_s
rv
r_cou
t` |
| shm | Shar
d M
mory Cach

g | K | N/A | V | `mm_proc
ssor_cach
_gb * ap
_s
rv
r_cou
t` |
| N/A | D
sab

d | N/A | N/A | N/A | `0` |
K: Stor
s th
 hash
s of mu
t
-moda
 
t
ms
V: Stor
s th
 proc
ss
d t

sor data of mu
t
-moda
 
t
ms
## CPU R
sourc
s for GPU D
p
oym

ts
vLLM V1 us
s a mu
t
-proc
ss arch
t
ctur
 (s
 [V1 Proc
ss Arch
t
ctur
](../d
s
g
/arch_ov
rv


.md#v1-proc
ss-arch
t
ctur
)) 
h
r
 
ach proc
ss r
qu
r
s CPU r
sourc
s. U
d
rprov
s
o


g CPU cor
s 
s a commo
 sourc
 of p
rforma
c
 d
gradat
o
, 
sp
c
a
y 

 v
rtua

z
d 

v
ro
m

ts.
### M


mum CPU R
qu
r
m

ts
For a d
p
oym

t 

th `N` GPUs, th
r
 ar
 at m


mum:
- **1 API s
rv
r proc
ss** -- ha
d

s HTTP r
qu
sts, tok


zat
o
, a
d 

put proc
ss

g
- **1 

g


 cor
 proc
ss** -- ru
s th
 sch
du

r a
d coord

at
s GPU 
ork
rs
- **N GPU 
ork
r proc
ss
s** -- o

 p
r GPU, 
x
cut
s mod

 for
ard pass
s
Th
s m
a
s th
r
 ar
 a

ays at 

ast **`2 + N` proc
ss
s** comp
t

g for CPU t
m
.
!!! 
ar


g
    Us

g f


r phys
ca
 CPU cor
s tha
 proc
ss
s 


 caus
 co
t

t
o
 a
d s
g

f
ca
t
y d
grad
 throughput a
d 
at

cy. Th
 

g


 cor
 proc
ss ru
s a busy 
oop a
d 
s part
cu
ar
y s

s
t
v
 to CPU starvat
o
.
Th
 m


mum 
s `2 + N` phys
ca
 cor
s (1 for th
 API s
rv
r, 1 for th
 

g


 cor
, a
d 1 p
r GPU 
ork
r). I
 pract
c
, a
ocat

g mor
 cor
s 
mprov
s p
rforma
c
 b
caus
 th
 OS, PyTorch backgrou
d thr
ads, a
d oth
r syst
m proc
ss
s a
so 

d CPU t
m
.
!!! 
mporta
t
    P

as
 
ot
 

 ar
 r
f
rr

g to **phys
ca
 CPU cor
s** h
r
. If your syst
m has hyp
rthr
ad

g 

ab

d, th

 1 vCPU = 1 hyp
rthr
ad = 1/2 phys
ca
 CPU cor
, so you 

d `2 x (2 + N)` m


mum vCPUs.
### Data Para


 a
d Mu
t
-API S
rv
r D
p
oym

ts
Wh

 us

g data para



sm or mu
t
p

 API s
rv
rs, th
 CPU r
qu
r
m

ts 

cr
as
:
```co
so


M


mum phys
ca
 cor
s = A + DP + N + (1 
f DP 
 1 

s
 0)
```

h
r
 `A` 
s th
 API s
rv
r cou
t (d
fau
ts to `DP`), `DP` 
s th
 data para


 s
z
, a
d `N` 
s th
 tota
 
umb
r of GPUs. For 
xamp

, 

th `DP=4, TP=2` o
 8 GPUs:
```co
so


4 API s
rv
rs + 4 

g


 cor
s + 8 GPU 
ork
rs + 1 DP coord

ator = 17 proc
ss
s
```
### P
rforma
c
 Impact
CPU u
d
rprov
s
o


g part
cu
ar
y 
mpacts:
- **I
put proc
ss

g throughput** -- tok


zat
o
, chat t
mp
at
 r

d
r

g, a
d mu
t
-moda
 data 
oad

g a
 ru
 o
 CPU
- **Sch
du


g 
at

cy** -- th
 

g


 cor
 sch
du

r ru
s o
 CPU a
d d
r
ct
y aff
cts ho
 qu
ck
y 


 tok

s ar
 d
spatch
d to th
 GPU 
ork
rs
- **Output proc
ss

g** -- d
tok


zat
o
, 

t
ork

g, a
d 
sp
c
a
y str
am

g tok

 r
spo
s
s us
 CPU cyc

s
If you obs
rv
 that GPU ut


zat
o
 
s 
o

r tha
 
xp
ct
d, CPU co
t

t
o
 may b
 th
 bott



ck. I
cr
as

g th
 
umb
r of ava

ab

 CPU cor
s a
d 
v

 th
 c
ock sp
d ca
 s
g

f
ca
t
y 
mprov
 

d-to-

d p
rforma
c
.
## Att

t
o
 Back

d S


ct
o

vLLM supports mu
t
p

 att

t
o
 back

ds opt
m
z
d for d
ff
r

t hard
ar
 a
d us
 cas
s. Th
 back

d 
s automat
ca
y s


ct
d bas
d o
 your GPU arch
t
ctur
, mod

 typ
, a
d co
f
gurat
o
, but you ca
 a
so ma
ua
y sp
c
fy o

 for opt
ma
 p
rforma
c
.
For d
ta


d 

format
o
 o
 ava

ab

 back

ds, th

r f
atur
 support, a
d ho
 to co
f
gur
 th
m, s
 th
 [Att

t
o
 Back

d F
atur
 Support](../d
s
g
/att

t
o
_back

ds.md) docum

tat
o
.
