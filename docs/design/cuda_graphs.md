# CUDA Graphs
Th
s 
r
t
-up 

troduc
s th
 


 CUDA Graphs mod
s 

 vLLM v1 b
yo
d pr
v
ous [torch.comp


 

t
grat
o
](torch_comp


.md). To summar
z
, 

:
1. Add
d f

x
b

 `cudagraph_mod
` co
f
gurat
o

2. Mad
 fu
 CUDA Graphs support orthogo
a
 to comp

at
o

3. I
troduc
d a CUDA Graphs d
spatch
r as a c

tra
 co
tro

r that p
cks th
 d
s
r
d ru
t
m
 mod
 a
d CUDA Graphs p
r batch automat
ca
y
I
 th
s docum

t 

 


 d
scuss th
:
* [Mot
vat
o
](#mot
vat
o
)
* [CUDA Graphs mod
s](#cudagraphmod
s)
* [D
ta


d d
s
g
](#d
ta


d-d
s
g
)
* [Examp

 usag
 of th
 d
ff
r

t CUDA Graphs mod
s](#usag
-gu
d
)
!!! 
ot

    I
 th
s docum

t, 

 r
f
r to pur
 d
cod
 (`max_qu
ry_


=1`) or sp
cu
at
v
 d
cod
 (`max_qu
ry_


 =1+
um_sp
c_tok

s`) as **u

form d
cod
** batch
s, a
d th
 oppos
t
 
ou
d b
 **
o
-u

form** batch
s (
.
., pr
f

 or m
x
d pr
f

-d
cod
 batch
s).
!!! 
ot

    Th
 fo
o


g co
t

ts ar
 most
y bas
d o
 th
 
ast comm
t of 
https://g
thub.com/v
m-proj
ct/v
m/pu
/20059
.
## Mot
vat
o

I

t
a
 p

c


s
 comp

at
o
 
as bu

t to a
o
 p

c


s
 cudagraph captur
, 
xc
ud

g cudagraph-u
support
d op
rat
o
s (ma


y att

t
o
). Th
s a
o

d som
 sp
dup from cudagraphs 
h


 ma

ta



g compat
b


ty 

th a
 att

t
o
 back

ds. W
 
at
r add
d support for "fu
 cudagraphs" by 
ot comp



g p

c


s
, so that 

 cou
d furth
r r
duc
 th
 
at

cy 

 cas
s 
h
r
 att

t
o
 support
d cudagraphs. Ho

v
r, th
s t
ght coup


g b
t


 comp

at
o
 a
d cudagraph captur
 

d to a
 a
-or-
oth

g 
xp
r


c
 

th 

tt

 f

x
b


ty. Ma
y att

t
o
 back

ds a
so 

r

’t r
ady for u

f

d "fu
" CUDA Graphs captur
 (
.g., o

y F
ashAtt

t
o
 3 supports 
t curr

t
y) or o

y support CUDA Graphs for pur
 d
cod
 batch
s (
.g., F
ash

f
r, F
ashMLA, a
d Mamba, 
tc.). That 

d to co
fus

g p
rforma
c
/compat
b


ty trad
offs, 

co
s
st

t CUDA Graphs support, a
d 

cr
as

g
y comp

x cod
 structur
.
Th
s 

d us to s
k a mor
 f


-gra


d CUDA Graphs so
ut
o
 

th th
 fo
o


g f
atur
s:
* Exp

c
t
y a
ar
 of CUDA Graphs for pr
f

/m
x
d or (u

form-)d
cod
 batch a
d captur
 th
m s
parat

y.
* S
parat
 CUDAGraph captur
 
og
c from comp

at
o
 (as much as f
as
b

) for f
atur
 orthogo
a

ty, 
h
ch sugg
st:
    * Captur

g p

c


s
 a
d fu
 cudagraphs us

g th
 sam
 comp


d graph, a
d
    * Fu
 cudagraph captur
 

thout comp

at
o
.
* D
spatch b
t


 fu
 a
d p

c


s
 cudagraph at ru
t
m
 d
p

d

g o
 batch compos
t
o
.
* C

tra

z
d co
tro
 of CUDAGraph b
hav
or for r
duc
d cod
 comp

x
ty a
d a
o

d mor
 
xt

d
b


ty.
Th
s
 f
atur
s a
o
 th
 most f

x
b


ty for cudagraph captur
 a
d comp

at
o
 for a
 k

ds of startup/p
rforma
c
 trad
offs a
d f
atur
 support.
## `CudagraphMod
s`
[CUDAGraphMod
][v
m.co
f
g.comp

at
o
.CUDAGraphMod
] 
s th
 s

g

 k
ob you tu

 

 `Comp

at
o
Co
f
g.cudagraph_mod
`:
* `NONE` — tur
 CUDA Graphs off. Good for d
bugg

g.
* `PIECEWISE` —  a s

g

-mod
 strat
gy (a
d past d
fau
t). It 
s th
 most f

x
b

: att

t
o
 or oth
r CUDA Graphs-

compat
b

 op
rat
o
s stay 
ag
r, 
v
ryth

g 

s
 go
s 

to CUDA Graphs. R
qu
r
s p

c


s
 comp

at
o
.
* `FULL` — a s

g

-mod
 strat
gy, 
h
ch o

y captur
s fu
 CUDA Graphs for 
o
-u

form batch
s, th

 u

form-d
cod
 batch
s r
us
 th
 CUDA Graph of 
o
-u

form batch of th
 sam
 batch_s
z
, s

c
 th
y ar
 compat
b

; ca
 b
 good for sma
 mod

s or 
ork
oads 

th sma
 prompts.
* `FULL_DECODE_ONLY` — fu
 CUDA Graph for u

form d
cod
, 
o cudagraph for pr
f

/m
x
d 
tc.; su
tab

 for d
cod
 

sta
c
s 

 a P/D s
tup 
h
r
 pr
f

 
s 
ot as 
mporta
t, th
s 
ay 

 ca
 sav
 th
 m
mory 

d
d for `PIECEWISE` CUDA Graphs.
* `FULL_AND_PIECEWISE` — (d
fau
t mod
) fu
 CUDA Graph for u

form d
cod
, p

c


s
 CUDA Graphs for oth
rs; g


ra
y th
 most p
rforma
t s
tt

g, 
sp
c
a
y for 
o
 
at

cy 

th sma
 mod

s or MoEs, but a
so r
qu
r
s th
 most m
mory a
d tak
s th
 
o
g
st to captur
.
D
fau
ts: If you’r
 o
 v1 

th p

c


s
 comp

at
o
, 

 d
fau
t to `FULL_AND_PIECEWISE` for b
tt
r p
rforma
c
, (for poo


g mod

s, 
t's st

 `PIECEWISE`). Oth
r

s
, 
.g. 
f p

c


s
 comp

at
o
 u
ava

ab

, 

 d
fau
t to `NONE`.
Wh


 `NONE` , `PIECEWISE`, a
d `FULL` ar
 s

g

-mod
 co
f
gurat
o
s a
d s
mp
y 
qu
va


t to past 
mp

m

tat
o
s of 
ag
r 
x
cut
o
, p

c


s
 CUDA Graphs, a
d fu
 CUDA Graphs r
sp
ct
v

y, `FULL_DECODE_ONLY` a
d `FULL_AND_PIECEWISE` ar
 



y app

d
d dua
-mod
 co
f
gurat
o
s, 
h
ch r
qu
r
 d
spatch

g to s

tch b
t


 co
cr
t
 ru
t
m
 mod
s accord

g to ru
t
m
 batch
s dy
am
ca
y.
!!! 
ot

    H
r
, th
 s

g

-mod
s `NONE`, `PIECEWISE`, a
d `FULL` ar
 tr
at
d as th
 ru
t
m
 mod
s for CUDA Graphs d
spatch

g. If us

g a dua
-mod
, th
 d
spatch
r 


 a

ays d
spatch to o

 of 
ts m
mb
r mod
s (p
us a pot

t
a
 `NONE` 
f 
o su
tab

 CUDA Graph ava

ab

), d
p

d

g o
 th
 batch compos
t
o
.
Wh


 cascad
 att

t
o
 
s 
ot cudagraph compat
b

, 
t 
s 
o
 compat
b

 

th a
 poss
b

 cudagraph mod
 co
f
gurat
o
s. If a batch us
s cascad
 att

t
o
, 
t a

ays g
ts d
spatch
d to `PIECEWISE` mod
 
f ava

ab

 (oth
r

s
 `NONE`).
!!! 
ot

    Not a
 CUDA Graph mod
s ar
 compat
b

 

th 
v
ry att

t
o
 back

d. W
 automat
ca
y "do

grad
" mod
s to th
 c
os
st support
d mod
. For 
xamp

, 
f a back

d o

y supports CUDA Graphs for pur
 d
cod
/u

form batch
s, 

 co
v
rt `FULL` to `FULL_AND_PIECEWISE` 
f p

c


s
 comp

at
o
 
s 

ab

d, a
d `FULL_DECODE_ONLY` oth
r

s
.
## D
ta


d D
s
g

### Ov
rv



Th
 


 CUDA Graphs 
og
c 
s bu

t o
 top of p

c


s
 comp

at
o
 a
d supports dua
 CUDA Graphs ru
t
m
 mod
 s

tch

g. Th
 syst
m co
ta

s th
 fo
o


g cor
 compo


ts:
* [CUDAGraphWrapp
r][v
m.comp

at
o
.cuda_graph.CUDAGraphWrapp
r]: 
rapp
r that ha
d

s CUDAGraph captur
 & r
p
ay o
 th
 
rapp
d ca
ab


* [CudagraphD
spatch
r][v
m.v1.cudagraph_d
spatch
r.CudagraphD
spatch
r]: th
 c

tra
 co
tro

r that co
ta

s th
 s

g

 sourc
 of truth about CUDA Graphs a
d ha
d

s d
spatch

g b
t


 th
m.
* [CUDAGraphMod
][v
m.co
f
g.comp

at
o
.CUDAGraphMod
]: 

um d
scr
b

g th
 support
d a
d ru
t
m
 mod
s (

troduc
d abov
).
* [BatchD
scr
ptor][v
m.for
ard_co
t
xt.BatchD
scr
ptor], s
rv

g as a u

qu
 r
pr
s

tat
o
 of th
 ru
t
m
 batch us
d for d
spatch

g.
S
 th
 fo
o


g f
gur
s for a qu
ck compar
so
 b
t


 th
 pr
v
ous a
d curr

t d
s
g
 patt
r
s of CUDA Graphs 

th 

ductor comp

at
o
. W
 ca
 s
 that pr
v
ous
y th
 CUDA Graphs 
og
c a
d comp

at
o
 
og
c 

r
 t
ght
y coup

d 

to th
 v
m `P

c


s
Back

d`, a
d CUDA Graphs 
as 
mp

c
t
y d
spatch
d by `batch_s
z
` 
d
y. No
 th
 CUDA Graphs 
og
c 
s s
parat
d 

to th
 `CUDAGraphWrapp
r` c
ass, r
spo
s
b

 for both fu
 a
d p

c


s
 CUDA Graphs ab


t

s, a
d d
spatch

g 
s **
xp

c
t
y** do

 v
a **ru
t
m
 mod
** p
us th
 `BatchD
scr
ptor` as th
 **d
spatch k
y** v
a `CudagraphD
spatch
r`.
**B
for
:**
![pr
v
ous_d
s
g
](../ass
ts/d
s
g
/cuda_graphs/pr
v
ous_d
s
g
.p
g)
**Aft
r:**
![


_d
s
g
](../ass
ts/d
s
g
/cuda_graphs/curr

t_d
s
g
.p
g)
### `BatchD
scr
ptor`
[BatchD
scr
ptor][v
m.for
ard_co
t
xt.BatchD
scr
ptor] 
s a compo


t 

th

 `For
ardCo
t
xt`, a
o
gs
d
 th
 CUDA Graphs ru
t
m
 mod
s, s
rv

g as th
 cor
 structur
 for d
spatch

g k
ys at ru
t
m
. Th
 prototyp
 
s:
```pytho

c
ass BatchD
scr
ptor(Nam
dTup

):
    
um_tok

s: 

t
    
um_r
qs: 

t
    u

form: boo
 = Fa
s

    has_
ora: boo
 = Fa
s

```

h
r
 `
um_tok

s` ca
 b
 th
 padd
d tok

 


gth, a
d `u

form` 

d
cat
s 
f a
 th
 r
qu
sts hav
 th
 sam
 qu
ry 


gths. Ma
y att

t
o
 back

ds o

y support fu
 cudagraphs 
h

 th
 batch
s ar
 u

form; pur
 d
cod
 batch
s ar
 u

form but may 
ot b
 qu
ry 


gth 1 (
.
. `
um_tok

s == 
um_r
qs`), th
s occurs 

 th
 va

dat
o
 pass of sp
c-d
cod
 
h
r
 "d
cod
" batch
s 


 hav
 a qu
ry 


gth of  `1+
um_sp
c_tok

s`.
Th
 goa
 of th
s structur
 
s to u

qu

y 
d

t
fy a (padd
d) batch 

th m


ma
 poss
b

 
t
ms corr
spo
d

g to a CUDA Graphs 
t
m.
!!! 
ot

    Th
 prototyp
 of `BatchD
scr
ptor` may b
 
xt

d
d for mor
 g


ra
 s
tuat
o
s 

 th
 futur
, 
.g., 

c
ud
 mor
 
t
ms, 

k
 `u

form_qu
ry_


` to support mu
t
p

 d
ff
r

t u

form d
cod
 


gths s
tt

gs (
https://g
thub.com/v
m-proj
ct/v
m/pu
/23679
), or oth
r mod
f
cat
o
s 

d
d to support CUDA Graphs for mod

s 
hos
 

puts ar
 
ot 

c
ssar

y tok

 


gth a
ar
 (for 
xamp

, som
 mu
t
-moda
 

puts).
### `CudagraphD
spatch
r`
Th
 [CudagraphD
spatch
r][v
m.v1.cudagraph_d
spatch
r.CudagraphD
spatch
r] tak
s r
spo
s
b


ty for ma

ta



g t
o s
ts of va

d d
spatch

g k
ys, o

 s
t for `FULL` ru
t
m
 mod
 a
d o

 s
t for `PIECEWISE` ru
t
m
 mod
, a
d d
spatch
s th
 corr
ct ru
t
m
 mod
 a
d th
 d
spatch

g k
ys b
for
 
x
cut

g th
 mod

's for
ards. It 


 tak
 

 th
 


t
a
 k
y (a rough batch_d
scr
ptor for th
 padd
d 

put) a
d r
tur
 th
 s


ct
d ru
t
m
 mod
 a
d th
 f

a
 batch_d
scr
ptor, th

 t

 th
 CUDAGraphWrapp
r 

sta
c
s that d
c
s
o
 through for
ard co
t
xts. Not
c
 that `CudagraphD
spatch
r` 
s th
 o

y sourc
 of truth for ava

ab

 CUDA Graph k
ys a
d `CUDAGraphWrapp
r` 

sta
c
s ca
 b


d
y trust th
 for
ard co
t
xt o
 
hat CUDA Graphs to d
spatch to. Th
s 

ts us s
mp

fy th
 
rapp
r cod
 a
d c

tra

z
 th
 
og
c 

 th
 d
spatch
r.
Th
 d
spatch

g k
ys ar
 


t
a

z
d through th
 d
spatch
r's `


t
a

z
_cudagraph_k
ys` m
thod, 
h
ch 
s ca

d by th
 gpu_mod

_ru

r aft
r a
 poss
b

 att

t
o
 back

ds ar
 


t
a

z
d. Th
s 
s 
h
r
 

 ca
 g
t much fa
c

r 

 th
 futur
 a
d “pr
par
” a
 k

ds of CUDA Graphs comb

at
o
s. For 
o
, 

 just app

d ava

ab

 k
ys bas
d o
 th
 va

d combos of `d
cod
_mod
`/`m
x
d_mod
` of `cudagraph_mod
` a
d `cudagraph_captur
_s
z
s` 

 th
 comp

at
o
 co
f
g.
Th
 d
spatch cod
 
ooks 

k
:
```pytho

batch_d
scr
ptor=BatchD
scr
ptor(
um_tok

s=
um_

put_tok

s, u

form_d
cod
=...)
ru
t
m
_mod
, batch_d
scr
ptor = cudagraphd
spatch
r.d
spatch(batch_d
scr
ptor)
# 
x
cut
o



th s
t_for
ard_co
t
xt(
    ...,
    cudagraph_ru
t
m
_mod
=ru
t
m
_mod
,
    batch_d
scr
ptor=batch_d
scr
ptor,
):
     output = s

f.mod

(...)
```
I
s
d
 th
 `d
spatch()` m
thod, th
 d
spatch
r 


 s
arch th
 prop
r CUDA Graphs ru
t
m
 mod
 a
d 
x
st

g d
spatch

g k
ys for a r
tur
. W
 bas
ca
y s
arch th
 
x
st

g k
ys fo
o


g th
 pr
or
ty: `FULL`
`PIECEWISE`
`No

`. If th
 d
spatch

g k
y do
s 
ot 
x
st, d
fau
t to r
tur
 `NONE` mod
 for 
ag
r 
x
cut
o
. Th
 
mp

m

tat
o
s ca
 b
 fou
d [h
r
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/v1/cudagraph_d
spatch
r.py#L91).
H
r
 
s a s
mp

f

d 

ustrat
o
 of th
 
orkf
o
 at ru
t
m
 

 th
 mod

 
x
cutor:
![
x
cutor_ru
t
m
](../ass
ts/d
s
g
/cuda_graphs/
x
cutor_ru
t
m
.p
g)
### `CUDAGraphWrapp
r`
A [CUDAGraphWrapp
r][v
m.comp

at
o
.cuda_graph.CUDAGraphWrapp
r] 

sta
c
 
raps a ru
ab

 a
d s
mp
y m
m
cs th
 ru
ab

 

th app

d
d CUDA Graphs ab


t

s. Each 
rapp
r 

sta
c
 
s bou
d to a sp
c
f
c `ru
t
m
_mod
`, 
h
ch 
s r
str
ct
d to `PIECEWISE` a
d `FULL` mod
, a
d tak
s r
spo
s
b


ty for captur

g/r
p
ay

g a
d pass

g through (d
r
ct
y ca


g) th
 ru
ab

.  At ru
t
m
, 
ach 
rapp
r 
ou
d:
1. 

sp
ct th
 ru
t
m
_mod
 a
d batch_d
scr
ptor(d
spatch

g k
y) from th
 g
oba
 for
ard co
t
xt.
2. If ru
t
m
_mod
 
s `NONE` or ru
t
m
_mod
 do
s 
ot match th
 mod
 of th
 
rapp
r, just ca
 th
 ru
ab

 d
r
ct
y.
3. Oth
r

s
, 
.
., th
 ru
t
m
_mod
 match
s th
 mod
 of th
 
rapp
r, th
 
rapp
r 


 p
rform CUDA Graphs captur
 (
f k
y do
s 
ot 
x
st, cr
at

a 


 

try a
d cach
 
t) or r
p
ay (
f k
y 
x
sts 

 th
 cach
).
Th
 abov
 st
ps ar
 bas
d o
 th
 assumpt
o
 that th
 CUDA Graphs 
rapp
r 
ou
d d
r
ct
y trust 
hat’s 

 th
 for
ard co
t
xt (co
tro

d by th
 d
spatch
r). Th
s 

ts us s
mp

fy a
d c

tra

z
 th
 
og
c, r
duc

g th
 comp

x
ty as 


 as th
 r
sk of m
smatch
d stat
 b
t


 th
 
rapp
rs a
d th
 d
spatch
r. It a
so a
o
s r
us

g th
 
rapp
r c
ass for both `FULL` a
d `PIECEWISE` ru
t
m
 mod
s. S
 th
 
mp

m

tat
o
 [h
r
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/f751
50b7a2aa
3110d83
d0d88202fc91b3
78a/v
m/comp

at
o
/cuda_graph.py#L106).
#### N
st
d Wrapp
r d
s
g

Th
 cor
 m
cha

sm of mak

g a fu
 CUDA Graphs a
d p

c


s
 CUDA Graphs co
x
st a
d compat
b

 
s th
 

st
d CUDA Graphs 
rapp
r d
s
g
, bu

d

g o
 top of p

c


s
 comp

at
o
 

th o

y a s

g

 p

c


s
 FX graph.  W
 
rap a FULL mod
 
rapp
r outs
d
 th
 

t
r
 mod

 for th
 fu
 CUDA Graphs fu
ct
o
a

ty; m
a

h


, 
ach p

c


s
 back

d 
s 
rapp
d v
a a `PIECEWISE` mod
 
rapp
r 

s
d
 th
 comp

at
o
.
Th
 f
o
 chart b

o
 shou
d c

ar
y d
scr
b
 ho
 
t 
orks.
![
rapp
r_f
o
](../ass
ts/d
s
g
/cuda_graphs/
rapp
r_f
o
.p
g)
Th
r
for
, for a `FULL` ru
t
m
 mod
, 
t 
s saf
 to captur
/r
p
ay a fu
 CUDA Graph s

c
 th
 p

c


s
 
rapp
r 
s 
ot act
vat
d. Th
 s
tuat
o
 
s s
m

ar for `PIECEWISE` mod
, as th
r
 ar
 
o co
f

cts b
t


 th
 `FULL` mod
 
rapp
r a
d `PIECEWISE` mod
 
rapp
rs.  For th
 `NONE` ru
t
m
 mod
, both `FULL` a
d `PIECEWISE` 
rapp
rs 
ou
d 
ot b
 act
vat
d, so 

 s
mp
y fa
 through to 
ag
r 
x
cut
o
.
### Fu
 CUDA Graph captur

g & 
arm-up
Th
 CUDA Graphs captur

g happ

s 
h

 th
 ru

r f
rst ca
s th
 mod

 for
ard (us

g `_dummy_ru
`) 

th a 
o
-`NONE` ru
t
m
 mod
. For fu
 CUDA Graph captur
, 

 
xp

c
t
y captur
 d
ff
r

t cas
s (
.
., pr
f

/m
x
d batch or u

form_d
cod
 batch) by prop
r
y s
tt

g att

t
o
 m
tadata to mak
 sur
 th
 u
d
r
y

g att

t
o
 back

ds 
au
ch th
 d
s
r
d k
r


 rout


s. To d
st

gu
sh pr
f

/m
x
d batch or u

form_d
cod
 batch, th
 most 
mporta
t prop
rty 
s th
 `max_qu
ry_


` 

 att
_m
tadata (tru
 for most att

t
o
 back

ds). W
 s
t 
t to th
 d
s
r
d `u

form_qu
ry_


` for u

form_d
cod
 oth
r

s
 

 mak
 
t just th
 `
um_tok

s` for a 
o
-u

form_d
cod
 batch.
Th
 CUDA Graphs 
rapp
r 
o 
o
g
r ma
ag
s th
 
arm-up 
og
c. Th
 
arm-up proc
ss 
s 
o
 co
tro

d d
r
ct
y by th
 GPU mod

 ru

r, 
h
r
 th
 `NONE` ru
t
m
 mod
 
s ass
g

d to p
ay a
 
ag
r 
x
cut
o
 for 
arm-up. Wh

 
arm

g up for a fu
 CUDA Graph, 
t 
s a
so 
mporta
t to 
xp

c
t
y ru
 att

t
o
 dur

g th
 
armup `dummy_ru
` ca
.
## CUDA Graphs Compat
b


ty of Att

t
o
 Back

ds
To s
g
a
 th
 CUDA Graphs compat
b


ty of th
 att

t
o
 back

ds, 

 

troduc
 a 


 

um typ
 [Att

t
o
CGSupport][v
m.v1.att

t
o
.back

d.Att

t
o
CGSupport], 
h
ch 
s a
 

um typ
 that tracks th
 capab


ty of th
 att

t
o
 back

d to support CUDA Graphs. Th
 va
u
 
s sort
d 

 th
 ord
r of th
 capab


ty, 
.
., `ALWAYS`
 `UNIFORM_BATCH`
 `UNIFORM_SINGLE_TOKEN_DECODE`
 `NEVER`.
```pytho

c
ass Att

t
o
CGSupport(

um.E
um):
    """ Co
sta
ts for th
 CUDA Graphs support of th
 att

t
o
 back

d
    H
r
 

 do 
ot co
s
d
r th
 cascad
 att

t
o
, as curr

t
y
    
t 
s 

v
r CUDA Graphs support
d."""
    ALWAYS = 3
    """CUDA Graphs a

ays support
d; supports m
x
d-pr
f

-d
cod
"""
    UNIFORM_BATCH = 2
    """CUDA Graphs support
d for batch
s th
 o

y co
ta

 qu
ry 


gths that ar

    th
 sam
, th
s ca
 b
 us
d for sp
c-d
cod

        
.
. "d
cod
s" ar
 1 + 
um_sp
cu
at
v
_tok

s"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """CUDA Graphs support
d for batch
s th
 o

y co
ta

 qu
ry_


==1 d
cod
s"""
    NEVER = 0
    """NO CUDA Graphs support"""
```
Suppos
 

 hav
 hybr
d att

t
o
 back

ds (
.g., 

 mamba m
x
r mod

s). I
 that cas
, 

 s
k th
 m


mum capab


ty of a
 back

ds to d
t
rm


 th
 f

a
 capab


ty of th
 mod

, a
d 

 m
ght r
so
v
 th
 

compat
b

 CUDA Graphs mod
 by do

grad

g th
 mod
 to th
 b
st f
t o

. For 
xamp

, do

grad

g `FULL` mod
 to `FULL_AND_PIECEWISE` mod
 
f th
 m


mum capab


ty 
s `UNIFORM_BATCH`, or `PIECEWISE` mod
 
f th
 m


mum capab


ty 
s `NEVER` for -O3 comp

at
o
 mod
. For th
 comp

t
 fa
back po

cy, p

as
 s
 th
 cod
 for [th
s][v
m.v1.
ork
r.gpu_mod

_ru

r.GPUMod

Ru

r._ch
ck_a
d_updat
_cudagraph_mod
].
Th
 fo
o


g tab

 

sts back

ds that support fu
 CUDA Graphs at th
 t
m
 of 
r
t

g.
| Att

t
o
 Back

d | cudagraph_support | Comm

ts |
|:---|:---|:---|
| F
ashAtt

t
o
 v2 | `UNIFORM_BATCH` | Actua
y `ALWAYS` but 
orkarou
d to fa
back to `FULL_AND_PIECEWISE` for p
rforma
c
 r
aso
 |
| F
ashAtt

t
o
 v3 | `ALWAYS` | has u

f

d rout


 for both batch
s, so `FULL` mod
 
s good |
| Tr
to
 Att

t
o
 | `ALWAYS` | pr
f
r `FULL_AND_PIECEWISE` s

c
 
t has d
ff
r

t k
r


s for pr
f

/m
x
d a
d pur
 d
cod
 batch
s |
| AITER F
ashAtt

t
o
 | `UNIFORM_BATCH`| |
| F
ashI
f
r | `UNIFORM_SINGLE_TOKEN_DECODE` | W

 b
 s
t to `UNIFORM_BATCH` 
h

 us

g TRTLLM att

t
o
 o
 B
ack


 |
| F
ashMLA | `UNIFORM_BATCH` | |
| F
ashI
f
rMLA | `UNIFORM_BATCH` | |
| F
ashI
f
rMLASpars
 | `UNIFORM_BATCH` | |
| AITER MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| Mamba att

t
o
| `UNIFORM_SINGLE_TOKEN_DECODE` | |
U


st
d back

ds ar
 a
 d
c
ar
d as `NEVER`.
## Usag
 gu
d

No
 th
 CLI 
s d
r
ct
y us

g th
 upp
rcas
 str

g of cudagraph_mod
 for comp

at
o
_co
f
g: `--comp

at
o
-co
f
g '{"cudagraph_mod
": "..."}'`, 
h
r
 `...` shou
d b
 o

 of `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, a
d `FULL_AND_PIECEWISE`. Not
 that a
 `PIECEWISE` r

at
d mod
s r
qu
r
 p

c


s
 comp

at
o
, a
d a
 `FULL` r

at
d mod
s 

d CUDA Graphs support of att

t
o
 back

ds. For 
xamp

:
```bash
v
m s
rv
 --mod

 m
ta-
ama/L
ama-3.1-8B-I
struct --comp

at
o
-co
f
g '{"cudagraph_mod
": "FULL_AND_PIECEWISE"}'
```
### Pytho
 
xamp

s
```pytho


mport os
os.

v
ro
.s
td
fau
t("VLLM_LOGGING_LEVEL", "DEBUG")

mport v
m
from v
m.co
f
g 
mport CUDAGraphMod

comp

at
o
_co
f
g = {"mod
": 3, "cudagraph_mod
": "FULL_AND_PIECEWISE"}
mod

 = v
m.LLM(
    mod

="m
ta-
ama/L
ama-3.1-8B-I
struct",
    dtyp
="auto",
    comp

at
o
_co
f
g=comp

at
o
_co
f
g,
)
samp


g_params = v
m.Samp


gParams(
    t
mp
ratur
=0,  # gr
dy d
cod

g
    max_tok

s=1024,
)
outputs = mod

.g


rat
(
    ["My 
am
 
s Joh
 a
d"],
    samp


g_params=samp


g_params,
)
```
### P

c


s
 comp

at
o
 a
d fu
 graph custom pass
s (att

t
o
 fus
o
, s
qu

c
 para



sm)
U
fortu
at

y, som
 custom comp


 pass
s hav
 to s
 th
 
ho

 graph to b
 
ff
ct
v
 a
d h

c
 ar

't compat
b

 

th p

c


s
 comp

at
o
. Th
s 

c
ud
s `Att
Fus
o
Pass` a
d `S
qu

c
Para



smPass`. As a short-t
rm so
ut
o
, 

 automat
ca
y d
sab

 p

c


s
 comp

at
o
 (by s
tt

g `sp

tt

g_ops=[]`) 
h

 att

t
o
 fus
o
 
s 

ab

d. W
 us
 CUDA Graph mod
s `FULL` or `FULL_DECODE_ONLY` (d
p

d

g o
 back

d support). Ho

v
r, th
s 

ads to a
oth
r opt
m
zat
o
 

compat
b


ty a
d co
fus

g p
rforma
c
 trad
offs.
Lo
g t
rm, 

'v
 add
d th
 ab


ty to part
t
o
 th
 graph 

 I
ductor 

st
ad of r
ght aft
r Dy
amo. It ca
 b
 

ab

d 

th `Comp

at
o
Co
f
g.us
_

ductor_graph_part
t
o
=Tru
` but 
s curr

t
y 
xp
r
m

ta
 a
d o

y ava

ab

 

th `torch
=2.9`. Th
s a
so 

cr
as
s comp

at
o
 t
m
 as 
t has to comp


 th
 
ho

 graph a
d ca
ot r
us
 p

c


s
 comp

at
o
 art
facts. O
c
 vLLM supports 2.9, 

 p
a
 to mak
 th
s th
 d
fau
t approach as 
t 


 a
so sp
d up p

c


s
 cudagraph captur
.
## About th
 P
rforma
c

S
 th
 fo
o


g 


ks for 
xamp

s:
* [20059#
ssu
comm

t-3160858458](https://g
thub.com/v
m-proj
ct/v
m/pu
/20059#
ssu
comm

t-3160858458)
* [20059#
ssu
comm

t-3188735226](https://g
thub.com/v
m-proj
ct/v
m/pu
/20059#
ssu
comm

t-3188735226)
* [20059#
ssu
comm

t-3219888738](https://g
thub.com/v
m-proj
ct/v
m/pu
/20059#
ssu
comm

t-3219888738)
