# Fus
d MoE K
r


 F
atur
s
Th
 purpos
 of th
s docum

t 
s to prov
d
 a
 ov
rv


 of th
 var
ous MoE k
r


s (both modu
ar a
d 
o
-modu
ar) so 
t 


 b
 
as

r to s


ct a
 appropr
at
 s
t of k
r


s for a
y part
cu
ar s
tuat
o
. Th
s 

c
ud
s 

format
o
 about th
 a
2a
 back

ds us
d by modu
ar k
r


s.
## Fus
d MoE Modu
ar A
2A
 back

ds
Th
r
 ar
 a 
umb
r of a
2a
 commu

cat
o
 back

ds that ar
 us
d to 
mp

m

t 
xp
rt para



sm (EP) for th
 `Fus
dMoE` 
ay
r. Th
 d
ff
r

t `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` subc
ass
s prov
d
 a
 

t
rfac
 for 
ach a
2a
 back

d.
Th
 fo
o


g tab

 d
scr
b
s th
 r


va
t f
atur
s of 
ach back

d, 
.
. act
vat
o
 format, support
d qua
t
zat
o
 sch
m
s a
d asy
c support.
Th
 output act
vat
o
 format (sta
dard or batch
d) corr
spo
ds to th
 output of th
 pr
par
 st
p of th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` subc
ass, a
d th
 f

a

z
 st
p r
qu
r
s th
 sam
 format. A
 th
 back

d `pr
par
` m
thods 
xp
ct act
vat
o
s 

 th
 sta
dard format a
d a
 th
 `f

a

z
` m
thods r
tur
 act
vat
o
s 

 sta
dard format. Mor
 d
ta

s o
 th
 formats ca
 b
 fou
d 

 th
 [Fus
d MoE Modu
ar K
r


](./fus
d_mo
_modu
ar_k
r


.md) docum

t.
Th
 qua
t
zat
o
 typ
s a
d formats 

um
rat
 
h
ch qua
t
zat
o
 sch
m
s ar
 support
d by 
ach `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` c
ass. Th
 qua
t
zat
o
 ca
 happ

 b
for
 or aft
r th
 d
spatch bas
d o
 th
 format th
 a
2a
 back

d supports, 
.g. d
p
p_h
gh_throughput supports o

y b
ock-qua
t
z
d fp8 format. A
y oth
r format 


 r
su
t 

 d
spatch

g 

 h
gh
r pr
c
s
o
 a
d qua
t
z

g aft
r
ards. Th
 output of th
 pr
par
 st
p for 
ach back

d 
s th
 qua
t
z
d typ
. Th
 f

a

z
 st
p g


ra
y r
qu
r
s th
 sam
 

put typ
 as th
 or
g

a
 act
vat
o
s, 
.g. 
f th
 or
g

a
 

put 
s bf
oat16 a
d th
 qua
t
zat
o
 sch
m
 
s fp8 

th p
r-t

sor sca

s, `pr
par
` 


 r
tur
 fp8/p
r-t

sor sca

 act
vat
o
s a
d `f

a

z
` 


 tak
 bf
oat16 act
vat
o
s. S
 th
 d
agrams 

 [Fus
d MoE Modu
ar K
r


](./fus
d_mo
_modu
ar_k
r


.md) for mor
 d
ta

s o
 th
 typ
s a
d formats of act
vat
o
s at 
ach st
p of th
 MoE proc
ss. If 
o qua
t
zat
o
 typ
 
s sp
c
f

d, th
 k
r


 op
rat
s o
 f
oat16 a
d/or bf
oat16.
Asy
c back

ds support th
 us
 of DBO (Dua
 Batch Ov
r
ap) a
d shar
d 
xp
rt ov
r
ap (
h
r
 shar
d 
xp
rts ar
 comput
d dur

g th
 comb


 st
p).
C
rta

 mod

s r
qu
r
 th
 topk 


ghts to b
 app


d to th
 

put act
vat
o
s rath
r tha
 th
 output act
vat
o
s 
h

 topk==1, 
.g. L
ama. For modu
ar k
r


s, th
s f
atur
 
s support
d by th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` subc
ass. For 
o
-modu
ar k
r


s, 
t 
s up to th
 
xp
rts fu
ct
o
 to d
a
 

th th
s f
ag.
U


ss oth
r

s
 sp
c
f

d, back

ds ar
 co
tro

d v
a th
 `--a
2a
-back

d` comma
d-



 argum

t (or th
 `a
2a
_back

d` param
t
r 

 `Para


Co
f
g`). A
 back

ds 
xc
pt `f
ash

f
r` o

y 
ork 

th EP+DP or EP+TP. `F
ash

f
r` ca
 
ork 

th EP or DP 

thout EP.
sty



td {
  padd

g: 0.5r
m !
mporta
t;
  
h
t
-spac
: 
o
rap;
}
th {
  padd

g: 0.5r
m !
mporta
t;
  m

-

dth: 0 !
mporta
t;
}
/sty



| Back

d | Output act. format | Qua
t. typ
s | Qua
t. format | Asy
c | App
y W

ght O
 I
put | Subc
ass |
|---------|--------------------|--------------|---------------|-------|-----------------------|-----------|
| 
a
v
 | sta
dard | a

sup
1
/sup
 | G,A,T | N | 
sup
6
/sup
 | [
ay
r.py][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.
ay
r.Fus
dMoE] |
| d
p
p_h
gh_throughput | sta
dard | fp8 | G(128),A,T
sup
2
/sup
 | Y | Y | [`D
pEPHTPr
par
A
dF

a

z
`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.d
p
p_ht_pr
par
_f

a

z
.D
pEPHTPr
par
A
dF

a

z
] |
| d
p
p_
o
_
at

cy | batch
d | fp8 | G(128),A,T
sup
3
/sup
 | Y | Y | [`D
pEPLLPr
par
A
dF

a

z
`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.d
p
p_
_pr
par
_f

a

z
.D
pEPLLPr
par
A
dF

a

z
] |
| f
ash

f
r_a
2a
v | sta
dard | 
vfp4,fp8 | G,A,T | N | N | [`F
ashI
f
rA2APr
par
A
dF

a

z
`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.f
ash

f
r_a2a_pr
par
_f

a

z
.F
ashI
f
rA2APr
par
A
dF

a

z
] |
!!! 

fo "Tab

 k
y"
    1. A
 typ
s: mxfp4, 
vfp4, 

t4, 

t8, fp8
    2. A,T qua
t
zat
o
 occurs aft
r d
spatch.
    3. A
 qua
t
zat
o
 happ

s aft
r d
spatch.
    4. Co
tro

d by d
ff
r

t 

v vars (`VLLM_FLASHINFER_MOE_BACKEND` "throughput" or "
at

cy")
    5. Th
s 
s a 
o-op d
spatch
r that ca
 b
 us
d to pa
r 

th a
y modu
ar 
xp
rts to produc
 a modu
ar k
r


 that ru
s 

thout d
spatch or comb


. Th
s
 ca
ot b
 s


ct
d v
a 

v
ro
m

t var
ab

. Th
s
 ar
 g


ra
y us
 for t
st

g or adapt

g a
 
xp
rt subc
ass to th
 `fus
d_
xp
rts` API.
    6. Th
s d
p

ds o
 th
 
xp
rts 
mp

m

tat
o
.
    ---
    - G - Group
d
    - G(N) - Group
d 
/b
ock s
z
 N
    - A - P
r act
vat
o
 tok


    - T - P
r t

sor
Modu
ar k
r


s ar
 support
d by th
 fo
o


g `Fus
dMoEM
thodBas
` c
ass
s.
    - [`Mod

OptFp8MoEM
thod`][v
m.mod

_
x
cutor.
ay
rs.qua
t
zat
o
.mod

opt.Mod

OptFp8MoEM
thod]
    - [`Fp8MoEM
thod`][v
m.mod

_
x
cutor.
ay
rs.qua
t
zat
o
.fp8.Fp8MoEM
thod]
    - [`Compr
ss
dT

sorsW4A4Nvfp4MoEM
thod`][v
m.mod

_
x
cutor.
ay
rs.qua
t
zat
o
.compr
ss
d_t

sors.compr
ss
d_t

sors_mo
.Compr
ss
dT

sorsW4A4Nvfp4MoEM
thod]
    - [`Compr
ss
dT

sorsW8A8Fp8MoEM
thod`][v
m.mod

_
x
cutor.
ay
rs.qua
t
zat
o
.compr
ss
d_t

sors.compr
ss
d_t

sors_mo
.Compr
ss
dT

sorsW8A8Fp8MoEM
thod]
    - [`Mxfp4MoEM
thod`][v
m.mod

_
x
cutor.
ay
rs.qua
t
zat
o
.mxfp4.Mxfp4MoEM
thod]
    - [`U
qua
t
z
dFus
dMoEM
thod`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.
ay
r.U
qua
t
z
dFus
dMoEM
thod]
## Fus
d Exp
rts K
r


s
Th
r
 ar
 a 
umb
r of MoE 
xp
rts k
r


 
mp

m

tat
o
s for d
ff
r

t qua
t
zat
o
 typ
s a
d arch
t
ctur
s. Most fo
o
 th
 g


ra
 API of th
 bas
 Tr
to
 [`fus
d_
xp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.fus
d_mo
.fus
d_
xp
rts] fu
ct
o
. Ma
y hav
 modu
ar k
r


 adapt
rs, so th
y ca
 b
 us
d 

th compat
b

 a
2a
 back

ds. Th
s tab

 

sts 
ach 
xp
rts k
r


 a
d 
ts part
cu
ar prop
rt

s.
Each k
r


 must b
 prov
d
d 

th o

 of th
 support
d 

put act
vat
o
 formats. Som
 f
avors of k
r


s support both sta
dard a
d batch
d formats through d
ff
r

t 

try po

ts, 
.g. `Tr
to
Exp
rts` a
d `Batch
dTr
to
Exp
rts`. Batch
d format k
r


s ar
 curr

t
y o

y 

d
d for match

g 

th c
rta

 a
2a
 back

ds, 
.g. `D
pEPLLPr
par
A
dF

a

z
`.
S
m

ar to th
 back

d k
r


s, 
ach 
xp
rts k
r


 o

y supports c
rta

 qua
t
zat
o
 formats. For 
o
-modu
ar 
xp
rts, th
 act
vat
o
s 


 b
 

 th
 or
g

a
 typ
 a
d qua
t
z
d 

t
r
a
y by th
 k
r


. Modu
ar 
xp
rts 


 
xp
ct th
 act
vat
o
s to a
r
ady b
 

 th
 qua
t
z
d format. Both typ
s of 
xp
rts 


 y


d outputs 

 th
 or
g

a
 act
vat
o
 typ
.
Each 
xp
rts k
r


 supports o

 or mor
 act
vat
o
 fu
ct
o
s, 
.g. s

u or g

u, 
h
ch ar
 app


d to th
 

t
rm
d
at
 r
su
ts.
As 

th th
 back

ds, som
 
xp
rts support app
y

g topk 


ghts o
 th
 

put act
vat
o
s. Th
 

tr

s 

 th
 co
um
 

 th
s tab

 o

y app
y to th
 
o
-modu
ar 
xp
rts.
Most 
xp
rts f
avors 

c
ud
 a
 
qu
va


t modu
ar 

t
rfac
 
h
ch 


 b
 a subc
ass of `Fus
dMoEExp
rtsModu
ar`.
To b
 us
d 

th a part
cu
ar `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` subc
ass, MoE k
r


s must hav
 compat
b

 act
vat
o
 formats, qua
t
zat
o
 typ
s a
d qua
t
zat
o
 formats.
| K
r


 | I
put act. format | Qua
t. typ
s | Qua
t. format | Act
vat
o
 fu
ct
o
 | App
y W

ght O
 I
put | Modu
ar | Sourc
 |
|--------|-------------------|--------------|---------------|---------------------|-----------------------|---------|--------|
| tr
to
 | sta
dard | a

sup
1
/sup
 | G,A,T | s

u, g

u,
/br
s

g
uoa
,
/br
s

u_
o_mu
,
/br
g

u_
o_mu
 | Y | Y | [`fus
d_
xp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.fus
d_mo
.fus
d_
xp
rts],
/br
[`Tr
to
Exp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.fus
d_mo
.Tr
to
Exp
rts] |
| tr
to
 (batch
d) | batch
d | a

sup
1
/sup
 | G,A,T | s

u, g

u | 
sup
6
/sup
 | Y | [`Batch
dTr
to
Exp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.fus
d_batch
d_mo
.Batch
dTr
to
Exp
rts] |
| d
p g
mm | sta
dard,
/br
batch
d | fp8 | G(128),A,T | s

u, g

u | 
sup
6
/sup
 | Y | 
/br
[`D
pG
mmExp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.d
p_g
mm_mo
.D
pG
mmExp
rts],
/br
[`Batch
dD
pG
mmExp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.batch
d_d
p_g
mm_mo
.Batch
dD
pG
mmExp
rts] |
| cut
ass_fp4 | sta
dard,
/br
batch
d | 
vfp4 | A,T | s

u | Y | Y | [`Cut
assExp
rtsFp4`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.cut
ass_mo
.Cut
assExp
rtsFp4] |
| cut
ass_fp8 | sta
dard,
/br
batch
d | fp8 | A,T | s

u, g

u | Y | Y | [`Cut
assExp
rtsFp8`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.cut
ass_mo
.Cut
assExp
rtsFp8],
/br
[`Cut
asBatch
dExp
rtsFp8`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.cut
ass_mo
.Cut
assBatch
dExp
rtsFp8] |
| f
ash

f
r | sta
dard | 
vfp4,
/br
fp8 | T | 
sup
5
/sup
 | N | Y | [`F
ashI
f
rExp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.f
ash

f
r_cut
ass_mo
.F
ashI
f
rExp
rts] |
| gpt oss tr
to
 | sta
dard | N/A | N/A | 
sup
5
/sup
 | Y | Y | [`tr
to
_k
r


_fus
d_
xp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.gpt_oss_tr
to
_k
r


s_mo
.tr
to
_k
r


_fus
d_
xp
rts],
/br
[`OAITr
to
Exp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.gpt_oss_tr
to
_k
r


s_mo
.OAITr
to
Exp
rts] |
| mar


 | sta
dard,
/br
batch
d | 
sup
3
/sup
 / N/A | 
sup
3
/sup
 / N/A | s

u,
/br
s

g
uoa
 | Y | Y | [`fus
d_mar


_mo
`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.fus
d_mar


_mo
.fus
d_mar


_mo
],
/br
[`Mar


Exp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.fus
d_mar


_mo
.Mar


Exp
rts],
/br
[`Batch
dMar


Exp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.fus
d_mar


_mo
.Batch
dMar


Exp
rts] |
| trt
m | sta
dard | mxfp4,
/br

vfp4 | G(16),G(32) | 
sup
5
/sup
 | N | Y | [`TrtL
mG

Exp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.trt
m_mo
.TrtL
mG

Exp
rts] |
| rocm a
t
r mo
 | sta
dard | fp8 | G(128),A,T | s

u, g

u | Y | N | [`rocm_a
t
r_fus
d_
xp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.rocm_a
t
r_fus
d_mo
.rocm_a
t
r_fus
d_
xp
rts] |
| cpu_fus
d_mo
 | sta
dard | N/A | N/A | s

u | N | N | [`CPUFus
dMOE`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.cpu_fus
d_mo
.CPUFus
dMOE] |
| 
a
v
 batch
d
sup
4
/sup
 | batch
d | 

t8,
/br
fp8 | G,A,T | s

u, g

u | 
sup
6
/sup
 | Y | [`Na
v
Batch
dExp
rts`][v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.fus
d_batch
d_mo
.Na
v
Batch
dExp
rts] |
!!! 

fo "Tab

 k
y"
    1. A
 typ
s: mxfp4, 
vfp4, 

t4, 

t8, fp8
    2. A d
spatch
r 
rapp
r arou
d tr
to
 a
d d
p g
mm 
xp
rts. W

 s


ct bas
d o
 typ
 + shap
 + qua
t
zat
o
 params
    3. u

t4, u

t8, fp8, fp4
    4. Th
s 
s a 
a
v
 
mp

m

tat
o
 of 
xp
rts that supports batch
d format. Ma


y us
d for t
st

g.
    5. Th
 `act
vat
o
` param
t
r 
s 
g
or
d a
d S

G
u 
s us
d by d
fau
t 

st
ad.
    6. O

y ha
d

d by or support
d 
h

 us
d 

th modu
ar k
r


s.
## Modu
ar K
r


 "fam



s"
Th
 fo
o


g tab

 sho
s "fam



s" of modu
ar k
r


s that ar
 

t

d
d to 
ork tog
th
r. Th
r
 ar
 som
 comb

at
o
s 
h
ch may 
ork but hav
 
ot y
t b

 t
st
d, 
.g. f
ash

f
r 

th oth
r fp8 
xp
rts. Not
 that th
 "
a
v
" back

d 


 
ork 

th a
y 
o
-modu
ar 
xp
rts.
| back

d | `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` subc
ass
s | `Fus
dMoEExp
rtsModu
ar` subc
ass
s |
|---------|-----------------------------------------|----------------------------------------------|
| d
p
p_h
gh_throughput | `D
pEPHTPr
par
A
dF

a

z
` |  `D
pG
mmExp
rts`,
/br
`Tr
to
Exp
rts`,
/br
`Tr
to
OrD
pG
mmExp
rts`,
/br
`Cut
assExp
rtsFp8`, 
/br
`Mar


Exp
rts` |
| d
p
p_
o
_
at

cy | `D
pEPLLPr
par
A
dF

a

z
` |  `Batch
dD
pG
mmExp
rts`,
/br
`Batch
dTr
to
Exp
rts`,
/br
`Cut
assBatch
dExp
rtsFp8`,
/br
`Batch
dMar


Exp
rts` |
| f
ash

f
r | `F
ashI
f
rCut
assMoEPr
par
A
dF

a

z
` | `F
ashI
f
rExp
rts` |
