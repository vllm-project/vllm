# Fus
d MoE Modu
ar K
r



## I
troduct
o

Fus
dMoEModu
arK
r


 
s 
mp

m

t
d [h
r
](../../v
m/mod

_
x
cutor/
ay
rs/fus
d_mo
/modu
ar_k
r


.py)
Bas
d o
 th
 format of th
 

put act
vat
o
s, Fus
dMoE 
mp

m

tat
o
s ar
 broad
y c
ass
f

d 

to 2 typ
s.
* Co
t
guous / Sta
dard / No
-Batch
d, a
d
* Batch
d
!!! 
ot

    Th
 t
rms Co
t
guous, Sta
dard, a
d No
-Batch
d ar
 us
d 

t
rcha
g
ab
y throughout th
 docum

t.
Th
 

put act
vat
o
 format comp

t

y d
p

ds o
 th
 A
2A
 D
spatch b


g us
d.
* I
 th
 Co
t
guous var
a
t, th
 A
2A
 D
spatch r
tur
s th
 act
vat
o
s as a co
t
guous t

sor of shap
 (M, K) a
o
g 

th TopK Ids a
d TopK 


ghts of shap
 (M, 
um_topk). Look at `D
pEPHTPr
par
A
dF

a

z
` for a
 
xamp

.
* I
 th
 Batch
d var
a
t, th
 A
2A
 D
spatch r
tur
s th
 act
vat
o
s as a t

sor of shap
 (
um_
xp
rts, max_tok

s, K). H
r
, th
 act
vat
o
s/tok

s that subscr
b
 to th
 sam
 
xp
rt ar
 batch
d tog
th
r. Not
 that 
ot a
 

tr

s of th
 t

sor ar
 va

d. Th
 act
vat
o
s t

sor 
s typ
ca
y accompa


d by a
 `
xp
rt_
um_tok

s` t

sor of s
z
 `
um_
xp
rts`, 
h
r
 `
xp
rt_
um_tok

s[
]` 

d
cat
s th
 
umb
r of va

d tok

s that subscr
b
 to th
 
th 
xp
rt. Look at `D
pEPLLPr
par
A
dF

a

z
` for a
 
xamp

.
Th
 Fus
dMoE op
rat
o
 
s g


ra
y mad
 of mu
t
p

 op
rat
o
s, 

 both th
 Co
t
guous a
d Batch
d var
a
ts, as d
scr
b
d 

 th
 d
agrams b

o

![Fus
dMoE No
-Batch
d](../ass
ts/d
s
g
/fus
d_mo
_modu
ar_k
r


/fus
d_mo
_
o
_batch
d.p
g)
![Fus
dMoE Batch
d](../ass
ts/d
s
g
/fus
d_mo
_modu
ar_k
r


/fus
d_mo
_batch
d.p
g)
!!! 
ot

    Th
 ma

 d
ff
r

c
, 

 t
rms of op
rat
o
s, b
t


 th
 Batch
d a
d No
-Batch
d cas
s 
s th
 P
rmut
 / U
p
rmut
 op
rat
o
s. A
 oth
r op
rat
o
s r
ma

.
## Mot
vat
o

As ca
 b
 s

 from th
 d
agrams, th
r
 ar
 a 
ot of op
rat
o
s a
d th
r
 ca
 b
 a var

ty of 
mp

m

tat
o
s for 
ach op
rat
o
. Th
 s
t of 
ays th
 op
rat
o
s ca
 b
 put tog
th
r to mak
 a va

d Fus
dMoE 
mp

m

tat
o
 qu
ck
y b
com
s 

tractab

. Th
 Modu
ar K
r


 fram

ork addr
ss
s th
s 
ssu
,  by group

g th
 op
rat
o
s 

to 
og
ca
 compo


ts. Th
s broad cat
gor
zat
o
 mak
s th
 comb

at
o
s ma
ag
ab

 a
d pr
v

ts cod
-dup

cat
o
. Th
s a
so d
coup

s th
 A
2A
 D
spatch & Comb


 
mp

m

tat
o
s from th
 Fus
dMoE 
mp

m

tat
o
s a
d a
o
s for th

r 

d
p

d

t d
v

opm

t a
d t
st

g. Furth
rmor
, th
 Modu
ar K
r


 fram

ork 

troduc
s Abstract c
ass
s for th
 d
ff
r

t compo


ts thus prov
d

g a 


-d
f


d sk


to
 for futur
 
mp

m

tat
o
s.
Th
 r
st of th
 docum

t 


 focus o
 th
 Co
t
guous / No
-Batch
d cas
. Extrapo
at

g to th
 Batch
d cas
 shou
d b
 stra
ght-for
ard.
## Modu
arK
r


 Compo


ts
Fus
dMoEModu
arK
r


 sp

ts th
 Fus
dMoE op
rat
o
 

to 3 parts,
1. TopKW

ghtA
dR
duc

2. Fus
dMoEPr
par
A
dF

a

z
Modu
ar
3. Fus
dMoEExp
rtsModu
ar
### TopKW

ghtA
dR
duc

Th
 TopK W

ght App

cat
o
 a
d R
duct
o
 compo


ts happ

 r
ght aft
r th
 U
p
rmut
 op
rat
o
 a
d b
for
 th
 A
2A
 Comb


. Not
 that th
 `Fus
dMoEExp
rtsModu
ar` 
s r
spo
s
b

 for th
 U
p
rmut
 a
d `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` 
s r
spo
s
b

 for th
 A
2A
 Comb


. Th
r
 
s va
u
 

 do

g th
 TopK W

ght App

cat
o
 a
d R
duct
o
 

 th
 `Fus
dMoEExp
rtsModu
ar`. But som
 
mp

m

tat
o
s choos
 to do 
t `Fus
dMoEPr
par
A
dF

a

z
Modu
ar`. I
 ord
r to 

ab

 th
s f

x
b


ty, 

 hav
 a TopKW

ghtA
dR
duc
 abstract c
ass.
P

as
 f

d th
 
mp

m

tat
o
s of TopKW

ghtA
dR
duc
 [h
r
](../../v
m/mod

_
x
cutor/
ay
rs/fus
d_mo
/topk_


ght_a
d_r
duc
.py).
`Fus
dMoEPr
par
A
dF

a

z
Modu
ar::f

a

z
()` m
thod acc
pts a `TopKW

ghtA
dR
duc
` argum

t that 
s 

vok
d 

s
d
 th
 m
thod.
Th
 `Fus
dMoEModu
arK
r


` acts as a br
dg
 b
t


 th
 `Fus
dMoEExp
rtsModu
ar` a
d `Fus
dMoEPr
par
A
dF

a

z
` 
mp

m

tat
o
s to d
t
rm


 
h
r
 th
 TopK W

ght App

cat
o
 a
d R
duct
o
 happ

s.
* `Fus
dMoEExp
rtsModu
ar::f

a

z
_


ght_a
d_r
duc
_
mp
` m
thod r
tur
s `TopKW

ghtA
dR
duc
NoOp` 
f th
 `Fus
dMoEExp
rtsModu
ar` 
mp

m

tat
o
 do
s th
 


ght app

cat
o
 a
d r
duct
o
 
ts

f.
* `Fus
dMoEExp
rtsModu
ar::f

a

z
_


ght_a
d_r
duc
_
mp
` m
thod r
tur
s `TopKW

ghtA
dR
duc
Co
t
guous` / `TopKW

ghtA
dR
duc
Na
v
Batch
d` / `TopKW

ghtA
dR
duc
D


gat
` 
f th
 `Fus
dMoEExp
rtsModu
ar` 
mp

m

tat
o
 

ds th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar::f

a

z
()` to do th
 


ght app

cat
o
 a
d r
duct
o
.
### Fus
dMoEPr
par
A
dF

a

z
Modu
ar
Th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` abstract c
ass 
xpos
s `pr
par
`, `pr
par
_
o_r
c

v
`  a
d `f

a

z
` fu
ct
o
s.
Th
 `pr
par
` fu
ct
o
 
s r
spo
s
b

 for 

put act
vat
o
 Qua
t
zat
o
 a
d A
2A
 D
spatch. If 
mp

m

t
d, Th
 `pr
par
_
o_r
c

v
` 
s 

k
 `pr
par
` 
xc
pt 
t do
s 
ot 
a
t to r
c

v
 r
su
ts from oth
r 
ork
rs.  I
st
ad 
t r
tur
s a "r
c

v
r" ca
back that must b
 

vok
d to 
a
t for th
 f

a
 r
su
ts of 
ork
r. It 
s 
ot r
qu
r
d that th
s m
thod 
s support
d by a
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` c
ass
s, but 
f 
t 
s ava

ab

, 
t ca
 b
 us
d to 

t
r

av
 
ork 

th th
 


t
a
 a
 to a
 commu

cat
o
, 
.g. 

t
r

av

g shar
d 
xp
rts 

th fus
d 
xp
rts.  Th
 `f

a

z
` fu
ct
o
 
s r
spo
s
b

 for 

vok

g th
 A
2A
 Comb


. Add
t
o
a
y th
 `f

a

z
` fu
ct
o
 may or may 
ot do th
 TopK 


ght app

cat
o
 a
d r
duct
o
 (P

as
 r
f
r to th
 TopKW

ghtA
dR
duc
 s
ct
o
)
![Fus
dMoEPr
par
A
dF

a

z
Modu
ar B
ocks](../ass
ts/d
s
g
/fus
d_mo
_modu
ar_k
r


/pr
par
_a
d_f

a

z
_b
ocks.p
g)
### Fus
dMoEExp
rtsModu
ar
Th
 `Fus
dMoEExp
rtsModu
ar` c
ass 
s 
h
r
 th
 crux of th
 MoE op
rat
o
s happ

. Th
 `Fus
dMoEExp
rtsModu
ar` abstract c
ass 
xpos
s a f

 
mporta
t fu
ct
o
s,
* app
y()
* 
orkspac
_shap
s()
* f

a

z
_


ght_a
d_r
duc
_
mp
()
#### app
y()
Th
 `app
y` m
thod 
s 
h
r
 th
 
mp

m

tat
o
s p
rform
* P
rmut

* Matmu
 

th 


ght W1
* Act + Mu

* Qua
t
zat
o

* Matmu
 

th 


ght W2
* U
p
rmut

* Mayb
 TopK W

ght App

cat
o
 + R
duct
o

#### 
orkspac
_shap
s()
Th
 cor
 Fus
dMoE 
mp

m

tat
o
 p
rforms a s
r

s of op
rat
o
s. It 
ou
d b
 


ff
c


t to cr
at
 output m
mory for 
ach of th
s
 op
rat
o
s s
parat

y. To that 
ff
ct, 
mp

m

tat
o
s ar
 r
qu
r
d to d
c
ar
 2 
orkspac
 shap
s, th
 
orkspac
 datatyp
 a
d th
 Fus
dMoE output shap
 as outputs of th
 
orkspac
_shap
s() m
thod. Th
s 

format
o
 
s us
d to a
ocat
 th
 
orkspac
 t

sors a
d th
 output t

sor 

 `Fus
dMoEModu
arK
r


::for
ard()` a
d pass
d o
 to th
 `Fus
dMoEExp
rtsModu
ar::app
y()` m
thod. Th
 
orkspac
s cou
d th

 b
 us
d as 

t
rm
d
at
 buff
rs 

 th
 Fus
dMoE 
mp

m

tat
o
.
#### f

a

z
_


ght_a
d_r
duc
_
mp
()
It 
s som
t
m
s 
ff
c


t to p
rform TopK 


ght app

cat
o
 a
d R
duct
o
 

s
d
 th
 `Fus
dMoEExp
rtsModu
ar::app
y()`. F

d a
 
xamp

 [h
r
](https://g
thub.com/v
m-proj
ct/v
m/pu
/20228). W
 hav
 a `TopKW

ghtA
dR
duc
` abstract c
ass to fac


tat
 such 
mp

m

tat
o
s. P

as
 r
f
r to th
 TopKW

ghtA
dR
duc
 s
ct
o
.
`Fus
dMoEExp
rtsModu
ar::f

a

z
_


ght_a
d_r
duc
_
mp
()` r
tur
s th
 `TopKW

ghtA
dR
duc
` obj
ct that th
 
mp

m

tat
o
 
a
ts th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar::f

a

z
()` to us
.
![Fus
dMoEExp
rtsModu
ar B
ocks](../ass
ts/d
s
g
/fus
d_mo
_modu
ar_k
r


/fus
d_
xp
rts_b
ocks.p
g)
### Fus
dMoEModu
arK
r



`Fus
dMoEModu
arK
r


` 
s compos
d of th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` a
d `Fus
dMoEExp
rtsModu
ar` obj
cts.
`Fus
dMoEModu
arK
r


` ps
udocod
/sk
tch,
```py
c
ass Fus
dMoEModu
arK
r


:
    d
f __


t__(s

f,
                 pr
par
_f

a

z
: Fus
dMoEPr
par
A
dF

a

z
Modu
ar,
                 fus
d_
xp
rts: Fus
dMoEExp
rtsModu
ar):
        s

f.pr
par
_f

a

z
 = pr
par
_f

a

z

        s

f.fus
d_
xp
rts = fus
d_
xp
rts
    d
f for
ard(s

f, DP_A):
        Aq, A_sca

, _, _, _ = s

f.pr
par
_f

a

z
.pr
par
(DP_A, ...)
        
orkspac
13_shap
, 
orkspac
2_shap
, _, _ = s

f.fus
d_
xp
rts.
orkspac
_shap
s(...)
        # a
ocat
 
orkspac
s
        
orkspac
_13 = torch.
mpty(
orkspac
13_shap
, ...)
        
orkspac
_2 = torch.
mpty(
orkspac
2_shap
, ...)
        # 
x
cut
 fus
d_
xp
rts
        f
_out = s

f.fus
d_
xp
rts.app
y(Aq, A_sca

, 
orkspac
13, 
orkspac
2, ...)
        # 
ar_
mp
 
s a
 obj
ct of typ
 TopKW

ghtA
dR
duc
NoOp 
f th
 fus
d_
xp
rts 
mp

m

tat
o
s
        # p
rforms th
 TopK W

ght App

cat
o
 a
d R
duct
o
.
        
ar_
mp
 = s

f.fus
d_
xp
rts.f

a

z
_


ght_a
d_r
duc
_
mp
()
        output = s

f.pr
par
_f

a

z
.f

a

z
(f
_out, 
ar_
mp
,...)
        r
tur
 output
```
## Ho
-To
### Ho
 To Add a Fus
dMoEPr
par
A
dF

a

z
Modu
ar Typ

Typ
ca
y a Fus
dMoEPr
par
A
dF

a

z
Modu
ar typ
 
s back
d by a
 A
2A
 D
spatch & Comb


 
mp

m

tat
o
 / k
r


. For 
xamp

,
* D
pEPHTPr
par
A
dF

a

z
 typ
 
s back
d by D
pEP H
gh-Throughput A
2A
 k
r


s, a
d
* D
pEPLLPr
par
A
dF

a

z
 typ
 
s back
d by D
pEP Lo
-Lat

cy A
2A
 k
r


s.
#### St
p 1: Add a
 A
2A
 ma
ag
r
Th
 purpos
 of th
 A
2A
 Ma
ag
r 
s to s
t up th
 A
2A
 k
r


 
mp

m

tat
o
s. Th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` 
mp

m

tat
o
s typ
ca
y f
tch a k
r


-
mp

m

tat
o
 "ha
d

" from th
 A
2A
 Ma
ag
r to 

vok
 th
 D
spatch a
d Comb


 fu
ct
o
s. P

as
 
ook at th
 A
2A
 Ma
ag
r 
mp

m

tat
o
s [h
r
](../../v
m/d
str
but
d/d
v
c
_commu

cators/a
2a
.py).
#### St
p 2: Add a Fus
dMoEPr
par
A
dF

a

z
Modu
ar Typ

Th
s s
ct
o
 d
scr
b
s th
 s
g

f
ca
c
 of th
 var
ous fu
ct
o
s 
xpos
d by th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` abstract c
ass.
`Fus
dMoEPr
par
A
dF

a

z
Modu
ar::pr
par
()`: Th
 pr
par
 m
thod 
mp

m

ts th
 Qua
t
zat
o
 a
d A
2A
 D
spatch. Typ
ca
y th
 D
spatch fu
ct
o
 from th
 r


va
t A
2A
 Ma
ag
r 
s 

vok
d.
`Fus
dMoEPr
par
A
dF

a

z
Modu
ar::has_pr
par
_
o_r
c

v
()`: I
d
cat
s 
h
th
r or 
ot th
s subc
ass 
mp

m

ts `pr
par
_
o_r
c

v
`. D
fau
ts to Fa
s
.
`Fus
dMoEPr
par
A
dF

a

z
Modu
ar::pr
par
_
o_r
c

v
()`: Th
 pr
par
_
o_r
c

v
 m
thod 
mp

m

ts th
 Qua
t
zat
o
 a
d A
2A
 D
spatch. It do
s 
ot 
a
t for th
 r
su
t of th
 d
spatch op
rat
o
 but 

st
ad r
tur
s a thu
k that ca
 b
 

vok
d to 
a
t for th
 f

a
 r
su
ts. Typ
ca
y th
 D
spatch fu
ct
o
 from th
 r


va
t A
2A
 Ma
ag
r 
s 

vok
d.
`Fus
dMoEPr
par
A
dF

a

z
Modu
ar::f

a

z
()`: Mayb
 p
rform TopK W

ght App

cat
o
 a
d R
duct
o
 a
d A
2A
 Comb


. Typ
ca
y th
 Comb


 fu
ct
o
 from th
 r


va
t A
2A
Ma
ag
r 
s 

vok
d.
`Fus
dMoEPr
par
A
dF

a

z
Modu
ar::act
vat
o
_format()`: R
tur
 `Fus
dMoEAct
vat
o
Format.Batch
dExp
rts` 
f th
 output of th
 pr
par
 m
thod (
.
. th
 A
2A
 d
spatch) 
s Batch
d. R
tur
 `Fus
dMoEAct
vat
o
Format.Sta
dard` oth
r

s
.
`Fus
dMoEPr
par
A
dF

a

z
Modu
ar::topk_

d
c
s_dtyp
()`: Data typ
 of th
 TopK 
ds. Som
 A
2A
 k
r


s hav
 str
ct r
qu
r
m

ts p
rta



g to th
 data typ
 of th
 TopK 
ds. Th
s r
qu
r
m

t 
s pass
d o
 to th
 `Fus
dMo
::s


ct_
xp
rts` fu
ct
o
 so 
t cou
d b
 r
sp
ct
d. If th
r
 ar
 
o str
ct r
qu
r
m

ts r
tur
 No

.
`Fus
dMoEPr
par
A
dF

a

z
Modu
ar::max_
um_tok

s_p
r_ra
k()`: Th
s 
s th
 max
mum 
umb
r of tok

s that 
ou
d b
 subm
tt
d to th
 A
2A
 D
spatch at o
c
.
`Fus
dMoEPr
par
A
dF

a

z
Modu
ar::
um_d
spatch
rs()`: Tota
 
umb
r of d
spatch

g u

ts. Th
s va
u
 d
t
rm


s th
 s
z
 of th
 D
spatch output. Th
 D
spatch output 
s of shap
 (
um_
oca
_
xp
rts, max_
um_tok

s, K). H
r
 max_
um_tok

s = 
um_d
spatch
rs() * max_
um_tok

s_p
r_ra
k().
W
 sugg
st p
ck

g a
 a
r
ady 
x
st

g `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` 
mp

m

tat
o
 that match
s your A
2A
 
mp

m

tat
o
 c
os

y a
d us

g 
t as a r
f
r

c
.
### Ho
 To Add a Fus
dMoEExp
rtsModu
ar Typ

Fus
dMoEExp
rtsModu
ar p
rforms th
 cor
 of th
 Fus
dMoE op
rat
o
s. Th
 var
ous fu
ct
o
s 
xpos
d by th
 abstract c
ass a
d th

r s
g

f
ca
c
 
s as fo
o
s,
`Fus
dMoEExp
rtsModu
ar::act
vat
o
_formats()`: R
tur
 th
 support
d I
put a
d Output act
vat
o
 formats. 
.
. Co
t
guous / Batch
d format.
`Fus
dMoEExp
rtsModu
ar::supports_chu
k

g()`: R
tur
 Tru
 
f th
 
mp

m

tat
o
 supports chu
k

g. Typ
ca
y

mp

m

tat
o
s that 

put `Fus
dMoEAct
vat
o
Format.Sta
dard` support chu
k

g a
d `Fus
dMoEAct
vat
o
Format.Batch
dExp
rts` do 
ot.
`Fus
dMoEExp
rtsModu
ar::supports_
xp
rt_map()`: R
tur
 Tru
 
f th
 
mp

m

tat
o
 supports 
xp
rt map.
`Fus
dMoEExp
rtsModu
ar::
orkspac
_shap
s()` /
`Fus
dMoEExp
rtsModu
ar::f

a

z
_


ght_a
d_r
duc
_
mp
` /
`Fus
dMoEExp
rtsModu
ar::app
y`: R
f
r to `Fus
dMoEExp
rtsModu
ar` s
ct
o
 abov
.
### Fus
dMoEModu
arK
r


 I

t
a

zat
o

`Fus
dMoEM
thodBas
` c
ass has 3 m
thods that ar
 co

ct
v

y r
spo
s
b

 

 cr
at

g th
 `Fus
dMoEModu
arK
r


` obj
ct. Th
y ar
,
* mayb
_mak
_pr
par
_f

a

z
,
* s


ct_g
mm_
mp
, a
d
* 


t_pr
par
_f

a

z

#### mayb
_mak
_pr
par
_f

a

z

Th
 `mayb
_mak
_pr
par
_f

a

z
` m
thod 
s r
spo
s
b

 for co
struct

g a
 

sta
c
 of `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` 
h

 appropr
at
 bas
d o
 th
 curr

t a
2a
 back

d, 
.g. 
h

 EP + DP 
s 

ab

d.  Th
 bas
 c
ass m
thod curr

t
y co
structs a
 th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` obj
cts for th
 EP+DP cas
.  D
r
v
d c
ass
s ca
 ov
rr
d
 th
s m
thod to co
struct pr
par
/f

a

z
 obj
cts for d
ff
r

t sc

ar
os, 
.g. `Mod

OptNvFp4Fus
dMoE` ca
 co
struct a `F
ashI
f
rCut
assMoEPr
par
A
dF

a

z
` for th
 EP+TP cas
.
P

as
 r
f
r to th
 
mp

m

tat
o
s 

,
* `Mod

OptNvFp4Fus
dMoE`
#### s


ct_g
mm_
mp

Th
 `s


ct_g
mm_
mp
` m
thod 
s u
d
f


d 

 th
 bas
 c
ass. It 
s th
 r
spo
s
b


ty of th
 d
r
v
d c
ass to 
mp

m

t a m
thod that co
structs a va

d/appropr
at
 `Fus
dMoEExp
rtsModu
ar` obj
ct.
P

as
 r
f
r to th
 
mp

m

tat
o
s 

,
* `U
qua
t
z
dFus
dMoEM
thod`
* `Compr
ss
dT

sorsW8A8Fp8MoEM
thod`
* `Compr
ss
dT

sorsW8A8Fp8MoECut
assM
thod`
* `Fp8MoEM
thod`
* `Mod

OptNvFp4Fus
dMoE`
d
r
v
d c
ass
s.
#### 


t_pr
par
_f

a

z

Bas
d o
 th
 

put a
d 

v s
tt

gs, th
 `


t_pr
par
_f

a

z
` m
thod cr
at
s th
 appropr
at
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` obj
ct. Th
 m
thod th

 qu
r

s `s


ct_g
mm_
mp
` for th
 appropr
at
 `Fus
dMoEExp
rtsModu
ar` obj
ct a
d bu

ds th
 `Fus
dMoEModu
arK
r


` obj
ct
P

as
 tak
 a 
ook at [


t_pr
par
_f

a

z
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/1cbf951ba272c230823b947631065b826409fa62/v
m/mod

_
x
cutor/
ay
rs/fus
d_mo
/
ay
r.py#L188).
**Importa
t**: Th
 `Fus
dMoEM
thodBas
` d
r
v
d c
ass
s us
 th
 `Fus
dMoEM
thodBas
::fus
d_
xp
rts` obj
ct 

 th

r `app
y` m
thods. Wh

 s
tt

gs p
rm
t th
 co
struct
o
 of a va

d `Fus
dMoEModu
arK
r


` obj
ct, 

 ov
rr
d
 `Fus
dMoEM
thodBas
::fus
d_
xp
rts` 

th 
t. Th
s 
ss

t
a
y mak
s th
 d
r
v
d c
ass
s ag
ost
c to 
hat Fus
dMoE 
mp

m

tat
o
 
s us
d.
### Ho
 To U

t T
st
W
 hav
 `Fus
dMoEModu
arK
r


` u

t t
sts at [t
st_modu
ar_k
r


_comb

at
o
s.py](../../t
sts/k
r


s/mo
/t
st_modu
ar_k
r


_comb

at
o
s.py).
Th
 u

t t
st 
t
rat
s through a
 comb

at
o
s of `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` a
d `Fus
dMoEPr
mut
Exp
rtsU
p
rmut
` typ
s a
d 
f th
y ar

compat
b

, ru
s som
 corr
ct

ss t
sts.
If you ar
 add

g som
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` / `Fus
dMoEExp
rtsModu
ar` 
mp

m

tat
o
s,
1. Add th
 
mp

m

tat
o
 typ
 to `MK_ALL_PREPARE_FINALIZE_TYPES` a
d `MK_FUSED_EXPERT_TYPES` 

 [mk_obj
cts.py](../../t
sts/k
r


s/mo
/modu
ar_k
r


_too
s/mk_obj
cts.py) r
sp
ct
v

y.
2. Updat
 `Co
f
g::
s_batch
d_pr
par
_f

a

z
()`, `Co
f
g::
s_batch
d_fus
d_
xp
rts()`, `Co
f
g::
s_sta
dard_fus
d_
xp
rts()`,
`Co
f
g::
s_f
_16b
t_support
d()`,  `Co
f
g::
s_f
_fp8_support
d()`, `Co
f
g::
s_f
_b
ock_fp8_support
d()`,
`Co
f
g::
s_f
_supports_chu
k

g()` m
thods 

 [/t
sts/k
r


s/mo
/modu
ar_k
r


_too
s/commo
.py](../../t
sts/k
r


s/mo
/modu
ar_k
r


_too
s/commo
.py)
Do

g th
s 


 add th
 


 
mp

m

tat
o
 to th
 t
st su
t
.
### Ho
 To Ch
ck `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` & `Fus
dMoEExp
rtsModu
ar` Compat
b


ty
Th
 u

t t
st f


 [t
st_modu
ar_k
r


_comb

at
o
s.py](../../t
sts/k
r


s/mo
/t
st_modu
ar_k
r


_comb

at
o
s.py) ca
 a
so b
 
x
cut
d as a sta
da
o

 scr
pt.
Examp

: `pytho
3 -m t
sts.k
r


s.mo
.t
st_modu
ar_k
r


_comb

at
o
s --pf-typ
 D
pEPLLPr
par
A
dF

a

z
 --
xp
rts-typ
 Batch
dTr
to
Exp
rts`
As a s
d
 
ff
ct, th
s scr
pt ca
 b
 us
d to t
st `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` & `Fus
dMoEExp
rtsModu
ar` compat
b


ty. Wh

 

vok
d


th 

compat
b

 typ
s, th
 scr
pt 


 
rror.
### Ho
 To Prof



P

as
 tak
 a 
ook at [prof


_modu
ar_k
r


.py](../../t
sts/k
r


s/mo
/modu
ar_k
r


_too
s/prof


_modu
ar_k
r


.py)
Th
 scr
pt ca
 b
 us
d to g


rat
 Torch trac
s for a s

g

 `Fus
dMoEModu
arK
r


::for
ard()` ca
 for a
y compat
b


`Fus
dMoEPr
par
A
dF

a

z
Modu
ar` a
d `Fus
dMoEExp
rtsModu
ar` typ
s.
Examp

: `pytho
3 -m t
sts.k
r


s.mo
.modu
ar_k
r


_too
s.prof


_modu
ar_k
r


 --pf-typ
 D
pEPLLPr
par
A
dF

a

z
 --
xp
rts-typ
 Batch
dTr
to
Exp
rts`
## Fus
dMoEPr
par
A
dF

a

z
Modu
ar Imp

m

tat
o
s
S
 [Fus
d MoE K
r


 f
atur
s](./mo
_k
r


_f
atur
s.md#fus
d-mo
-modu
ar-a
2a
-back

ds) for a 

st of a
 th
 ava

ab

 modu
ar pr
par
 a
d f

a

z
 subc
ass
s.
## Fus
dMoEExp
rtsModu
ar
S
 [Fus
d MoE K
r


 f
atur
s](./mo
_k
r


_f
atur
s.md#fus
d-mo
-
xp
rts-k
r


s) for a 

st of a
 th
 ava

ab

 modu
ar 
xp
rts.
