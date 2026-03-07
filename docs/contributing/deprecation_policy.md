# D
pr
cat
o
 Po

cy
Th
s docum

t out



s th
 off
c
a
 po

cy a
d proc
ss for d
pr
cat

g f
atur
s


 th
 vLLM proj
ct.
## Ov
rv



vLLM us
s a structur
d "d
pr
cat
o
 p
p




" to gu
d
 th
 

f
cyc

 of
d
pr
cat
d f
atur
s. Th
s po

cy 

sur
s that us
rs ar
 g
v

 c

ar a
d
suff
c


t 
ot
c
 
h

 a f
atur
 
s d
pr
cat
d a
d that d
pr
cat
o
s proc
d 


a co
s
st

t a
d pr
d
ctab

 ma

r.
W
 a
m to str
k
 a ba
a
c
 b
t


 co
t

u
d 

ovat
o
 a
d r
sp
ct

g us
rs’
r


a
c
 o
 
x
st

g fu
ct
o
a

ty. D
pr
cat
o
s ar
 t

d to our **m

or (Y)
r


as
s** fo
o


g s
ma
t
c v
rs
o


g (X.Y.Z), 
h
r
:
- **X** 
s a major v
rs
o
 (rar
)
- **Y** 
s a m

or v
rs
o
 (us
d for s
g

f
ca
t cha
g
s, 

c
ud

g d
pr
cat
o
s/r
mova
s)
- **Z** 
s a patch v
rs
o
 (us
d for f
x
s a
d saf
r 

ha
c
m

ts)
F
atur
s that fa
 u
d
r th
s po

cy 

c
ud
 (at a m


mum) th
 fo
o


g:
- CLI f
ags
- E
v
ro
m

t var
ab

s
- Co
f
gurat
o
 f


s
- APIs 

 th
 Op

AI-compat
b

 API s
rv
r
- Pub

c Pytho
 APIs for th
 `v
m` 

brary
## D
pr
cat
o
 P
p





Th
 d
pr
cat
o
 proc
ss co
s
sts of s
v
ra
 c

ar
y d
f


d stag
s that spa

mu
t
p

 Y r


as
s:
### 1. D
pr
cat
d (St

 O
 By D
fau
t)
- **Act
o
**: F
atur
 
s mark
d as d
pr
cat
d.
- **T
m




**: A r
mova
 v
rs
o
 
s 
xp

c
t
y stat
d 

 th
 d
pr
cat
o


ar


g (
.g., "Th
s 


 b
 r
mov
d 

 v0.10.0").
- **Commu

cat
o
**: D
pr
cat
o
 
s 
ot
d 

 th
 fo
o


g, as app

cab

:
    - H

p str

gs
    - Log output
    - API r
spo
s
s
    - `/m
tr
cs` output (for m
tr
cs f
atur
s)
    - Us
r-fac

g docum

tat
o

    - R


as
 
ot
s
    - G
tHub Issu
 (RFC) for f
dback
    - Docum

tat
o
 a
d us
 of th
 `@typ

g_
xt

s
o
s.d
pr
cat
d` d
corator for Pytho
 APIs
### 2. D
pr
cat
d (Off By D
fau
t)
- **Act
o
**: F
atur
 
s d
sab

d by d
fau
t, but ca
 st

 b
 r
-

ab

d v
a a
CLI f
ag or 

v
ro
m

t var
ab

. F
atur
 thro
s a
 
rror 
h

 us
d 

thout
r
-

ab


g.
- **Purpos
**: A
o
s us
rs 
ho m
ss
d 
ar


r 
ar


gs a t
mporary 
scap
 hatch

h


 s
g
a


g 
mm



t r
mova
. E
sur
s a
y r
ma



g usag
 
s c

ar
y
surfac
d a
d b
ocks s



t br
akag
 b
for
 fu
 r
mova
.
### 3. R
mov
d
- **Act
o
**: F
atur
 
s comp

t

y r
mov
d from th
 cod
bas
.
- **Not
**: O

y f
atur
s that hav
 pass
d through th
 pr
v
ous d
pr
cat
o

stag
s 


 b
 r
mov
d.
## Examp

 T
m





Assum
 a f
atur
 
s d
pr
cat
d 

 `v0.9.0`.
| R


as
       | Status                                                                                          |
|---------------|-------------------------------------------------------------------------------------------------|
| `v0.9.0`      | F
atur
 
s d
pr
cat
d 

th c

ar r
mova
 v
rs
o
 

st
d.                                        |
| `v0.10.0`     | F
atur
 
s 
o
 off by d
fau
t, thro
s a
 
rror 
h

 us
d, a
d ca
 b
 r
-

ab

d for 

gacy us
. |
| `v0.11.0`     | F
atur
 
s r
mov
d.                                                                             |
## Importa
t Gu
d




s
- **No R
mova
s 

 Patch R


as
s**: R
mov

g d
pr
cat
d f
atur
s 

 patch
(`.Z`) r


as
s 
s d
sa
o

d to avo
d surpr
s

g us
rs.
- **Grac
 P
r
od for Ex
st

g D
pr
cat
o
s**: A
y f
atur
 d
pr
cat
d **b
for

th
s po

cy** 


 hav
 
ts grac
 p
r
od start **
o
**, 
ot r
troact
v

y.
- **Docum

tat
o
 
s Cr
t
ca
**: E
sur
 
v
ry stag
 of th
 p
p




 
s
docum

t
d c

ar
y for us
rs.
## F

a
 Not
s
Th
s po

cy 
s a 

v

g docum

t a
d may 
vo
v
 as th
 

ds of th
 proj
ct a
d

ts us
rs cha
g
. Commu

ty f
dback 
s 


com
 a
d 

courag
d as 

 r
f


 th

proc
ss.
