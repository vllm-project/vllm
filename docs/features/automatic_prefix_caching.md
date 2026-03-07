# Automat
c Pr
f
x Cach

g
## I
troduct
o

Automat
c Pr
f
x Cach

g (APC 

 short) cach
s th
 KV cach
 of 
x
st

g qu
r

s, so that a 


 qu
ry ca
 d
r
ct
y r
us
 th
 KV cach
 
f 
t shar
s th
 sam
 pr
f
x 

th o

 of th
 
x
st

g qu
r

s, a
o


g th
 


 qu
ry to sk
p th
 computat
o
 of th
 shar
d part.
!!! 
ot

    T
ch

ca
 d
ta

s o
 ho
 vLLM 
mp

m

ts APC ca
 b
 fou
d [h
r
](../d
s
g
/pr
f
x_cach

g.md).
## E
ab


g APC 

 vLLM
S
t `

ab

_pr
f
x_cach

g=Tru
` 

 vLLM 

g


 to 

ab

 APC. H
r
 
s a
 
xamp

:
[
xamp

s/off



_

f
r

c
/automat
c_pr
f
x_cach

g.py](../../
xamp

s/off



_

f
r

c
/automat
c_pr
f
x_cach

g.py)
## Examp

 
ork
oads
W
 d
scr
b
 t
o 
xamp

 
ork
oads, 
h
r
 APC ca
 prov
d
 hug
 p
rforma
c
 b


f
t:
    - Lo
g docum

t qu
ry, 
h
r
 th
 us
r r
p
at
d
y qu
r

s th
 sam
 
o
g docum

t (
.g. soft
ar
 ma
ua
 or a
ua
 r
port) 

th d
ff
r

t qu
r

s. I
 th
s cas
, 

st
ad of proc
ss

g th
 
o
g docum

t aga

 a
d aga

, APC a
o
s vLLM to proc
ss th
s 
o
g docum

t *o

y o
c
*, a
d a
 futur
 r
qu
sts ca
 avo
d r
comput

g th
s 
o
g docum

t by r
us

g 
ts KV cach
. Th
s a
o
s vLLM to s
rv
 futur
 r
qu
sts 

th much h
gh
r throughput a
d much 
o

r 
at

cy.
    - Mu
t
-rou
d co
v
rsat
o
, 
h
r
 th
 us
r may chat 

th th
 app

cat
o
 mu
t
p

 t
m
s 

 th
 sam
 chatt

g s
ss
o
. I
 th
s cas
, 

st
ad of proc
ss

g th
 
ho

 chatt

g h
story aga

 a
d aga

, APC a
o
s vLLM to r
us
 th
 proc
ss

g r
su
ts of th
 chat h
story across a
 futur
 rou
ds of co
v
rsat
o
, a
o


g vLLM to s
rv
 futur
 r
qu
sts 

th much h
gh
r throughput a
d much 
o

r 
at

cy.
## L
m
ts
APC 

 g


ra
 do
s 
ot r
duc
 th
 p
rforma
c
 of vLLM. W
th that b


g sa
d, APC o

y r
duc
s th
 t
m
 of proc
ss

g th
 qu
r

s (th
 pr
f



g phas
) a
d do
s 
ot r
duc
 th
 t
m
 of g


rat

g 


 tok

s (th
 d
cod

g phas
). So APC do
s 
ot br

g p
rforma
c
 ga

 
h

 vLLM sp

ds most of th
 t
m
 g


rat

g a
s

rs to th
 qu
r

s (
.g. 
h

 th
 


gth of th
 a
s

r 
s 
o
g), or 


 qu
r

s do 
ot shar
 th
 sam
 pr
f
x 

th a
y of 
x
st

g qu
r

s (so that th
 computat
o
 ca
ot b
 r
us
d).
