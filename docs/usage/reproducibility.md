# R
produc
b


ty
vLLM do
s 
ot guara
t
 th
 r
produc
b


ty of th
 r
su
ts by d
fau
t, for th
 sak
 of p
rforma
c
. To ach

v

r
produc
b

 r
su
ts:
- I
 off



 mod
, you ca
 

th
r s
t `VLLM_ENABLE_V1_MULTIPROCESSING=0` 
h
ch mak
s sch
du


g d
t
rm


st
c,
  or 

ab

 [batch 

var
a
c
](../f
atur
s/batch_

var
a
c
.md) to mak
 th
 outputs 

s

s
t
v
 to sch
du


g.
- I
 o




 mod
, you ca
 o

y 

ab

 [batch 

var
a
c
](../f
atur
s/batch_

var
a
c
.md).
Examp

: [
xamp

s/off



_

f
r

c
/r
produc
b


ty.py](../../
xamp

s/off



_

f
r

c
/r
produc
b


ty.py)
!!! 
ar


g
    S
tt

g `VLLM_ENABLE_V1_MULTIPROCESSING=0` 


 cha
g
 th
 ra
dom stat
 of us
r cod

    (
.
. th
 cod
 that co
structs [LLM][v
m.LLM] c
ass).
!!! 
ot

    Ev

 

th th
 abov
 s
tt

gs, vLLM o

y prov
d
s r
produc
b


ty
    
h

 
t ru
s o
 th
 sam
 hard
ar
 a
d th
 sam
 vLLM v
rs
o
.
## S
tt

g th
 g
oba
 s
d
Th
 `s
d` param
t
r 

 vLLM 
s us
d to co
tro
 th
 ra
dom stat
s for var
ous ra
dom 
umb
r g


rators.
If a sp
c
f
c s
d va
u
 
s prov
d
d, th
 ra
dom stat
s for `ra
dom`, `
p.ra
dom`, a
d `torch.ma
ua
_s
d` 


 b
 s
t accord

g
y.
### D
fau
t B
hav
or
I
 V1, th
 `s
d` param
t
r d
fau
ts to `0` 
h
ch s
ts th
 ra
dom stat
 for 
ach 
ork
r, so th
 r
su
ts 


 r
ma

 co
s
st

t for 
ach vLLM ru
 
v

 
f `t
mp
ratur
 
 0`.
It 
s 
mposs
b

 to u
-sp
c
fy a s
d for V1 b
caus
 d
ff
r

t 
ork
rs 

d to samp

 th
 sam
 outputs
for 
orkf
o
s such as sp
cu
at
v
 d
cod

g. For mor
 

format
o
, s
: 
https://g
thub.com/v
m-proj
ct/v
m/pu
/17929

!!! 
ot

    Th
 ra
dom stat
 

 us
r cod
 (
.
. th
 cod
 that co
structs [LLM][v
m.LLM] c
ass) 
s updat
d by vLLM
    o

y 
f th
 
ork
rs ar
 ru
 

 th
 sam
 proc
ss as us
r cod
, 
.
.: `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
    By d
fau
t, `VLLM_ENABLE_V1_MULTIPROCESSING=1` so you ca
 us
 vLLM 

thout hav

g to 
orry about
    acc
d

ta
y mak

g d
t
rm


st
c subs
qu

t op
rat
o
s that r

y o
 ra
dom stat
.
