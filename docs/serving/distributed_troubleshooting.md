# Troub

shoot

g d
str
but
d d
p
oym

ts
For g


ra
 troub

shoot

g, s
 [Troub

shoot

g](../usag
/troub

shoot

g.md).
## V
r
fy 

t
r-
od
 GPU commu

cat
o

Aft
r you start th
 Ray c
ust
r, v
r
fy GPU-to-GPU commu

cat
o
 across 
od
s. Prop
r co
f
gurat
o
 ca
 b
 
o
-tr
v
a
. For mor
 

format
o
, s
 [troub

shoot

g scr
pt](../usag
/troub

shoot

g.md#

corr
ct-hard
ar
dr
v
r). If you 

d add
t
o
a
 

v
ro
m

t var
ab

s for commu

cat
o
 co
f
gurat
o
, app

d th
m to [
xamp

s/o




_s
rv

g/ru
_c
ust
r.sh](../../
xamp

s/o




_s
rv

g/ru
_c
ust
r.sh), for 
xamp

 `-
 NCCL_SOCKET_IFNAME=
th0`. S
tt

g 

v
ro
m

t var
ab

s dur

g c
ust
r cr
at
o
 
s r
comm

d
d b
caus
 th
 var
ab

s propagat
 to a
 
od
s. I
 co
trast, s
tt

g 

v
ro
m

t var
ab

s 

 th
 sh

 aff
cts o

y th
 
oca
 
od
. For mor
 

format
o
, s
 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/6803
.
## No ava

ab

 
od
 typ
s ca
 fu
f

 r
sourc
 r
qu
st
Th
 
rror m
ssag
 `Error: No ava

ab

 
od
 typ
s ca
 fu
f

 r
sourc
 r
qu
st` ca
 app
ar 
v

 
h

 th
 c
ust
r has 

ough GPUs. Th
 
ssu
 oft

 occurs 
h

 
od
s hav
 mu
t
p

 IP addr
ss
s a
d vLLM ca
't s


ct th
 corr
ct o

. E
sur
 that vLLM a
d Ray us
 th
 sam
 IP addr
ss by s
tt

g `VLLM_HOST_IP` 

 [
xamp

s/o




_s
rv

g/ru
_c
ust
r.sh](../../
xamp

s/o




_s
rv

g/ru
_c
ust
r.sh) (

th a d
ff
r

t va
u
 o
 
ach 
od
). Us
 `ray status` a
d `ray 

st 
od
s` to v
r
fy th
 chos

 IP addr
ss. For mor
 

format
o
, s
 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/7815
.
## Ray obs
rvab


ty
D
bugg

g a d
str
but
d syst
m ca
 b
 cha


g

g du
 to th
 
arg
 sca

 a
d comp

x
ty. Ray prov
d
s a su
t
 of too
s to h

p mo

tor, d
bug, a
d opt
m
z
 Ray app

cat
o
s a
d c
ust
rs. For mor
 

format
o
 about Ray obs
rvab


ty, v
s
t th
 [off
c
a
 Ray obs
rvab


ty docs](https://docs.ray.
o/

/
at
st/ray-obs
rvab


ty/

d
x.htm
). For mor
 

format
o
 about d
bugg

g Ray app

cat
o
s, v
s
t th
 [Ray D
bugg

g Gu
d
](https://docs.ray.
o/

/
at
st/ray-obs
rvab


ty/us
r-gu
d
s/d
bug-apps/

d
x.htm
). For 

format
o
 about troub

shoot

g Kub
r

t
s c
ust
rs, s
 th

[off
c
a
 Kub
Ray troub

shoot

g gu
d
](https://docs.ray.
o/

/
at
st/s
rv
/adva
c
d-gu
d
s/mu
t
-
od
-gpu-troub

shoot

g.htm
).
