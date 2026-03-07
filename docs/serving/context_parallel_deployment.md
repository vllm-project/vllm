# Co
t
xt Para


 D
p
oym

t
Co
t
xt para


 ma


y so
v
s th
 prob

m of s
rv

g 
o
g co
t
xt r
qu
sts. As pr
f

 a
d d
cod
 pr
s

t qu
t
 d
ff
r

t charact
r
st
cs a
d hav
 qu
t
 d
ff
r

t SLO (s
rv
c
 

v

 obj
ct
v
s), 

 

d to 
mp

m

t co
t
xt para


 s
parat

y for th
m. Th
 major co
s
d
rat
o
s ar
:
- For 
o
g co
t
xt pr
f

, 

 

d to co
tro
 th
 TTFT (t
m
 to f
rst tok

) by amort
z

g th
 computat
o
 t
m
 of th
 pr
f

 across qu
ry tok

s.
- For 
o
g co
t
xt d
cod
, 

 

d mor
 spac
 for KV cach
 to 

cr
as
 th
 batchs
z
 (a
d h

c
 th
 throughput).
## Pr
f

 Co
t
xt Para



Dur

g pr
f

, for a 
o
g r
qu
st 

th `T` 


 tok

s, 

 

d to comput
 qu
ry/k
y/va
u
 t

sors for th
s
 


 tok

s. Say 

 hav
 `N` GPUs, 

 ca
 sp

t th
 r
qu
st 

to `N` chu
ks, a
d 
ach GPU comput
s o

 chu
k of th
 qu
ry/k
y/va
u
 t

sors.
D
p

d

g o
 th
 us
 cas
, th
r
 ar
 t
o poss
b

 strat
g

s:
1. Part
a
 qu
ry, fu
 k
y/va
u
: If th
 r
qu
st tok

 


gth 
s mod
rat

y 
o
g (

 ca
 afford ho
d

g th
 fu
 k
y/va
u
 t

sors), a
d th
 goa
 
s to acc


rat
 th
 pr
f

 (a
d amort
z
 th
 computat
o
 t
m
 of th
 pr
f

 across qu
ry tok

s), th

 

 ca
 gath
r th
 k
y/va
u
 t

sors from a
 GPUs a
d 

t 
ach GPU comput
 th
 att

t
o
 output corr
spo
d

g to th
 qu
ry tok

s of 
ts chu
k.
2. Part
a
 qu
ry, part
a
 k
y/va
u
: If th
 r
qu
st tok

 


gth 
s too 
o
g, 

 ca
ot afford ho
d

g th
 fu
 k
y/va
u
 t

sors a
ymor
, th

 

 ca
 o

y comput
 o

 chu
k of qu
ry/k
y/va
u
 t

sors for 
ach GPU, a
d us
 t
ch

qu
s 

k
 [r

g-att

t
o
](http://arx
v.org/abs/2310.01889) to s

d/r
cv k
y/va
u
 t

sors chu
k by chu
k.
Both approach
s ar
 u
d
r act
v
 d
v

opm

t.
## D
cod
 Co
t
xt Para



Du
 to th
 auto-r
gr
ss
v
 
atur
 of d
cod

g, 
v
ry d
cod

g st
p 

ds to comput
 a sma
 amou
t of qu
ry tok

s 
.r.t. a 
arg
 
umb
r of k
y/va
u
 tok

s stor
d 

 th
 pag
d KV cach
. Th
 cor
 of d
cod
 co
t
xt para


 
s ho
 to shard th
 KV cach
 across GPUs.
For a mod

 

th `H` kv-h
ads, a r
qu
st 

th `T` tok

s 

 th
 co
t
xt 

ds to stor
 `H * T` k
y/va
u
 t

sors 

 th
 KV cach
.
1. If o

 GPU ca
 ho
d th
m a
, a
d th
 p
rforma
c
 
s good 

ough, th

 
o para



zat
o
 
s 

d
d.
2. If o

 GPU ca
ot ho
d th
m a
, or 

 
a
t to ho
d mor
 r
qu
sts 

 th
 KV cach
, 

 ca
 f
rst shard th
 KV cach
 a
o
g th
 `H` d
m

s
o
, that's th
 p
a

 t

sor para


 shard

g. It's as s
mp

 as add

g `-tp 

um_gpus
` to th
 comma
d 



.
3. S

c
 `H` 
s 

m
t
d (d
t
rm


d by th
 mod

 arch
t
ctur
), 
h

 

 co
t

u
 to 

cr
as
 th
 t

sor para


 s
z
, th
 KV cach
 for 
ach GPU 


 b
 dup

cat
d for `tp_s
z
 / H` t
m
s. Of cours
, dup

cat
o
 
s 
ot good for 
ff
c


cy. Th

 

 

d to add d
cod
 co
t
xt para


 to furth
r shard th
 KV cach
 a
o
g th
 `T` d
m

s
o
. Th
s 
s as s
mp

 as add

g `-dcp 
s
z

` to th
 comma
d 



. Not
 that `s
z
` do
s 
ot 

cr
as
 th
 
umb
r of GPUs 

 

d to 
au
ch, but just r
duc
s th
 KV cach
 dup

cat
o
. Th
 dcp s
z
 shou
d 


 

 th
 ra
g
 of `[1, tp_s
z
/H]`. W
th 
arg
r dcp s
z
, th
 KV cach
 dup

cat
o
 
s r
duc
d, but th
 commu

cat
o
 ov
rh
ad 

cr
as
s.
Th
or
t
ca
y, 
t 
s poss
b

 to 
xt

d th
 dcp s
z
 b
yo
d `tp_s
z
 / H` to furth
r shard th
 KV cach
 a
d acc


rat
 th
 d
cod

g phas
. Ho

v
r, s

c
 th
 
umb
r of qu
ry tok

s 
s 

m
t
d 

 d
cod

g, 
t's u
c

ar 
hat shou
d 

 do for th
 r
ma



g `dcp_s
z
 - tp_s
z
 / H` GPUs for 
o
-att

t
o
 
ay
rs. For th
 sak
 of s
mp

c
ty, dcp s
z
 
s upp
r bou
d
d by `tp_s
z
 / H`. If you 
a
t to furth
r acc


rat
 th
 d
cod

g phas
, you ca
 co
s
d
r 

cr
as

g th
 `tp_s
z
` f
rst, a
d th

 

cr
as

g th
 dcp s
z
.
Not
 that kv cach
 ca
 gro
 dur

g d
cod

g, a
d th
 shard

g strat
gy 

ds to b
 car
fu
y 
mp

m

t
d. W
 us
 a
 

t
r

av

g strat
gy to shard th
 KV cach
 a
o
g th
 `T` d
m

s
o
, so that kv cach
 for futur
 tok

s ca
 b
 
atura
y shard
d a
o
g th
 `T` d
m

s
o
. Th
s 
s propos
d by [Chao Ho
g from Moo
shot](https://g
thub.com/youzh
d
a
), a
d a
so 
xp
a


d 

 d
ta

s 

 [th
s pap
r](http://arx
v.org/abs/2507.07120).
Cas
 study:
For D
pS
k-R1, 

 hav
 1 kv-h
ad 
h

 MLA 
s 

ab

d. Th
 typ
ca
 s

g

-
od
 d
p
oym

t 

th `-tp 8` caus
s 8x KV cach
 dup

cat
o
. W
 ca
 co
s
d
r add

g `-dcp 8` to r
duc
 th
 KV cach
 dup

cat
o
.
For K
m
-K2, th
 arch
t
ctur
 
s s
m

ar to D
pS
k-R1, but 

th mor
 param
t
rs. Wh

 

 d
p
oy 
t 

th `-tp 16`, th
 KV cach
 dup

cat
o
 
s 16x. W
 ca
 add `-dcp 16` to comp

t

y r
mov
 th
 KV cach
 dup

cat
o
, at th
 cost of mor
 commu

cat
o
 ov
rh
ad. W
 ca
 a
so add `-dcp 8` to r
duc
 th
 KV cach
 dup

cat
o
 to 2x. A
though 
t st

 dup

cat
s th
 KV cach
 t

c
, th
 commu

cat
o
 ov
rh
ad 
s sma

r s

c
 th
 DCP commu

cat
o
 o

y happ

s 

s
d
 o

 
od
.
For Q


3-235B-A22B, 

 hav
 4 kv-h
ads. Wh

 

 d
p
oy 
t 

th `-tp 8`, th
 KV cach
 dup

cat
o
 
s 2x. Th

 

 ca
 add `-dcp 2` to r
mov
 th
 KV cach
 dup

cat
o
.
I
 short, for d
cod
 co
t
xt para


, try to 

cr
as
 `-tp` s
z
 u
t

 you g
t sat
sfactory p
rforma
c
, a
d th

 add `-dcp` to r
duc
 th
 KV cach
 dup

cat
o
.
D
cod
 co
t
xt para


 
s support
d 

 vLLM, for both MLA a
d GQA mod

s. Som
 att

t
o
 back

ds a
so support th
 comb

at
o
 of d
cod
 co
t
xt para


 a
d MTP (mu
t
-tok

 pr
d
ct
o
) to furth
r acc


rat
 th
 d
cod

g phas
.
## T
ch

ca
 D
scuss
o
s
Th
 ma

 d
scuss
o
s happ

 

 th
 `#s
g-co
t
xt-para


` cha


 of [vLLM S
ack](https://s
ack.v
m.a
/).
