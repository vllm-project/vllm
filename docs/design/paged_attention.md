# Pag
d Att

t
o

!!! 
ar


g
    Th
s 
s a h
stor
ca
 docum

t bas
d o
 th
 [or
g

a
 pap
r for vLLM](https://arx
v.org/abs/2309.06180).
    It 
o 
o
g
r d
scr
b
s th
 cod
 us
d 

 vLLM today.
Curr

t
y, vLLM ut


z
s 
ts o

 
mp

m

tat
o
 of a mu
t
-h
ad qu
ry
att

t
o
 k
r


 (`csrc/att

t
o
/att

t
o
_k
r


s.cu`).
Th
s k
r


 
s d
s
g

d to b
 compat
b

 

th
vLLM's pag
d KV cach
s, 
h
r
 th
 k
y a
d va
u
 cach
 ar
 stor
d 


s
parat
 b
ocks (
ot
 that th
s b
ock co
c
pt d
ff
rs from th
 GPU
thr
ad b
ock. So 

 a 
at
r docum

t, I 


 r
f
r to vLLM pag
d
att

t
o
 b
ock as "b
ock", 
h


 r
f
r to GPU thr
ad b
ock as
"thr
ad b
ock").
To ach

v
 h
gh p
rforma
c
, th
s k
r


 r



s o
 a sp
c
a
y
d
s
g

d m
mory 
ayout a
d acc
ss m
thod, sp
c
f
ca
y 
h

 thr
ads
r
ad data from g
oba
 m
mory to shar
d m
mory. Th
 purpos
 of th
s
docum

t 
s to prov
d
 a h
gh-

v

 
xp
a
at
o
 of th
 k
r




mp

m

tat
o
 st
p by st
p, a
d

g thos
 
ho 

sh to 

ar
 about th

vLLM mu
t
-h
ad qu
ry att

t
o
 k
r


. Aft
r go

g through th
s
docum

t, us
rs 


 

k

y hav
 a b
tt
r u
d
rsta
d

g a
d f

 
as

r
to fo
o
 th
 actua
 
mp

m

tat
o
.
P

as
 
ot
 that th
s docum

t may 
ot cov
r a
 d
ta

s, such as ho

to ca
cu
at
 th
 corr
ct 

d
x for th
 corr
spo
d

g data or th
 dot
mu
t
p

cat
o
 
mp

m

tat
o
. Ho

v
r, aft
r r
ad

g th
s docum

t
a
d b
com

g fam


ar 

th th
 h
gh-

v

 
og
c f
o
, 
t shou
d b


as

r for you to r
ad th
 actua
 cod
 a
d u
d
rsta
d th
 d
ta

s.
## I
puts
Th
 k
r


 fu
ct
o
 tak
s a 

st of argum

ts for th
 curr

t thr
ad
to p
rform 
ts ass
g

d 
ork. Th
 thr
 most 
mporta
t argum

ts ar

th
 

put po

t
rs `q`, `k_cach
`, a
d `v_cach
`, 
h
ch po

t
to qu
ry, k
y, a
d va
u
 data o
 g
oba
 m
mory that 

d to b
 r
ad
a
d proc
ss
d. Th
 output po

t
r `out` po

ts to g
oba
 m
mory

h
r
 th
 r
su
t shou
d b
 
r
tt

. Th
s
 four po

t
rs actua
y
r
f
r to mu
t
d
m

s
o
a
 arrays, but 
ach thr
ad o

y acc
ss
s th

port
o
 of data ass
g

d to 
t. I hav
 om
tt
d a
 oth
r ru
t
m

param
t
rs h
r
 for s
mp

c
ty.
```cpp
t
mp
at

typ

am
 sca
ar_t, 

t HEAD_SIZE, 

t BLOCK_SIZE, 

t NUM_THREADS, 

t PARTITION_SIZE = 0

__d
v
c
__ vo
d pag
d_att

t
o
_k
r


(
    ... // Oth
r s
d
 args.
    co
st sca
ar_t* __r
str
ct__ out,       // [
um_s
qs, 
um_h
ads, max_
um_part
t
o
s, h
ad_s
z
]
    co
st sca
ar_t* __r
str
ct__ q,         // [
um_s
qs, 
um_h
ads, h
ad_s
z
]
    co
st sca
ar_t* __r
str
ct__ k_cach
,   // [
um_b
ocks, 
um_kv_h
ads, h
ad_s
z
/x, b
ock_s
z
, x]
    co
st sca
ar_t* __r
str
ct__ v_cach
,   // [
um_b
ocks, 
um_kv_h
ads, h
ad_s
z
, b
ock_s
z
]
    ... // Oth
r s
d
 args.
)
```
Th
r
 ar
 a
so a 

st of t
mp
at
 argum

ts abov
 th
 fu
ct
o

s
g
atur
 that ar
 d
t
rm


d dur

g comp

at
o
 t
m
. `sca
ar_t`
r
pr
s

ts th
 data typ
 of th
 qu
ry, k
y, a
d va
u
 data 


m

ts,
such as FP16. `HEAD_SIZE` 

d
cat
s th
 
umb
r of 


m

ts 

 
ach
h
ad. `BLOCK_SIZE` r
f
rs to th
 
umb
r of tok

s 

 
ach b
ock.
`NUM_THREADS` d

ot
s th
 
umb
r of thr
ads 

 
ach thr
ad b
ock.
`PARTITION_SIZE` r
pr
s

ts th
 
umb
r of t

sor para


 GPUs (For
s
mp

c
ty, 

 assum
 th
s 
s 0 a
d t

sor para


 
s d
sab

d).
W
th th
s
 argum

ts, 

 

d to p
rform a s
qu

c
 of pr
parat
o
s.
Th
s 

c
ud
s ca
cu
at

g th
 curr

t h
ad 

d
x, b
ock 

d
x, a
d
oth
r 

c
ssary var
ab

s. Ho

v
r, for 
o
, 

 ca
 
g
or
 th
s

pr
parat
o
s a
d proc
d d
r
ct
y to th
 actua
 ca
cu
at
o
s. It 



b
 
as

r to u
d
rsta
d th
m o
c
 

 grasp th
 

t
r
 f
o
.
## Co
c
pts
Just b
for
 

 d
v
 

to th
 ca
cu
at
o
 f
o
, I 
a
t to d
scr
b
 a
f

 co
c
pts that ar
 

d
d for 
at
r s
ct
o
s. Ho

v
r, you may
sk
p th
s s
ct
o
 a
d r
tur
 
at
r 
f you 

cou
t
r a
y co
fus

g
t
rm

o
og

s.
- **S
qu

c
**: A s
qu

c
 r
pr
s

ts a c



t r
qu
st. For 
xamp

,
  th
 data po

t
d to by `q` has a shap
 of
  `[
um_s
qs, 
um_h
ads, h
ad_s
z
]`. That r
pr
s

ts th
r
 ar
 tota

  `
um_s
qs` of qu
ry s
qu

c
 data ar
 po

t
d by `q`. S

c
 th
s
  k
r


 
s a s

g

 qu
ry att

t
o
 k
r


, 
ach s
qu

c
 o

y has o


  qu
ry tok

. H

c
, th
 `
um_s
qs` 
qua
s th
 tota
 
umb
r of tok

s
  that ar
 proc
ss
d 

 th
 batch.
- **Co
t
xt**: Th
 co
t
xt co
s
sts of th
 g


rat
d tok

s from th

  s
qu

c
. For 

sta
c
, `["What", "
s", "your"]` ar
 th
 co
t
xt
  tok

s, a
d th
 

put qu
ry tok

 
s `"
am
"`. Th
 mod

 m
ght
  g


rat
 th
 tok

 `"?"`.
- **V
c**: Th
 v
c 
s a 

st of 


m

ts that ar
 f
tch
d a
d
  ca
cu
at
d tog
th
r. For qu
ry a
d k
y data, th
 v
c s
z

  (`VEC_SIZE`) 
s d
t
rm


d so that 
ach thr
ad group ca
 f
tch a
d
  ca
cu
at
 16 byt
s of data at a t
m
. For va
u
 data, th
 v
c s
z

  (`V_VEC_SIZE`) 
s d
t
rm


d so that 
ach thr
ad ca
 f
tch a
d
  ca
cu
at
 16 byt
s of data at a t
m
. For 
xamp

, 
f th

  `sca
ar_t` 
s FP16 (2 byt
s) a
d `THREAD_GROUP_SIZE` 
s 2, th

  `VEC_SIZE` 


 b
 4, 
h


 th
 `V_VEC_SIZE` 


 b
 8.
- **Thr
ad group**: Th
 thr
ad group 
s a sma
 group of
  thr
ads(`THREAD_GROUP_SIZE`) that f
tch
s a
d ca
cu
at
s o


  qu
ry tok

 a
d o

 k
y tok

 at a t
m
. Each thr
ad ha
d

s o

y a
  port
o
 of th
 tok

 data. Th
 tota
 
umb
r of 


m

ts proc
ss
d by
  o

 thr
ad group 
s r
f
rr
d as `x`. For 
xamp

, 
f th
 thr
ad
  group co
ta

s 2 thr
ads a
d th
 h
ad s
z
 
s 8, th

 thr
ad 0
  ha
d

s th
 qu
ry a
d k
y 


m

ts at 

d
x 0, 2, 4, 6, 
h


 thr
ad
  1 ha
d

s th
 


m

ts at 

d
x 1, 3, 5, 7.
- **B
ock**: Th
 k
y a
d va
u
 cach
 data 

 vLLM ar
 sp

t 

to
  b
ocks. Each b
ock stor
s data for a f
x
d 
umb
r(`BLOCK_SIZE`)
  of tok

s at o

 h
ad. Each b
ock may co
ta

 o

y a port
o
 of th

  
ho

 co
t
xt tok

s. For 
xamp

, 
f th
 b
ock s
z
 
s 16 a
d th

  h
ad s
z
 
s 128, th

 for o

 h
ad, o

 b
ock ca
 stor
 16 * 128 =
  2048 


m

ts.
- **Warp**: A 
arp 
s a group of 32 thr
ads(`WARP_SIZE`) that
  
x
cut
 s
mu
ta

ous
y o
 a str
am mu
t
proc
ssor (SM). I
 th
s
  k
r


, 
ach 
arp proc
ss
s th
 ca
cu
at
o
 b
t


 o

 qu
ry tok


  a
d k
y tok

s of o

 

t
r
 b
ock at a t
m
 (
t may proc
ss mu
t
p


  b
ocks 

 mu
t
p

 
t
rat
o
s). For 
xamp

, 
f th
r
 ar
 4 
arps a
d
  6 b
ocks for o

 co
t
xt, th
 ass
g
m

t 
ou
d b
 

k
 
arp 0 ha
d

s
  th
 0th, 4th b
ocks, 
arp 1 ha
d

s th
 1st, 5th b
ocks, 
arp 2
  ha
d

s th
 2
d b
ock a
d 
arp 3 ha
d

s th
 3rd b
ock.
- **Thr
ad b
ock**: A thr
ad b
ock 
s a group of
  thr
ads(`NUM_THREADS`) that ca
 acc
ss th
 sam
 shar
d m
mory.
  Each thr
ad b
ock co
ta

s mu
t
p

 
arps(`NUM_WARPS`), a
d 


  th
s k
r


, 
ach thr
ad b
ock proc
ss
s th
 ca
cu
at
o
 b
t


 o


  qu
ry tok

 a
d k
y tok

s of a 
ho

 co
t
xt.
- **Gr
d**: A gr
d 
s a co

ct
o
 of thr
ad b
ocks a
d d
f


s th

  shap
 of th
 co

ct
o
. I
 th
s k
r


, th
 shap
 
s
  `(
um_h
ads, 
um_s
qs, max_
um_part
t
o
s)`. Th
r
for
, 
ach thr
ad
  b
ock o

y ha
d

s th
 ca
cu
at
o
 for o

 h
ad, o

 s
qu

c
, a
d
  o

 part
t
o
.
## Qu
ry
Th
s s
ct
o
 


 

troduc
 ho
 qu
ry data 
s stor
d 

 m
mory a
d
f
tch
d by 
ach thr
ad. As m

t
o

d abov
, 
ach thr
ad group f
tch
s
o

 qu
ry tok

 data, 
h


 
ach thr
ad 
ts

f o

y ha
d

s a part of
o

 qu
ry tok

 data. W
th

 
ach 
arp, 
v
ry thr
ad group 


 f
tch
th
 sam
 qu
ry tok

 data, but 


 mu
t
p
y 
t 

th d
ff
r

t k
y
tok

 data.
```cpp
co
st sca
ar_t* q_ptr = q + s
q_
dx * q_str
d
 + h
ad_
dx * HEAD_SIZE;
```
![qu
ry](../ass
ts/d
s
g
/pag
d_att

t
o
/qu
ry.p
g)
Each thr
ad d
f


s 
ts o

 `q_ptr` 
h
ch po

ts to th
 ass
g

d
qu
ry tok

 data o
 g
oba
 m
mory. For 
xamp

, 
f `VEC_SIZE` 
s 4
a
d `HEAD_SIZE` 
s 128, th
 `q_ptr` po

ts to data that co
ta

s
tota
 of 128 


m

ts d
v
d
d 

to 128 / 4 = 32 v
cs.
![q_v
cs](../ass
ts/d
s
g
/pag
d_att

t
o
/q_v
cs.p
g)
```cpp
__shar
d__ Q_v
c q_v
cs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```
N
xt, 

 

d to r
ad th
 g
oba
 m
mory data po

t
d to by `q_ptr`


to shar
d m
mory as `q_v
cs`. It 
s 
mporta
t to 
ot
 that 
ach
v
cs 
s ass
g

d to a d
ff
r

t ro
. For 
xamp

, 
f th

`THREAD_GROUP_SIZE` 
s 2, thr
ad 0 


 ha
d

 th
 0th ro
 v
cs,

h


 thr
ad 1 ha
d

s th
 1st ro
 v
cs. By r
ad

g th
 qu
ry data 


th
s 
ay, 


ghbor

g thr
ads 

k
 thr
ad 0 a
d thr
ad 1 ca
 r
ad



ghbor m
mory, ach

v

g th
 m
mory coa

sc

g to 
mprov

p
rforma
c
.
## K
y
S
m

ar to th
 "Qu
ry" s
ct
o
, th
s s
ct
o
 

troduc
s m
mory 
ayout
a
d ass
g
m

t for k
ys. Wh


 
ach thr
ad group o

y ha
d

 o


qu
ry tok

 o

 k
r


 ru
, 
t may ha
d

 mu
t
p

 k
y tok

s across
mu
t
p

 
t
rat
o
s. M
a

h


, 
ach 
arp 


 proc
ss mu
t
p

 b
ocks
of k
y tok

s 

 mu
t
p

 
t
rat
o
s, 

sur

g that a
 co
t
xt
tok

s ar
 proc
ss
d by th
 

t
r
 thr
ad group aft
r th
 k
r


 ru
.
I
 th
s co
t
xt, "ha
d

" r
f
rs to p
rform

g th
 dot mu
t
p

cat
o

b
t


 qu
ry data a
d k
y data.
```cpp
co
st sca
ar_t* k_ptr = k_cach
 + phys
ca
_b
ock_
umb
r * kv_b
ock_str
d

                    + kv_h
ad_
dx * kv_h
ad_str
d

                    + phys
ca
_b
ock_offs
t * x;
```
U


k
 to `q_ptr`, `k_ptr` 

 
ach thr
ad 


 po

t to d
ff
r

t
k
y tok

 at d
ff
r

t 
t
rat
o
s. As sho

 abov
, that `k_ptr`
po

ts to k
y tok

 data bas
d o
 `k_cach
` at ass
g

d b
ock,
ass
g

d h
ad a
d ass
g

d tok

.
![k
y](../ass
ts/d
s
g
/pag
d_att

t
o
/k
y.p
g)
Th
 d
agram abov
 

ustrat
s th
 m
mory 
ayout for k
y data. It
assum
s that th
 `BLOCK_SIZE` 
s 16, `HEAD_SIZE` 
s 128, `x` 
s
8, `THREAD_GROUP_SIZE` 
s 2, a
d th
r
 ar
 a tota
 of 4 
arps. Each
r
cta
g

 r
pr
s

ts a
 th
 


m

ts for o

 k
y tok

 at o

 h
ad,

h
ch 


 b
 proc
ss
d by o

 thr
ad group. Th
 

ft ha
f sho
s th

tota
 16 b
ocks of k
y tok

 data for 
arp 0, 
h


 th
 r
ght ha
f
r
pr
s

ts th
 r
ma



g k
y tok

 data for oth
r 
arps or

t
rat
o
s. I
s
d
 
ach r
cta
g

, th
r
 ar
 a tota
 32 v
cs (128



m

ts for o

 tok

) that 


 b
 proc
ss
d by 2 thr
ads (o


thr
ad group) s
parat

y.
![k_v
cs](../ass
ts/d
s
g
/pag
d_att

t
o
/k_v
cs.p
g)
```cpp
K_v
c k_v
cs[NUM_VECS_PER_THREAD]
```
N
xt, 

 

d to r
ad th
 k
y tok

 data from `k_ptr` a
d stor

th
m o
 r
g
st
r m
mory as `k_v
cs`. W
 us
 r
g
st
r m
mory for
`k_v
cs` b
caus
 
t 


 o

y b
 acc
ss
d by o

 thr
ad o
c
,

h
r
as `q_v
cs` 


 b
 acc
ss
d by mu
t
p

 thr
ads mu
t
p


t
m
s. Each `k_v
cs` 


 co
ta

 mu
t
p

 v
ctors for 
at
r
ca
cu
at
o
. Each v
c 


 b
 s
t at 
ach 


r 
t
rat
o
. Th

ass
g
m

t of v
cs a
o
s 


ghbor

g thr
ads 

 a 
arp to r
ad



ghbor

g m
mory tog
th
r, 
h
ch aga

 promot
s th
 m
mory
coa

sc

g. For 

sta
c
, thr
ad 0 


 r
ad v
c 0, 
h


 thr
ad 1



 r
ad v
c 1. I
 th
 

xt 


r 
oop, thr
ad 0 


 r
ad v
c 2,

h


 thr
ad 1 


 r
ad v
c 3, a
d so o
.
You may st

 b
 a 

tt

 co
fus
d about th
 ov
ra
 f
o
. Do
't

orry, p

as
 k
p r
ad

g th
 

xt "QK" s
ct
o
. It 


 

ustrat

th
 qu
ry a
d k
y ca
cu
at
o
 f
o
 

 a c

ar
r a
d h
gh
r-

v


ma

r.
## QK
As sho

 th
 ps
udocod
 b

o
, b
for
 th
 

t
r
 for 
oop b
ock, 


f
tch th
 qu
ry data for o

 tok

 a
d stor
 
t 

 `q_v
cs`. Th

,


 th
 out
r for 
oop, 

 
t
rat
 through d
ff
r

t `k_ptrs` that
po

t to d
ff
r

t tok

s a
d pr
par
 th
 `k_v
cs` 

 th
 


r for

oop. F

a
y, 

 p
rform th
 dot mu
t
p

cat
o
 b
t


 th

`q_v
cs` a
d 
ach `k_v
cs`.
```cpp
q_v
cs = ...
for ... {
    k_ptr = ...
    for ... {
        k_v
cs[
] = ...
    }
    ...
    f
oat qk = sca

 * Qk_dot
sca
ar_t, THREAD_GROUP_SIZE
::dot(q_v
cs[thr
ad_group_offs
t], k_v
cs);
}
```
As m

t
o

d b
for
, for 
ach thr
ad, 
t o

y f
tch
s part of th

qu
ry a
d k
y tok

 data at a t
m
. Ho

v
r, th
r
 


 b
 a cross
thr
ad group r
duct
o
 happ

 

 th
 `Qk_dot

::dot` . So `qk`
r
tur

d h
r
 
s 
ot just b
t


 part of th
 qu
ry a
d k
y tok

 dot
mu
t
p

cat
o
, but actua
y a fu
 r
su
t b
t


 

t
r
 qu
ry a
d
k
y tok

 data.
For 
xamp

, 
f th
 va
u
 of `HEAD_SIZE` 
s 128 a
d
`THREAD_GROUP_SIZE` 
s 2, 
ach thr
ad's `k_v
cs` 


 co
ta


tota
 64 


m

ts. Ho

v
r, th
 r
tur

d `qk` 
s actua
y th

r
su
t of dot mu
t
p

cat
o
 b
t


 128 qu
ry 


m

ts a
d 128 k
y



m

ts. If you 
a
t to 

ar
 mor
 about th
 d
ta

s of th
 dot
mu
t
p

cat
o
 a
d r
duct
o
, you may r
f
r to th
 
mp

m

tat
o
 of
`Qk_dot

::dot`. Ho

v
r, for th
 sak
 of s
mp

c
ty, I 


 
ot
cov
r 
t 

 th
s docum

t.
## Softmax
N
xt, 

 

d to ca
cu
at
 th
 
orma

z
d softmax for a
 `qk`s,
as sho

 abov
, 
h
r
 
ach $x$ r
pr
s

ts a `qk`. To do th
s,


 must obta

 th
 r
duc
d va
u
 of `qk_max`($m(x)$) a
d
th
 `
xp_sum`($\

(x)$) of a
 `qk`s. Th
 r
duct
o

shou
d b
 p
rform
d across th
 

t
r
 thr
ad b
ock, 

compass

g
r
su
ts b
t


 th
 qu
ry tok

 a
d a
 co
t
xt k
y tok

s.
$$
\b
g

{gath
r*}
m(x):=\max _
 \quad x_
 \\ \quad f(x):=\

ft[\b
g

{array}{
}
^{x_1-m(x)} & \
dots & 
^{x_B-m(x)}\

d{array}\r
ght]\\ \quad \

(x):=\sum_
 f(x)_
 \\
\quad \op
rator
am
{softmax}(x):=\frac{f(x)}{\

(x)}
\

d{gath
r*}
$$
### `qk_max` a
d `
og
ts`
Just r
ght aft
r 

 g
t th
 `qk` r
su
t, 

 ca
 s
t th
 t
mporary
`
og
ts` r
su
t 

th `qk` (I
 th
 

d, th
 `
og
ts` shou
d
stor
 th
 
orma

z
d softmax r
su
t). A
so 

 ca
 compar
 a
d co

ct
th
 `qk_max` for a
 `qk`s that ar
 ca
cu
at
d by curr

t
thr
ad group.
```cpp

f (thr
ad_group_offs
t == 0) {
    co
st boo
 mask = tok

_
dx 
= co
t
xt_


;
    
og
ts[tok

_
dx - start_tok

_
dx] = mask ? 0.f : qk;
    qk_max = mask ? qk_max : fmaxf(qk_max, qk);
}
```
P

as
 
ot
 that th
 `
og
ts` h
r
 
s o
 shar
d m
mory, so 
ach
thr
ad group 


 s
t th
 f


ds for 
ts o

 ass
g

d co
t
xt tok

s.
Ov
ra
, th
 s
z
 of 
og
ts shou
d b
 
umb
r of co
t
xt tok

s.
```cpp
for (

t mask = WARP_SIZE / 2; mask 
= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}

f (
a

 == 0) {
    r
d_sm
m[
arp_
dx] = qk_max;
}
```
Th

 

 

d to g
t th
 r
duc
d `qk_max` across 
ach 
arp. Th
 ma



d
a 
s to mak
 thr
ads 

 
arp to commu

cat
 

th 
ach oth
r a
d
g
t th
 f

a
 max `qk` .
```cpp
for (

t mask = NUM_WARPS / 2; mask 
= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}
qk_max = VLLM_SHFL_SYNC(qk_max, 0);
```
F

a
y, 

 ca
 g
t th
 r
duc
d `qk_max` from 
ho

 thr
ad b
ock by
compar
 th
 `qk_max` from a
 
arps 

 th
s thr
ad b
ock. Th

 




d to broadcast th
 f

a
 r
su
t to 
ach thr
ad.
### `
xp_sum`
S
m

ar to `qk_max`, 

 

d to g
t th
 r
duc
d sum va
u
 from th



t
r
 thr
ad b
ock too.
```cpp
for (

t 
 = thr
ad_
dx; 
 
 
um_tok

s; 
 += NUM_THREADS) {
    f
oat va
 = __
xpf(
og
ts[
] - qk_max);
    
og
ts[
] = va
;
    
xp_sum += va
;
}
...

xp_sum = b
ock_sum
NUM_WARPS
(&r
d_sm
m[NUM_WARPS], 
xp_sum);
```
F
rst
y, sum a
 
xp va
u
s from 
ach thr
ad group, a
d m
a

h


,
co
v
rt 
ach 

try of `
og
ts` from `qk` to `
xp(qk - qk_max)`.
P

as
 
ot
, th
 `qk_max` h
r
 
s a
r
ady th
 max `qk` across th


ho

 thr
ad b
ock. A
d th

 

 ca
 do r
duct
o
 for `
xp_sum`
across 
ho

 thr
ad b
ock just 

k
 th
 `qk_max`.
```cpp
co
st f
oat 

v_sum = __fd
v
d
f(1.f, 
xp_sum + 1
-6f);
for (

t 
 = thr
ad_
dx; 
 
 
um_tok

s; 
 += NUM_THREADS) {
    
og
ts[
] *= 

v_sum;
}
```
F

a
y, 

th th
 r
duc
d `qk_max` a
d `
xp_sum`, 

 ca
 obta


th
 f

a
 
orma

z
d softmax r
su
t as `
og
ts`. Th
s `
og
ts`
var
ab

 


 b
 us
d for dot mu
t
p

cat
o
 

th th
 va
u
 data 



at
r st
ps. No
, 
t shou
d stor
 th
 
orma

z
d softmax r
su
t of
`qk` for a
 ass
g

d co
t
xt tok

s.
## Va
u

![va
u
](../ass
ts/d
s
g
/pag
d_att

t
o
/va
u
.p
g)
![
og
ts_v
c](../ass
ts/d
s
g
/pag
d_att

t
o
/
og
ts_v
c.p
g)
![v_v
c](../ass
ts/d
s
g
/pag
d_att

t
o
/v_v
c.p
g)
No
 

 

d to r
tr

v
 th
 va
u
 data a
d p
rform dot mu
t
p

cat
o



th `
og
ts`. U


k
 qu
ry a
d k
y, th
r
 
s 
o thr
ad group
co
c
pt for va
u
 data. As sho

 

 d
agram, d
ff
r

t from k
y tok


m
mory 
ayout, 


m

ts from th
 sam
 co
um
 corr
spo
d to th
 sam

va
u
 tok

. For o

 b
ock of va
u
 data, th
r
 ar
 `HEAD_SIZE` of
ro
s a
d `BLOCK_SIZE` of co
um
s that ar
 sp

t 

to mu
t
p


`v_v
cs`.
Each thr
ad a

ays f
tch
s `V_VEC_SIZE` 


m

ts from th
 sam

`V_VEC_SIZE` of tok

s at a t
m
. As a r
su
t, a s

g

 thr
ad
r
tr

v
s mu
t
p

 `v_v
c`s from d
ff
r

t ro
s a
d th
 sam

co
um
s through mu
t
p

 


r 
t
rat
o
s. For 
ach `v_v
c`, 
t


ds to b
 dot mu
t
p


d 

th th
 corr
spo
d

g `
og
ts_v
c`,

h
ch 
s a
so `V_VEC_SIZE` 


m

ts from `
og
ts`. Ov
ra
, 

th
mu
t
p

 


r 
t
rat
o
s, 
ach 
arp 


 proc
ss o

 b
ock of va
u

tok

s. A
d 

th mu
t
p

 out
r 
t
rat
o
s, th
 
ho

 co
t
xt va
u

tok

s ar
 proc
ss
d
```cpp
f
oat accs[NUM_ROWS_PER_THREAD];
for ... { // It
rat
o
 ov
r d
ff
r

t b
ocks.
    
og
ts_v
c = ...
    for ... { // It
rat
o
 ov
r d
ff
r

t ro
s.
        v_v
c = ...
        ...
        accs[
] += dot(
og
ts_v
c, v_v
c);
    }
}
```
As sho

 

 th
 abov
 ps
udocod
, 

 th
 out
r 
oop, s
m

ar to
`k_ptr`, `
og
ts_v
c` 
t
rat
s ov
r d
ff
r

t b
ocks a
d r
ads
`V_VEC_SIZE` 


m

ts from `
og
ts`. I
 th
 


r 
oop, 
ach
thr
ad r
ads `V_VEC_SIZE` 


m

ts from th
 sam
 tok

s as a
`v_v
c` a
d p
rforms dot mu
t
p

cat
o
. It 
s 
mporta
t to 
ot

that 

 
ach 


r 
t
rat
o
, th
 thr
ad f
tch
s d
ff
r

t h
ad
pos
t
o
 


m

ts for th
 sam
 tok

s. Th
 dot r
su
t 
s th


accumu
at
d 

 `accs`. Th
r
for
, 
ach 

try of `accs` 
s mapp
d
to a h
ad pos
t
o
 ass
g

d to th
 curr

t thr
ad.
For 
xamp

, 
f `BLOCK_SIZE` 
s 16 a
d `V_VEC_SIZE` 
s 8, 
ach
thr
ad f
tch
s 8 va
u
 


m

ts for 8 tok

s at a t
m
. Each 


m

t

s from d
ff
r

t tok

s at th
 sam
 h
ad pos
t
o
. If `HEAD_SIZE`

s 128 a
d `WARP_SIZE` 
s 32, for 
ach 


r 
oop, a 
arp 

ds to
f
tch `WARP_SIZE * V_VEC_SIZE = 256` 


m

ts. Th
s m
a
s th
r
 ar

a tota
 of 128 * 16 / 256 = 8 


r 
t
rat
o
s for a 
arp to ha
d


a 
ho

 b
ock of va
u
 tok

s. A
d 
ach `accs` 

 
ach thr
ad
co
ta

s 8 


m

ts that accumu
at
d at 8 d
ff
r

t h
ad pos
t
o
s.
For th
 thr
ad 0, th
 `accs` var
ab

 


 hav
 8 


m

ts, 
h
ch
ar
 0th, 32
d … 224th 


m

ts of a va
u
 h
ad that ar
 accumu
at
d
from a
 ass
g

d 8 tok

s.
## LV
No
, 

 

d to p
rform r
duct
o
 for `accs` 

th

 
ach 
arp. Th
s
proc
ss a
o
s 
ach thr
ad to accumu
at
 th
 `accs` for th

ass
g

d h
ad pos
t
o
s of a
 tok

s 

 o

 b
ock.
```cpp
for (

t 
 = 0; 
 
 NUM_ROWS_PER_THREAD; 
++) {
    f
oat acc = accs[
];
    for (

t mask = NUM_V_VECS_PER_ROW / 2; mask 
= 1; mask /= 2) {
        acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[
] = acc;
}
```
N
xt, 

 p
rform r
duct
o
 for `accs` across a
 
arps, a
o


g

ach thr
ad to hav
 th
 accumu
at
o
 of `accs` for th
 ass
g

d
h
ad pos
t
o
s of a
 co
t
xt tok

s. P

as
 
ot
 that 
ach `accs`


 
v
ry thr
ad o

y stor
s th
 accumu
at
o
 for a port
o
 of



m

ts of th
 

t
r
 h
ad for a
 co
t
xt tok

s. Ho

v
r, ov
ra
,
a
 r
su
ts for output hav
 b

 ca
cu
at
d but ar
 just stor
d 


d
ff
r

t thr
ad r
g
st
r m
mory.
??? cod

    ```cpp
    f
oat* out_sm
m = r


t
rpr
t_cast
f
oat*
(shar
d_m
m);
    for (

t 
 = NUM_WARPS; 
 
 1; 
 /= 2) {
        // Upp
r 
arps 
r
t
 to shar
d m
mory.
        ...
        f
oat* dst = &out_sm
m[(
arp_
dx - m
d) * HEAD_SIZE];
        for (

t 
 = 0; 
 
 NUM_ROWS_PER_THREAD; 
++) {
            ...
            dst[ro
_
dx] = accs[
];
        }
        // Lo

r 
arps updat
 th
 output.
        co
st f
oat* src = &out_sm
m[
arp_
dx * HEAD_SIZE];
        for (

t 
 = 0; 
 
 NUM_ROWS_PER_THREAD; 
++) {
            ...
            accs[
] += src[ro
_
dx];
        }
        // Wr
t
 out th
 accs.
    }
    ```
## Output
No
 

 ca
 
r
t
 a
 of ca
cu
at
d r
su
t from 
oca
 r
g
st
r m
mory
to f

a
 output g
oba
 m
mory.
```cpp
sca
ar_t* out_ptr = out + s
q_
dx * 
um_h
ads * max_
um_part
t
o
s * HEAD_SIZE
                + h
ad_
dx * max_
um_part
t
o
s * HEAD_SIZE
                + part
t
o
_
dx * HEAD_SIZE;
```
F
rst, 

 

d to d
f


 th
 `out_ptr` var
ab

, 
h
ch po

ts to
th
 start addr
ss of th
 ass
g

d s
qu

c
 a
d ass
g

d h
ad.
```cpp
for (

t 
 = 0; 
 
 NUM_ROWS_PER_THREAD; 
++) {
    co
st 

t ro
_
dx = 
a

 / NUM_V_VECS_PER_ROW + 
 * NUM_ROWS_PER_ITER;
    
f (ro
_
dx 
 HEAD_SIZE && 
a

 % NUM_V_VECS_PER_ROW == 0) {
        from_f
oat(*(out_ptr + ro
_
dx), accs[
]);
    }
}
```
F

a
y, 

 

d to 
t
rat
 ov
r d
ff
r

t ass
g

d h
ad pos
t
o
s
a
d 
r
t
 out th
 corr
spo
d

g accumu
at
d r
su
t bas
d o
 th

`out_ptr`.
## C
tat
o

```b
bt
x
@

proc
d

gs{k
o
2023
ff
c


t,
  t
t

={Eff
c


t M
mory Ma
ag
m

t for Larg
 La
guag
 Mod

 S
rv

g 

th Pag
dAtt

t
o
},
  author={Woosuk K
o
 a
d Zhuoha
 L
 a
d S
yua
 Zhua
g a
d Y

g Sh

g a
d L
a
m

 Zh

g a
d Cody Hao Yu a
d Jos
ph E. Go
za

z a
d Hao Zha
g a
d Io
 Sto
ca},
  bookt
t

={Proc
d

gs of th
 ACM SIGOPS 29th Sympos
um o
 Op
rat

g Syst
ms Pr

c
p

s},
  y
ar={2023}
}
```
