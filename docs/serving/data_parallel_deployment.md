# Data Para


 D
p
oym

t
vLLM supports Data Para


 d
p
oym

t, 
h
r
 mod

 


ghts ar
 r
p

cat
d across s
parat
 

sta
c
s/GPUs to proc
ss 

d
p

d

t batch
s of r
qu
sts.
Th
s 


 
ork 

th both d

s
 a
d MoE mod

s.
For MoE mod

s, part
cu
ar
y thos
 

k
 D
pS
k that 
mp
oy MLA (Mu
t
-h
ad Lat

t Att

t
o
), 
t ca
 b
 adva
tag
ous to us
 data para


 for th
 att

t
o
 
ay
rs a
d 
xp
rt or t

sor para


 (EP or TP) for th
 
xp
rt 
ay
rs.
I
 th
s
 cas
s, th
 data para


 ra
ks ar
 
ot comp

t

y 

d
p

d

t. For
ard pass
s must b
 a

g

d, a
d 
xp
rt 
ay
rs across a
 ra
ks ar
 r
qu
r
d to sy
chro

z
 dur

g 
v
ry for
ard pass, 
v

 
h

 th
r
 ar
 f


r r
qu
sts to b
 proc
ss
d tha
 DP ra
ks.
By d
fau
t, 
xp
rt 
ay
rs form a t

sor para


 group of s
z
 `DP × TP`. To us
 
xp
rt para



sm 

st
ad, 

c
ud
 th
 `--

ab

-
xp
rt-para


` CLI arg (o
 a
 
od
s 

 th
 mu
t
-
od
 cas
). S
 [Exp
rt Para


 D
p
oym

t](
xp
rt_para


_d
p
oym

t.md) for d
ta

s o
 ho
 att

t
o
 a
d 
xp
rt 
ay
rs b
hav
 d
ff
r

t
y 

th EP 

ab

d.
I
 vLLM, 
ach DP ra
k 
s d
p
oy
d as a s
parat
 "cor
 

g


" proc
ss that commu

cat
s 

th fro
t-

d proc
ss(
s) v
a ZMQ sock
ts. Data Para


 att

t
o
 ca
 b
 comb


d 

th T

sor Para


 att

t
o
, 

 
h
ch cas
 
ach DP 

g


 o

s a 
umb
r of p
r-GPU 
ork
r proc
ss
s 
qua
 to th
 co
f
gur
d TP s
z
.
For MoE mod

s, 
h

 a
y r
qu
sts ar
 

 progr
ss 

 a
y ra
k, 

 must 

sur
 that 
mpty "dummy" for
ard pass
s ar
 p
rform
d 

 a
 ra
ks that do
't curr

t
y hav
 a
y r
qu
sts sch
du

d. Th
s 
s ha
d

d v
a a s
parat
 DP Coord

ator proc
ss that commu

cat
s 

th a
 ra
ks, a
d a co

ct
v
 op
rat
o
 p
rform
d 
v
ry N st
ps to d
t
rm


 
h

 a
 ra
ks b
com
 
d

 a
d ca
 b
 paus
d. Wh

 TP 
s us
d 

 co
ju
ct
o
 

th DP, 
xp
rt 
ay
rs form a group of s
z
 `DP × TP` (us

g 

th
r t

sor para



sm by d
fau
t, or 
xp
rt para



sm 
f `--

ab

-
xp
rt-para


` 
s s
t).
I
 a
 cas
s, 
t 
s b


f
c
a
 to 
oad-ba
a
c
 r
qu
sts b
t


 DP ra
ks. For o




 d
p
oym

ts, th
s ba
a
c

g ca
 b
 opt
m
z
d by tak

g 

to accou
t th
 stat
 of 
ach DP 

g


 - 

 part
cu
ar 
ts curr

t
y sch
du

d a
d 
a
t

g (qu
u
d) r
qu
sts, a
d KV cach
 stat
. Each DP 

g


 has a
 

d
p

d

t KV cach
, a
d th
 b


f
t of pr
f
x cach

g ca
 b
 max
m
z
d by d
r
ct

g prompts 

t


g

t
y.
Th
s docum

t focus
s o
 o




 d
p
oym

ts (

th th
 API s
rv
r). DP + EP 
s a
so support
d for off



 usag
 (v
a th
 LLM c
ass), for a
 
xamp

 s
 [
xamp

s/off



_

f
r

c
/data_para


.py](../../
xamp

s/off



_

f
r

c
/data_para


.py).
Th
r
 ar
 t
o d
st

ct mod
s support
d for o




 d
p
oym

ts - s

f-co
ta


d 

th 

t
r
a
 
oad ba
a
c

g, or 
xt
r
a
y p
r-ra
k proc
ss d
p
oym

t a
d 
oad ba
a
c

g.
## I
t
r
a
 Load Ba
a
c

g
vLLM supports "s

f-co
ta


d" data para


 d
p
oym

ts that 
xpos
 a s

g

 API 

dpo

t.
It ca
 b
 co
f
gur
d by s
mp
y 

c
ud

g 
.g. `--data-para


-s
z
=4` 

 th
 v
m s
rv
 comma
d 



 argum

ts. Th
s 


 r
qu
r
 4 GPUs. It ca
 b
 comb


d 

th t

sor para


, for 
xamp

 `--data-para


-s
z
=4 --t

sor-para


-s
z
=2`, 
h
ch 
ou
d r
qu
r
 8 GPUs. Wh

 s
z

g DP d
p
oym

ts, r
m
mb
r that `--max-
um-s
qs` app


s p
r DP ra
k.
Ru


g a s

g

 data para


 d
p
oym

t across mu
t
p

 
od
s r
qu
r
s a d
ff
r

t `v
m s
rv
` to b
 ru
 o
 
ach 
od
, sp
c
fy

g 
h
ch DP ra
ks shou
d ru
 o
 that 
od
. I
 th
s cas
, th
r
 


 st

 b
 a s

g

 HTTP 

trypo

t - th
 API s
rv
r(s) 


 ru
 o

y o
 o

 
od
, but 
t do
s
't 

c
ssar

y 

d to b
 co-
ocat
d 

th th
 DP ra
ks.
Th
s 


 ru
 DP=4, TP=2 o
 a s

g

 8-GPU 
od
:
```bash
v
m s
rv
 $MODEL --data-para


-s
z
 4 --t

sor-para


-s
z
 2
```
Th
s 


 ru
 DP=4 

th DP ra
ks 0 a
d 1 o
 th
 h
ad 
od
 a
d ra
ks 2 a
d 3 o
 th
 s
co
d 
od
:
```bash
# Nod
 0  (

th 
p addr
ss 10.99.48.128)
v
m s
rv
 $MODEL --data-para


-s
z
 4 --data-para


-s
z
-
oca
 2 \
                  --data-para


-addr
ss 10.99.48.128 --data-para


-rpc-port 13345
# Nod
 1
v
m s
rv
 $MODEL --h
ad

ss --data-para


-s
z
 4 --data-para


-s
z
-
oca
 2 \
                  --data-para


-start-ra
k 2 \
                  --data-para


-addr
ss 10.99.48.128 --data-para


-rpc-port 13345
```
Th
s 


 ru
 DP=4 

th o

y th
 API s
rv
r o
 th
 f
rst 
od
 a
d a
 

g


s o
 th
 s
co
d 
od
:
```bash
# Nod
 0  (

th 
p addr
ss 10.99.48.128)
v
m s
rv
 $MODEL --data-para


-s
z
 4 --data-para


-s
z
-
oca
 0 \
                  --data-para


-addr
ss 10.99.48.128 --data-para


-rpc-port 13345
# Nod
 1
v
m s
rv
 $MODEL --h
ad

ss --data-para


-s
z
 4 --data-para


-s
z
-
oca
 4 \
                  --data-para


-addr
ss 10.99.48.128 --data-para


-rpc-port 13345
```
Th
s DP mod
 ca
 a
so b
 us
d 

th Ray by sp
c
fy

g `--data-para


-back

d=ray`:
```bash
v
m s
rv
 $MODEL --data-para


-s
z
 4 --data-para


-s
z
-
oca
 2 \
                  --data-para


-back

d=ray
```
Th
r
 ar
 s
v
ra
 
otab

 d
ff
r

c
s 
h

 us

g Ray:
- A s

g

 
au
ch comma
d (o
 a
y 
od
) 
s 

d
d to start a
 
oca
 a
d r
mot
 DP ra
ks, th
r
for
 
t 
s mor
 co
v




t compar
d to 
au
ch

g o
 
ach 
od

- Th
r
 
s 
o 

d to sp
c
fy `--data-para


-addr
ss`, a
d th
 
od
 
h
r
 th
 comma
d 
s ru
 
s us
d as `--data-para


-addr
ss`
- Th
r
 
s 
o 

d to sp
c
fy `--data-para


-rpc-port`
- Wh

 a s

g

 DP group r
qu
r
s mu
t
p

 
od
s, *
.g.* 

 cas
 a s

g

 mod

 r
p

ca 

ds to ru
 o
 at 

ast t
o 
od
s, mak
 sur
 to s
t `VLLM_RAY_DP_PACK_STRATEGY="spa
"` 

 
h
ch cas
 `--data-para


-s
z
-
oca
` 
s 
g
or
d a
d 


 b
 automat
ca
y d
t
rm


d
- R
mot
 DP ra
ks 


 b
 a
ocat
d bas
d o
 
od
 r
sourc
s of th
 Ray c
ust
r
Curr

t
y, th
 

t
r
a
 DP 
oad ba
a
c

g 
s do

 

th

 th
 API s
rv
r proc
ss(
s) a
d 
s bas
d o
 th
 ru


g a
d 
a
t

g qu
u
s 

 
ach of th
 

g


s. Th
s cou
d b
 mad
 mor
 soph
st
cat
d 

 futur
 by 

corporat

g KV cach
 a
ar
 
og
c.
Wh

 d
p
oy

g 
arg
 DP s
z
s us

g th
s m
thod, th
 API s
rv
r proc
ss ca
 b
com
 a bott



ck. I
 th
s cas
, th
 orthogo
a
 `--ap
-s
rv
r-cou
t` comma
d 



 opt
o
 ca
 b
 us
d to sca

 th
s out (for 
xamp

 `--ap
-s
rv
r-cou
t=4`). Th
s 
s tra
spar

t to us
rs - a s

g

 HTTP 

dpo

t / port 
s st

 
xpos
d. Not
 that th
s API s
rv
r sca

-out 
s "

t
r
a
" a
d st

 co
f


d to th
 "h
ad" 
od
.
f
gur
 markdo

="1"

![DP I
t
r
a
 LB D
agram](../ass
ts/d
p
oym

t/dp_

t
r
a
_
b.p
g)
/f
gur


## Hybr
d Load Ba
a
c

g
Hybr
d 
oad ba
a
c

g s
ts b
t


 th
 

t
r
a
 a
d 
xt
r
a
 approach
s. Each 
od
 ru
s 
ts o

 API s
rv
r(s) that o

y qu
u
 r
qu
sts to th
 data-para


 

g


s co
ocat
d o
 that 
od
. A
 upstr
am 
oad ba
a
c
r (for 
xamp

, a
 

gr
ss co
tro

r or traff
c rout
r) spr
ads us
r r
qu
sts across thos
 p
r-
od
 

dpo

ts.
E
ab

 th
s mod
 

th `--data-para


-hybr
d-
b` 
h


 st

 
au
ch

g 
v
ry 
od
 

th th
 g
oba
 data-para


 s
z
. Th
 k
y d
ff
r

c
s from 

t
r
a
 
oad ba
a
c

g ar
:
- You must prov
d
 `--data-para


-s
z
-
oca
` a
d `--data-para


-start-ra
k` so 
ach 
od
 k
o
s 
h
ch ra
ks 
t o

s.
- Not compat
b

 

th `--h
ad

ss` s

c
 
v
ry 
od
 
xpos
s a
 API 

dpo

t.
- Sca

 `--ap
-s
rv
r-cou
t` p
r 
od
 bas
d o
 th
 
umb
r of 
oca
 ra
ks
I
 th
s co
f
gurat
o
, 
ach 
od
 k
ps sch
du


g d
c
s
o
s 
oca
, 
h
ch r
duc
s cross-
od
 traff
c a
d avo
ds s

g

 
od
 bott



cks at 
arg
r DP s
z
s.
## Ext
r
a
 Load Ba
a
c

g
For 
arg
r sca

 d
p
oym

ts 
sp
c
a
y, 
t ca
 mak
 s

s
 to ha
d

 th
 orch
strat
o
 a
d 
oad ba
a
c

g of data para


 ra
ks 
xt
r
a
y.
I
 th
s cas
, 
t's mor
 co
v




t to tr
at 
ach DP ra
k 

k
 a s
parat
 vLLM d
p
oym

t, 

th 
ts o

 

dpo

t, a
d hav
 a
 
xt
r
a
 rout
r ba
a
c
 HTTP r
qu
sts b
t


 th
m, mak

g us
 of appropr
at
 r
a
-t
m
 t


m
try from 
ach s
rv
r for rout

g d
c
s
o
s.
Th
s ca
 a
r
ady b
 do

 tr
v
a
y for 
o
-MoE mod

s, s

c
 
ach d
p
oy
d s
rv
r 
s fu
y 

d
p

d

t. No data para


 CLI opt
o
s 

d to b
 us
d for th
s.
W
 support a
 
qu
va


t topo
ogy for MoE DP+EP 
h
ch ca
 b
 co
f
gur
d v
a th
 fo
o


g CLI argum

ts.
If DP ra
ks ar
 co-
ocat
d (sam
 
od
 / 
p addr
ss), a d
fau
t RPC port 
s us
d, but a d
ff
r

t HTTP s
rv
r port must b
 sp
c
f

d for 
ach ra
k:
```bash
# Ra
k 0
CUDA_VISIBLE_DEVICES=0 v
m s
rv
 $MODEL --data-para


-s
z
 2 --data-para


-ra
k 0 \
                                         --port 8000
# Ra
k 1
CUDA_VISIBLE_DEVICES=1 v
m s
rv
 $MODEL --data-para


-s
z
 2 --data-para


-ra
k 1 \
                                         --port 8001
```
For mu
t
-
od
 cas
s, th
 addr
ss/port of ra
k 0 must a
so b
 sp
c
f

d:
```bash
# Ra
k 0  (

th 
p addr
ss 10.99.48.128)
v
m s
rv
 $MODEL --data-para


-s
z
 2 --data-para


-ra
k 0 \
                  --data-para


-addr
ss 10.99.48.128 --data-para


-rpc-port 13345
# Ra
k 1
v
m s
rv
 $MODEL --data-para


-s
z
 2 --data-para


-ra
k 1 \
                  --data-para


-addr
ss 10.99.48.128 --data-para


-rpc-port 13345
```
Th
 coord

ator proc
ss a
so ru
s 

 th
s sc

ar
o, co-
ocat
d 

th th
 DP ra
k 0 

g


.
f
gur
 markdo

="1"

![DP Ext
r
a
 LB D
agram](../ass
ts/d
p
oym

t/dp_
xt
r
a
_
b.p
g)
/f
gur


I
 th
 abov
 d
agram, 
ach of th
 dott
d box
s corr
spo
ds to a s
parat
 
au
ch of `v
m s
rv
` - th
s
 cou
d b
 s
parat
 Kub
r

t
s pods, for 
xamp

.
