# Para



sm a
d Sca


g
## D
str
but
d 

f
r

c
 strat
g

s for a s

g

-mod

 r
p

ca
To choos
 a d
str
but
d 

f
r

c
 strat
gy for a s

g

-mod

 r
p

ca, us
 th
 fo
o


g gu
d




s:
- **S

g

 GPU (
o d
str
but
d 

f
r

c
):** 
f th
 mod

 f
ts o
 a s

g

 GPU, d
str
but
d 

f
r

c
 
s probab
y u

c
ssary. Ru
 

f
r

c
 o
 that GPU.
- **S

g

-
od
 mu
t
-GPU us

g t

sor para


 

f
r

c
:** 
f th
 mod

 
s too 
arg
 for a s

g

 GPU but f
ts o
 a s

g

 
od
 

th mu
t
p

 GPUs, us
 *t

sor para



sm*. For 
xamp

, s
t `t

sor_para


_s
z
=4` 
h

 us

g a 
od
 

th 4 GPUs.
- **Mu
t
-
od
 mu
t
-GPU us

g t

sor para


 a
d p
p




 para


 

f
r

c
:** 
f th
 mod

 
s too 
arg
 for a s

g

 
od
, comb


 *t

sor para



sm* 

th *p
p




 para



sm*. S
t `t

sor_para


_s
z
` to th
 
umb
r of GPUs p
r 
od
 a
d `p
p




_para


_s
z
` to th
 
umb
r of 
od
s. For 
xamp

, s
t `t

sor_para


_s
z
=8` a
d `p
p




_para


_s
z
=2` 
h

 us

g 2 
od
s 

th 8 GPUs p
r 
od
.
I
cr
as
 th
 
umb
r of GPUs a
d 
od
s u
t

 th
r
 
s 

ough GPU m
mory for th
 mod

. S
t `t

sor_para


_s
z
` to th
 
umb
r of GPUs p
r 
od
 a
d `p
p




_para


_s
z
` to th
 
umb
r of 
od
s.
Aft
r you prov
s
o
 suff
c


t r
sourc
s to f
t th
 mod

, ru
 `v
m`. Look for 
og m
ssag
s 

k
:
```t
xt
INFO 07-23 13:56:04 [kv_cach
_ut

s.py:775] GPU KV cach
 s
z
: 643,232 tok

s
INFO 07-23 13:56:04 [kv_cach
_ut

s.py:779] Max
mum co
curr

cy for 40,960 tok

s p
r r
qu
st: 15.70x
```
Th
 `GPU KV cach
 s
z
` 



 r
ports th
 tota
 
umb
r of tok

s that ca
 b
 stor
d 

 th
 GPU KV cach
 at o
c
. Th
 `Max
mum co
curr

cy` 



 prov
d
s a
 
st
mat
 of ho
 ma
y r
qu
sts ca
 b
 s
rv
d co
curr

t
y 
f 
ach r
qu
st r
qu
r
s th
 sp
c
f

d 
umb
r of tok

s (40,960 

 th
 
xamp

 abov
). Th
 tok

s-p
r-r
qu
st 
umb
r 
s tak

 from th
 mod

 co
f
gurat
o
's max
mum s
qu

c
 


gth, `Mod

Co
f
g.max_mod

_


`. If th
s
 
umb
rs ar
 
o

r tha
 your throughput r
qu
r
m

ts, add mor
 GPUs or 
od
s to your c
ust
r.
!!! 
ot
 "Edg
 cas
: u

v

 GPU sp

ts"
    If th
 mod

 f
ts 

th

 a s

g

 
od
 but th
 GPU cou
t do
s
't 
v


y d
v
d
 th
 mod

 s
z
, 

ab

 p
p




 para



sm, 
h
ch sp

ts th
 mod

 a
o
g 
ay
rs a
d supports u

v

 sp

ts. I
 th
s sc

ar
o, s
t `t

sor_para


_s
z
=1` a
d `p
p




_para


_s
z
` to th
 
umb
r of GPUs. Furth
rmor
, 
f th
 GPUs o
 th
 
od
 do 
ot hav
 NVLINK 

t
rco

ct (
.g. L40S), 

v
rag
 p
p




 para



sm 

st
ad of t

sor para



sm for h
gh
r throughput a
d 
o

r commu

cat
o
 ov
rh
ad.
### D
str
but
d s
rv

g of *M
xtur
 of Exp
rts* (*MoE*) mod

s
It's oft

 adva
tag
ous to 
xp
o
t th
 

h
r

t para



sm of 
xp
rts by us

g a s
parat
 para



sm strat
gy for th
 
xp
rt 
ay
rs. vLLM supports 
arg
-sca

 d
p
oym

t comb



g Data Para


 att

t
o
 

th Exp
rt or T

sor Para


 MoE 
ay
rs. For mor
 

format
o
, s
 [Data Para


 D
p
oym

t](data_para


_d
p
oym

t.md).
## S

g

-
od
 d
p
oym

t
vLLM supports d
str
but
d t

sor-para


 a
d p
p




-para


 

f
r

c
 a
d s
rv

g. Th
 
mp

m

tat
o
 

c
ud
s [M
gatro
-LM's t

sor para


 a
gor
thm](https://arx
v.org/pdf/1909.08053.pdf).
Th
 d
fau
t d
str
but
d ru
t
m
s ar
 [Ray](https://g
thub.com/ray-proj
ct/ray) for mu
t
-
od
 

f
r

c
 a
d 
at
v
 Pytho
 `mu
t
proc
ss

g` for s

g

-
od
 

f
r

c
. You ca
 ov
rr
d
 th
 d
fau
ts by s
tt

g `d
str
but
d_
x
cutor_back

d` 

 th
 `LLM` c
ass or `--d
str
but
d-
x
cutor-back

d` 

 th
 API s
rv
r. Us
 `mp` for `mu
t
proc
ss

g` or `ray` for Ray.
For mu
t
-GPU 

f
r

c
, s
t `t

sor_para


_s
z
` 

 th
 `LLM` c
ass to th
 d
s
r
d GPU cou
t. For 
xamp

, to ru
 

f
r

c
 o
 4 GPUs:
```pytho

from v
m 
mport LLM

m = LLM("fac
book/opt-13b", t

sor_para


_s
z
=4)
output = 
m.g


rat
("Sa
 Fra
c
sco 
s a")
```
For mu
t
-GPU s
rv

g, 

c
ud
 `--t

sor-para


-s
z
` 
h

 start

g th
 s
rv
r. For 
xamp

, to ru
 th
 API s
rv
r o
 4 GPUs:
```bash
v
m s
rv
 fac
book/opt-13b \
     --t

sor-para


-s
z
 4
```
To 

ab

 p
p




 para



sm, add `--p
p




-para


-s
z
`. For 
xamp

, to ru
 th
 API s
rv
r o
 8 GPUs 

th p
p




 para



sm a
d t

sor para



sm:
```bash
# E
ght GPUs tota

v
m s
rv
 gpt2 \
     --t

sor-para


-s
z
 4 \
     --p
p




-para


-s
z
 2
```
## Mu
t
-
od
 d
p
oym

t
If a s

g

 
od
 
acks suff
c


t GPUs to ho
d th
 mod

, d
p
oy vLLM across mu
t
p

 
od
s. E
sur
 that 
v
ry 
od
 prov
d
s a
 
d

t
ca
 
x
cut
o
 

v
ro
m

t, 

c
ud

g th
 mod

 path a
d Pytho
 packag
s. Us

g co
ta


r 
mag
s 
s r
comm

d
d b
caus
 th
y prov
d
 a co
v




t 
ay to k
p 

v
ro
m

ts co
s
st

t a
d to h
d
 host h
t
rog



ty.
### What 
s Ray?
Ray 
s a d
str
but
d comput

g fram

ork for sca


g Pytho
 programs. Mu
t
-
od
 vLLM d
p
oym

ts ca
 us
 Ray as th
 ru
t
m
 

g


.
vLLM us
s Ray to ma
ag
 th
 d
str
but
d 
x
cut
o
 of tasks across mu
t
p

 
od
s a
d co
tro
 
h
r
 
x
cut
o
 happ

s.
Ray a
so off
rs h
gh-

v

 APIs for 
arg
-sca

 [off



 batch 

f
r

c
](https://docs.ray.
o/

/
at
st/data/
ork

g-

th-
ms.htm
) a
d [o




 s
rv

g](https://docs.ray.
o/

/
at
st/s
rv
/
m) that ca
 

v
rag
 vLLM as th
 

g


. Th
s
 APIs add product
o
-grad
 fau
t to

ra
c
, sca


g, a
d d
str
but
d obs
rvab


ty to vLLM 
ork
oads.
For d
ta

s, s
 th
 [Ray docum

tat
o
](https://docs.ray.
o/

/
at
st/

d
x.htm
).
### Ray c
ust
r s
tup 

th co
ta


rs
Th
 h

p
r scr
pt [
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
r.sh) starts co
ta


rs across 
od
s a
d 


t
a

z
s Ray. By d
fau
t, th
 scr
pt ru
s Dock
r 

thout adm


strat
v
 pr
v


g
s, 
h
ch pr
v

ts acc
ss to th
 GPU p
rforma
c
 cou
t
rs 
h

 prof



g or trac

g. To 

ab

 adm

 pr
v


g
s, add th
 `--cap-add=CAP_SYS_ADMIN` f
ag to th
 Dock
r comma
d.
Choos
 o

 
od
 as th
 h
ad 
od
 a
d ru
:
```bash
bash ru
_c
ust
r.sh \
                v
m/v
m-op

a
 \
                
HEAD_NODE_IP
 \
                --h
ad \
                /path/to/th
/hugg

gfac
/hom
/

/th
s/
od
 \
                -
 VLLM_HOST_IP=
HEAD_NODE_IP

```
O
 
ach 
ork
r 
od
, ru
:
```bash
bash ru
_c
ust
r.sh \
                v
m/v
m-op

a
 \
                
HEAD_NODE_IP
 \
                --
ork
r \
                /path/to/th
/hugg

gfac
/hom
/

/th
s/
od
 \
                -
 VLLM_HOST_IP=
WORKER_NODE_IP

```
Not
 that `VLLM_HOST_IP` 
s u

qu
 for 
ach 
ork
r. K
p th
 sh

s ru


g th
s
 comma
ds op

; c
os

g a
y sh

 t
rm

at
s th
 c
ust
r. E
sur
 that a
 
od
s ca
 commu

cat
 

th 
ach oth
r through th

r IP addr
ss
s.
!!! 
ar


g "N
t
ork s
cur
ty"
    For s
cur
ty, s
t `VLLM_HOST_IP` to a
 addr
ss o
 a pr
vat
 

t
ork s
gm

t. Traff
c s

t ov
r th
s 

t
ork 
s u


crypt
d, a
d th
 

dpo

ts 
xcha
g
 data 

 a format that ca
 b
 
xp
o
t
d to 
x
cut
 arb
trary cod
 
f a
 adv
rsary ga

s 

t
ork acc
ss. E
sur
 that u
trust
d part

s ca
ot r
ach th
 

t
ork.
From a
y 
od
, 

t
r a co
ta


r a
d ru
 `ray status` a
d `ray 

st 
od
s` to v
r
fy that Ray f

ds th
 
xp
ct
d 
umb
r of 
od
s a
d GPUs.
!!! t
p
    A
t
r
at
v

y, s
t up th
 Ray c
ust
r us

g Kub
Ray. For mor
 

format
o
, s
 [Kub
Ray vLLM docum

tat
o
](https://docs.ray.
o/

/
at
st/c
ust
r/kub
r

t
s/
xamp

s/rays
rv
-
m-
xamp

.htm
).
### Ru


g vLLM o
 a Ray c
ust
r
!!! t
p
    If Ray 
s ru


g 

s
d
 co
ta


rs, ru
 th
 comma
ds 

 th
 r
ma

d
r of th
s gu
d
 *

s
d
 th
 co
ta


rs*, 
ot o
 th
 host. To op

 a sh

 

s
d
 a co
ta


r, co

ct to a 
od
 a
d us
 `dock
r 
x
c -
t 
co
ta


r_
am

 /b

/bash`.
O
c
 a Ray c
ust
r 
s ru


g, us
 vLLM as you 
ou
d 

 a s

g

-
od
 s
tt

g. A
 r
sourc
s across th
 Ray c
ust
r ar
 v
s
b

 to vLLM, so a s

g

 `v
m` comma
d o
 a s

g

 
od
 
s suff
c


t.
Th
 commo
 pract
c
 
s to s
t th
 t

sor para


 s
z
 to th
 
umb
r of GPUs 

 
ach 
od
, a
d th
 p
p




 para


 s
z
 to th
 
umb
r of 
od
s. For 
xamp

, 
f you hav
 16 GPUs across 2 
od
s (8 GPUs p
r 
od
), s
t th
 t

sor para


 s
z
 to 8 a
d th
 p
p




 para


 s
z
 to 2:
```bash
v
m s
rv
 /path/to/th
/mod

/

/th
/co
ta


r \
    --t

sor-para


-s
z
 8 \
    --p
p




-para


-s
z
 2 \
    --d
str
but
d-
x
cutor-back

d ray
```
A
t
r
at
v

y, you ca
 s
t `t

sor_para


_s
z
` to th
 tota
 
umb
r of GPUs 

 th
 c
ust
r:
```bash
v
m s
rv
 /path/to/th
/mod

/

/th
/co
ta


r \
     --t

sor-para


-s
z
 16 \
     --d
str
but
d-
x
cutor-back

d ray
```
### Ru


g vLLM 

th Mu
t
Proc
ss

g
B
s
d
s Ray, Mu
t
-
od
 vLLM d
p
oym

ts ca
 a
so us
 `mu
t
proc
ss

g` as th
 ru
t
m
 

g


. H
r
's a
 
xamp

 to d
p
oy mod

 across 2 
od
s (8 GPUs p
r 
od
) 

th `tp_s
z
=8` a
d `pp_s
z
=2`.
Choos
 o

 
od
 as th
 h
ad 
od
 a
d ru
:
```bash
v
m s
rv
 /path/to/th
/mod

/

/th
/co
ta


r \
  --t

sor-para


-s
z
 8 --p
p




-para


-s
z
 2 \
  --
od
s 2 --
od
-ra
k 0 \
  --mast
r-addr 
HEAD_NODE_IP

```
O
 th
 oth
r 
ork
r 
od
, ru
:
```bash
v
m s
rv
 /path/to/th
/mod

/

/th
/co
ta


r \
  --t

sor-para


-s
z
 8 --p
p




-para


-s
z
 2 \
  --
od
s 2 --
od
-ra
k 1 \
  --mast
r-addr 
HEAD_NODE_IP
 --h
ad

ss
```
## Opt
m
z

g 

t
ork commu

cat
o
 for t

sor para



sm
Eff
c


t t

sor para



sm r
qu
r
s fast 

t
r
od
 commu

cat
o
, pr
f
rab
y through h
gh-sp
d 

t
ork adapt
rs such as I
f


Ba
d.
To s
t up th
 c
ust
r to us
 I
f


Ba
d, app

d add
t
o
a
 argum

ts 

k
 `--pr
v


g
d -
 NCCL_IB_HCA=m
x5` to th

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
r.sh) h

p
r scr
pt.
Co
tact your syst
m adm


strator for mor
 

format
o
 about th
 r
qu
r
d f
ags.
## E
ab


g GPUD
r
ct RDMA
GPUD
r
ct RDMA (R
mot
 D
r
ct M
mory Acc
ss) 
s a
 NVIDIA t
ch
o
ogy that a
o
s 

t
ork adapt
rs to d
r
ct
y acc
ss GPU m
mory, bypass

g th
 CPU a
d syst
m m
mory. Th
s d
r
ct acc
ss r
duc
s 
at

cy a
d CPU ov
rh
ad, 
h
ch 
s b


f
c
a
 for 
arg
 data tra
sf
rs b
t


 GPUs across 
od
s.
To 

ab

 GPUD
r
ct RDMA 

th vLLM, co
f
gur
 th
 fo
o


g s
tt

gs:
- `IPC_LOCK` s
cur
ty co
t
xt: add th
 `IPC_LOCK` capab


ty to th
 co
ta


r's s
cur
ty co
t
xt to 
ock m
mory pag
s a
d pr
v

t s
app

g to d
sk.
- Shar
d m
mory 

th `/d
v/shm`: mou
t `/d
v/shm` 

 th
 pod sp
c to prov
d
 shar
d m
mory for 

t
rproc
ss commu

cat
o
 (IPC).
If you us
 Dock
r, s
t up th
 co
ta


r as fo
o
s:
```bash
dock
r ru
 --gpus a
 \
    --
pc=host \
    --shm-s
z
=16G \
    -v /d
v/shm:/d
v/shm \
    v
m/v
m-op

a

```
If you us
 Kub
r

t
s, s
t up th
 pod sp
c as fo
o
s:
```yam

...
sp
c:
  co
ta


rs:
    - 
am
: v
m
      
mag
: v
m/v
m-op

a

      s
cur
tyCo
t
xt:
        capab


t

s:
          add: ["IPC_LOCK"]
      vo
um
Mou
ts:
        - mou
tPath: /d
v/shm
          
am
: dshm
      r
sourc
s:
        

m
ts:
          
v
d
a.com/gpu: 8
        r
qu
sts:
          
v
d
a.com/gpu: 8
  vo
um
s:
    - 
am
: dshm
      
mptyD
r:
        m
d
um: M
mory
...
```
!!! t
p "Co
f
rm GPUD
r
ct RDMA op
rat
o
"
    To co
f
rm your I
f


Ba
d card 
s us

g GPUD
r
ct RDMA, ru
 vLLM 

th d
ta


d NCCL 
ogs: `NCCL_DEBUG=TRACE v
m s
rv
 ...`.
    Th

 
ook for th
 NCCL v
rs
o
 a
d th
 

t
ork us
d.
    - If you f

d `[s

d] v
a NET/IB/GDRDMA` 

 th
 
ogs, th

 NCCL 
s us

g I
f


Ba
d 

th GPUD
r
ct RDMA, 
h
ch *
s* 
ff
c


t.
    - If you f

d `[s

d] v
a NET/Sock
t` 

 th
 
ogs, NCCL us
d a ra
 TCP sock
t, 
h
ch *
s 
ot* 
ff
c


t for cross-
od
 t

sor para



sm.
!!! t
p "Pr
-do


oad Hugg

g Fac
 mod

s"
    If you us
 Hugg

g Fac
 mod

s, do


oad

g th
 mod

 b
for
 start

g vLLM 
s r
comm

d
d. Do


oad th
 mod

 o
 
v
ry 
od
 to th
 sam
 path, or stor
 th
 mod

 o
 a d
str
but
d f


 syst
m acc
ss
b

 by a
 
od
s. Th

 pass th
 path to th
 mod

 

 p
ac
 of th
 r
pos
tory ID. Oth
r

s
, supp
y a Hugg

g Fac
 tok

 by app

d

g `-
 HF_TOKEN=
TOKEN
` to `ru
_c
ust
r.sh`.
## Troub

shoot

g d
str
but
d d
p
oym

ts
For 

format
o
 about d
str
but
d d
bugg

g, s
 [Troub

shoot

g d
str
but
d d
p
oym

ts](d
str
but
d_troub

shoot

g.md).
