# P2P NCCL Co

ctor
A
 
mp

m

tat
o
 of xPyD 

th dy
am
c sca


g bas
d o
 po

t-to-po

t commu

cat
o
, part
y 

sp
r
d by Dy
amo.
## D
ta


d D
s
g

### Ov
ra
 Proc
ss
As sho

 

 F
gur
 1, th
 ov
ra
 proc
ss of th
s **PD d
saggr
gat
o
** so
ut
o
 
s d
scr
b
d through a r
qu
st f
o
:
1. Th
 c



t s

ds a
 HTTP r
qu
st to th
 Proxy/Rout
r's `/v1/comp

t
o
s` 

t
rfac
.
2. Th
 Proxy/Rout
r s


cts a **1P1D (1 Pr
f

 

sta
c
 + 1 D
cod
 

sta
c
)** through 

th
r through rou
d-rob

 or ra
dom s


ct
o
, g


rat
s a `r
qu
st_
d` (ru

s to b
 

troduc
d 
at
r), mod
f

s th
 `max_tok

s` 

 th
 HTTP r
qu
st m
ssag
 to **1**, a
d th

 for
ards th
 r
qu
st to th
 **P 

sta
c
**.
3. Imm
d
at

y aft
r
ard, th
 Proxy/Rout
r for
ards th
 **or
g

a
 HTTP r
qu
st** to th
 **D 

sta
c
**.
4. Th
 **P 

sta
c
** p
rforms **Pr
f

** a
d th

 **act
v

y s

ds th
 g


rat
d KV cach
** to th
 D 

sta
c
 (us

g **PUT_ASYNC** mod
). Th
 D 

sta
c
's `zmq_addr` ca
 b
 r
so
v
d through th
 `r
qu
st_
d`.
5. Th
 **D 

sta
c
** has a **d
d
cat
d thr
ad** for r
c

v

g th
 KV cach
 (to avo
d b
ock

g th
 ma

 proc
ss). Th
 r
c

v
d KV cach
 
s sav
d 

to th
 **GPU m
mory buff
r**, th
 s
z
 of 
h
ch 
s d
t
rm


d by th
 vLLM startup param
t
r `kv_buff
r_s
z
`. Wh

 th
 GPU buff
r 
s fu
, th
 KV cach
 
s stor
d 

 th
 **
oca
 T

sor m
mory poo
**.
6. Dur

g th
 **D
cod
**, th
 D 

sta
c
's ma

 proc
ss r
tr

v
s th
 KV cach
 (tra
sm
tt
d by th
 P 

sta
c
) from 

th
r th
 **GPU buff
r** or th
 **m
mory poo
**, th
r
by **sk
pp

g Pr
f

**.
7. Aft
r comp

t

g **D
cod
**, th
 D 

sta
c
 r
tur
s th
 r
su
t to th
 **Proxy/Rout
r**, 
h
ch th

 for
ards 
t to th
 **c



t**.
![
mag
1](https://g
thub.com/us
r-attachm

ts/ass
ts/fb01bd
6-755b-49f7-ad45-48a94b1
10a7)
### Proxy/Rout
r (D
mo)
A s
mp

 HTTP s
rv
c
 acts as th
 

try po

t for c



t r
qu
sts a
d starts a backgrou
d thr
ad to 

st

 for P/D 

sta
c
s r
port

g th

r HTTP IP a
d PORT, as 


 as ZMQ IP a
d PORT. It ma

ta

s a d
ct
o
ary of `http_addr -
 zmq_addr`. Th
 `http_addr` 
s th
 IP:PORT for th
 vLLM 

sta
c
's r
qu
st, 
h


 th
 `zmq_addr` 
s th
 addr
ss for KV cach
 ha
dshak
 a
d m
tadata r
c
pt
o
.
Th
 Proxy/Rout
r 
s r
spo
s
b

 for s


ct

g 1P1D bas
d o
 th
 charact
r
st
cs of th
 c



t r
qu
st, such as th
 prompt, a
d g


rat

g a corr
spo
d

g `r
qu
st_
d`, for 
xamp

:
```t
xt
cmp
-___pr
f

_addr_10.0.1.2:21001___d
cod
_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0
```
Curr

t
y, to qu
ck
y v
r
fy 
h
th
r xPyD ca
 
ork, a rou
d-rob

 s


ct
o
 of 1P1D 
s us
d. I
 th
 futur
, 
t 
s p
a

d to us
 a tr

 comb


d 

th th
 
oad status of 

sta
c
s to s


ct appropr
at
 P a
d D.
Each P/D 

sta
c
 p
r
od
ca
y s

ds a h
artb
at pack
t to th
 Proxy/Rout
r (curr

t
y 
v
ry 3 s
co
ds) to r
g
st
r (
.
., r
port `http_addr -
 zmq_addr`) a
d k
p th
 co

ct
o
 a

v
. If a
 

sta
c
 crash
s a
d fa

s to s

d a p

g for a c
rta

 p
r
od of t
m
, th
 Proxy/Rout
r 


 r
mov
 th
 t
m
d-out 

sta
c
 (th
s f
atur
 has 
ot y
t b

 d
v

op
d).
### KV Cach
 Tra
sf
r M
thods
Th
r
 ar
 thr
 m
thods for KVCach
 tra
sf
r: PUT, GET, a
d PUT_ASYNC. Th
s
 m
thods ca
 b
 sp
c
f

d us

g th
 `--kv-tra
sf
r-co
f
g` a
d `kv_co

ctor_
xtra_co
f
g` param
t
rs, sp
c
f
ca
y through th
 `s

d_typ
` f


d. Both PUT a
d PUT_ASYNC 

vo
v
 th
 P 

sta
c
 act
v

y s

d

g KVCach
 to th
 D 

sta
c
. Th
 d
ff
r

c
 
s that PUT 
s a sy
chro
ous tra
sf
r m
thod that b
ocks th
 ma

 proc
ss, 
h


 PUT_ASYNC 
s a
 asy
chro
ous tra
sf
r m
thod. PUT_ASYNC us
s a d
d
cat
d thr
ad for s

d

g KVCach
, 
h
ch m
a
s 
t do
s 
ot b
ock th
 ma

 proc
ss. I
 co
trast, th
 GET m
thod 

vo
v
s th
 P 

sta
c
 sav

g th
 KVCach
 to th
 m
mory buff
r aft
r comput

g th
 pr
f

. Th
 D 

sta
c
 th

 act
v

y r
tr

v
s th
 comput
d KVCach
 from th
 P 

sta
c
 o
c
 
t has a
ocat
d spac
 for th
 KVCach
.
Exp
r
m

ta
 r
su
ts hav
 sho

 that th
 p
rforma
c
 of th
s
 m
thods, from h
gh
st to 
o

st, 
s as fo
o
s: PUT_ASYNC → GET → PUT.
### P2P Commu

cat
o
 v
a ZMQ & NCCL
As 
o
g as th
 addr
ss of th
 cou
t
rpart 
s k
o

, po

t-to-po

t KV cach
 tra
sf
r (us

g NCCL) ca
 b
 p
rform
d, 

thout b


g co
stra


d by ra
k a
d 
or
d s
z
. To support dy
am
c sca


g (
xpa
s
o
 a
d co
tract
o
) of 

sta
c
s 

th PD d
saggr
gat
o
. Th
s m
a
s that add

g or r
mov

g P/D 

sta
c
s do
s 
ot r
qu
r
 a fu
 syst
m r
start.
Each P/D 

sta
c
 o

y 

ds to cr
at
 a s

g

 `P2pNcc
E
g


` 

sta
c
. Th
s 

sta
c
 ma

ta

s a ZMQ S
rv
r, 
h
ch ru
s a d
d
cat
d thr
ad to 

st

 o
 th
 `zmq_addr` addr
ss a
d r
c

v
 co
tro
 f
o
 r
qu
sts from oth
r 

sta
c
s. Th
s
 r
qu
sts 

c
ud
 r
qu
sts to 
stab

sh a
 NCCL co

ct
o
 a
d r
qu
sts to s

d KVCach
 m
tadata (such as t

sor shap
s a
d data typ
s). Ho

v
r, 
t do
s 
ot actua
y tra
sm
t th
 KVCach
 data 
ts

f.
Wh

 a P 

sta
c
 a
d a D 

sta
c
 tra
sm
t KVCach
 for th
 f
rst t
m
, th
y 

d to 
stab

sh a ZMQ co

ct
o
 a
d a
 NCCL group. For subs
qu

t KVCach
 tra
sm
ss
o
s, th
s ZMQ co

ct
o
 a
d NCCL group ar
 r
us
d. Th
 NCCL group co
s
sts of o

y t
o ra
ks, m
a


g th
 
or
d s
z
 
s 
qua
 to 2. Th
s d
s
g
 
s 

t

d
d to support dy
am
c sca


g, 
h
ch m
a
s that add

g or r
mov

g P/D 

sta
c
s do
s 
ot r
qu
r
 a fu
 syst
m r
start. As 
o
g as th
 addr
ss of th
 cou
t
rpart 
s k
o

, po

t-to-po

t KVCach
 tra
sm
ss
o
 ca
 b
 p
rform
d, 

thout b


g r
str
ct
d by ra
k or 
or
d s
z
.
### NCCL Group Topo
ogy
Curr

t
y, o

y symm
tr
c TP (T

sor Para



sm) m
thods ar
 support
d for KVCach
 tra
sm
ss
o
. Asymm
tr
c TP a
d PP (P
p




 Para



sm) m
thods 


 b
 support
d 

 th
 futur
. F
gur
 2 

ustrat
s th
 1P2D s
tup, 
h
r
 
ach 

sta
c
 has a TP (T

sor Para



sm) d
gr
 of 2. Th
r
 ar
 a tota
 of 7 NCCL groups: thr
 vLLM 

sta
c
s 
ach hav
 o

 NCCL group 

th TP=2. Add
t
o
a
y, th
 0th GPU card of th
 P 

sta
c
 
stab

sh
s a
 NCCL group 

th th
 0th GPU card of 
ach D 

sta
c
. S
m

ar
y, th
 1st GPU card of th
 P 

sta
c
 
stab

sh
s a
 NCCL group 

th th
 1st GPU card of 
ach D 

sta
c
.
![
mag
2](https://g
thub.com/us
r-attachm

ts/ass
ts/837
61d6-365
-4cbf-8640-6dd7ab295b36)
Each NCCL group occup

s a c
rta

 amou
t of GPU m
mory buff
r for commu

cat
o
, th
 s
z
 of 
h
ch 
s pr
mar

y 

f
u

c
d by th
 `NCCL_MAX_NCHANNELS` 

v
ro
m

t var
ab

. Wh

 `NCCL_MAX_NCHANNELS=16`, a
 NCCL group typ
ca
y occup

s 100MB, 
h


 
h

 `NCCL_MAX_NCHANNELS=8`, 
t usua
y tak
s up 52MB. For 
arg
-sca

 xPyD co
f
gurat
o
s—such as D
pS
k's 96P144D—th
s 
mp

m

tat
o
 
s curr

t
y 
ot f
as
b

. Mov

g for
ard, 

 ar
 co
s
d
r

g us

g RDMA for po

t-to-po

t commu

cat
o
 a
d ar
 a
so k
p

g a
 
y
 o
 UCCL.
### GPU M
mory Buff
r a
d T

sor M
mory Poo

Th
 trad
-off 

 th
 s
z
 of th
 m
mory buff
r 
s as fo
o
s: For P 

sta
c
s, th
 m
mory buff
r 
s 
ot r
qu
r
d 

 PUT a
d PUT_ASYNC mod
s, but 
t 
s 

c
ssary 

 GET mod
. For D 

sta
c
s, a m
mory buff
r 
s 

d
d 

 a
 thr
 mod
s. Th
 m
mory buff
r for D 

sta
c
s shou
d 
ot b
 too 
arg
. S
m

ar
y, for P 

sta
c
s 

 GET mod
, th
 m
mory buff
r shou
d a
so 
ot b
 too 
arg
. Th
 m
mory buff
r of D 

sta
c
s 
s us
d to t
mporar

y stor
 KVCach
 s

t by P 

sta
c
s. If 
t 
s too 
arg
, 
t 


 r
duc
 th
 KVCach
 spac
 ava

ab

 for 
orma
 

f
r

c
 by D 

sta
c
s, th
r
by d
cr
as

g th
 

f
r

c
 batch s
z
 a
d u
t
mat

y 

ad

g to a r
duct
o
 

 output throughput. Th
 s
z
 of th
 m
mory buff
r 
s co
f
gur
d by th
 param
t
r `kv_buff
r_s
z
`, m
asur
d 

 byt
s, a
d 
s typ
ca
y s
t to 5%～10% of th
 m
mory s
z
.
If th
 `--max-
um-s
qs` param
t
r for P 

sta
c
s 
s s
t to a 
arg
 va
u
, du
 to th
 
arg
 batch s
z
, P 

sta
c
s 


 g


rat
 a 
arg
 amou
t of KVCach
 s
mu
ta

ous
y. Th
s may 
xc
d th
 capac
ty of th
 m
mory buff
r of D 

sta
c
s, r
su
t

g 

 KVCach
 
oss. O
c
 KVCach
 
s 
ost, D 

sta
c
s 

d to r
comput
 Pr
f

, 
h
ch 
s 
qu
va


t to p
rform

g Pr
f

 t

c
. Co
s
qu

t
y, th
 t
m
-to-f
rst-tok

 (TTFT) 


 s
g

f
ca
t
y 

cr
as
, 

ad

g to d
grad
d p
rforma
c
.
To addr
ss th
 abov
 
ssu
s, I hav
 d
s
g

d a
d d
v

op
d a 
oca
 T

sor m
mory poo
 for stor

g KVCach
, 

sp
r
d by th
 buddy syst
m us
d 

 L

ux m
mory modu

s. S

c
 th
 m
mory 
s suff
c


t
y 
arg
, typ
ca
y 

 th
 TB ra
g
 o
 s
rv
rs, th
r
 
s 
o 

d to co
s
d
r pr
f
x cach

g or us

g b
ock-bas
d d
s
g
s to r
us
 m
mory, th
r
by sav

g spac
. Wh

 th
 m
mory buff
r 
s 

suff
c


t, KVCach
 ca
 b
 d
r
ct
y stor
d 

 th
 T

sor m
mory poo
, a
d D 

sta
c
s ca
 subs
qu

t
y r
tr

v
 KVCach
 from 
t. Th
 r
ad a
d 
r
t
 sp
d 
s that of PCI
, 

th PCI
 4.0 hav

g a sp
d of approx
mat

y 21 GB/s, 
h
ch 
s usua
y fast
r tha
 th
 Pr
f

 sp
d. Oth
r

s
, so
ut
o
s 

k
 Moo
cak
 a
d 
mcach
 
ou
d 
ot b
 

c
ssary. Th
 T

sor m
mory poo
 acts as a f
ood d
v
rs
o
 ar
a, typ
ca
y u
us
d 
xc
pt dur

g sudd

 traff
c surg
s. I
 th
 
orst-cas
 sc

ar
o, my so
ut
o
 p
rforms 
o 
ors
 tha
 th
 
orma
 s
tuat
o
 

th a Cach
 stor
.
## I
sta
 vLLM
```sh


p
p 

sta
 "v
m
=0.9.2"
```
## Ru
 xPyD
### I
struct
o
s
- Th
 fo
o


g 
xamp

s ar
 ru
 o
 a
 A800 (80GB) d
v
c
, us

g th
 M
ta-L
ama-3.1-8B-I
struct mod

.
- Pay att

t
o
 to th
 s
tt

g of th
 `kv_buff
r_s
z
` (

 byt
s). Th
 
mp
r
ca
 va
u
 
s 10% of th
 GPU m
mory s
z
. Th
s 
s r

at
d to th
 kvcach
 s
z
. If 
t 
s too sma
, th
 GPU m
mory buff
r for t
mporar

y stor

g th
 r
c

v
d kvcach
 


 ov
rf
o
, caus

g th
 kvcach
 to b
 stor
d 

 th
 t

sor m
mory poo
, 
h
ch 

cr
as
s 
at

cy. If 
t 
s too 
arg
, th
 kvcach
 ava

ab

 for 

f
r

c
 


 b
 r
duc
d, 

ad

g to a sma

r batch s
z
 a
d d
cr
as
d throughput.
- For Pr
f

 

sta
c
s, 
h

 us

g 
o
-GET mod
, th
 `kv_buff
r_s
z
` ca
 b
 s
t to 1, as Pr
f

 curr

t
y do
s 
ot 

d to r
c

v
 kvcach
. Ho

v
r, 
h

 us

g GET mod
, a 
arg
r `kv_buff
r_s
z
` 
s r
qu
r
d b
caus
 
t 

ds to stor
 th
 kvcach
 s

t to th
 D 

sta
c
.
- You may 

d to mod
fy th
 `kv_buff
r_s
z
` a
d `port` 

 th
 fo
o


g comma
ds (
f th
r
 
s a co
f

ct).
- `PUT_ASYNC` off
rs th
 b
st p
rforma
c
 a
d shou
d b
 pr
or
t
z
d.
- Th
 `--port` must b
 co
s
st

t 

th th
 `http_port` 

 th
 `--kv-tra
sf
r-co
f
g`.
- Th
 `d
sagg_proxy_p2p_
cc
_xpyd.py` scr
pt 


 us
 port 10001 (for r
c

v

g c



t r
qu
sts) a
d port 30001 (for r
c

v

g s
rv
c
 d
scov
ry from P a
d D 

sta
c
s).
- Th
 
od
 ru


g th
 proxy must hav
 `quart` 

sta

d.
- Supports mu
t
p

 
od
s; you just 

d to mod
fy th
 `proxy_
p` a
d `proxy_port` 

 `--kv-tra
sf
r-co
f
g`.
- I
 th
 fo
o


g 
xamp

s, 
t 
s assum
d that **th
 proxy's IP 
s 10.0.1.1**.
### Ru
 1P3D
#### Proxy (
.g. 10.0.1.1)
```sh


cd {your v
m d
r
ctory}/
xamp

s/o




_s
rv

g/d
saggr
gat
d_s
rv

g_p2p_
cc
_xpyd/
pytho
3 d
sagg_proxy_p2p_
cc
_xpyd.py &
```
#### Pr
f

1 (
.g. 10.0.1.2 or 10.0.1.1)
??? co
so

 "Comma
d"
    ```sh


    CUDA_VISIBLE_DEVICES=0 v
m s
rv
 {your mod

 d
r
ctory} \
        --host 0.0.0.0 \
        --port 20001 \
        --t

sor-para


-s
z
 1 \
        --s
d 1024 \
        --s
rv
d-mod

-
am
 bas
_mod

 \
        --dtyp
 f
oat16 \
        --max-mod

-


 10000 \
        --max-
um-batch
d-tok

s 10000 \
        --max-
um-s
qs 256 \
        --trust-r
mot
-cod
 \
        --gpu-m
mory-ut


zat
o
 0.9 \
        --kv-tra
sf
r-co
f
g \
        '{"kv_co

ctor":"P2pNcc
Co

ctor","kv_ro

":"kv_produc
r","kv_buff
r_s
z
":"1
1","kv_port":"21001","kv_co

ctor_
xtra_co
f
g":{"proxy_
p":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}' 
 /var/v
m.
og 2
&1 &
    ```
#### D
cod
1 (
.g. 10.0.1.3 or 10.0.1.1)
??? co
so

 "Comma
d"
    ```sh


    CUDA_VISIBLE_DEVICES=1 v
m s
rv
 {your mod

 d
r
ctory} \
        --host 0.0.0.0 \
        --port 20002 \
        --t

sor-para


-s
z
 1 \
        --s
d 1024 \
        --s
rv
d-mod

-
am
 bas
_mod

 \
        --dtyp
 f
oat16 \
        --max-mod

-


 10000 \
        --max-
um-batch
d-tok

s 10000 \
        --max-
um-s
qs 256 \
        --trust-r
mot
-cod
 \
        --gpu-m
mory-ut


zat
o
 0.7 \
        --kv-tra
sf
r-co
f
g \
        '{"kv_co

ctor":"P2pNcc
Co

ctor","kv_ro

":"kv_co
sum
r","kv_buff
r_s
z
":"8
9","kv_port":"22001","kv_co

ctor_
xtra_co
f
g":{"proxy_
p":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}' 
 /var/v
m.
og 2
&1 &
    ```
#### D
cod
2 (
.g. 10.0.1.4 or 10.0.1.1)
??? co
so

 "Comma
d"
    ```sh


    CUDA_VISIBLE_DEVICES=2 v
m s
rv
 {your mod

 d
r
ctory} \
        --host 0.0.0.0 \
        --port 20003 \
        --t

sor-para


-s
z
 1 \
        --s
d 1024 \
        --s
rv
d-mod

-
am
 bas
_mod

 \
        --dtyp
 f
oat16 \
        --max-mod

-


 10000 \
        --max-
um-batch
d-tok

s 10000 \
        --max-
um-s
qs 256 \
        --trust-r
mot
-cod
 \
        --gpu-m
mory-ut


zat
o
 0.7 \
        --kv-tra
sf
r-co
f
g \
        '{"kv_co

ctor":"P2pNcc
Co

ctor","kv_ro

":"kv_co
sum
r","kv_buff
r_s
z
":"8
9","kv_port":"23001","kv_co

ctor_
xtra_co
f
g":{"proxy_
p":"10.0.1.1","proxy_port":"30001","http_port":"20003"}}' 
 /var/v
m.
og 2
&1 &
    ```
#### D
cod
3 (
.g. 10.0.1.5 or 10.0.1.1)
??? co
so

 "Comma
d"
    ```sh


    CUDA_VISIBLE_DEVICES=3 v
m s
rv
 {your mod

 d
r
ctory} \
        --host 0.0.0.0 \
        --port 20004 \
        --t

sor-para


-s
z
 1 \
        --s
d 1024 \
        --s
rv
d-mod

-
am
 bas
_mod

 \
        --dtyp
 f
oat16 \
        --max-mod

-


 10000 \
        --max-
um-batch
d-tok

s 10000 \
        --max-
um-s
qs 256 \
        --trust-r
mot
-cod
 \
        --gpu-m
mory-ut


zat
o
 0.7 \
        --kv-tra
sf
r-co
f
g \
        '{"kv_co

ctor":"P2pNcc
Co

ctor","kv_ro

":"kv_co
sum
r","kv_buff
r_s
z
":"8
9","kv_port":"24001","kv_co

ctor_
xtra_co
f
g":{"proxy_
p":"10.0.1.1","proxy_port":"30001","http_port":"20004"}}' 
 /var/v
m.
og 2
&1 &
    ```
### Ru
 3P1D
#### Proxy (
.g. 10.0.1.1)
```sh


cd {your v
m d
r
ctory}/
xamp

s/o




_s
rv

g/d
saggr
gat
d_s
rv

g_p2p_
cc
_xpyd/
pytho
3 d
sagg_proxy_p2p_
cc
_xpyd.py &
```
#### Pr
f

1 (
.g. 10.0.1.2 or 10.0.1.1)
??? co
so

 "Comma
d"
    ```sh


    CUDA_VISIBLE_DEVICES=0 v
m s
rv
 {your mod

 d
r
ctory} \
        --host 0.0.0.0 \
        --port 20001 \
        --t

sor-para


-s
z
 1 \
        --s
d 1024 \
        --s
rv
d-mod

-
am
 bas
_mod

 \
        --dtyp
 f
oat16 \
        --max-mod

-


 10000 \
        --max-
um-batch
d-tok

s 10000 \
        --max-
um-s
qs 256 \
        --trust-r
mot
-cod
 \
        --gpu-m
mory-ut


zat
o
 0.9 \
        --kv-tra
sf
r-co
f
g \
        '{"kv_co

ctor":"P2pNcc
Co

ctor","kv_ro

":"kv_produc
r","kv_buff
r_s
z
":"1
1","kv_port":"21001","kv_co

ctor_
xtra_co
f
g":{"proxy_
p":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}' 
 /var/v
m.
og 2
&1 &
    ```
#### Pr
f

2 (
.g. 10.0.1.3 or 10.0.1.1)
??? co
so

 "Comma
d"
    ```sh


    CUDA_VISIBLE_DEVICES=1 v
m s
rv
 {your mod

 d
r
ctory} \
        --host 0.0.0.0 \
        --port 20002 \
        --t

sor-para


-s
z
 1 \
        --s
d 1024 \
        --s
rv
d-mod

-
am
 bas
_mod

 \
        --dtyp
 f
oat16 \
        --max-mod

-


 10000 \
        --max-
um-batch
d-tok

s 10000 \
        --max-
um-s
qs 256 \
        --trust-r
mot
-cod
 \
        --gpu-m
mory-ut


zat
o
 0.9 \
        --kv-tra
sf
r-co
f
g \
        '{"kv_co

ctor":"P2pNcc
Co

ctor","kv_ro

":"kv_produc
r","kv_buff
r_s
z
":"1
1","kv_port":"22001","kv_co

ctor_
xtra_co
f
g":{"proxy_
p":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}' 
 /var/v
m.
og 2
&1 &
    ```
#### Pr
f

3 (
.g. 10.0.1.4 or 10.0.1.1)
??? co
so

 "Comma
d"
    ```sh


    CUDA_VISIBLE_DEVICES=2 v
m s
rv
 {your mod

 d
r
ctory} \
        --host 0.0.0.0 \
        --port 20003 \
        --t

sor-para


-s
z
 1 \
        --s
d 1024 \
        --s
rv
d-mod

-
am
 bas
_mod

 \
        --dtyp
 f
oat16 \
        --max-mod

-


 10000 \
        --max-
um-batch
d-tok

s 10000 \
        --max-
um-s
qs 256 \
        --trust-r
mot
-cod
 \
        --gpu-m
mory-ut


zat
o
 0.9 \
        --kv-tra
sf
r-co
f
g \
        '{"kv_co

ctor":"P2pNcc
Co

ctor","kv_ro

":"kv_produc
r","kv_buff
r_s
z
":"1
1","kv_port":"23001","kv_co

ctor_
xtra_co
f
g":{"proxy_
p":"10.0.1.1","proxy_port":"30001","http_port":"20003"}}' 
 /var/v
m.
og 2
&1 &
    ```
#### D
cod
1 (
.g. 10.0.1.5 or 10.0.1.1)
??? co
so

 "Comma
d"
    ```sh


    CUDA_VISIBLE_DEVICES=3 v
m s
rv
 {your mod

 d
r
ctory} \
        --host 0.0.0.0 \
        --port 20004 \
        --t

sor-para


-s
z
 1 \
        --s
d 1024 \
        --s
rv
d-mod

-
am
 bas
_mod

 \
        --dtyp
 f
oat16 \
        --max-mod

-


 10000 \
        --max-
um-batch
d-tok

s 10000 \
        --max-
um-s
qs 256 \
        --trust-r
mot
-cod
 \
        --gpu-m
mory-ut


zat
o
 0.7 \
        --kv-tra
sf
r-co
f
g \
        '{"kv_co

ctor":"P2pNcc
Co

ctor","kv_ro

":"kv_co
sum
r","kv_buff
r_s
z
":"8
9","kv_port":"24001","kv_co

ctor_
xtra_co
f
g":{"proxy_
p":"10.0.1.1","proxy_port":"30001","http_port":"20004"}}' 
 /var/v
m.
og 2
&1 &
    ```
## S

g

 r
qu
st
```sh


cur
 -X POST -s http://10.0.1.1:10001/v1/comp

t
o
s \
-H "Co
t

t-Typ
: app

cat
o
/jso
" \
-d '{
    "mod

": "bas
_mod

",
    "prompt": "Sa
 Fra
c
sco 
s a",
    "max_tok

s": 10,
    "t
mp
ratur
": 0
}'
```
## B

chmark
??? co
so

 "Comma
d"
    ```sh


    v
m b

ch s
rv
 \
        --back

d v
m \
        --mod

 bas
_mod

 \
        --tok


z
r m
ta-
ama/L
ama-3.1-8B-I
struct \
        --datas
t-
am
 "ra
dom" \
        --host 10.0.1.1 \
        --port 10001 \
        --ra
dom-

put-


 1024 \
        --ra
dom-output-


 1024 \
        --
g
or
-
os \
        --burst


ss 100 \
        --p
rc

t


-m
tr
cs "ttft,tpot,
t
,
2

" \
        --m
tr
c-p
rc

t


s "90,95,99" \
        --s
d $(dat
 +%s) \
        --trust-r
mot
-cod
 \
        --r
qu
st-rat
 3 \
        --
um-prompts 1000
    ```
## Shut do


```sh


pgr
p pytho
 | xargs k

 -9 && pk

 -f pytho

```
## T
st data
### **Sc

ar
o**: 1K 

put & 200 output tok

s, E2E P99 
at

cy ~2s
![t
stdata](https://g
thub.com/us
r-attachm

ts/ass
ts/c
f0953b-4567-4bf9-b940-405b92a28
b1)
