# N
x
Co

ctor Usag
 Gu
d

N
x
Co

ctor 
s a h
gh-p
rforma
c
 KV cach
 tra
sf
r co

ctor for vLLM's d
saggr
gat
d pr
f



g f
atur
. It prov
d
s fu
y asy
chro
ous s

d/r
c

v
 op
rat
o
s us

g th
 NIXL 

brary for 
ff
c


t cross-proc
ss KV cach
 tra
sf
r.
## Pr
r
qu
s
t
s
### I
sta
at
o

I
sta
 th
 NIXL 

brary: `uv p
p 

sta
 

x
`, as a qu
ck start o
 Nv
d
a p
atform.
    - R
f
r to [NIXL off
c
a
 r
pos
tory](https://g
thub.com/a
-dy
amo/

x
) for mor
 

sta
at
o
 

struct
o
s
    - Th
 sp
c
f

d r
qu
r
d NIXL v
rs
o
 ca
 b
 fou
d 

 [r
qu
r
m

ts/kv_co

ctors.txt](../../r
qu
r
m

ts/kv_co

ctors.txt) a
d oth
r r


va
t co
f
g f


s
For ROCm p
atform, th
 [bas
 ROCm dock
r f


](../../dock
r/Dock
rf


.rocm_bas
) 

c
ud
s RIXL a
d ucx a
r
ady.
    - R
f
r to [RIXL off
c
a
 r
pos
tory](https://g
thub.com/rocm/r
x
) for mor
 

format
o

    - Th
 support
v
 

brar

s for RIXL ca
 b
 fou
d 

 [r
qu
r
m

ts/kv_co

ctors_rocm.txt](../../r
qu
r
m

ts/kv_co

ctors_rocm.txt)
    - I
 th
 futur
 

 may r
mov
 RIXL from dock
r 
mag
 f


 a
d us
rs 


 b
 ab

 to 

sta
 from pr
-comp


d b

ary packag
s
For 
o
-cuda p
atform, p

as
 

sta
 

x
 

th ucx bu

d from sourc
, 

struct
d as b

o
.
```bash
pytho
 too
s/

sta
_

x
_from_sourc
_ubu
tu.py
```
### Tra
sport Co
f
gurat
o

N
x
Co

ctor us
s NIXL 

brary for u
d
r
y

g commu

cat
o
, 
h
ch supports mu
t
p

 tra
sport back

ds. UCX (U

f

d Commu

cat
o
 X) 
s th
 pr
mary d
fau
t tra
sport 

brary us
d by NIXL. Co
f
gur
 tra
sport 

v
ro
m

t var
ab

s:
```bash
# Examp

 UCX co
f
gurat
o
, adjust accord

g to your 

v
ro
m

t

xport UCX_TLS=a
  # or sp
c
fy sp
c
f
c tra
sports 

k
 "rc,ud,sm,^cuda_
pc" ..
tc

xport UCX_NET_DEVICES=a
  # or sp
c
fy 

t
ork d
v
c
s 

k
 "m
x5_0:1,m
x5_1:1"
```
!!! t
p
    Wh

 us

g UCX as th
 tra
sport back

d, NCCL 

v
ro
m

t var
ab

s (

k
 `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`) ar
 
ot app

cab

 to N
x
Co

ctor, so co
f
gur
 UCX-sp
c
f
c 

v
ro
m

t var
ab

s 

st
ad of NCCL var
ab

s.
#### S


ct

g a NIXL tra
sport back

d (p
ug

)
N
x
Co

ctor ca
 us
 d
ff
r

t NIXL tra
sport back

ds (p
ug

s). By d
fau
t, N
x
Co

ctor us
s UCX as th
 tra
sport back

d.
To s


ct a d
ff
r

t back

d, s
t `kv_co

ctor_
xtra_co
f
g.back

ds` 

 `--kv-tra
sf
r-co
f
g`.
### Examp

: us

g LIBFABRIC back

d
```bash
v
m s
rv
 
MODEL
 \
  --kv-tra
sf
r-co
f
g '{
    "kv_co

ctor":"N
x
Co

ctor",
    "kv_ro

":"kv_both",
    "kv_co

ctor_
xtra_co
f
g":{"back

ds":["LIBFABRIC"]}
  }'
```
You ca
 a
so pass JSON k
ys 

d
v
dua
y us

g dott
d argum

ts, a
d you ca
 app

d 

st 


m

ts us

g `+`:
```bash
v
m s
rv
 
MODEL
 \
  --kv-tra
sf
r-co
f
g.kv_co

ctor N
x
Co

ctor \
  --kv-tra
sf
r-co
f
g.kv_ro

 kv_both \
  --kv-tra
sf
r-co
f
g.kv_co

ctor_
xtra_co
f
g.back

ds+ LIBFABRIC
```
!!! 
ot

    Back

d ava

ab


ty d
p

ds o
 ho
 NIXL 
as bu

t a
d 
hat p
ug

s ar
 pr
s

t 

 your 

v
ro
m

t. R
f
r to th
 [NIXL r
pos
tory](https://g
thub.com/a
-dy
amo/

x
) for ava

ab

 back

ds a
d bu

d 

struct
o
s.
## Bas
c Usag
 (o
 th
 sam
 host)
### Produc
r (Pr
f


r) Co
f
gurat
o

Start a pr
f


r 

sta
c
 that produc
s KV cach
s
```bash
# 1st GPU as pr
f


r
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=a
 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
v
m s
rv
 Q


/Q


3-0.6B \
  --port 8100 \
  --

forc
-
ag
r \
  --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_both","kv_
oad_fa

ur
_po

cy":"fa

"}'
```
### Co
sum
r (D
cod
r) Co
f
gurat
o

Start a d
cod
r 

sta
c
 that co
sum
s KV cach
s:
```bash
# 2
d GPU as d
cod
r
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=a
 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
v
m s
rv
 Q


/Q


3-0.6B \
  --port 8200 \
  --

forc
-
ag
r \
  --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_both","kv_
oad_fa

ur
_po

cy":"fa

"}'
```
### Proxy S
rv
r
Us
 a proxy s
rv
r to rout
 r
qu
sts b
t


 pr
f


r a
d d
cod
r:
```bash
pytho
 t
sts/v1/kv_co

ctor/

x
_

t
grat
o
/toy_proxy_s
rv
r.py \
  --port 8192 \
  --pr
f


r-hosts 
oca
host \
  --pr
f


r-ports 8100 \
  --d
cod
r-hosts 
oca
host \
  --d
cod
r-ports 8200
```
## E
v
ro
m

t Var
ab

s
    - `VLLM_NIXL_SIDE_CHANNEL_PORT`: Port for NIXL ha
dshak
 commu

cat
o

    - D
fau
t: 5600
    - **R
qu
r
d for both pr
f


r a
d d
cod
r 

sta
c
s**
    - Each vLLM 
ork
r 

ds a u

qu
 port o
 
ts host; us

g th
 sam
 port 
umb
r across d
ff
r

t hosts 
s f



    - For TP/DP d
p
oym

ts, 
ach 
ork
r's port o
 a 
od
 
s comput
d as: bas
_port + dp_ra
k (
.g., 

th `--data-para


-s
z
=2` a
d bas
_port=5600, dp_ra
k 0..1 us
 port 5600, 5601 o
 that 
od
).
    - Us
d for th
 


t
a
 NIXL ha
dshak
 b
t


 th
 pr
f


r a
d th
 d
cod
r
    - `VLLM_NIXL_SIDE_CHANNEL_HOST`: Host for s
d
 cha


 commu

cat
o

    - D
fau
t: "
oca
host"
    - S
t 
h

 pr
f


r a
d d
cod
r ar
 o
 d
ff
r

t mach


s
    - Co

ct
o
 

fo 
s pass
d v
a KVTra
sf
rParams from pr
f


r to d
cod
r for ha
dshak

    - `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`: T
m
out (

 s
co
ds) for automat
ca
y r


as

g th
 pr
f


r’s KV cach
 for a part
cu
ar r
qu
st. (Opt
o
a
)
    - D
fau
t: 480
    - If a r
qu
st 
s abort
d a
d th
 d
cod
r has 
ot y
t r
ad th
 KV-cach
 b
ocks through th
 

x
 cha


, th
 pr
f

 

sta
c
 


 r


as
 
ts KV-cach
 b
ocks aft
r th
s t
m
out to avo
d ho
d

g th
m 

d
f


t

y.
## Mu
t
-I
sta
c
 S
tup
### Mu
t
p

 Pr
f


r I
sta
c
s o
 D
ff
r

t Mach


s
```bash
# Pr
f


r 1 o
 Mach


 A (
xamp

 IP: ${IP1})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP1} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=a
 \
v
m s
rv
 Q


/Q


3-0.6B --port 8000 \
  --t

sor-para


-s
z
 8 \
  --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_produc
r","kv_
oad_fa

ur
_po

cy":"fa

"}'
# Pr
f


r 2 o
 Mach


 B (
xamp

 IP: ${IP2})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP2} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=a
 \
v
m s
rv
 Q


/Q


3-0.6B --port 8000 \
  --t

sor-para


-s
z
 8 \
  --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_produc
r","kv_
oad_fa

ur
_po

cy":"fa

"}'
```
### Mu
t
p

 D
cod
r I
sta
c
s o
 D
ff
r

t Mach


s
```bash
# D
cod
r 1 o
 Mach


 C (
xamp

 IP: ${IP3})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP3} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=a
 \
v
m s
rv
 Q


/Q


3-0.6B --port 8000 \
  --t

sor-para


-s
z
 8 \
  --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_co
sum
r","kv_
oad_fa

ur
_po

cy":"fa

"}'
# D
cod
r 2 o
 Mach


 D (
xamp

 IP: ${IP4})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP4} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=a
 \
v
m s
rv
 Q


/Q


3-0.6B --port 8000 \
  --t

sor-para


-s
z
 8 \
  --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_co
sum
r","kv_
oad_fa

ur
_po

cy":"fa

"}'
```
### Proxy for Mu
t
p

 I
sta
c
s
```bash
pytho
 t
sts/v1/kv_co

ctor/

x
_

t
grat
o
/toy_proxy_s
rv
r.py \
  --port 8192 \
  --pr
f


r-hosts ${IP1} ${IP2} \
  --pr
f


r-ports 8000 8000 \
  --d
cod
r-hosts ${IP3} ${IP4} \
  --d
cod
r-ports 8000 8000
```
For mu
t
-host DP d
p
oym

t, o

y 

d to prov
d
 th
 host/port of th
 h
ad 

sta
c
s.
### KV Ro

 Opt
o
s
    - **kv_produc
r**: For pr
f


r 

sta
c
s that g


rat
 KV cach
s
    - **kv_co
sum
r**: For d
cod
r 

sta
c
s that co
sum
 KV cach
s from pr
f


r
    - **kv_both**: E
ab

s symm
tr
c fu
ct
o
a

ty 
h
r
 th
 co

ctor ca
 act as both produc
r a
d co
sum
r. Th
s prov
d
s f

x
b


ty for 
xp
r
m

ta
 s
tups a
d sc

ar
os 
h
r
 th
 ro

 d
st

ct
o
 
s 
ot pr
d
t
rm


d.
!!! t
p
    N
x
Co

ctor curr

t
y do
s 
ot d
st

gu
sh `kv_ro

`; th
 actua
 pr
f


r/d
cod
r ro

s ar
 d
t
rm


d by th
 upp
r-

v

 proxy (
.g., `toy_proxy_s
rv
r.py` us

g `--pr
f


r-hosts` a
d `--d
cod
r-hosts`).
    Th
r
for
, `kv_ro

` 

 `--kv-tra
sf
r-co
f
g` 
s 
ff
ct
v

y a p
ac
ho
d
r a
d do
s 
ot aff
ct N
x
Co

ctor's b
hav
or.
### KV Load Fa

ur
 Po

cy
Th
 `kv_
oad_fa

ur
_po

cy` s
tt

g co
tro
s ho
 th
 syst
m ha
d

s fa

ur
s 
h

 th
 d
cod
r 

sta
c
 
oads KV cach
 b
ocks from th
 pr
f


r 

sta
c
:
    - **fa

** (d
fau
t): Imm
d
at

y fa

 th
 r
qu
st 

th a
 
rror 
h

 KV 
oad fa

s. Th
s pr
v

ts p
rforma
c
 d
gradat
o
 by avo
d

g r
computat
o
 of pr
f

 
ork o
 th
 d
cod
 

sta
c
.
    - **r
comput
**: R
comput
 fa


d b
ocks 
oca
y o
 th
 d
cod
 

sta
c
. Th
s may caus
 p
rforma
c
 _j
tt
r_ o
 d
cod
 

sta
c
s as th
 sch
du

d pr
f

 


 d

ay a
d 

t
rf
r
 

th oth
r d
cod
s. Furth
rmor
, d
cod
 

sta
c
s ar
 typ
ca
y co
f
gur
d 

th 
o
-
at

cy opt
m
zat
o
s.
!!! 
ar


g
    Us

g `kv_
oad_fa

ur
_po

cy="r
comput
"` ca
 

ad to p
rforma
c
 d
gradat
o
 

 product
o
 d
p
oym

ts. Wh

 KV 
oads fa

, th
 d
cod
 

sta
c
 


 
x
cut
 pr
f

 
ork 

th d
cod
-opt
m
z
d co
f
gurat
o
s, 
h
ch 
s 


ff
c


t a
d d
f
ats th
 purpos
 of d
saggr
gat
d pr
f



g. Th
s a
so 

cr
as
s ta

 
at

cy for oth
r o
go

g d
cod
 r
qu
sts.
## Exp
r
m

ta
 F
atur

### H
t
rog


ous KV Layout support
Support us
 cas
: Pr
f

 

th 'HND' a
d d
cod
 

th 'NHD' 

th 
xp
r
m

ta
 co
f
gurat
o

```bash
--kv-tra
sf
r-co
f
g '{..., "

ab

_p
rmut
_
oca
_kv":"Tru
"}'
```
### Cross 
ay
rs b
ocks
By d
fau
t, th
s f
atur
 
s d
sab

d. O
 att

t
o
 back

ds that support th
s f
atur
, 
ach 
og
ca
 b
ock 
s co
t
guous 

 phys
ca
 m
mory. Th
s r
duc
s th
 
umb
r of buff
rs that 

d to b
 tra
sf
rr
d.
To 

ab

 th
s f
atur
:
```bash
--kv-tra
sf
r-co
f
g '{..., "kv_co

ctor_
xtra_co
f
g": {"

ab

_cross_
ay
rs_b
ocks": "Tru
"}}'
```
## Examp

 Scr
pts/Cod

R
f
r to th
s
 
xamp

 scr
pts 

 th
 vLLM r
pos
tory:
    - [ru
_accuracy_t
st.sh](../../t
sts/v1/kv_co

ctor/

x
_

t
grat
o
/ru
_accuracy_t
st.sh)
    - [toy_proxy_s
rv
r.py](../../t
sts/v1/kv_co

ctor/

x
_

t
grat
o
/toy_proxy_s
rv
r.py)
    - [t
st_accuracy.py](../../t
sts/v1/kv_co

ctor/

x
_

t
grat
o
/t
st_accuracy.py)
