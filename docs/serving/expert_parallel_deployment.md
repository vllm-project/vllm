# Exp
rt Para


 D
p
oym

t
vLLM supports Exp
rt Para



sm (EP), 
h
ch a
o
s 
xp
rts 

 M
xtur
-of-Exp
rts (MoE) mod

s to b
 d
p
oy
d o
 s
parat
 GPUs, 

cr
as

g 
oca

ty, 
ff
c


cy, a
d throughput ov
ra
.
EP 
s typ
ca
y coup

d 

th Data Para



sm (DP). Wh


 DP ca
 b
 us
d 

d
p

d

t
y of EP, EP 
s mor
 
ff
c


t 
h

 us
d 

 co
ju
ct
o
 

th DP. You ca
 r
ad mor
 about data para



sm [h
r
](data_para


_d
p
oym

t.md).
## Pr
r
qu
s
t
s
B
for
 us

g EP, you 

d to 

sta
 th
 

c
ssary d
p

d

c

s. W
 ar
 act
v

y 
ork

g o
 mak

g th
s 
as

r 

 th
 futur
:
1. **I
sta
 D
pEP**: S
t up host 

v
ro
m

t fo
o


g vLLM's gu
d
 for EP k
r


s [h
r
](../../too
s/
p_k
r


s).
2. **I
sta
 D
pGEMM 

brary**: Fo
o
 th
 [off
c
a
 

struct
o
s](https://g
thub.com/d
ps
k-a
/D
pGEMM#

sta
at
o
).
3. **For d
saggr
gat
d s
rv

g**: I
sta
 `gdrcopy` by ru


g th
 [`

sta
_gdrcopy.sh`](../../too
s/

sta
_gdrcopy.sh) scr
pt (
.g., `

sta
_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"`). You ca
 f

d ava

ab

 OS v
rs
o
s [h
r
](https://d
v

op
r.do


oad.
v
d
a.com/comput
/r
d
st/gdrcopy/CUDA%2012.8/).
### Back

d S


ct
o
 Gu
d

vLLM prov
d
s mu
t
p

 commu

cat
o
 back

ds for EP. Us
 `--a
2a
-back

d` to s


ct o

:
| Back

d | Us
 Cas
 | F
atur
s | B
st For |
|---------|----------|----------|----------|
| `a
gath
r_r
duc
scatt
r` | D
fau
t back

d | Sta
dard a
2a
 us

g a
gath
r/r
duc
scatt
r pr
m
t
v
s | G


ra
 purpos
, 
orks 

th a
y EP+DP co
f
gurat
o
 |
| `d
p
p_h
gh_throughput` | Mu
t
-
od
 pr
f

 | Group
d GEMM 

th co
t

uous 
ayout, opt
m
z
d for pr
f

 | Pr
f

-dom

at
d 
ork
oads, h
gh-throughput sc

ar
os |
| `d
p
p_
o
_
at

cy` | Mu
t
-
od
 d
cod
 | CUDA graph support, mask
d 
ayout, opt
m
z
d for d
cod
 | D
cod
-dom

at
d 
ork
oads, 
o
-
at

cy sc

ar
os |
| `f
ash

f
r_a
2a
v` | MNNVL syst
ms | F
ashI
f
r a
toa
v k
r


s for mu
t
-
od
 NVL

k | Syst
ms 

th NVL

k across 
od
s |
| `
a
v
` | T
st

g/d
bugg

g | S
mp

 broadcast-bas
d 
mp

m

tat
o
 | D
bugg

g, 
ot r
comm

d
d for product
o
 |
## S

g

 Nod
 D
p
oym

t
!!! 
ar


g
    EP 
s a
 
xp
r
m

ta
 f
atur
. Argum

t 
am
s a
d d
fau
t va
u
s may cha
g
 

 th
 futur
.
### Co
f
gurat
o

E
ab

 EP by s
tt

g th
 `--

ab

-
xp
rt-para


` f
ag. Th
 EP s
z
 
s automat
ca
y ca
cu
at
d as:
```t
xt
EP_SIZE = TP_SIZE × DP_SIZE
```
Wh
r
:
    - `TP_SIZE`: T

sor para


 s
z

    - `DP_SIZE`: Data para


 s
z

    - `EP_SIZE`: Exp
rt para


 s
z
 (comput
d automat
ca
y)
### Lay
r B
hav
or 

th EP E
ab

d
Wh

 EP 
s 

ab

d, d
ff
r

t 
ay
rs 

 MoE mod

s b
hav
 d
ff
r

t
y:
| Lay
r Typ
 | B
hav
or | Para



sm Us
d |
|------------|----------|------------------|
| **Exp
rt (MoE) Lay
rs** | Shard
d across a
 EP ra
ks | Exp
rt Para


 (EP) of s
z
 `TP × DP` |
| **Att

t
o
 Lay
rs** | B
hav
or d
p

ds o
 TP s
z
 | S
 b

o
 |
**Att

t
o
 
ay
r para



sm:**
    - **Wh

 `TP = 1`**: Att

t
o
 


ghts ar
 **r
p

cat
d** across a
 DP ra
ks (data para



sm)
    - **Wh

 `TP 
 1`**: Att

t
o
 


ghts ar
 **shard
d** us

g t

sor para



sm across TP ra
ks 

th

 
ach DP group
For 
xamp

, 

th `TP=2, DP=4` (8 GPUs tota
):
    - Exp
rt 
ay
rs form a
 EP group of s
z
 8, 

th 
xp
rts d
str
but
d across a
 GPUs
    - Att

t
o
 
ay
rs us
 TP=2 

th

 
ach of th
 4 DP groups
!!! 
ot
 "K
y D
ff
r

c
 from Data Para


 D
p
oym

t"
    W
thout `--

ab

-
xp
rt-para


`, MoE 
ay
rs 
ou
d us
 t

sor para



sm (form

g a TP group of s
z
 `TP × DP`), s
m

ar to d

s
 mod

s. W
th EP 

ab

d, 
xp
rt 
ay
rs s

tch to 
xp
rt para



sm, 
h
ch ca
 prov
d
 b
tt
r 
ff
c


cy a
d 
oca

ty for MoE mod

s.
### Examp

 Comma
d
Th
 fo
o


g comma
d s
rv
s a `D
pS
k-V3-0324` mod

 

th 1-
ay t

sor para


, 8-
ay (att

t
o
) data para


, a
d 8-
ay 
xp
rt para


. Th
 att

t
o
 


ghts ar
 r
p

cat
d across a
 GPUs, 
h


 th
 
xp
rt 


ghts ar
 sp

t across GPUs. It 


 
ork o
 a H200 (or H20) 
od
 

th 8 GPUs. For H100, you ca
 try to s
rv
 a sma

r mod

 or r
f
r to th
 mu
t
-
od
 d
p
oym

t s
ct
o
.
```bash
# S

g

 
od
 EP d
p
oym

t
v
m s
rv
 d
ps
k-a
/D
pS
k-V3-0324 \
    --t

sor-para


-s
z
 1 \       # T

sor para



sm across 1 GPU
    --data-para


-s
z
 8 \         # Data para



sm across 8 proc
ss
s
    --

ab

-
xp
rt-para


         # E
ab

 
xp
rt para



sm
```
## Mu
t
-Nod
 D
p
oym

t
For mu
t
-
od
 d
p
oym

t, us
 th
 D
pEP commu

cat
o
 k
r


 

th o

 of t
o mod
s (s
 [Back

d S


ct
o
 Gu
d
](#back

d-s


ct
o
-gu
d
) abov
).
### D
p
oym

t St
ps
1. **Ru
 o

 comma
d p
r 
od
** - Each 
od
 r
qu
r
s 
ts o

 
au
ch comma
d
2. **Co
f
gur
 

t
ork

g** - E
sur
 prop
r IP addr
ss
s a
d port co
f
gurat
o
s
3. **S
t 
od
 ro

s** - F
rst 
od
 ha
d

s r
qu
sts, add
t
o
a
 
od
s ru
 

 h
ad

ss mod

### Examp

: 2-Nod
 D
p
oym

t
Th
 fo
o


g 
xamp

 d
p
oys `D
pS
k-V3-0324` across 2 
od
s us

g `d
p
p_
o
_
at

cy` mod
:
```bash
# Nod
 1 (Pr
mary - ha
d

s 

com

g r
qu
sts)
v
m s
rv
 d
ps
k-a
/D
pS
k-V3-0324 \
    --a
2a
-back

d d
p
p_
o
_
at

cy \
    --t

sor-para


-s
z
 1 \               # TP s
z
 p
r 
od

    --

ab

-
xp
rt-para


 \               # E
ab

 EP
    --data-para


-s
z
 16 \                # Tota
 DP s
z
 across a
 
od
s
    --data-para


-s
z
-
oca
 8 \           # Loca
 DP s
z
 o
 th
s 
od
 (8 GPUs p
r 
od
)
    --data-para


-addr
ss 192.168.1.100 \  # R
p
ac
 

th actua
 IP of Nod
 1
    --data-para


-rpc-port 13345 \         # RPC commu

cat
o
 port, ca
 b
 a
y port as 
o
g as r
achab

 by a
 
od
s
    --ap
-s
rv
r-cou
t=8                     # Numb
r of API s
rv
rs for 
oad ha
d


g (sca


g th
s out to # 
oca
 ra
ks 
s r
comm

d
d)
# Nod
 2 (S
co
dary - h
ad

ss mod
, 
o API s
rv
r)
v
m s
rv
 d
ps
k-a
/D
pS
k-V3-0324 \
    --a
2a
-back

d d
p
p_
o
_
at

cy \
    --t

sor-para


-s
z
 1 \               # TP s
z
 p
r 
od

    --

ab

-
xp
rt-para


 \               # E
ab

 EP
    --data-para


-s
z
 16 \                # Tota
 DP s
z
 across a
 
od
s
    --data-para


-s
z
-
oca
 8 \           # Loca
 DP s
z
 o
 th
s 
od

    --data-para


-start-ra
k 8 \           # Start

g ra
k offs
t for th
s 
od

    --data-para


-addr
ss 192.168.1.100 \  # IP of pr
mary 
od
 (Nod
 1)
    --data-para


-rpc-port 13345 \         # Sam
 RPC port as pr
mary
    --h
ad

ss                               # No API s
rv
r, 
ork
r o

y
```
### K
y Co
f
gurat
o
 Not
s
    - **H
ad

ss mod
**: S
co
dary 
od
s ru
 

th `--h
ad

ss` f
ag, m
a


g a
 c



t r
qu
sts ar
 ha
d

d by th
 pr
mary 
od

    - **Ra
k ca
cu
at
o
**: `--data-para


-start-ra
k` shou
d 
qua
 th
 cumu
at
v
 
oca
 DP s
z
 of pr
v
ous 
od
s
    - **Load sca


g**: Adjust `--ap
-s
rv
r-cou
t` o
 th
 pr
mary 
od
 to ha
d

 h
gh
r r
qu
st 
oads
### N
t
ork Co
f
gurat
o

!!! 
mporta
t "I
f


Ba
d C
ust
rs"
    O
 I
f


Ba
d 

t
ork
d c
ust
rs, s
t th
s 

v
ro
m

t var
ab

 to pr
v

t 


t
a

zat
o
 ha
gs:
    ```bash
    
xport GLOO_SOCKET_IFNAME=
th0
```
    Th
s 

sur
s torch d
str
but
d group d
scov
ry us
s Eth
r

t 

st
ad of I
f


Ba
d for 


t
a
 s
tup.
## Exp
rt Para


 Load Ba
a
c
r (EPLB)
Wh


 MoE mod

s ar
 typ
ca
y tra


d so that 
ach 
xp
rt r
c

v
s a s
m

ar 
umb
r of tok

s, 

 pract
c
 th
 d
str
but
o
 of tok

s across 
xp
rts ca
 b
 h
gh
y sk


d. vLLM prov
d
s a
 Exp
rt Para


 Load Ba
a
c
r (EPLB) to r
d
str
but
 
xp
rt mapp

gs across EP ra
ks, 
v



g th
 
oad across 
xp
rts.
### Co
f
gurat
o

E
ab

 EPLB 

th th
 `--

ab

-
p
b` f
ag.
Wh

 

ab

d, vLLM co

cts 
oad stat
st
cs 

th 
v
ry for
ard pass a
d p
r
od
ca
y r
ba
a
c
s 
xp
rt d
str
but
o
.
### EPLB Param
t
rs
Co
f
gur
 EPLB 

th th
 `--
p
b-co
f
g` argum

t, 
h
ch acc
pts a JSON str

g. Th
 ava

ab

 k
ys a
d th

r d
scr
pt
o
s ar
:
| Param
t
r | D
scr
pt
o
 | D
fau
t |
|-----------|-------------|---------|
| `


do
_s
z
`| Numb
r of 

g


 st
ps to track for r
ba
a
c

g d
c
s
o
s | 1000 |
| `st
p_

t
rva
`| Fr
qu

cy of r
ba
a
c

g (
v
ry N 

g


 st
ps) | 3000 |
| `
og_ba
a
c
d

ss` | Log ba
a
c
d

ss m
tr
cs (avg tok

s p
r 
xp
rt ÷ max tok

s p
r 
xp
rt) | `fa
s
` |
| `
um_r
du
da
t_
xp
rts` | Add
t
o
a
 g
oba
 
xp
rts p
r EP ra
k b
yo
d 
qua
 d
str
but
o
 | `0` |
| `us
_asy
c` | Us
 
o
-b
ock

g EPLB for r
duc
d 
at

cy ov
rh
ad | `fa
s
` |
| `po

cy` | Th
 po

cy typ
 for 
xp
rt para


 
oad ba
a
c

g | `"d
fau
t"` |
For 
xamp

:
```bash
v
m s
rv
 Q


/Q


3-30B-A3B \
  --

ab

-
p
b \
  --
p
b-co
f
g '{"


do
_s
z
":1000,"st
p_

t
rva
":3000,"
um_r
du
da
t_
xp
rts":2,"
og_ba
a
c
d

ss":tru
}'
```
??? t
p "Pr
f
r 

d
v
dua
 argum

ts 

st
ad of JSON?"
    ```bash
    v
m s
rv
 Q


/Q


3-30B-A3B \
            --

ab

-
p
b \
            --
p
b-co
f
g.


do
_s
z
 1000 \
            --
p
b-co
f
g.st
p_

t
rva
 3000 \
            --
p
b-co
f
g.
um_r
du
da
t_
xp
rts 2 \
            --
p
b-co
f
g.
og_ba
a
c
d

ss tru

```
### Exp
rt D
str
but
o
 Formu
a
    - **D
fau
t**: Each EP ra
k has `NUM_TOTAL_EXPERTS ÷ NUM_EP_RANKS` 
xp
rts
    - **W
th r
du
da
cy**: Each EP ra
k has `(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS` 
xp
rts
### M
mory Footpr

t Ov
rh
ad
EPLB us
s r
du
da
t 
xp
rts that 

d to f
t 

 GPU m
mory. Th
s m
a
s that EPLB may 
ot b
 a good f
t for m
mory co
stra


d 

v
ro
m

ts or 
h

 KV cach
 spac
 
s at a pr
m
um.
Th
s ov
rh
ad 
qua
s `NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`.
For D
pS
kV3, th
s 
s approx
mat

y `2.4 GB` for o

 r
du
da
t 
xp
rt p
r EP ra
k.
### Examp

 Comma
d
S

g

 
od
 d
p
oym

t 

th EPLB 

ab

d:
```bash
# S

g

 
od
 

th EPLB 
oad ba
a
c

g
v
m s
rv
 d
ps
k-a
/D
pS
k-V3-0324 \
    --t

sor-para


-s
z
 1 \       # T

sor para



sm
    --data-para


-s
z
 8 \         # Data para



sm
    --

ab

-
xp
rt-para


 \       # E
ab

 EP
    --

ab

-
p
b \                  # E
ab

 
oad ba
a
c
r
    --
p
b-co
f
g '{"


do
_s
z
":1000,"st
p_

t
rva
":3000,"
um_r
du
da
t_
xp
rts":2,"
og_ba
a
c
d

ss":tru
}'
```
For mu
t
-
od
 d
p
oym

t, add th
s
 EPLB f
ags to 
ach 
od
's comma
d. W
 r
comm

d s
tt

g `--
p
b-co
f
g '{"
um_r
du
da
t_
xp
rts":32}'` to 32 

 
arg
 sca

 us
 cas
s so th
 most popu
ar 
xp
rts ar
 a

ays ava

ab

.
## Adva
c
d Co
f
gurat
o

### P
rforma
c
 Opt
m
zat
o

    - **D
pEP k
r


s**: Th
 `h
gh_throughput` a
d `
o
_
at

cy` k
r


s ar
 opt
m
z
d for d
saggr
gat
d s
rv

g a
d may sho
 poor p
rforma
c
 for m
x
d 
ork
oads
    - **Dua
 Batch Ov
r
ap**: Us
 `--

ab

-dbo` to ov
r
ap a
-to-a
 commu

cat
o
 

th comput
. S
 [Dua
 Batch Ov
r
ap](../d
s
g
/dbo.md) for mor
 d
ta

s.
    - **Asy
c sch
du


g (
xp
r
m

ta
)**: Try `--asy
c-sch
du


g` to ov
r
ap sch
du


g 

th mod

 
x
cut
o
.
### Troub

shoot

g
    - **`
o
-z
ro status: 7 ca
ot r
g
st
r cq buf`**: Wh

 us

g I
f


ba
d/RoCE, mak
 sur
 host VM a
d pods sho
 `u

m
t -
` "u


m
t
d".
    - **`


t fa


d for tra
sport: IBGDA`**: Th
 I
f


Ba
d GDA k
r


 modu

s ar
 m
ss

g. Ru
 `too
s/
p_k
r


s/co
f
gur
_syst
m_dr
v
rs.sh` o
 
ach GPU 
od
 a
d r
boot. A
so f
x
s 
rror `NVSHMEM API ca

d b
for
 NVSHMEM 


t
a

zat
o
 has comp

t
d`.
    - **NVSHMEM p
r d
sco

ct**: Usua
y a 

t
ork

g m
sco
f
gurat
o
. If d
p
oy

g v
a Kub
r

t
s, v
r
fy that 
v
ry pod ru
s 

th `hostN
t
ork: tru
`, `s
cur
tyCo
t
xt.pr
v


g
d: tru
` to acc
ss I
f


ba
d.
### B

chmark

g
    - Us
 s
mu
ator f
ags `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=u

form_ra
dom` a
d `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` so tok

 rout

g 
s ba
a
c
d across EP ra
ks.
    - I
cr
as

g `VLLM_MOE_DP_CHUNK_SIZE` may 

cr
as
 throughput by 

cr
as

g th
 max
mum batch s
z
 for 

t
r-ra
k tok

 tra
sf
rs. Th
s may caus
 D
pEP  to thro
 `ass
rt s

f.
vshm
m_qp_d
pth 
= (
um_max_d
spatch_tok

s_p
r_ra
k + 1) * 2`, 
h
ch ca
 b
 f
x
d by 

cr
as

g 

v
ro
m

t var
ab

 `NVSHMEM_QP_DEPTH`.
## D
saggr
gat
d S
rv

g (Pr
f

/D
cod
 Sp

t)
For product
o
 d
p
oym

ts r
qu
r

g str
ct SLA guara
t
s for t
m
-to-f
rst-tok

 a
d 

t
r-tok

 
at

cy, d
saggr
gat
d s
rv

g a
o
s 

d
p

d

t sca


g of pr
f

 a
d d
cod
 op
rat
o
s.
### Arch
t
ctur
 Ov
rv



    - **Pr
f

 I
sta
c
**: Us
s `d
p
p_h
gh_throughput` back

d for opt
ma
 pr
f

 p
rforma
c

    - **D
cod
 I
sta
c
**: Us
s `d
p
p_
o
_
at

cy` back

d for m


ma
 d
cod
 
at

cy
    - **KV Cach
 Tra
sf
r**: Co

cts 

sta
c
s v
a NIXL or oth
r KV co

ctors
### S
tup St
ps
1. **I
sta
 gdrcopy/ucx/

x
**: For max
mum p
rforma
c
, ru
 th
 [

sta
_gdrcopy.sh](../../too
s/

sta
_gdrcopy.sh) scr
pt to 

sta
 `gdrcopy` (
.g., `

sta
_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"`). You ca
 f

d ava

ab

 OS v
rs
o
s [h
r
](https://d
v

op
r.do


oad.
v
d
a.com/comput
/r
d
st/gdrcopy/CUDA%2012.8/). If `gdrcopy` 
s 
ot 

sta

d, th

gs 


 st

 
ork 

th a p
a

 `p
p 

sta
 

x
`, just 

th 
o

r p
rforma
c
. `

x
` a
d `ucx` ar
 

sta

d as d
p

d

c

s v
a p
p. For 
o
-cuda p
atform to 

sta
 

x
 

th 
o
-cuda UCX bu

d, ru
 th
 [

sta
_

x
_from_sourc
_ubu
tu.py](../../too
s/

sta
_

x
_from_sourc
_ubu
tu.py) scr
pt.
2. **Co
f
gur
 Both I
sta
c
s**: Add th
s f
ag to both pr
f

 a
d d
cod
 

sta
c
s `--kv-tra
sf
r-co
f
g '{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_both"}`. Not
d, you may a
so sp
c
fy o

 or mu
t
p

 NIXL_Back

d. Such as: `--kv-tra
sf
r-co
f
g '{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_both", "kv_co

ctor_
xtra_co
f
g":{"back

ds":["UCX", "GDS"]}}'`
3. **C



t Orch
strat
o
**: Us
 th
 c



t-s
d
 scr
pt b

o
 to coord

at
 pr
f

/d
cod
 op
rat
o
s. W
 ar
 act
v

y 
ork

g o
 rout

g so
ut
o
s.
### C



t Orch
strat
o
 Examp


```pytho

from op

a
 
mport Op

AI

mport uu
d
try:
    # 1: S
t up c



ts for pr
f

 a
d d
cod
 

sta
c
s
    op

a
_ap
_k
y = "EMPTY"  # vLLM do
s
't r
qu
r
 a r
a
 API k
y
    # R
p
ac
 th
s
 IP addr
ss
s 

th your actua
 

sta
c
 addr
ss
s
    pr
f

_c



t = Op

AI(
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
="http://192.168.1.100:8000/v1",  # Pr
f

 

sta
c
 URL
    )
    d
cod
_c



t = Op

AI(
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
="http://192.168.1.101:8001/v1",  # D
cod
 

sta
c
 URL
    )
    # G
t mod

 
am
 from pr
f

 

sta
c

    mod

s = pr
f

_c



t.mod

s.

st()
    mod

 = mod

s.data[0].
d
    pr

t(f"Us

g mod

: {mod

}")
    # 2: Pr
f

 Phas

    # G


rat
 u

qu
 r
qu
st ID to 


k pr
f

 a
d d
cod
 op
rat
o
s
    r
qu
st_
d = str(uu
d.uu
d4())
    pr

t(f"R
qu
st ID: {r
qu
st_
d}")
    pr
f

_r
spo
s
 = pr
f

_c



t.comp

t
o
s.cr
at
(
        mod

=mod

,
        # Prompt must 
xc
d vLLM's b
ock s
z
 (16 tok

s) for PD to 
ork
        prompt="Wr
t
 a d
ta


d 
xp
a
at
o
 of Pag
d Att

t
o
 for Tra
sform
rs 
orks 

c
ud

g th
 ma
ag
m

t of KV cach
 for mu
t
-tur
 co
v
rsat
o
s",
        max_tok

s=1,  # Forc
 pr
f

-o

y op
rat
o

        
xtra_body={
            "kv_tra
sf
r_params": {
                "do_r
mot
_d
cod
": Tru
,     # E
ab

 r
mot
 d
cod

                "do_r
mot
_pr
f

": Fa
s
,   # Th
s 
s th
 pr
f

 

sta
c

                "r
mot
_

g


_
d": No

,     # W

 b
 popu
at
d by vLLM
                "r
mot
_b
ock_
ds": No

,     # W

 b
 popu
at
d by vLLM
                "r
mot
_host": No

,          # W

 b
 popu
at
d by vLLM
                "r
mot
_port": No

,          # W

 b
 popu
at
d by vLLM
            }
        },
        
xtra_h
ad
rs={"X-R
qu
st-Id": r
qu
st_
d},
    )
    pr

t("-" * 50)
    pr

t("✓ Pr
f

 comp

t
d succ
ssfu
y")
    pr

t(f"Pr
f

 r
spo
s
: {pr
f

_r
spo
s
.cho
c
s[0].t
xt}")
    # 3: D
cod
 Phas

    # Tra
sf
r KV cach
 param
t
rs from pr
f

 to d
cod
 

sta
c

    d
cod
_r
spo
s
 = d
cod
_c



t.comp

t
o
s.cr
at
(
        mod

=mod

,
        prompt="Th
s prompt 
s 
g
or
d dur

g d
cod
",  # Or
g

a
 prompt 
ot 

d
d
        max_tok

s=150,  # G


rat
 up to 150 tok

s
        
xtra_body={
            "kv_tra
sf
r_params": pr
f

_r
spo
s
.kv_tra
sf
r_params  # Pass KV cach
 

fo
        },
        
xtra_h
ad
rs={"X-R
qu
st-Id": r
qu
st_
d},  # Sam
 r
qu
st ID
    )
    pr

t("-" * 50)
    pr

t("✓ D
cod
 comp

t
d succ
ssfu
y")
    pr

t(f"F

a
 r
spo
s
: {d
cod
_r
spo
s
.cho
c
s[0].t
xt}")

xc
pt Exc
pt
o
 as 
:
    pr

t(f"❌ Error dur

g d
saggr
gat
d s
rv

g: {
}")
    pr

t("Ch
ck that both pr
f

 a
d d
cod
 

sta
c
s ar
 ru


g a
d acc
ss
b

")
```
### B

chmark

g
    - To s
mu
at
 th
 d
cod
 d
p
oym

t of d
saggr
gat
d s
rv

g, pass `--kv-tra
sf
r-co
f
g '{"kv_co

ctor":"D
cod
B

chCo

ctor","kv_ro

":"kv_both"}'` to th
 `v
m s
rv
` 

vocat
o
. Th
 co

ctor popu
at
s KV cach
 

th ra
dom va
u
s so d
cod
 ca
 b
 prof


d 

 
so
at
o
.
    - **CUDAGraph captur
**: Us
 `--comp

at
o
_co
f
g '{"cudagraph_mod
": "FULL_DECODE_ONLY"}'` to 

ab

 CUDA graph captur
 for d
cod
 o

y a
d sav
 KV cach
.
