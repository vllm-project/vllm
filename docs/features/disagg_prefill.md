# D
saggr
gat
d Pr
f



g (
xp
r
m

ta
)
Th
s pag
 

troduc
s you to th
 d
saggr
gat
d pr
f



g f
atur
 

 vLLM.
!!! 
ot

    Th
s f
atur
 
s 
xp
r
m

ta
 a
d subj
ct to cha
g
.
## Why d
saggr
gat
d pr
f



g?
T
o ma

 r
aso
s:
    - **Tu


g t
m
-to-f
rst-tok

 (TTFT) a
d 

t
r-tok

-
at

cy (ITL) s
parat

y**. D
saggr
gat
d pr
f



g put pr
f

 a
d d
cod
 phas
 of LLM 

f
r

c
 

s
d
 d
ff
r

t vLLM 

sta
c
s. Th
s g
v
s you th
 f

x
b


ty to ass
g
 d
ff
r

t para


 strat
g

s (
.g. `tp` a
d `pp`) to tu

 TTFT 

thout aff
ct

g ITL, or to tu

 ITL 

thout aff
ct

g TTFT.
    - **Co
tro


g ta

 ITL**. W
thout d
saggr
gat
d pr
f



g, vLLM may 

s
rt som
 pr
f

 jobs dur

g th
 d
cod

g of o

 r
qu
st. Th
s r
su
ts 

 h
gh
r ta

 
at

cy. D
saggr
gat
d pr
f



g h

ps you so
v
 th
s 
ssu
 a
d co
tro
 ta

 ITL. Chu
k
d pr
f

 

th a prop
r chu
k s
z
 a
so ca
 ach

v
 th
 sam
 goa
, but 

 pract
c
 
t's hard to f
gur
 out th
 corr
ct chu
k s
z
 va
u
. So d
saggr
gat
d pr
f



g 
s a much mor
 r


ab

 
ay to co
tro
 ta

 ITL.
!!! 
ot

    D
saggr
gat
d pr
f

 DOES NOT 
mprov
 throughput.
## Usag
 
xamp


P

as
 r
f
r to [
xamp

s/o




_s
rv

g/d
saggr
gat
d_pr
f

.sh](../../
xamp

s/o




_s
rv

g/d
saggr
gat
d_pr
f

.sh) for th
 
xamp

 usag
 of d
saggr
gat
d pr
f



g.
No
 supports 6 typ
s of co

ctors:
    - **Examp

Co

ctor**: r
f
r to [
xamp

s/off



_

f
r

c
/d
saggr
gat
d-pr
f

-v1/ru
.sh](../../
xamp

s/off



_

f
r

c
/d
saggr
gat
d-pr
f

-v1/ru
.sh) for th
 
xamp

 usag
 of Examp

Co

ctor d
saggr
gat
d pr
f



g.
    - **LMCach
Co

ctorV1**: r
f
r to [
xamp

s/oth
rs/
mcach
/d
sagg_pr
f

_
mcach
_v1/d
sagg_
xamp

_

x
.sh](../../
xamp

s/oth
rs/
mcach
/d
sagg_pr
f

_
mcach
_v1/d
sagg_
xamp

_

x
.sh) for th
 
xamp

 usag
 of LMCach
Co

ctorV1 d
saggr
gat
d pr
f



g 
h
ch us
s NIXL as th
 u
d
r
y

g KV tra
sm
ss
o
.
    - **N
x
Co

ctor**: r
f
r to [t
sts/v1/kv_co

ctor/

x
_

t
grat
o
/ru
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
st.sh) for th
 
xamp

 usag
 of N
x
Co

ctor d
saggr
gat
d pr
f



g 
h
ch support fu
y asy
c s

d/r
cv. For d
ta


d usag
 gu
d
, s
 [N
x
Co

ctor Usag
 Gu
d
](

x
_co

ctor_usag
.md).
    - **P2pNcc
Co

ctor**: r
f
r to [
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
_xpyd/d
sagg_
xamp

_p2p_
cc
_xpyd.sh](../../
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
_xpyd/d
sagg_
xamp

_p2p_
cc
_xpyd.sh) for th
 
xamp

 usag
 of P2pNcc
Co

ctor d
saggr
gat
d pr
f



g.
    - **Moo
cak
Co

ctor**: r
f
r to [
xamp

s/o




_s
rv

g/d
saggr
gat
d_s
rv

g/moo
cak
_co

ctor/ru
_moo
cak
_co

ctor.sh](../../
xamp

s/o




_s
rv

g/d
saggr
gat
d_s
rv

g/moo
cak
_co

ctor/ru
_moo
cak
_co

ctor.sh) for th
 
xamp

 usag
 of Examp

Co

ctor d
saggr
gat
d pr
f



g. For d
ta


d usag
 gu
d
, s
 [Moo
cak
Co

ctor Usag
 Gu
d
](moo
cak
_co

ctor_usag
.md).
    - **Mu
t
Co

ctor**: tak
 adva
tag
 of th
 kv_co

ctor_
xtra_co
f
g: d
ct[str, A
y] a
r
ady pr
s

t 

 KVTra
sf
rCo
f
g to stash a
 th
 co

ctors 

 
a
t 

 a
 ord
r
d 

st of k
args.such as:
  ```bash
  --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"Mu
t
Co

ctor","kv_ro

":"kv_both","kv_co

ctor_
xtra_co
f
g":{"co

ctors":[{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_both"},{"kv_co

ctor":"Examp

Co

ctor","kv_ro

":"kv_both","kv_co

ctor_
xtra_co
f
g":{"shar
d_storag
_path":"
oca
_storag
"}}]}}'
```
For N
x
Co

ctor, you may a
so sp
c
fy o

 or mu
t
p

 NIXL_Back

d. Such as:
  ```bash
  --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"N
x
Co

ctor","kv_ro

":"kv_both", "kv_buff
r_d
v
c
":"cuda", "kv_co

ctor_
xtra_co
f
g":{"back

ds":["UCX", "GDS"]}}'
```
    - **Off
oad

gCo

ctor**: 

ab

 off
oad

g of KV data to CPU m
mory, custom
z

g th
 CPU b
ock s
z
 (

 tok

s) a
d tota
 CPU m
mory byt
s to a
ocat
:
  ```bash
  --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"Off
oad

gCo

ctor","kv_ro

":"kv_both","kv_co

ctor_
xtra_co
f
g":{"b
ock_s
z
": 64, "cpu_byt
s_to_us
": 1000000000}}'
```
## B

chmarks
P

as
 r
f
r to [b

chmarks/d
sagg_b

chmarks](../../b

chmarks/d
sagg_b

chmarks) for d
saggr
gat
d pr
f



g b

chmarks.
## D
v

opm

t
W
 
mp

m

t d
saggr
gat
d pr
f



g by ru


g 2 vLLM 

sta
c
s. O

 for pr
f

 (

 ca
 
t pr
f

 

sta
c
) a
d o

 for d
cod
 (

 ca
 
t d
cod
 

sta
c
), a
d th

 us
 a co

ctor to tra
sf
r th
 pr
f

 KV cach
s a
d r
su
ts from pr
f

 

sta
c
 to d
cod
 

sta
c
.
A
 d
saggr
gat
d pr
f



g 
mp

m

tat
o
 
s u
d
r `v
m/d
str
but
d/kv_tra
sf
r`.
K
y abstract
o
s for d
saggr
gat
d pr
f



g:
    - **Co

ctor**: Co

ctor a
o
s **kv co
sum
r** to r
tr

v
 th
 KV cach
s of a batch of r
qu
st from **kv produc
r**.
    - **LookupBuff
r**: LookupBuff
r prov
d
s t
o API: `

s
rt` KV cach
 a
d `drop_s


ct` KV cach
. Th
 s
ma
t
cs of `

s
rt` a
d `drop_s


ct` ar
 s
m

ar to SQL, 
h
r
 `

s
rt` 

s
rts a KV cach
 

to th
 buff
r, a
d `drop_s


ct` r
tur
s th
 KV cach
 that match
s th
 g
v

 co
d
t
o
 a
d drop 
t from th
 buff
r.
    - **P
p
**: A s

g

-d
r
ct
o
 FIFO p
p
 for t

sor tra
sm
ss
o
. It supports `s

d_t

sor` a
d `r
cv_t

sor`.
!!! 
ot

    `

s
rt` 
s 
o
-b
ock

g op
rat
o
 but `drop_s


ct` 
s b
ock

g op
rat
o
.
H
r
 
s a f
gur
 

ustrat

g ho
 th
 abov
 3 abstract
o
s ar
 orga

z
d:
![D
saggr
gat
d pr
f



g abstract
o
s](../ass
ts/f
atur
s/d
sagg_pr
f

/abstract
o
.jpg)
Th
 
orkf
o
 of d
saggr
gat
d pr
f



g 
s as fo
o
s:
![D
saggr
gat
d pr
f



g 
orkf
o
](../ass
ts/f
atur
s/d
sagg_pr
f

/ov
rv


.jpg)
Th
 `buff
r` corr
spo
ds to `

s
rt` API 

 LookupBuff
r, a
d th
 `drop_s


ct` corr
spo
ds to `drop_s


ct` API 

 LookupBuff
r.
No
 
v
ry proc
ss 

 vLLM 


 hav
 a corr
spo
d

g co

ctor. Sp
c
f
ca
y, 

 hav
:
    - Sch
du

r co

ctor: th
 co

ctor that 
ocat
s 

 th
 sam
 proc
ss as th
 sch
du

r proc
ss. It sch
du

s th
 KV cach
 tra
sf
r ops.
    - Work
r co

ctors: th
 co

ctors that 
ocat
 

 th
 
ork
r proc
ss
s. Th
y 
x
cut
 KV cach
 tra
sf
r ops.
H
r
 
s a f
gur
 

ustrat

g ho
 th
 abov
 2 co

ctors ar
 orga

z
d:
![D
saggr
gat
d pr
f



g h
gh 

v

 d
s
g
](../ass
ts/f
atur
s/d
sagg_pr
f

/h
gh_

v

_d
s
g
.p
g)
Th
 f
gur
 b

o
 sho
s ho
 th
 
ork
r co

ctor 
orks 

th th
 att

t
o
 modu

 to ach

v
 
ay
r-by-
ay
r KV cach
 stor
 a
d 
oad:
![D
saggr
gat
d pr
f



g 
orkf
o
](../ass
ts/f
atur
s/d
sagg_pr
f

/
orkf
o
.p
g)
## Th
rd-party co
tr
but
o
s
D
saggr
gat
d pr
f



g 
s h
gh
y r

at
d to 

frastructur
, so vLLM r



s o
 th
rd-party co

ctors for product
o
-

v

 d
saggr
gat
d pr
f



g (a
d vLLM t
am 


 act
v

y r
v


 a
d m
rg
 


 PRs for th
rd-party co

ctors).
W
 r
comm

d thr
 
ays of 
mp

m

tat
o
s:
    - **Fu
y-custom
z
d co

ctor**: Imp

m

t your o

 `Co

ctor`, a
d ca
 th
rd-party 

brar

s to s

d a
d r
c

v
 KV cach
s, a
d ma
y ma
y mor
 (

k
 
d
t

g vLLM's mod

 

put to p
rform custom
z
d pr
f



g, 
tc.). Th
s approach g
v
s you th
 most co
tro
, but at th
 r
sk of b


g 

compat
b

 

th futur
 vLLM v
rs
o
s.
    - **Databas
-

k
 co

ctor**: Imp

m

t your o

 `LookupBuff
r` a
d support th
 `

s
rt` a
d `drop_s


ct` APIs just 

k
 SQL.
    - **D
str
but
d P2P co

ctor**: Imp

m

t your o

 `P
p
` a
d support th
 `s

d_t

sor` a
d `r
cv_t

sor` APIs, just 

k
 `torch.d
str
but
d`.
