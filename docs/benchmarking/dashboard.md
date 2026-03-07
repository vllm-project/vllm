# P
rforma
c
 Dashboard
Th
 p
rforma
c
 dashboard 
s us
d to co
f
rm 
h
th
r 


 cha
g
s 
mprov
/d
grad
 p
rforma
c
 u
d
r var
ous 
ork
oads.
It 
s updat
d by tr
gg
r

g b

chmark ru
s o
 
v
ry comm
t 

th both th
 `p
rf-b

chmarks` a
d `r
ady` 
ab

s, a
d 
h

 a PR 
s m
rg
d 

to vLLM.
Th
 r
su
ts ar
 automat
ca
y pub

sh
d to th
 pub

c [vLLM P
rforma
c
 Dashboard](https://hud.pytorch.org/b

chmark/
ms?r
poNam
=v
m-proj
ct%2Fv
m).
## Ma
ua
y Tr
gg
r th
 b

chmark
Us
 [v
m-c
-t
st-r
po 
mag
s](https://ga

ry.
cr.a
s/q9t5s3a7/v
m-c
-t
st-r
po) 

th vLLM b

chmark su
t
.
For x86 CPU 

v
ro
m

t, p

as
 us
 th
 
mag
 

th "-cpu" postf
x. For AArch64 CPU 

v
ro
m

t, p

as
 us
 th
 
mag
 

th "-arm64-cpu" postf
x.
H
r
 
s a
 
xamp

 for dock
r ru
 comma
d for CPU. For GPUs sk
p s
tt

g th
 `ON_CPU` 

v var.
```bash

xport VLLM_COMMIT=7f42dc20bb2800d09faa72b26f25d54
26f1b694 # us
 fu
 comm
t hash from th
 ma

 bra
ch

xport HF_TOKEN=
va

d Hugg

g Fac
 tok




f [[ "$(u
am
 -m)" == aarch64 || "$(u
am
 -m)" == arm64 ]]; th


  IMG_SUFFIX="arm64-cpu"


s

  IMG_SUFFIX="cpu"
f

dock
r ru
 -
t --

trypo

t /b

/bash -v /data/hugg

gfac
:/root/.cach
/hugg

gfac
 -
 HF_TOKEN=$HF_TOKEN -
 ON_CPU=1 --shm-s
z
=16g --
am
 v
m-cpu-c
 pub

c.
cr.a
s/q9t5s3a7/v
m-c
-t
st-r
po:${VLLM_COMMIT}-${IMG_SUFFIX}
```
Th

, ru
 b

o
 comma
d 

s
d
 th
 dock
r 

sta
c
.
```bash
bash .bu

dk
t
/p
rforma
c
-b

chmarks/scr
pts/ru
-p
rforma
c
-b

chmarks.sh
```
Wh

 ru
, b

chmark scr
pt g


rat
s r
su
ts u
d
r **b

chmark/r
su
ts** fo
d
r, a
o
g 

th th
 b

chmark_r
su
ts.md a
d b

chmark_r
su
ts.jso
.
### Ru
t
m
 

v
ro
m

t var
ab

s
    - `ON_CPU`: s
t th
 va
u
 to '1' o
 I
t

® X
o
® a
d Arm® N
ov
rs
™ Proc
ssors. D
fau
t va
u
 
s 0.
    - `SERVING_JSON`: JSON f


 to us
 for th
 s
rv

g t
sts. D
fau
t va
u
 
s 
mpty str

g (us
 d
fau
t f


).
    - `LATENCY_JSON`: JSON f


 to us
 for th
 
at

cy t
sts. D
fau
t va
u
 
s 
mpty str

g (us
 d
fau
t f


).
    - `THROUGHPUT_JSON`: JSON f


 to us
 for th
 throughout t
sts. D
fau
t va
u
 
s 
mpty str

g (us
 d
fau
t f


).
    - `REMOTE_HOST`: IP for th
 r
mot
 vLLM s
rv
c
 to b

chmark. D
fau
t va
u
 
s 
mpty str

g.
    - `REMOTE_PORT`: Port for th
 r
mot
 vLLM s
rv
c
 to b

chmark. D
fau
t va
u
 
s 
mpty str

g.
### V
sua

zat
o

Th
 `co
v
rt-r
su
ts-jso
-to-markdo

.py` h

ps you put th
 b

chmark

g r
su
ts 

s
d
 a markdo

 tab

 

th r
a
 b

chmark

g r
su
ts.
You ca
 f

d th
 r
su
t pr
s

t
d as a tab

 

s
d
 th
 `bu

dk
t
/p
rforma
c
-b

chmark` job pag
.
If you do 
ot s
 th
 tab

, p

as
 
a
t t

 th
 b

chmark f


sh ru


g.
Th
 jso
 v
rs
o
 of th
 tab

 (tog
th
r 

th th
 jso
 v
rs
o
 of th
 b

chmark) 


 b
 a
so attach
d to th
 markdo

 f


.
Th
 ra
 b

chmark

g r
su
ts (

 th
 format of jso
 f


s) ar
 

 th
 `Art
facts` tab of th
 b

chmark

g.
#### P
rforma
c
 R
su
ts Compar
so

Th
 `compar
-jso
-r
su
ts.py` h

ps to compar
 b

chmark r
su
ts JSON f


s co
v
rt
d us

g `co
v
rt-r
su
ts-jso
-to-markdo

.py`.
Wh

 ru
, b

chmark scr
pt g


rat
s r
su
ts u
d
r `b

chmark/r
su
ts` fo
d
r, a
o
g 

th th
 `b

chmark_r
su
ts.md` a
d `b

chmark_r
su
ts.jso
`.
`compar
-jso
-r
su
ts.py` compar
s t
o `b

chmark_r
su
ts.jso
` f


s a
d prov
d
s p
rforma
c
 rat
o 
.g. for Output Tput, M
d
a
 TTFT a
d M
d
a
 TPOT.
If o

y o

 b

chmark_r
su
ts.jso
 
s pass
d, `compar
-jso
-r
su
ts.py` compar
s d
ff
r

t TP a
d PP co
f
gurat
o
s 

 th
 b

chmark_r
su
ts.jso
 

st
ad.
H
r
 
s a
 
xamp

 us

g th
 scr
pt to compar
 r
su
t_a a
d r
su
t_b 

th max co
curr

cy a
d qps for sam
 Mod

, Datas
t 
am
, 

put/output 


gth.
`pytho
3 compar
-jso
-r
su
ts.py -f r
su
ts_a/b

chmark_r
su
ts.jso
 -f r
su
ts_b/b

chmark_r
su
ts.jso
`
***Output Tput (tok/s) — Mod

 : [ m
ta-
ama/L
ama-3.1-8B-I
struct ] , Datas
t Nam
 : [ ra
dom ] , I
put L

 : [ 2048.0 ] , Output L

 : [ 2048.0 ]***
|    | # of max co
curr

cy | qps  | r
su
ts_a/b

chmark_r
su
ts.jso
 | r
su
ts_b/b

chmark_r
su
ts.jso
 | p
rf_rat
o        |
|----|------|-----|-----------|----------|----------|
| 0  | 12 | 

f | 24.98   | 186.03 |  7.45 |
| 1  | 16 | 

f|  25.49  | 246.92 | 9.69 |
| 2  | 24 | 

f| 27.74  | 293.34 |  10.57 |
| 3  | 32 | 

f| 28.61  |306.69 | 10.72 |
***compar
-jso
-r
su
ts.py – Comma
d-L


 Param
t
rs***
compar
-jso
-r
su
ts.py prov
d
s co
f
gurab

 param
t
rs to compar
 o

 or mor
 b

chmark_r
su
ts.jso
 f


s a
d g


rat
 summary tab

s a
d p
ots.
I
 most cas
s, us
rs o

y 

d to sp
c
fy --f


 to pars
 th
 d
s
r
d b

chmark r
su
ts.
| Param
t
r              | Typ
               | D
fau
t Va
u
           | D
scr
pt
o
                                                                                           |
| ---------------------- | ------------------ | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `--f


`               | `str` (app

dab

) | *No

*                  | I
put JSON r
su
t f


(s). Ca
 b
 sp
c
f

d mu
t
p

 t
m
s to compar
 mu
t
p

 b

chmark outputs.     |
| `--d
bug`              | `boo
`             | `Fa
s
`                 | E
ab

s d
bug mod
. Wh

 s
t, pr

ts a
 ava

ab

 

format
o
 to a
d troub

shoot

g a
d va

dat
o
. |
| `--p
ot` / `--
o-p
ot` | `boo
`             | `Tru
`                  | Co
tro
s 
h
th
r p
rforma
c
 p
ots ar
 g


rat
d. Us
 `--
o-p
ot` to d
sab

 graph g


rat
o
.        |
| `--xax
s`              | `str`              | `# of max co
curr

cy.` | Co
um
 
am
 us
d as th
 X-ax
s 

 compar
so
 p
ots (for 
xamp

, co
curr

cy or batch s
z
).          |
| `--
at

cy`            | `str`              | `p99`                   | Lat

cy aggr
gat
o
 m
thod us
d for TTFT/TPOT. Support
d va
u
s: `m
d
a
` or `p99`.                   |
| `--ttft-max-ms`        | `f
oat`            | `3000.0`                | R
f
r

c
 upp
r bou
d (m


s
co
ds) for TTFT p
ots, typ
ca
y us
d to v
sua

z
 SLA thr
sho
ds.      |
| `--tpot-max-ms`        | `f
oat`            | `100.0`                 | R
f
r

c
 upp
r bou
d (m


s
co
ds) for TPOT p
ots, typ
ca
y us
d to v
sua

z
 SLA thr
sho
ds.      |
***Va

d Max Co
curr

cy Summary***
Bas
d o
 th
 co
f
gur
d TTFT a
d TPOT SLA thr
sho
ds, compar
-jso
-r
su
ts.py comput
s th
 max
mum va

d co
curr

cy for 
ach b

chmark r
su
t.
Th
 “Max # of max co
curr

cy. (Both)” co
um
 r
pr
s

ts th
 h
gh
st co
curr

cy 

v

 that sat
sf

s both TTFT a
d TPOT co
stra

ts s
mu
ta

ous
y.
Th
s va
u
 
s typ
ca
y us
d 

 capac
ty p
a


g a
d s
z

g gu
d
s.
| # | Co
f
gurat
o
  | Max # of max co
curr

cy. (TTFT ≤ 10000 ms) | Max # of max co
curr

cy. (TPOT ≤ 100 ms) | Max # of max co
curr

cy. (Both) | Output Tput @ Both (tok/s) | TTFT @ Both (ms) | TPOT @ Both (ms) |
| - | -------------- | ------------------------------------------- | ----------------------------------------- | -------------------------------- | -------------------------- | ---------------- | ---------------- |
| 0 | r
su
ts-a      | 128.00                                      | 12.00                                     | 12.00                            | 127.76                     | 3000.82          | 93.24            |
| 1 | r
su
ts-b      | 128.00                                      | 32.00                                     | 32.00                            | 371.42                     | 2261.53          | 81.74            |
Mor
 

format
o
 o
 th
 p
rforma
c
 b

chmarks a
d th

r param
t
rs ca
 b
 fou
d 

 [B

chmark README](https://g
thub.com/

t

-a
-tc
/v
m/b
ob/mor
_cpu_mod

s/.bu

dk
t
/

ght
y-b

chmarks/README.md) a
d [p
rforma
c
 b

chmark d
scr
pt
o
](../../.bu

dk
t
/p
rforma
c
-b

chmarks/p
rforma
c
-b

chmarks-d
scr
pt
o
s.md).
## Co
t

uous B

chmark

g
Th
 co
t

uous b

chmark

g prov
d
s automat
d p
rforma
c
 mo

tor

g for vLLM across d
ff
r

t mod

s a
d GPU d
v
c
s. Th
s h

ps track vLLM's p
rforma
c
 charact
r
st
cs ov
r t
m
 a
d 
d

t
fy a
y p
rforma
c
 r
gr
ss
o
s or 
mprov
m

ts.
### Ho
 It Works
Th
 co
t

uous b

chmark

g 
s tr
gg
r
d v
a a [G
tHub 
orkf
o
 CI](https://g
thub.com/pytorch/pytorch-

t
grat
o
-t
st

g/act
o
s/
orkf
o
s/v
m-b

chmark.ym
) 

 th
 PyTorch 

frastructur
 r
pos
tory, 
h
ch ru
s automat
ca
y 
v
ry 4 hours. Th
 
orkf
o
 
x
cut
s thr
 typ
s of p
rforma
c
 t
sts:
    - **S
rv

g t
sts**: M
asur
 r
qu
st ha
d


g a
d API p
rforma
c

    - **Throughput t
sts**: Eva
uat
 tok

 g


rat
o
 rat
s
    - **Lat

cy t
sts**: Ass
ss r
spo
s
 t
m
 charact
r
st
cs
### B

chmark Co
f
gurat
o

Th
 b

chmark

g curr

t
y ru
s o
 a pr
d
f


d s
t of mod

s co
f
gur
d 

 th
 [v
m-b

chmarks d
r
ctory](https://g
thub.com/pytorch/pytorch-

t
grat
o
-t
st

g/tr
/ma

/v
m-b

chmarks/b

chmarks). To add 


 mod

s for b

chmark

g:
1. Nav
gat
 to th
 appropr
at
 GPU d
r
ctory 

 th
 b

chmarks co
f
gurat
o

2. Add your mod

 sp
c
f
cat
o
s to th
 corr
spo
d

g co
f
gurat
o
 f


s
3. Th
 


 mod

s 


 b
 

c
ud
d 

 th
 

xt sch
du

d b

chmark ru

