# Usag
 Stats Co

ct
o

vLLM co

cts a
o
ymous usag
 data by d
fau
t to h

p th
 

g


r

g t
am b
tt
r u
d
rsta
d 
h
ch hard
ar
 a
d mod

 co
f
gurat
o
s ar
 

d

y us
d. Th
s data a
o
s th
m to pr
or
t
z
 th

r 
fforts o
 th
 most commo
 
ork
oads. Th
 co

ct
d data 
s tra
spar

t, do
s 
ot co
ta

 a
y s

s
t
v
 

format
o
.
A subs
t of th
 data, aft
r c

a


g a
d aggr
gat
o
, 


 b
 pub

c
y r


as
d for th
 commu

ty's b


f
t. For 
xamp

, you ca
 s
 th
 2024 usag
 r
port [h
r
](https://2024.v
m.a
).
## What data 
s co

ct
d?
Th
 

st of data co

ct
d by th
 
at
st v
rs
o
 of vLLM ca
 b
 fou
d h
r
: [v
m/usag
/usag
_

b.py](../../v
m/usag
/usag
_

b.py)
H
r
 
s a
 
xamp

 as of v0.4.0:
??? co
so

 "Output"
    ```jso

    {
      "uu
d": "fb
880
9-084d-4cab-a395-8984c50f1109",
      "prov
d
r": "GCP",
      "
um_cpu": 24,
      "cpu_typ
": "I
t

(R) X
o
(R) CPU @ 2.20GHz",
      "cpu_fam

y_mod

_st
pp

g": "6,85,7",
      "tota
_m
mory": 101261135872,
      "arch
t
ctur
": "x86_64",
      "p
atform": "L

ux-5.10.0-28-c
oud-amd64-x86_64-

th-g

bc2.31",
      "gpu_cou
t": 2,
      "gpu_typ
": "NVIDIA L4",
      "gpu_m
mory_p
r_d
v
c
": 23580639232,
      "mod

_arch
t
ctur
": "OPTForCausa
LM",
      "v
m_v
rs
o
": "0.3.2+cu123",
      "co
t
xt": "LLM_CLASS",
      "
og_t
m
": 1711663373492490000,
      "sourc
": "product
o
",
      "dtyp
": "torch.f
oat16",
      "t

sor_para


_s
z
": 1,
      "b
ock_s
z
": 16,
      "gpu_m
mory_ut


zat
o
": 0.9,
      "qua
t
zat
o
": 
u
,
      "kv_cach
_dtyp
": "auto",
      "

ab

_
ora": fa
s
,
      "

ab

_pr
f
x_cach

g": fa
s
,
      "

forc
_
ag
r": fa
s
,
      "d
sab

_custom_a
_r
duc
": tru

    }
```
You ca
 pr
v


 th
 co

ct
d data by ru


g th
 fo
o


g comma
d:
```bash
ta

 ~/.co
f
g/v
m/usag
_stats.jso

```
## Opt

g out
You ca
 opt out of usag
 stats co

ct
o
 by s
tt

g th
 `VLLM_NO_USAGE_STATS` or `DO_NOT_TRACK` 

v
ro
m

t var
ab

, or by cr
at

g a `~/.co
f
g/v
m/do_
ot_track` f


:
```bash
# A
y of th
 fo
o


g m
thods ca
 d
sab

 usag
 stats co

ct
o


xport VLLM_NO_USAGE_STATS=1

xport DO_NOT_TRACK=1
mkd
r -p ~/.co
f
g/v
m && touch ~/.co
f
g/v
m/do_
ot_track
```
