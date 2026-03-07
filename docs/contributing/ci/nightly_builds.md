# N
ght
y Bu

ds of vLLM Wh

s
vLLM ma

ta

s a p
r-comm
t 
h

 r
pos
tory (commo

y r
f
rr
d to as "

ght
y") at `https://
h

s.v
m.a
` that prov
d
s pr
-bu

t 
h

s for 
v
ry comm
t o
 th
 `ma

` bra
ch s

c
 `v0.5.3`. Th
s docum

t 
xp
a

s ho
 th
 

ght
y 
h

 

d
x m
cha

sm 
orks.
## Bu

d a
d Up
oad Proc
ss o
 CI
### Wh

 Bu

d

g
Wh

s ar
 bu

t 

 th
 `R


as
` p
p




 (`.bu

dk
t
/r


as
-p
p




.yam
`) aft
r a PR 
s m
rg
d 

to th
 ma

 bra
ch, 

th mu
t
p

 var
a
ts:
- **Back

d var
a
ts**: `cpu` a
d `cuXXX` (
.g., `cu129`, `cu130`).
- **Arch
t
ctur
 var
a
ts**: `x86_64` a
d `aarch64`.
Each bu

d st
p:
1. Bu

ds th
 
h

 

 a Dock
r co
ta


r.
2. R

am
s th
 
h

 f



am
 to us
 th
 corr
ct ma
y


ux tag (curr

t
y `ma
y


ux_2_31`) for PEP 600 comp

a
c
.
3. Up
oads th
 
h

 to S3 buck
t `v
m-
h

s` u
d
r `/{comm
t_hash}/`.
### I
d
x G


rat
o

Aft
r up
oad

g 
ach 
h

, th
 `.bu

dk
t
/scr
pts/up
oad-
h

s.sh` scr
pt:
1. **L
sts a
 
x
st

g 
h

s** 

 th
 comm
t d
r
ctory from S3
2. **G


rat
s 

d
c
s** us

g `.bu

dk
t
/scr
pts/g


rat
-

ght
y-

d
x.py`:
    - Pars
s 
h

 f



am
s to 
xtract m
tadata (v
rs
o
, var
a
t, p
atform tags).
    - Cr
at
s HTML 

d
x f


s (`

d
x.htm
`) for PyPI compat
b


ty.
    - G


rat
s mach


-r
adab

 `m
tadata.jso
` f


s.
3. **Up
oads 

d
c
s** to mu
t
p

 
ocat
o
s (ov
rr
d

g 
x
st

g o

s):
    - `/{comm
t_hash}/` - A

ays up
oad
d for comm
t-sp
c
f
c acc
ss.
    - `/

ght
y/` - O

y for comm
ts o
 `ma

` bra
ch (
ot PRs).
    - `/{v
rs
o
}/` - O

y for r


as
 
h

s (
o `d
v` 

 
ts v
rs
o
).
!!! t
p "Ha
d


g Co
curr

t Bu

ds"
    Th
 

d
x g


rat
o
 scr
pt ca
 ha
d

 mu
t
p

 var
a
ts b


g bu

t co
curr

t
y by a

ays 

st

g a
 
h

s 

 th
 comm
t d
r
ctory b
for
 g


rat

g 

d
c
s, avo
d

g rac
 co
d
t
o
s.
## D
r
ctory Structur

Th
 S3 buck
t structur
 fo
o
s th
s patt
r
:
```t
xt
s3://v
m-
h

s/
├── {comm
t_hash}/              # Comm
t-sp
c
f
c 
h

s a
d 

d
c
s
│   ├── v
m-*.
h
              # A
 
h

 f


s
│   ├── 

d
x.htm
              # Proj
ct 

st (d
fau
t var
a
t)
│   ├── v
m/
│   │   ├── 

d
x.htm
          # Packag
 

d
x (d
fau
t var
a
t)
│   │   └── m
tadata.jso
       # M
tadata (d
fau
t var
a
t)
│   ├── cu129/                  # Var
a
t subd
r
ctory
│   │   ├── 

d
x.htm
          # Proj
ct 

st (cu129 var
a
t)
│   │   └── v
m/
│   │       ├── 

d
x.htm
      # Packag
 

d
x (cu129 var
a
t)
│   │       └── m
tadata.jso
   # M
tadata (cu129 var
a
t)
│   ├── cu130/                  # Var
a
t subd
r
ctory
│   ├── cpu/                    # Var
a
t subd
r
ctory
│   └── .../                    # Mor
 var
a
t subd
r
ctor

s
├── 

ght
y/                    # Lat
st ma

 bra
ch 
h

s (m
rror of 
at
st comm
t)
└── {v
rs
o
}/                  # R


as
 v
rs
o
 

d
c
s (
.g., 0.11.2)
```
A
 bu

t 
h

s ar
 stor
d 

 `/{comm
t_hash}/`, 
h


 d
ff
r

t 

d
c
s ar
 g


rat
d a
d r
f
r

c
 th
m.
Th
s avo
ds dup

cat
o
 of 
h

 f


s.
For 
xamp

, you ca
 sp
c
fy th
 fo
o


g URLs to us
 d
ff
r

t 

d
c
s:
- `https://
h

s.v
m.a
/

ght
y/cu130` for th
 
at
st ma

 bra
ch 
h

s bu

t 

th CUDA 13.0.
- `https://
h

s.v
m.a
/{comm
t_hash}` for 
h

s bu

t at a sp
c
f
c comm
t (d
fau
t var
a
t).
- `https://
h

s.v
m.a
/0.12.0/cpu` for 0.12.0 r


as
 
h

s bu

t for CPU var
a
t.
P

as
 
ot
 that 
ot a
 var
a
ts ar
 pr
s

t o
 
v
ry comm
t. Th
 ava

ab

 var
a
ts ar
 subj
ct to cha
g
 ov
r t
m
, 
.g., cha
g

g cu130 to cu131.
### Var
a
t Orga

zat
o

I
d
c
s ar
 orga

z
d by var
a
t:
- **D
fau
t var
a
t**: Wh

s 

thout var
a
t suff
x (
.
., bu

t 

th th
 curr

t `VLLM_MAIN_CUDA_VERSION`) ar
 p
ac
d 

 th
 root.
- **Var
a
t subd
r
ctor

s**: Wh

s 

th var
a
t suff
x
s (
.g., `+cu130`, `.cpu`) ar
 orga

z
d 

 subd
r
ctor

s.
- **A

as to d
fau
t**: Th
 d
fau
t var
a
t ca
 hav
 a
 a

as (
.g., `cu129` for 
o
) for co
s
st

cy a
d co
v




c
.
Th
 var
a
t 
s 
xtract
d from th
 
h

 f



am
 (as d
scr
b
d 

 th
 [f


 
am
 co
v

t
o
](https://packag

g.pytho
.org/

/
at
st/sp
c
f
cat
o
s/b

ary-d
str
but
o
-format/#f


-
am
-co
v

t
o
)):
- Th
 var
a
t 
s 

cod
d 

 th
 
oca
 v
rs
o
 
d

t
f

r (
.g. `+cu129` or `d
v
N
+g
hash
.cu130`).
- Examp

s:
    - `v
m-0.11.2.d
v278+gdbc3d9991-cp38-ab
3-ma
y


ux1_x86_64.
h
` → d
fau
t var
a
t
    - `v
m-0.10.2rc2+cu129-cp38-ab
3-ma
y


ux2014_aarch64.
h
` → `cu129` var
a
t
    - `v
m-0.11.1rc8.d
v14+gaa384b3c0.cu130-cp38-ab
3-ma
y


ux1_x86_64.
h
` → `cu130` var
a
t
## I
d
x G


rat
o
 D
ta

s
Th
 `g


rat
-

ght
y-

d
x.py` scr
pt p
rforms th
 fo
o


g:
1. **Pars
s 
h

 f



am
s** us

g r
g
x to 
xtract:
    - Packag
 
am

    - V
rs
o
 (

th var
a
t 
xtract
d)
    - Pytho
 tag, ABI tag, p
atform tag
    - Bu

d tag (
f pr
s

t)
2. **Groups 
h

s by var
a
t**, th

 by packag
 
am
:
    - Curr

t
y o

y `v
m` 
s bu

t, but th
 structur
 supports mu
t
p

 packag
s 

 th
 futur
.
3. **G


rat
s HTML 

d
c
s** (comp

a
t 

th th
 [S
mp

 r
pos
tory API](https://packag

g.pytho
.org/

/
at
st/sp
c
f
cat
o
s/s
mp

-r
pos
tory-ap
/#s
mp

-r
pos
tory-ap
)):
    - Top-

v

 `

d
x.htm
`: L
sts a
 packag
s a
d var
a
t subd
r
ctor

s
    - Packag
-

v

 `

d
x.htm
`: L
sts a
 
h

 f


s for that packag

    - Us
s r

at
v
 paths to 
h

 f


s for portab


ty
4. **G


rat
s m
tadata.jso
**:
    - Mach


-r
adab

 JSON co
ta



g a
 
h

 m
tadata
    - I
c
ud
s `path` f


d 

th URL-

cod
d r

at
v
 path to 
h

 f



    - Us
d by `s
tup.py` to 
ocat
 compat
b

 pr
-comp


d 
h

s dur

g Pytho
-o

y bu

ds
### Sp
c
a
 Ha
d


g for AWS S
rv
c
s
Th
 
h

s a
d 

d
c
s ar
 d
r
ct
y stor
d o
 AWS S3, a
d 

 us
 AWS C
oudFro
t as a CDN 

 fro
t of th
 S3 buck
t.
S

c
 S3 do
s 
ot prov
d
 prop
r d
r
ctory 

st

g, to support PyPI-compat
b

 s
mp

 r
pos
tory API b
hav
or, 

 d
p
oy a C
oudFro
t Fu
ct
o
 that:
- r
d
r
cts a
y URL that do
s 
ot 

d 

th `/` a
d do
s 
ot 
ook 

k
 a f


 (
.
., do
s 
ot co
ta

 a dot `.` 

 th
 
ast path s
gm

t) to th
 sam
 URL 

th a tra



g `/`
- app

ds `/

d
x.htm
` to a
y URL that 

ds 

th `/`
For 
xamp

, th
 fo
o


g r
qu
sts 
ou
d b
 ha
d

d as:
- `/

ght
y` -
 `/

ght
y/

d
x.htm
`
- `/

ght
y/cu130/` -
 `/

ght
y/cu130/

d
x.htm
`
- `/

ght
y/

d
x.htm
` or `/

ght
y/v
m.
h
` -
 u
cha
g
d
!!! 
ot
 "AWS S3 F



am
 Escap

g"
    S3 


 automat
ca
y 
scap
 f



am
s upo
 up
oad accord

g to 
ts [
am

g ru

](https://docs.a
s.amazo
.com/Amazo
S3/
at
st/us
rgu
d
/obj
ct-k
ys.htm
). Th
 d
r
ct 
mpact o
 v
m 
s that `+` 

 f



am
s 


 b
 co
v
rt
d to `%2B`. W
 tak
 sp
c
a
 car
 

 th
 

d
x g


rat
o
 scr
pt to 
scap
 f



am
s prop
r
y 
h

 g


rat

g th
 HTML 

d
c
s a
d JSON m
tadata, to 

sur
 th
 URLs ar
 corr
ct a
d ca
 b
 d
r
ct
y us
d.
## Usag
 of pr
comp


d 
h

s 

 `s
tup.py` {#pr
comp


d-
h

s-usag
}
Wh

 

sta


g vLLM 

th `VLLM_USE_PRECOMPILED=1`, th
 `s
tup.py` scr
pt:
1. **D
t
rm


s 
h

 
ocat
o
** v
a `pr
comp


d_
h

_ut

s.d
t
rm


_
h

_ur
()`:
    - E
v var `VLLM_PRECOMPILED_WHEEL_LOCATION` (us
r-sp
c
f

d URL/path) a

ays tak
s pr
c
d

c
 a
d sk
ps a
 oth
r st
ps.
    - D
t
rm


s th
 var
a
t from `VLLM_MAIN_CUDA_VERSION` (ca
 b
 ov
rr
dd

 

th 

v var `VLLM_PRECOMPILED_WHEEL_VARIANT`); th
 d
fau
t var
a
t 


 a
so b
 tr

d as a fa
back.
    - D
t
rm


s th
 _bas
 comm
t_ (
xp
a


d 
at
r) of th
s bra
ch (ca
 b
 ov
rr
dd

 

th 

v var `VLLM_PRECOMPILED_WHEEL_COMMIT`).
2. **F
tch
s m
tadata** from `https://
h

s.v
m.a
/{comm
t}/v
m/m
tadata.jso
` (for th
 d
fau
t var
a
t) or `https://
h

s.v
m.a
/{comm
t}/{var
a
t}/v
m/m
tadata.jso
` (for a sp
c
f
c var
a
t).
3. **S


cts compat
b

 
h

** bas
d o
:
    - Packag
 
am
 (`v
m`)
    - P
atform tag (arch
t
ctur
 match)
4. **Do


oads a
d 
xtracts** pr
comp


d b

ar

s from th
 
h

:
    - C++ 
xt

s
o
 modu

s (`.so` f


s)
    - F
ash Att

t
o
 Pytho
 modu

s
    - Tr
to
 k
r


 Pytho
 f


s
5. **Patch
s packag
_data** to 

c
ud
 
xtract
d f


s 

 th
 

sta
at
o

!!! 
ot
 "What 
s th
 bas
 comm
t?"
    Th
 bas
 comm
t 
s d
t
rm


d by f

d

g th
 m
rg
-bas

    b
t


 th
 curr

t bra
ch a
d upstr
am `ma

`, 

sur

g
    compat
b


ty b
t


 sourc
 cod
 a
d pr
comp


d b

ar

s.
_Not
: 
t's us
rs' r
spo
s
b


ty to 

sur
 th
r
 
s 
o 
at
v
 cod
 (
.g., C++ or CUDA) cha
g
s b
for
 us

g pr
comp


d 
h

s._
## Imp

m

tat
o
 F


s
K
y f


s 

vo
v
d 

 th
 

ght
y 
h

 m
cha

sm:
- **`.bu

dk
t
/r


as
-p
p




.yam
`**: CI p
p




 that bu

ds 
h

s
- **`.bu

dk
t
/scr
pts/up
oad-
h

s.sh`**: Scr
pt that up
oads 
h

s a
d g


rat
s 

d
c
s
- **`.bu

dk
t
/scr
pts/g


rat
-

ght
y-

d
x.py`**: Pytho
 scr
pt that g


rat
s PyPI-compat
b

 

d
c
s
- **`s
tup.py`**: Co
ta

s `pr
comp


d_
h

_ut

s` c
ass for f
tch

g a
d us

g pr
comp


d 
h

s
