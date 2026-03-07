# S

p Mod

vLLM's S

p Mod
 a
o
s you to t
mporar

y r


as
 most GPU m
mory us
d by a mod

, 

c
ud

g mod

 


ghts a
d KV cach
, 

thout stopp

g th
 s
rv
r or u

oad

g th
 Dock
r co
ta


r. Th
s 
s 
sp
c
a
y us
fu
 for RLHF, tra



g, or cost-sav

g sc

ar
os 
h
r
 GPU r
sourc
s 

d to b
 fr
d b
t


 

f
r

c
 
ork
oads.
K
y b


f
ts:
- **Fr
s GPU m
mory**: Off
oads mod

 


ghts to CPU RAM a
d d
scards KV cach
, r


as

g up to 90%+ of GPU m
mory for oth
r tasks.
- **Fast r
sum
**: Qu
ck
y 
ak
 up th
 

g


 a
d r
sum
 

f
r

c
 

thout fu
 mod

 r

oad.
- **API 

dpo

ts**: Co
tro
 s

p/
ak
_up stat
 v
a HTTP 

dpo

ts or Pytho
 API.
- **Supports d
str
but
d 
ork
oads**: Works 

th t

sor para



sm, p
p




 para



sm, 
tc.
- **F


-gra


d co
tro
**: Opt
o
a
y 
ak
 up o

y mod

 


ghts or KV cach
 to avo
d OOM dur

g 


ght updat
s.
!!! 
ot

    Th
s f
atur
 
s 
o
 support
d o
 CUDA a
d ROCm p
atform.
!!! 
ot

    For mor
 

format
o
, s
 th
s [B
og Post](https://b
og.v
m.a
/2025/10/26/s

p-mod
.htm
).
## S

p 

v

s
L
v

 1 s

p 


 off
oad th
 mod

 


ghts a
d d
scard th
 KV cach
. Th
 co
t

t of KV cach
 
s forgott

. L
v

 1 s

p 
s good for s

p

g a
d 
ak

g up th
 

g


 to ru
 th
 sam
 mod

 aga

. Th
 mod

 


ghts ar
 back
d up 

 CPU m
mory. P

as
 mak
 sur
 th
r
's 

ough CPU m
mory to stor
 th
 mod

 


ghts. L
v

 2 s

p 


 d
scard both th
 mod

 


ghts a
d th
 KV cach
 (
h


 th
 mod

's buff
rs ar
 k
pt 

 CPU, 

k
 rop
 sca


g t

sors). Th
 co
t

t of both th
 mod

 


ghts a
d KV cach
 
s forgott

. L
v

 2 s

p 
s good for s

p

g a
d 
ak

g up th
 

g


 to ru
 a d
ff
r

t mod

 or updat
 th
 mod

, 
h
r
 pr
v
ous mod

 


ghts ar
 
ot 

d
d, 
.g. RLHF 


ght updat
.
## Usag

### Off



 

f
r

c

E
ab

 s

p mod
 by pass

g `

ab

_s

p_mod
=Tru
` to th
 `LLM` c
ass.
```pytho

from v
m 
mport LLM

m = LLM("Q


/Q


3-0.6B", 

ab

_s

p_mod
=Tru
)
```
#### Pytho
 API
```pytho

# S

p 

v

 1
# Put th
 

g


 to s

p (

v

=1: off
oad 


ghts to CPU RAM, d
scard KV cach
)

m.s

p(

v

=1)
# Wak
 up th
 

g


 (r
stor
 


ghts)

m.
ak
_up()
```
```pytho

# S

p 

v

 2
# Put th
 

g


 to s

p (

v

=2: d
scard both 


ghts a
d KV cach
)

m.s

p(

v

=2)
# R
a
ocat
 


ghts m
mory o

y

m.
ak
_up(tags=["


ghts"])
# Load 


ghts 

-p
ac


m.co

ct
v
_rpc("r

oad_


ghts")
# R
a
ocat
 KV cach


m.
ak
_up(tags=["kv_cach
"])
```
#### RLHF 


ght updat
s
Dur

g RLHF tra



g, vLLM a
o
s you to s


ct
v

y 
ak
 up o

y th
 mod

 


ghts or th
 KV cach
 us

g th
 tags argum

t 

 
ak
_up(). Th
s f


-gra


d co
tro
 
s 
sp
c
a
y us
fu
 
h

 updat

g mod

 


ghts: by 
ak

g up just th
 


ghts (
.g., 
m.
ak
_up(tags=["


ghts"])), you avo
d a
ocat

g m
mory for th
 KV cach
 u
t

 aft
r th
 


ght updat
 
s comp

t
. Th
s approach h

ps pr
v

t GPU out-of-m
mory (OOM) 
rrors, part
cu
ar
y 

th 
arg
 mod

s, by m


m
z

g p
ak m
mory usag
 dur

g 


ght sy
chro

zat
o
 a
d updat
 op
rat
o
s.
Us
 `tags=["


ghts"]` or `tags=["kv_cach
"]` to co
tro
 
h
ch r
sourc
s ar
 r
stor
d, us
fu
 for RLHF a
d 


ght updat
s. **Not
** that `
s_s

p

g` 


 r
port `tru
` u
t

 a
 compo


ts ar
 a
ak
.
```pytho

# Put 

g


 to d
p s

p (

v

=2)

m.s

p(

v

=2)
# ... G
t th
 


 


ghts
# Wak
 up o

y 


ghts to avo
d OOM

m.
ak
_up(tags=["


ghts"])
# ... Updat
 th
 


ghts
# 
ak
 up KV cach
 aft
r 


ghts ar
 updat
d

m.
ak
_up(tags=["kv_cach
"])
```
### O




 S
rv

g
To 

ab

 s

p mod
 

 a vLLM s
rv
r you 

d to 


t
a

z
 
t 

th th
 f
ag `VLLM_SERVER_DEV_MODE=1` a
d pass `--

ab

-s

p-mod
` to th
 vLLM s
rv
r.
#### S
rv
r 

 d
v

opm

t mod

Wh

 us

g th
 f
ag `VLLM_SERVER_DEV_MODE=1` you 

ab

 d
v

opm

t 

dpo

ts, a
d th
s
 

dpo

ts shou
d 
ot b
 
xpos
d to us
rs.
```bash
VLLM_SERVER_DEV_MODE=1 v
m s
rv
 Q


/Q


3-0.6B \
  --

ab

-s

p-mod
 \
  --port 8000
```
B

o
 
s a
 
xamp

 of ho
 to s

p a
d 
ak
 up a mod

 

 

v

 1.
```bash
cur
 -X POST 'http://
oca
host:8000/s

p?

v

=1'
cur
 -X POST 'http://
oca
host:8000/
ak
_up'
```
A
d th
s 
s a
 
xamp

 of ho
 to s

p a
d 
ak
 up a mod

 

 

v

 2.
```bash
cur
 -X POST 'http://
oca
host:8000/s

p?

v

=2'
# R
a
ocat
 


ghts m
mory o

y
cur
 -X POST 'http://
oca
host:8000/
ak
_up?tags=


ghts'
# Load 


ghts 

-p
ac

cur
 -X POST 'http://
oca
host:8000/co

ct
v
_rpc' -H 'Co
t

t-Typ
: app

cat
o
/jso
' -d '{"m
thod":"r

oad_


ghts"}'
# R
a
ocat
 KV cach

cur
 -X POST 'http://
oca
host:8000/
ak
_up?tags=kv_cach
'
```
#### HTTP 

dpo

ts
- `POST /s

p?

v

=1` — Put th
 mod

 to s

p (`

v

=1`).
- `POST /
ak
_up` — Wak
 up th
 mod

. Supports opt
o
a
 `tags` qu
ry param
t
rs for part
a
 
ak
-up (
.g., `?tags=


ghts`).
- `POST /co

ct
v
_rpc` — P
rform a co

ct
v
 r
mot
 proc
dur
 ca
 (RPC).
- `GET /
s_s

p

g` — Ch
ck 
f th
 mod

 
s s

p

g.
!!! 
ot

    Th
s
 

dpo

ts ar
 o

y ava

ab

 
h

 pass

g `VLLM_SERVER_DEV_MODE=1`.
## L
m
tat
o

O
 ROCm, th
 v
rtua
 m
mory a
ocat
o
 o
 ROCm 
s do

 through chu
k
d m
mory a
ocat
o
. You ca
 co
tro
 th
 chu
k s
z
 through `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` (

 MB). Th
 d
fau
t va
u
 
s s
t at 256MB. Th
 
arg
r th
 chu
k s
z
 th
 fast
r th
 p
rforma
c
. Ho

v
r, s
tt

g 
t too 
arg
 


 caus
 OOM. So 
f you 

cou
t
r OOM 
h

 us

g s

p mod
. Try r
duc

g th
 chu
k s
z
. It 
s r
comm

d
d to d
f


 th
 chu
k s
z
 as a po

r of 2.
