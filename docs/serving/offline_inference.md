# Off



 I
f
r

c

Off



 

f
r

c
 
s poss
b

 

 your o

 cod
 us

g vLLM's [`LLM`][v
m.LLM] c
ass.
For 
xamp

, th
 fo
o


g cod
 do


oads th
 [`fac
book/opt-125m`](https://hugg

gfac
.co/fac
book/opt-125m) mod

 from Hugg

gFac

a
d ru
s 
t 

 vLLM us

g th
 d
fau
t co
f
gurat
o
.
```pytho

from v
m 
mport LLM
# I

t
a

z
 th
 vLLM 

g


.

m = LLM(mod

="fac
book/opt-125m")
```
Aft
r 


t
a

z

g th
 `LLM` 

sta
c
, us
 th
 ava

ab

 APIs to p
rform mod

 

f
r

c
.
Th
 ava

ab

 APIs d
p

d o
 th
 mod

 typ
:
    - [G


rat
v
 mod

s](../mod

s/g


rat
v
_mod

s.md) output 
ogprobs 
h
ch ar
 samp

d from to obta

 th
 f

a
 output t
xt.
    - [Poo


g mod

s](../mod

s/poo


g_mod

s.md) output th

r h
dd

 stat
s d
r
ct
y.
!!! 

fo
    [API R
f
r

c
](../ap
/README.md#off



-

f
r

c
)
## Ray Data LLM API
Ray Data LLM 
s a
 a
t
r
at
v
 off



 

f
r

c
 API that us
s vLLM as th
 u
d
r
y

g 

g


.
Th
s API adds s
v
ra
 batt
r

s-

c
ud
d capab


t

s that s
mp

fy 
arg
-sca

, GPU-
ff
c


t 

f
r

c
:
    - Str
am

g 
x
cut
o
 proc
ss
s datas
ts that 
xc
d aggr
gat
 c
ust
r m
mory.
    - Automat
c shard

g, 
oad ba
a
c

g, a
d autosca


g d
str
but
 
ork across a Ray c
ust
r 

th bu

t-

 fau
t to

ra
c
.
    - Co
t

uous batch

g k
ps vLLM r
p

cas saturat
d a
d max
m
z
s GPU ut


zat
o
.
    - Tra
spar

t support for t

sor a
d p
p




 para



sm 

ab

s 
ff
c


t mu
t
-GPU 

f
r

c
.
    - R
ad

g a
d 
r
t

g to most popu
ar f


 formats a
d c
oud obj
ct storag
.
    - Sca


g up th
 
ork
oad 

thout cod
 cha
g
s.
??? cod

    ```pytho

    
mport ray  # R
qu
r
s ray
=2.44.1
    from ray.data.
m 
mport vLLME
g


Proc
ssorCo
f
g, bu

d_
m_proc
ssor
    co
f
g = vLLME
g


Proc
ssorCo
f
g(mod

_sourc
="u
s
oth/L
ama-3.2-1B-I
struct")
    proc
ssor = bu

d_
m_proc
ssor(
        co
f
g,
        pr
proc
ss=
ambda ro
: {
            "m
ssag
s": [
                {"ro

": "syst
m", "co
t

t": "You ar
 a bot that comp

t
s u
f


sh
d ha
kus."},
                {"ro

": "us
r", "co
t

t": ro
["
t
m"]},
            ],
            "samp


g_params": {"t
mp
ratur
": 0.3, "max_tok

s": 250},
        },
        postproc
ss=
ambda ro
: {"a
s

r": ro
["g


rat
d_t
xt"]},
    )
    ds = ray.data.from_
t
ms(["A
 o
d s



t po
d..."])
    ds = proc
ssor(ds)
    ds.
r
t
_parqu
t("
oca
:///tmp/data/")
```
For mor
 

format
o
 about th
 Ray Data LLM API, s
 th
 [Ray Data LLM docum

tat
o
](https://docs.ray.
o/

/
at
st/data/
ork

g-

th-
ms.htm
).
