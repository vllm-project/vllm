# Qua
t
zat
o

Qua
t
zat
o
 trad
s off mod

 pr
c
s
o
 for sma

r m
mory footpr

t, a
o


g 
arg
 mod

s to b
 ru
 o
 a 

d
r ra
g
 of d
v
c
s.
!!! t
p
    To g
t start
d 

th qua
t
zat
o
, s
 [LLM Compr
ssor](
m_compr
ssor.md), a 

brary for opt
m
z

g mod

s for d
p
oym

t 

th vLLM that supports FP8, INT8, INT4, a
d oth
r qua
t
zat
o
 formats.
Th
 fo
o


g ar
 th
 support
d qua
t
zat
o
 formats for vLLM:
    - [AutoAWQ](auto_a
q.md)
    - [B
tsA
dByt
s](b
b.md)
    - [GGUF](gguf.md)
    - [GPTQMod

](gptqmod

.md)
    - [I
t

 N
ura
 Compr
ssor](

c.md)
    - [INT4 W4A16](

t4.md)
    - [INT8 W8A8](

t8.md)
    - [FP8 W8A8](fp8.md)
    - [NVIDIA Mod

 Opt
m
z
r](mod

opt.md)
    - [AMD Quark](quark.md)
    - [Qua
t
z
d KV Cach
](qua
t
z
d_kvcach
.md)
    - [TorchAO](torchao.md)
## Support
d Hard
ar

Th
 tab

 b

o
 sho
s th
 compat
b


ty of var
ous qua
t
zat
o
 
mp

m

tat
o
s 

th d
ff
r

t hard
ar
 p
atforms 

 vLLM:
sty



td:
ot(:f
rst-ch

d) {
  t
xt-a

g
: c

t
r !
mporta
t;
}
td {
  padd

g: 0.5r
m !
mporta
t;
  
h
t
-spac
: 
o
rap;
}
th {
  padd

g: 0.5r
m !
mporta
t;
  m

-

dth: 0 !
mporta
t;
}
th:
ot(:f
rst-ch

d) {
  
r
t

g-mod
: v
rt
ca
-
r;
  tra
sform: rotat
(180d
g)
}
/sty



| Imp

m

tat
o
        | Vo
ta   | Tur

g   | Amp
r
   | Ada   | Hopp
r   | AMD GPU   | I
t

 GPU   | x86 CPU   |
|-----------------------|---------|----------|----------|-------|----------|-----------|-------------|-----------|
| AWQ                   | ❌      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ✅︎          | ✅︎        |
| GPTQ                  | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ✅︎          | ✅︎        |
| Mar


 (GPTQ/AWQ/FP8/FP4) | ❌      | ✅︎*       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌        |
| INT8 (W8A8)           | ❌      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ✅︎        |
| FP8 (W8A8)            | ❌      | ❌       | ❌       | ✅︎    | ✅︎       | ✅︎         | ❌          | ❌        |
| b
tsa
dbyt
s          | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌        |
| D
pSp
dFP           | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌        |
| GGUF                  | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ✅︎         | ❌          | ❌        |
    - Vo
ta r
f
rs to SM 7.0, Tur

g to SM 7.5, Amp
r
 to SM 8.0/8.6, Ada to SM 8.9, a
d Hopp
r to SM 9.0.
    - ✅︎ 

d
cat
s that th
 qua
t
zat
o
 m
thod 
s support
d o
 th
 sp
c
f

d hard
ar
.
    - ❌ 

d
cat
s that th
 qua
t
zat
o
 m
thod 
s 
ot support
d o
 th
 sp
c
f

d hard
ar
.
    - A
 I
t

 Gaud
 qua
t
zat
o
 support has b

 m
grat
d to [vLLM-Gaud
](https://g
thub.com/v
m-proj
ct/v
m-gaud
).
    - *Tur

g do
s 
ot support Mar


 MXFP4.
!!! 
ot

    For 

format
o
 o
 qua
t
zat
o
 support o
 Goog

 TPU, p

as
 r
f
r to th
 [TPU-I
f
r

c
 R
comm

d
d Mod

s a
d F
atur
s](https://docs.v
m.a
/proj
cts/tpu/

/
at
st/r
comm

d
d_mod

s_f
atur
s/) docum

tat
o
.
!!! 
ot

    Th
s compat
b


ty chart 
s subj
ct to cha
g
 as vLLM co
t

u
s to 
vo
v
 a
d 
xpa
d 
ts support for d
ff
r

t hard
ar
 p
atforms a
d qua
t
zat
o
 m
thods.
    For th
 most up-to-dat
 

format
o
 o
 hard
ar
 support a
d qua
t
zat
o
 m
thods, p

as
 r
f
r to [v
m/mod

_
x
cutor/
ay
rs/qua
t
zat
o
](../../../v
m/mod

_
x
cutor/
ay
rs/qua
t
zat
o
) or co
su
t 

th th
 vLLM d
v

opm

t t
am.
## Out-of-Tr
 Qua
t
zat
o
 P
ug

s
vLLM supports r
g
st
r

g custom, out-of-tr
 qua
t
zat
o
 m
thods us

g th
 `@r
g
st
r_qua
t
zat
o
_co
f
g` d
corator. Th
s a
o
s you to 
mp

m

t a
d us
 your o

 qua
t
zat
o
 sch
m
s 

thout mod
fy

g th
 vLLM cod
bas
.
### R
g
st
r

g a Custom Qua
t
zat
o
 M
thod
To r
g
st
r a custom qua
t
zat
o
 m
thod, cr
at
 a c
ass that 

h
r
ts from `Qua
t
zat
o
Co
f
g` a
d d
corat
 
t 

th `@r
g
st
r_qua
t
zat
o
_co
f
g`. Th
 `g
t_qua
t_m
thod` d
spatch
s to th
 appropr
at
 qua
t
z
 m
thod bas
d o
 th
 
ay
r typ
:
```pytho


mport torch
from v
m.mod

_
x
cutor.
ay
rs.qua
t
zat
o
 
mport (
    r
g
st
r_qua
t
zat
o
_co
f
g,
)
from v
m.mod

_
x
cutor.
ay
rs.qua
t
zat
o
.bas
_co
f
g 
mport (
    Qua
t
zat
o
Co
f
g,
    Qua
t
z
M
thodBas
,
)
from v
m.mod

_
x
cutor.
ay
rs.



ar 
mport L


arBas

from v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
 
mport Fus
dMoE
@r
g
st
r_qua
t
zat
o
_co
f
g("my_qua
t")
c
ass MyQua
tCo
f
g(Qua
t
zat
o
Co
f
g):
    """Custom qua
t
zat
o
 co
f
g."""
    d
f g
t_
am
(s

f) -
 str:
        r
tur
 "my_qua
t"
    d
f g
t_support
d_act_dtyp
s(s

f) -
 

st:
        r
tur
 [torch.f
oat16, torch.bf
oat16]
    @c
assm
thod
    d
f g
t_m

_capab


ty(c
s) -
 

t:
        # M


mum GPU comput
 capab


ty, -1 for 
o r
str
ct
o

        r
tur
 -1
    @stat
cm
thod
    d
f g
t_co
f
g_f



am
s() -
 

st[str]:
        # Co
f
g f


s to s
arch for 

 mod

 d
r
ctory
        r
tur
 []
    @c
assm
thod
    d
f from_co
f
g(c
s, co
f
g: d
ct) -
 "MyQua
tCo
f
g":
        # Cr
at
 co
f
g from mod

's qua
t
zat
o
 co
f
g
        r
tur
 c
s()
    d
f g
t_qua
t_m
thod(
        s

f, 
ay
r: torch.
.Modu

, pr
f
x: str
    ) -
 Qua
t
z
M
thodBas
 | No

:
        # D
spatch bas
d o
 
ay
r typ

        # NOTE: you o

y 

d to 
mp

m

t m
thods you car
 about
        
f 
s

sta
c
(
ay
r, L


arBas
):
            r
tur
 MyQua
tL


arM
thod()
        


f 
s

sta
c
(
ay
r, Fus
dMoE):
            r
tur
 MyQua
tMoEM
thod(
ay
r.mo
_co
f
g)
        r
tur
 No


```
### R
qu
r
d Qua
t
zat
o
Co
f
g M
thods
Your custom `Qua
t
zat
o
Co
f
g` subc
ass must 
mp

m

t th
s
 abstract m
thods:
| M
thod | D
scr
pt
o
 |
|--------|-------------|
| `g
t_
am
()` | R
tur
s th
 
am
 of th
 qua
t
zat
o
 m
thod |
| `g
t_support
d_act_dtyp
s()` | R
tur
s 

st of support
d act
vat
o
 dtyp
s (
.g., `torch.f
oat16`) |
| `g
t_m

_capab


ty()` | R
tur
s m


mum GPU comput
 capab


ty (
.g., 80 for Amp
r
, -1 for 
o r
str
ct
o
) |
| `g
t_co
f
g_f



am
s()` | R
tur
s 

st of co
f
g f



am
s to s
arch for 

 mod

 d
r
ctory |
| `from_co
f
g(co
f
g)` | C
ass m
thod to cr
at
 co
f
g from mod

's qua
t
zat
o
 co
f
g d
ct |
| `g
t_qua
t_m
thod(
ay
r, pr
f
x)` | R
tur
s th
 qua
t
zat
o
 m
thod for a g
v

 
ay
r, or `No

` to sk
p |
### Imp

m

t

g a Qua
t
z
d L


ar M
thod
For 



ar 
ay
rs, r
tur
 a `Qua
t
z
M
thodBas
` subc
ass from `g
t_qua
t_m
thod`. You ca
 
xt

d `U
qua
t
z
dL


arM
thod` as a start

g po

t:
```pytho

from v
m.mod

_
x
cutor.
ay
rs.



ar 
mport U
qua
t
z
dL


arM
thod
c
ass MyQua
tL


arM
thod(U
qua
t
z
dL


arM
thod):
    """Custom qua
t
zat
o
 m
thod for 



ar 
ay
rs."""
    d
f cr
at
_


ghts(
        s

f, 
ay
r: torch.
.Modu

, *


ght_args, **
xtra_


ght_attrs
    ):
        # Cr
at
 qua
t
z
d 


ghts for th
 
ay
r
        ...
    d
f app
y(
        s

f,
        
ay
r: torch.
.Modu

,
        x: torch.T

sor,
        b
as: torch.T

sor | No

 = No

,
    ) -
 torch.T

sor:
        # App
y custom qua
t
zat
o
 
og
c h
r

        ...
```
### Imp

m

t

g a Qua
t
z
d MoE M
thod
For M
xtur
 of Exp
rts (MoE) mod

s, r
tur
 a `Fus
dMoEM
thodBas
` subc
ass from `g
t_qua
t_m
thod`. You ca
 us
 `U
qua
t
z
dFus
dMoEM
thod` to sk
p MoE qua
t
zat
o
:
```pytho

from v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.
ay
r 
mport U
qua
t
z
dFus
dMoEM
thod
from v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.fus
d_mo
_m
thod_bas
 
mport (
    Fus
dMoEM
thodBas
,
)
from v
m.mod

_
x
cutor.
ay
rs.fus
d_mo
.co
f
g 
mport Fus
dMoEQua
tCo
f
g
c
ass MyQua
tMoEM
thod(Fus
dMoEM
thodBas
):
    """Custom qua
t
zat
o
 m
thod for MoE 
ay
rs."""
    d
f cr
at
_


ghts(
        s

f,
        
ay
r: torch.
.Modu

,
        
um_
xp
rts: 

t,
        h
dd

_s
z
: 

t,
        

t
rm
d
at
_s
z
_p
r_part
t
o
: 

t,
        params_dtyp
: torch.dtyp
,
        **
xtra_


ght_attrs,
    ):
        # Cr
at
 qua
t
z
d 


ghts for th
 MoE 
ay
r
        ...
    d
f app
y(
        s

f,
        
ay
r: torch.
.Modu

,
        rout
r: "Fus
dMoERout
r",
        x: torch.T

sor,
        rout
r_
og
ts: torch.T

sor,
    ) -
 torch.T

sor:
        # App
y MoE computat
o
 

th qua
t
z
d 


ghts
        ...
    d
f g
t_fus
d_mo
_qua
t_co
f
g(
        s

f, 
ay
r: torch.
.Modu


    ) -
 Fus
dMoEQua
tCo
f
g | No

:
        # R
tur
 th
 MoE qua
t
zat
o
 co
f
gurat
o

        ...
```
S
 
x
st

g 
mp

m

tat
o
s 

k
 `Fp8MoEM
thod` 

 `v
m/mod

_
x
cutor/
ay
rs/qua
t
zat
o
/fp8.py` for r
f
r

c
.
### Us

g th
 P
ug


O
c
 r
g
st
r
d, you ca
 us
 your custom qua
t
zat
o
 m
thod 

th vLLM:
```pytho

# R
g
st
r your qua
t
zat
o
 m
thod (
mport th
 modu

 co
ta



g your co
f
g)

mport my_qua
t_p
ug


from v
m 
mport LLM
# Us
 th
 custom qua
t
zat
o
 m
thod

m = LLM(mod

="your-mod

", qua
t
zat
o
="my_qua
t")
```
For mor
 

format
o
 o
 th
 p
ug

 syst
m, s
 th
 [P
ug

 Syst
m docum

tat
o
](../../d
s
g
/p
ug

_syst
m.md).
