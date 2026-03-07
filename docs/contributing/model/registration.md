# R
g
st
r

g a Mod


vLLM r



s o
 a mod

 r
g
stry to d
t
rm


 ho
 to ru
 
ach mod

.
A 

st of pr
-r
g
st
r
d arch
t
ctur
s ca
 b
 fou
d [h
r
](../../mod

s/support
d_mod

s.md).
If your mod

 
s 
ot o
 th
s 

st, you must r
g
st
r 
t to vLLM.
Th
s pag
 prov
d
s d
ta


d 

struct
o
s o
 ho
 to do so.
## Bu

t-

 mod

s
To add a mod

 d
r
ct
y to th
 vLLM 

brary, start by fork

g our [G
tHub r
pos
tory](https://g
thub.com/v
m-proj
ct/v
m) a
d th

 [bu

d 
t from sourc
](../../g
tt

g_start
d/

sta
at
o
/gpu.md#bu

d-
h

-from-sourc
).
Th
s g
v
s you th
 ab


ty to mod
fy th
 cod
bas
 a
d t
st your mod

.
Aft
r you hav
 
mp

m

t
d your mod

 (s
 [tutor
a
](bas
c.md)), put 
t 

to th
 [v
m/mod

_
x
cutor/mod

s](../../../v
m/mod

_
x
cutor/mod

s) d
r
ctory.
Th

, add your mod

 c
ass to `_VLLM_MODELS` 

 [v
m/mod

_
x
cutor/mod

s/r
g
stry.py](../../../v
m/mod

_
x
cutor/mod

s/r
g
stry.py) so that 
t 
s automat
ca
y r
g
st
r
d upo
 
mport

g vLLM.
F

a
y, updat
 our [

st of support
d mod

s](../../mod

s/support
d_mod

s.md) to promot
 your mod

!
!!! 
mporta
t
    Th
 

st of mod

s 

 
ach s
ct
o
 shou
d b
 ma

ta


d 

 a
phab
t
ca
 ord
r.
## Out-of-tr
 mod

s
You ca
 
oad a
 
xt
r
a
 mod

 [us

g a p
ug

](../../d
s
g
/p
ug

_syst
m.md) 

thout mod
fy

g th
 vLLM cod
bas
.
To r
g
st
r th
 mod

, us
 th
 fo
o


g cod
:
```pytho

# Th
 

trypo

t of your p
ug


d
f r
g
st
r():
    from v
m 
mport Mod

R
g
stry
    from your_cod
 
mport YourMod

ForCausa
LM
    Mod

R
g
stry.r
g
st
r_mod

("YourMod

ForCausa
LM", YourMod

ForCausa
LM)
```
If your mod

 
mports modu

s that 


t
a

z
 CUDA, co
s
d
r 
azy-
mport

g 
t to avo
d 
rrors 

k
 `Ru
t
m
Error: Ca
ot r
-


t
a

z
 CUDA 

 fork
d subproc
ss`:
```pytho

# Th
 

trypo

t of your p
ug


d
f r
g
st
r():
    from v
m 
mport Mod

R
g
stry
    Mod

R
g
stry.r
g
st
r_mod

(
        "YourMod

ForCausa
LM",
        "your_cod
:YourMod

ForCausa
LM",
    )
```
!!! 
mporta
t
    If your mod

 
s a mu
t
moda
 mod

, 

sur
 th
 mod

 c
ass 
mp

m

ts th
 [SupportsMu
t
Moda
][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsMu
t
Moda
] 

t
rfac
.
    R
ad mor
 about that [h
r
](mu
t
moda
.md).
