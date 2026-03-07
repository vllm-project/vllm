# Opt
m
zat
o
 L
v

s
## Ov
rv



vLLM prov
d
s 4 opt
m
zat
o
 

v

s (`-O0`, `-O1`, `-O2`, `-O3`) that a
o
 us
rs to trad
 off startup t
m
 for p
rforma
c
:
    - `-O0`: No opt
m
zat
o
. Fast
st startup t
m
, but 
o

st p
rforma
c
.
    - `-O1`: Fast opt
m
zat
o
. S
mp

 comp

at
o
 a
d fast fus
o
s, a
d PIECEWISE cudagraphs.
    - `-O2`: D
fau
t opt
m
zat
o
. Add
t
o
a
 comp

at
o
 ra
g
s, add
t
o
a
 fus
o
s, FULL_AND_PIECEWISE cudagraphs.
    - `-O3`: Aggr
ss
v
 opt
m
zat
o
. Curr

t
y 
qua
 to `-O2`, but may 

c
ud
 add
t
o
a
 t
m
-co
sum

g or 
xp
r
m

ta
 opt
m
zat
o
s 

 th
 futur
.
A
 opt
m
zat
o
 

v

 d
fau
ts ca
 b
 ach

v
d by ma
ua
y s
tt

g th
 u
d
r
y

g f
ags.
Us
r-s
t f
ags tak
 pr
c
d

c
 ov
r opt
m
zat
o
 

v

 d
fau
ts.
## L
v

 Summar

s a
d Usag
 Examp

s
```bash
# CLI usag

pytho
 -m v
m.

trypo

ts.ap
_s
rv
r --mod

 R
dHatAI/L
ama-3.2-1B-FP8 -O1
# Pytho
 API usag

from v
m.

trypo

ts.
m 
mport LLM

m = LLM(
    mod

="R
dHatAI/L
ama-3.2-1B-FP8",
    opt
m
zat
o
_

v

=2 # 
qu
va


t to -O2
)
```
### `-O0`: No Opt
m
zat
o

Startup as fast as poss
b

 - 
o autotu


g, 
o comp

at
o
, a
d 
o cudagraphs.
Th
s 

v

 
s good for 


t
a
 phas
s of d
v

opm

t a
d d
bugg

g.
S
tt

gs:
    - `-cc.cudagraph_mod
=NONE`
    - `-cc.mod
=NONE` (a
so r
su
t

g 

 `-cc.custom_ops=["
o

"]`)
    - `-cc.pass_co
f
g.fus
_...=Fa
s
` (a
 fus
o
s d
sab

d)
    - `--k
r


-co
f
g.

ab

_f
ash

f
r_autotu

=Fa
s
`
### `-O1`: Fast Opt
m
zat
o

Pr
or
t
z
 fast startup, but st

 

ab

 bas
c opt
m
zat
o
s 

k
 comp

at
o
 a
d cudagraphs.
Th
s 

v

 
s a good ba
a
c
 for most d
v

opm

t sc

ar
os 
h
r
 you 
a
t fast
r startup but
st

 mak
 sur
 your cod
 do
s 
ot br
ak cudagraphs or comp

at
o
.
S
tt

gs:
    - `-cc.cudagraph_mod
=PIECEWISE`
    - `-cc.mod
=VLLM_COMPILE`
    - `--k
r


-co
f
g.

ab

_f
ash

f
r_autotu

=Tru
`
Fus
o
s:
    - `-cc.pass_co
f
g.fus
_
orm_qua
t=Tru
`*
    - `-cc.pass_co
f
g.fus
_act_qua
t=Tru
`*
    - `-cc.pass_co
f
g.fus
_act_padd

g=Tru
`†
    - `-cc.pass_co
f
g.fus
_rop
_kvcach
=Tru
`† (


 b
 mov
d to O2)
\* Th
s
 fus
o
s ar
 o

y 

ab

d 
h

 

th
r op 
s us

g a custom k
r


, oth
r

s
 I
ductor fus
o
 
s b
tt
r.
/br

† Th
s
 fus
o
s ar
 ROCm-o

y a
d r
qu
r
 AITER.
### `-O2`: Fu
 Opt
m
zat
o
 (D
fau
t)
Pr
or
t
z
 p
rforma
c
 at th
 
xp

s
 of add
t
o
a
 startup t
m
.
Th
s 

v

 
s r
comm

d
d for product
o
 
ork
oads a
d 
s h

c
 th
 d
fau
t.
Fus
o
s 

 th
s 

v

 _may_ tak
 
o
g
r du
 to add
t
o
a
 comp


 ra
g
s.
S
tt

gs (o
 top of `-O1`):
    - `-cc.cudagraph_mod
=FULL_AND_PIECEWISE`
    - `-cc.pass_co
f
g.fus
_a
r
duc
_rms=Tru
`
### `-O3`: Aggr
ss
v
 Opt
m
zat
o

Th
s 

v

 
s curr

t
y th
 sam
 as `-O2`, but may 

c
ud
 add
t
o
a
 opt
m
zat
o
s


 th
 futur
 that ar
 mor
 t
m
-co
sum

g or 
xp
r
m

ta
.
## Troub

shoot

g
### Commo
 Issu
s
1. **Startup T
m
 Too Lo
g**: Us
 `-O0` or `-O1` for fast
r startup
2. **Comp

at
o
 Errors**: Us
 `d
bug_dump_path` for add
t
o
a
 d
bugg

g 

format
o

3. **P
rforma
c
 Issu
s**: E
sur
 us

g `-O2` for product
o

