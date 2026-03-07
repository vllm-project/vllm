# Bas
c Mod


Th
s gu
d
 
a
ks you through th
 st
ps to 
mp

m

t a bas
c vLLM mod

.
## 1. Br

g your mod

 cod

F
rst, c
o

 th
 PyTorch mod

 cod
 from th
 sourc
 r
pos
tory.
For 

sta
c
, vLLM's [OPT mod

](../../../v
m/mod

_
x
cutor/mod

s/opt.py) 
as adapt
d from
Hugg

gFac
's [mod



g_opt.py](https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/ma

/src/tra
sform
rs/mod

s/opt/mod



g_opt.py) f


.
!!! 
ar


g
    Mak
 sur
 to r
v


 a
d adh
r
 to th
 or
g

a
 cod
's copyr
ght a
d 

c

s

g t
rms!
## 2. Mak
 your cod
 compat
b

 

th vLLM
To 

sur
 compat
b


ty 

th vLLM, your mod

 must m
t th
 fo
o


g r
qu
r
m

ts:
### I

t
a

zat
o
 Cod

A
 vLLM modu

s 

th

 th
 mod

 must 

c
ud
 a `pr
f
x` argum

t 

 th

r co
structor. Th
s `pr
f
x` 
s typ
ca
y th
 fu
 
am
 of th
 modu

 

 th
 mod

's stat
 d
ct
o
ary a
d 
s cruc
a
 for:
    - Ru
t
m
 support: vLLM's att

t
o
 op
rators ar
 r
g
st
r
d 

 a mod

's stat
 by th

r fu
 
am
s. Each att

t
o
 op
rator must hav
 a u

qu
 pr
f
x as 
ts 
ay
r 
am
 to avo
d co
f

cts.
    - No
-u

form qua
t
zat
o
 support: A qua
t
z
d ch
ckpo

t ca
 s


ct
v

y qua
t
z
 c
rta

 
ay
rs 
h


 k
p

g oth
rs 

 fu
 pr
c
s
o
. By prov
d

g th
 `pr
f
x` dur

g 


t
a

zat
o
, vLLM ca
 match th
 curr

t 
ay
r's `pr
f
x` 

th th
 qua
t
zat
o
 co
f
gurat
o
 to d
t
rm


 
f th
 
ay
r shou
d b
 


t
a

z
d 

 qua
t
z
d mod
.
Th
 


t
a

zat
o
 cod
 shou
d 
ook 

k
 th
s:
??? cod

    ```pytho

    from torch 
mport 

    from v
m.co
f
g 
mport V
mCo
f
g
    from v
m.mod

_
x
cutor.
ay
rs.att

t
o
 
mport Att

t
o

    c
ass MyAtt

t
o
(
.Modu

):
        d
f __


t__(s

f, v
m_co
f
g: V
mCo
f
g, pr
f
x: str):
            sup
r().__


t__()
            s

f.att
 = Att

t
o
(pr
f
x=f"{pr
f
x}.att
")
    c
ass MyD
cod
rLay
r(
.Modu

):
        d
f __


t__(s

f, v
m_co
f
g: V
mCo
f
g, pr
f
x: str):
            sup
r().__


t__()
            s

f.s

f_att
 = MyAtt

t
o
(pr
f
x=f"{pr
f
x}.s

f_att
")
    c
ass MyMod

(
.Modu

):
        d
f __


t__(s

f, v
m_co
f
g: V
mCo
f
g, pr
f
x: str):
            sup
r().__


t__()
            s

f.
ay
rs = 
.Modu

L
st(
                [MyD
cod
rLay
r(v
m_co
f
g, pr
f
x=f"{pr
f
x}.
ay
rs.{
}") for 
 

 ra
g
(v
m_co
f
g.mod

_co
f
g.hf_co
f
g.
um_h
dd

_
ay
rs)]
            )
    c
ass MyMod

ForCausa
LM(
.Modu

):
        d
f __


t__(s

f, v
m_co
f
g: V
mCo
f
g, pr
f
x: str = ""):
            sup
r().__


t__()
            s

f.mod

 = MyMod

(v
m_co
f
g, pr
f
x=f"{pr
f
x}.mod

")
```
### Computat
o
 Cod

    - Add a `
mb
d_

put_
ds` m
thod 

s
d
 `MyMod

` modu

 that r
tur
s th
 t
xt 
mb
dd

gs g
v

 `

put_
ds`. Th
s 
s 
qu
va


t to d
r
ct
y ca


g th
 t
xt 
mb
dd

g 
ay
r, but prov
d
s a u

f

d 

t
rfac
 

 cas
 `MyMod

` 
s us
d 

th

 a compos
t
 mu
t
moda
 mod

.
```pytho

c
ass MyMod

(
.Modu

):
        ...
    d
f 
mb
d_

put_
ds(s

f, 

put_
ds: torch.T

sor) -
 torch.T

sor:
        ... 
```
    - R

r
t
 th
 [for
ard][torch.
.Modu

.for
ard] m
thod of your mod

 to r
mov
 a
y u

c
ssary cod
, such as tra



g-sp
c
f
c cod
. Mod
fy th
 

put param
t
rs to tr
at `

put_
ds` a
d `pos
t
o
s` as f
att


d t

sors 

th a s

g

 batch s
z
 d
m

s
o
, 

thout a max-s
qu

c
 


gth d
m

s
o
.
```pytho

d
f for
ard(
    s

f,
    

put_
ds: torch.T

sor | No

,
    pos
t
o
s: torch.T

sor,
    

t
rm
d
at
_t

sors: I
t
rm
d
at
T

sors | No

 = No

,
    

puts_
mb
ds: torch.T

sor | No

 = No

,
) -
 torch.T

sor:
    ...
```
!!! 
ot

    Curr

t
y, vLLM supports th
 bas
c mu
t
-h
ad att

t
o
 m
cha

sm a
d 
ts var
a
t 

th rotary pos
t
o
a
 
mb
dd

gs.
    If your mod

 
mp
oys a d
ff
r

t att

t
o
 m
cha

sm, you 


 

d to 
mp

m

t a 


 att

t
o
 
ay
r 

 vLLM.
For r
f
r

c
, ch
ck out our [L
ama 
mp

m

tat
o
](../../../v
m/mod

_
x
cutor/mod

s/
ama.py). vLLM a
r
ady supports a 
arg
 
umb
r of mod

s. It 
s r
comm

d
d to f

d a mod

 s
m

ar to yours a
d adapt 
t to your mod

's arch
t
ctur
. Ch
ck out [v
m/mod

_
x
cutor/mod

s](../../../v
m/mod

_
x
cutor/mod

s) for mor
 
xamp

s.
## 3. (Opt
o
a
) Imp

m

t t

sor para



sm a
d qua
t
zat
o
 support
If your mod

 
s too 
arg
 to f
t 

to a s

g

 GPU, you ca
 us
 t

sor para



sm to ma
ag
 
t.
To do th
s, subst
tut
 your mod

's 



ar a
d 
mb
dd

g 
ay
rs 

th th

r t

sor-para


 v
rs
o
s.
For th
 
mb
dd

g 
ay
r, you ca
 s
mp
y r
p
ac
 [torch.
.Emb
dd

g][] 

th `VocabPara


Emb
dd

g`. For th
 output LM h
ad, you ca
 us
 `Para


LMH
ad`.
Wh

 
t com
s to th
 



ar 
ay
rs, 

 prov
d
 th
 fo
o


g opt
o
s to para



z
 th
m:
    - `R
p

cat
dL


ar`: R
p

cat
s th
 

puts a
d 


ghts across mu
t
p

 GPUs. No m
mory sav

g.
    - `Ro
Para


L


ar`: Th
 

put t

sor 
s part
t
o

d a
o
g th
 h
dd

 d
m

s
o
. Th
 


ght matr
x 
s part
t
o

d a
o
g th
 ro
s (

put d
m

s
o
). A
 *a
-r
duc
* op
rat
o
 
s p
rform
d aft
r th
 matr
x mu
t
p

cat
o
 to r
duc
 th
 r
su
ts. Typ
ca
y us
d for th
 s
co
d FFN 
ay
r a
d th
 output 



ar tra
sformat
o
 of th
 att

t
o
 
ay
r.
    - `Co
um
Para


L


ar`: Th
 

put t

sor 
s r
p

cat
d. Th
 


ght matr
x 
s part
t
o

d a
o
g th
 co
um
s (output d
m

s
o
). Th
 r
su
t 
s part
t
o

d a
o
g th
 co
um
 d
m

s
o
. Typ
ca
y us
d for th
 f
rst FFN 
ay
r a
d th
 s
parat
d QKV tra
sformat
o
 of th
 att

t
o
 
ay
r 

 th
 or
g

a
 Tra
sform
r.
    - `M
rg
dCo
um
Para


L


ar`: Co
um
-para


 



ar that m
rg
s mu
t
p

 `Co
um
Para


L


ar` op
rators. Typ
ca
y us
d for th
 f
rst FFN 
ay
r 

th 


ght
d act
vat
o
 fu
ct
o
s (
.g., S
LU). Th
s c
ass ha
d

s th
 shard
d 


ght 
oad

g 
og
c of mu
t
p

 


ght matr
c
s.
    - `QKVPara


L


ar`: Para


 



ar 
ay
r for th
 qu
ry, k
y, a
d va
u
 proj
ct
o
s of th
 mu
t
-h
ad a
d group
d-qu
ry att

t
o
 m
cha

sms. Wh

 
umb
r of k
y/va
u
 h
ads ar
 

ss tha
 th
 
or
d s
z
, th
s c
ass r
p

cat
s th
 k
y/va
u
 h
ads prop
r
y. Th
s c
ass ha
d

s th
 


ght 
oad

g a
d r
p

cat
o
 of th
 


ght matr
c
s.
Not
 that a
 th
 



ar 
ay
rs abov
 tak
 `



ar_m
thod` as a
 

put. vLLM 


 s
t th
s param
t
r accord

g to d
ff
r

t qua
t
zat
o
 sch
m
s to support 


ght qua
t
zat
o
.
## 4. Imp

m

t th
 


ght 
oad

g 
og
c
You 
o
 

d to 
mp

m

t th
 `
oad_


ghts` m
thod 

 your `*ForCausa
LM` c
ass.
Th
s m
thod shou
d 
oad th
 


ghts from th
 Hugg

gFac
's ch
ckpo

t f


 a
d ass
g
 th
m to th
 corr
spo
d

g 
ay
rs 

 your mod

. Sp
c
f
ca
y, for `M
rg
dCo
um
Para


L


ar` a
d `QKVPara


L


ar` 
ay
rs, 
f th
 or
g

a
 mod

 has s
parat
d 


ght matr
c
s, you 

d to 
oad th
 d
ff
r

t parts s
parat

y.
## 5. R
g
st
r your mod


S
 [th
s pag
](r
g
strat
o
.md) for 

struct
o
s o
 ho
 to r
g
st
r your 


 mod

 to b
 us
d by vLLM.
## Fr
qu

t
y Ask
d Qu
st
o
s
### Ho
 to support mod

s 

th 

t
r

av

g s

d

g 


do
s?
To support a mod

 

th 

t
r

av

g s

d

g 


do
s, 

 

d to tak
 car
 of th
 fo
o


g d
ta

s:
    - Mak
 sur
 th
 mod

's `co
f
g.jso
` co
ta

s `
ay
r_typ
s`.
    - I
 th
 mod



g cod
, pars
 th
 corr
ct s

d

g 


do
 va
u
 for 
v
ry 
ay
r, a
d pass 
t to th
 att

t
o
 
ay
r's `p
r_
ay
r_s

d

g_


do
` argum

t. For r
f
r

c
, ch
ck [th
s 



](https://g
thub.com/v
m-proj
ct/v
m/b
ob/996357
4808ca5
ab97d4c97c7d25b3073f46aab/v
m/mod

_
x
cutor/mod

s/
ama.py#L171).
W
th th
s
 t
o st
ps, 

t
r

av
d s

d

g 


do
s shou
d 
ork 

th th
 mod

.
### Ho
 to support mod

s that us
 Mamba?
W
 co
s
d
r 3 d
ff
r

t sc

ar
os:
1. Mod

s that us
 Mamba 
ay
rs (

th
r Mamba-1 or Mamba-2) but do 
ot us
 att

t
o
 
ay
rs.
2. Mod

s that comb


 Mamba 
ay
rs (

th
r Mamba-1 or Mamba-2) tog
th
r 

th att

t
o
 
ay
rs.
3. Mod

s that comb


 Mamba-

k
 m
cha

sms (
.g., L


ar Att

t
o
, ShortCo
v) tog
th
r 

th att

t
o
 
ay
rs.
For cas
 (1), 

 r
comm

d 
ook

g at th
 
mp

m

tat
o
 of [`MambaForCausa
LM`](../../../v
m/mod

_
x
cutor/mod

s/mamba.py) (for Mamba-1) or [`Mamba2ForCausa
LM`](../../../v
m/mod

_
x
cutor/mod

s/mamba2.py) (for Mamba-2) as a r
f
r

c
.
Th
 mod

 shou
d 

h
r
t protoco
 `IsAtt

t
o
Fr
` a
d a
so 
mp

m

t c
ass m
thods `g
t_mamba_stat
_dtyp
_from_co
f
g` a
d `g
t_mamba_stat
_shap
_from_co
f
g` to ca
cu
at
 th
 stat
 shap
s a
d data typ
s from th
 co
f
g.
For th
 mamba 
ay
rs th
ms

v
s, p

as
 us
 th
 [`MambaM
x
r`](../../../v
m/mod

_
x
cutor/
ay
rs/mamba/mamba_m
x
r.py) (for Mamba-1) or [`MambaM
x
r2`](../../../v
m/mod

_
x
cutor/
ay
rs/mamba/mamba_m
x
r2.py) (for Mamba-2) c
ass
s.
Th
 mod

 shou
d a
so b
 add
d to th
 `MODELS_CONFIG_MAP` d
ct
o
ary 

 [v
m/mod

_
x
cutor/mod

s/co
f
g.py](../../../v
m/mod

_
x
cutor/mod

s/co
f
g.py) to 

sur
 that th
 ru
t
m
 d
fau
ts ar
 opt
m
z
d.
For cas
 (2), 

 r
comm

d us

g as a r
f
r

c
 th
 
mp

m

tat
o
 of [`JambaForCausa
LM`](../../../v
m/mod

_
x
cutor/mod

s/jamba.py) (for a
 
xamp

 of a mod

 that us
s Mamba-1 a
d att

t
o
 tog
th
r) or [`BambaForCausa
LM`](../../../v
m/mod

_
x
cutor/mod

s/bamba.py) (for a
 
xamp

 of a mod

 that us
s Mamba-2 a
d att

t
o
 tog
th
r).
Th
s
 mod

s shou
d fo
o
 th
 sam
 

struct
o
s as cas
 (1), but th
y shou
d 

h
r
t protoco
 `IsHybr
d` (

st
ad of `IsAtt

t
o
Fr
`) a
d 
t 
s *
ot* 

c
ssary to add th
m to th
 `MODELS_CONFIG_MAP` (th

r ru
t
m
 d
fau
ts 


 b
 

f
rr
d from th
 protoco
).
For cas
 (3), 

 r
comm

d 
ook

g at th
 
mp

m

tat
o
 of [`M


MaxT
xt01ForCausa
LM`](../../../v
m/mod

_
x
cutor/mod

s/m


max_t
xt_01.py) or [`Lfm2ForCausa
LM`](../../../v
m/mod

_
x
cutor/mod

s/
fm2.py) as a r
f
r

c
, 
h
ch us
 custom "mamba-

k
" 
ay
rs `M


MaxT
xt01L


arAtt

t
o
` a
d `ShortCo
v` r
sp
ct
v

y.
P

as
 fo
o
 th
 sam
 gu
d




s as cas
 (2) for 
mp

m

t

g th
s
 mod

s.
W
 us
 "mamba-

k
" to r
f
r to 
ay
rs that poss
ss a stat
 that 
s updat
d 

-p
ac
, rath
r tha
 b


g app

d
d-to (

k
 KV cach
 for att

t
o
).
For 
mp

m

t

g 


 custom mamba-

k
 
ay
rs, o

 shou
d 

h
r
t from `MambaBas
` a
d 
mp

m

t th
 m
thods `g
t_stat
_dtyp
`, `g
t_stat
_shap
` to ca
cu
at
 th
 data typ
s a
d stat
 shap
s at ru
t
m
, as 


 as `mamba_typ
` a
d `g
t_att
_back

d`.
It 
s a
so 

c
ssary to 
mp

m

t th
 "att

t
o
 m
ta-data" c
ass 
h
ch ha
d

s th
 m
ta-data that 
s commo
 across a
 
ay
rs.
P

as
 s
 [`L


arAtt

t
o
M
tadata`](../../../v
m/v1/att

t
o
/back

ds/



ar_att
.py) or [`ShortCo
vAtt

t
o
M
tadata`](../../../v
m/v1/att

t
o
/back

ds/short_co
v_att
.py) for 
xamp

s of th
s.
It 
s a
so 
orth 
ot

g that 

 shou
d updat
 `MAMBA_TYPE_TO_BACKEND_MAP` a
d `MambaAtt

t
o
Back

dE
um` 

 [`r
g
stry.py`](../../../v
m/v1/att

t
o
/back

ds/r
g
stry.py) 
h

 add

g a 


 mamba back

d.
F

a
y, 
f o

 
a
ts to support torch comp


 a
d CUDA graphs, 
t 

c
ssary to 
rap th
 ca
 to th
 mamba-

k
 
ay
r 

s
d
 a custom op a
d r
g
st
r 
t.
P

as
 s
 th
 ca
s to `d
r
ct_r
g
st
r_custom_op` 

 [v
m/mod

_
x
cutor/mod

s/m


max_t
xt_01.py](../../../v
m/mod

_
x
cutor/mod

s/m


max_t
xt_01.py) or [v
m/mod

_
x
cutor/
ay
rs/mamba/short_co
v.py](../../../v
m/mod

_
x
cutor/
ay
rs/mamba/short_co
v.py) for 
xamp

s of th
s.
Th
 


 custom op shou
d th

 b
 add
d to th
 

st `_att

t
o
_ops` 

 [v
m/co
f
g/comp

at
o
.py](../../../v
m/co
f
g/comp

at
o
.py) to 

sur
 that p

c


s
 CUDA graphs 
orks as 

t

d
d.
