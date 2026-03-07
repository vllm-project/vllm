# `torch.comp


` 

t
grat
o

I
 vLLM's V1 arch
t
ctur
, `torch.comp


` 
s 

ab

d by d
fau
t a
d 
s a cr
t
ca
 part of th
 fram

ork. Th
s docum

t g
v
s a s
mp

 
a
k-through 
xamp

 to sho
 ho
 to u
d
rsta
d th
 `torch.comp


` usag
.
Throughout th
 
xamp

, 

 


 ru
 a commo
 L
ama mod

, a
d tur
 o
 d
bug 

v

 
ogg

g to sho
 a
 th
 d
ta

s. Th
 comma
d to b
 us
d 
s `VLLM_LOGGING_LEVEL=DEBUG v
m s
rv
 m
ta-
ama/L
ama-3.2-1B`.
!!! 
ot

    For mor
 

format
o
 a
d th
 
at
st progr
ss of `torch.comp


` 

t
grat
o
, s
 th
s [B
og Post](https://b
og.v
m.a
/2025/08/20/torch-comp


.htm
).
## Comp

at
o
 Cach

I
 th
 v
ry v
rbos
 
ogs, 

 ca
 s
:
```co
so


INFO 03-07 03:06:55 [back

ds.py:409] Us

g cach
 d
r
ctory: ~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0 for vLLM's torch.comp



```
vLLM 


 tak
 a
 th
 ava

ab

 factors 

to co
s
d
rat
o
, a
d d
c
d
 a d
r
ctory to stor
 a
 th
 comp

at
o
 art
fact. Th
s m
a
s, you ca
 d
r
ct
y copy th
 
ho

 `~/.cach
/v
m/torch_comp


_cach
` d
r
ctory 

 your d
p
oym

t sc

ar
o to sav
 a gr
at amou
t of comp

at
o
 t
m
, a
d h

c
 acc


rat

g th
 start

g t
m
 of th
 vLLM 

sta
c
.
Th
 factors co
s
d
r
d 

c
ud
:
- A
 th
 r

at
d co
f
gs (s
 th
 `comput
_hash` fu
ct
o
s 

 th

r r
sp
ct
v
 co
f
gs 

 th
 [co
f
g fo
d
r](../../v
m/co
f
g))
- PyTorch co
f
gs (s
 th
 `comput
_hash` fu
ct
o
s 

 th
 [comp


r_

t
rfac
.py](../../v
m/comp

at
o
/comp


r_

t
rfac
.py))
- Th
 mod

's for
ard fu
ct
o
 a
d th
 r


va
t fu
ct
o
s ca

d by th
 for
ard fu
ct
o
 (s
 b

o
)
W
th a
 th
s
 factors tak

 

to co
s
d
rat
o
, usua
y 

 ca
 guara
t
 that th
 cach
 
s saf
 to us
, a
d 


 
ot caus
 a
y u

xp
ct
d b
hav
or. Th
r
for
, th
 cach
 
s 

ab

d by d
fau
t. If you 
a
t to d
bug th
 comp

at
o
 proc
ss, or 
f you susp
ct th
 cach
 
s caus

g som
 
ssu
s, you ca
 d
sab

 
t by s
tt

g th
 

v
ro
m

t var
ab

 `VLLM_DISABLE_COMPILE_CACHE=1`.
A u

qu
 asp
ct of vLLM's `torch.comp


` 

t
grat
o
, 
s that 

 guara
t
 a
 th
 comp

at
o
 f


sh
s b
for
 

 s
rv
 a
y r
qu
sts. No r
qu
sts 


 tr
gg
r 


 comp

at
o
s. Oth
r

s
, th
 

g


 
ou
d b
 b
ock
d o
 that r
qu
st, a
d th
 r
spo
s
 t
m
 


 hav
 u

xp
ct
d sp
k
s.
By d
fau
t, th
 cach
 sav
s comp


d art
facts as b

ary f


s. If you 
ou
d 

k
 to 

t
ract 

th th
 g


rat
d cod
 for d
bugg

g purpos
s, s
t th
 f


d `comp


_cach
_sav
_format=u
pack
d` 

 th
 comp

at
o
 co
f
g, or om
t th
s a
d s
t th
 

v var
ab

 `VLLM_COMPILE_CACHE_SAVE_FORMAT=u
pack
d`.
## Dy
am
c shap
s a
d v
m guard dropp

g
`torch.comp


` 
s d
s
g

d to guard o
 dy
am
c shap
s 

th 
o h
s
tat
o


h

 

d
d. Th
s co
trad
cts 

th vLLM's `torch.comp


` approach of
dropp

g th
 guards s

c
 ma
y of thos
 guards cou
d b
 mat
r
a
.
`torch.comp


` prov
d
s t
o k

ds of dy
am
c shap
s: `back
d` a
d `u
back
d`.
`torch.comp


` guards o
 `back
d` dy
am
c shap
s a
d do
s 
ot prov
d
 a
guara
t
 that 
o guards 


 b
 add
d to th
m. Us
r cod
, dy
amo,


ductor, a
d autograd a
 ca
 add guards. Mor
ov
r, for 0/1
sp
c
a

zat
o
s, back
d symbo
s ar
 sp
c
a

z
d u
co
d
t
o
a
y to 0, 1,
or 
=2 
v

 

thout 

cou
t
r

g a bra
ch

g o
 thos
 ra
g
s.
O
 th
 co
trary, `u
back
d` dy
am
c shap
s ar
 guara
t
d 
ot to b
 guard
d
o
 a
d ar
 
ot 0/1 sp
c
a

z
d. Ho

v
r, th
r
 
s a poss
b


ty of
thro


g a data d
p

d

t 
rror 
h

 a bra
ch that r
qu
r
s th

r va
u
 
s


cou
t
r
d a
d 
o 
xp

c
t u
back
d ha
d


g 
s d
f


d. Th
 fram

ork 
s
co
v
rg

g to a stat
 
h
r
 
t 
o
't thro
 DDE but rath
r p
ck g


ra

paths. O

 do

s
d
 of us

g u
back
d 
s m
ss
d opt
m
zat
o
 opportu

t

s
du
 to 

th
r p
rf bugs or p
ck

g g


ra
 paths, a
so us

g a f
x
d

o
-
xamp

 

put-bas
d h

t (th
s 


 b
 f
x
d soo
 

th ov
rr
d
_h

t
API). A
 
xamp

 of p
ck

g g


ra
 paths 
s assum

g 

put 
ot co
t
guous


 fu
ct
o
s ca
 co
t
guous() a
d r
shap
() 
h

 ca
't b
 symbo

ca
y prov




th a cha
g
 of 

troduc

g a c
o

.
`back
d_s
z
_ob

v
ous` 
s a f
ag that 

ab

s tr
at

g back
d symbo
s as
u
back
d 
h
r
v
r 
xp

c
t ha
d


g for u
back
d 
s d
f


d. W
th th
s
mod
, 0/1 sp
c
a

zat
o
s ar
 most
y avo
d
d 

 fram

ork cod
 a
d th

d
fau
t 0/1 sp
c
a

zat
o
 do
s 
ot happ

. Ho

v
r, th
r
 
s st

 
o
guara
t
 that torch.comp


 
o
't guard, 
sp
c
a
y du
 to us
r cod
 or
custom pass
s. `back
d_s
z
_ob

v
ous` 
s 
xp
r
m

ta
 

 PyTorch comp



a
d cou
d b
 d
pr
cat
d. That sa
d, 
t's a saf
r opt
o
 to us
 tha

`back
d` a
d th
 probab


ty of r
duc

g p
rforma
c
 
s 
o

r tha

`u
back
d`.
### Co
f
gur

g Dy
am
c Shap
s
Th
 `Dy
am
cShap
sCo
f
g` a
o
s you to co
tro
 th
 dy
am
c shap
s b
hav
or by
s
tt

g th
 `typ
` f


d. You ca
 choos
 b
t


 thr
 mod
s:
`BACKED`(d
fau
t), `UNBACKED` , a
d `BACKED_SIZE_OBLIVIOUS`.
#### Off



 I
f
r

c
 Examp

 (Us

g LLM c
ass)
Wh

 us

g th
 `LLM` c
ass for off



 

f
r

c
, you ca
 co
f
gur
 dy
am
c
shap
s through th
 `comp

at
o
_co
f
g` param
t
r:
```pytho

from v
m 
mport LLM, Samp


gParams
from v
m.co
f
g.comp

at
o
 
mport Comp

at
o
Co
f
g, Dy
am
cShap
sCo
f
g, Dy
am
cShap
sTyp

# Examp

: Us

g back
d_s
z
_ob

v
ous (
xp
r
m

ta
, saf
r tha
 back
d)

m = LLM(
    mod

="m
ta-
ama/L
ama-3.2-1B",
    comp

at
o
_co
f
g=Comp

at
o
Co
f
g(
        dy
am
c_shap
s_co
f
g=Dy
am
cShap
sCo
f
g(
            typ
=Dy
am
cShap
sTyp
.BACKED_SIZE_OBLIVIOUS
        )
    )
)
# Examp

: Us

g u
back
d (stro
g
st guara
t
 aga

st guards)

m = LLM(
    mod

="m
ta-
ama/L
ama-3.2-1B",
    comp

at
o
_co
f
g=Comp

at
o
Co
f
g(
        dy
am
c_shap
s_co
f
g=Dy
am
cShap
sCo
f
g(
            typ
=Dy
am
cShap
sTyp
.UNBACKED
        )
    )
)
# G


rat
 outputs
prompts = ["H

o, my 
am
 
s", "Th
 futur
 of AI 
s"]
samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.95)
outputs = 
m.g


rat
(prompts, samp


g_params)
```
#### O




 S
rv

g Examp

 (Us

g v
m s
rv
)
Wh

 us

g `v
m s
rv
` for o




 s
rv

g, you ca
 co
f
gur
 dy
am
c shap
s
through th
 `--comp

at
o
-co
f
g` f
ag:
```bash
# Examp

: Us

g u
back
d
v
m s
rv
 m
ta-
ama/L
ama-3.2-1B \
  --comp

at
o
-co
f
g '{"dy
am
c_shap
s_co
f
g": {"typ
": "u
back
d"}}'
# A
t
r
at
v
: Us

g dot 
otat
o
 (s
mp

r for s

g

 va
u
s)
v
m s
rv
 m
ta-
ama/L
ama-3.2-1B -cc.dy
am
c_shap
s_co
f
g.typ
=u
back
d
```
#### Choos

g th
 R
ght Mod

- **BACKED** (d
fau
t): Us
 
h

 you'r
 




g to acc
pt pot

t
a
 u
saf
 dropp

g of guards
for max
ma
 p
rforma
c
. Guard cou
d b
 u
sou
d
y add
d a
d th

 
g
or
d.
- **UNBACKED**  Us
 
h

 you 

d th
 stro
g
st guara
t
 aga

st guards.
  Th
s 
s th
 most co
s
rvat
v
 opt
o
 but may m
ss som
 opt
m
zat
o
 opportu

t

s.
- **BACKED_SIZE_OBLIVIOUS**: Us
 
h

 you 
a
t a ba
a
c
 b
t


 avo
d

g guards
  a
d p
rforma
c
. Th
s 
xp
r
m

ta
 mod
 
s saf
r tha
 BACKED but st

 
ot as
  co
s
rvat
v
 as UNBACKED.
## Pytho
 Cod
 Comp

at
o

I
 th
 v
ry v
rbos
 
ogs, 

 ca
 s
:
??? co
so

 "Logs"
      ```t
xt
      DEBUG 03-07 03:06:52 [d
corators.py:203] Start comp



g fu
ct
o
 
cod
 obj
ct for
ard at 0x7f08acf40c90, f


 "xxx/v
m/mod

_
x
cutor/mod

s/
ama.py", 



 339

      DEBUG 03-07 03:06:54 [back

ds.py:370] Trac
d f


s (to b
 co
s
d
r
d for comp

at
o
 cach
):
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/torch/_dy
amo/po
yf

s/bu

t

s.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/torch/
/modu

s/co
ta


r.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/torch/
/modu

s/modu

.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/att

t
o
/
ay
r.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/d
str
but
d/commu

cat
o
_op.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/d
str
but
d/para


_stat
.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/mod

_
x
cutor/custom_op.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/mod

_
x
cutor/
ay
rs/
ay
r
orm.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/mod

_
x
cutor/
ay
rs/



ar.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/mod

_
x
cutor/
ay
rs/rotary_
mb
dd

g.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/mod

_
x
cutor/
ay
rs/vocab_para


_
mb
dd

g.py
      DEBUG 03-07 03:06:54 [back

ds.py:370] xxx/v
m/mod

_
x
cutor/mod

s/
ama.py
      DEBUG 03-07 03:07:07 [back

ds.py:462] Computat
o
 graph sav
d to ~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/computat
o
_graph.py
      DEBUG 03-07 03:07:07 [
rapp
r.py:105] Dy
amo tra
sform
d cod
 sav
d to ~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/tra
sform
d_cod
.py
      ```
Th
s 
s about th
 Pytho
 cod
 comp

at
o
, 
.
. graph captur
 by Dy
amo. It tr

s to trac
 th
 fu
ct
o
 

th cod
 `xxx/v
m/mod

_
x
cutor/mod

s/
ama.py:339`, 
h
ch 
s th
 `for
ard` fu
ct
o
 of th
 mod

 

 comp


. Dur

g th
 for
ard pass, th
r
 ar
 a
so oth
r fu
ct
o
s ca

d a
d 





d by Dy
amo, as sho

 by th
 
ogs, 

c
ud

g som
 PyTorch fu
ct
o
s from `xxx/torch/
/modu

s/modu

.py` (us
d by PyTorch `
.Modu

`, b
caus
 modu

 attr
but
 acc
ss 


 tr
gg
r a fu
ct
o
 ca
), som
 commu

cat
o
 / att

t
o
 / act
vat
o
 fu
ct
o
s from vLLM. A
 th
 trac
d f


s 


 b
 co
s
d
r
d 
h

 

 d
c
d
 th
 cach
 d
r
ctory to us
. Th
s 
ay, a
y cod
 cha
g
 

 th
 abov
 f


s 


 tr
gg
r comp

at
o
 cach
 m
ss, a
d th
r
for
 r
comp

at
o
.
Th
 r
su
t of th
 Dy
amo comp

at
o
, 
s a 


 fu
ct
o
 stor
d 

 `~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/tra
sform
d_cod
.py`. Usua
y, th
s fu
ct
o
 u
packs t

sors from th
 modu

, a
d th

 pass 
t to th
 trac
d computat
o
 graph. Th
 computat
o
 graph 
s stor
d 

 `~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/computat
o
_graph.py`.
## Computat
o
 Graph Proc
ss

g
Th
 computat
o
 graph has shap
 a
otat
o
s for 
v
ry t

sor. Th
 

puts ar
 

put 
ds, pos
t
o
 
ds, 


ghts a
d buff
rs from th
 mod

, a
d th
 outputs ar
 th
 f

a
 h
dd

 stat
s. Not
 that 
m h
ad proj
ct
o
 a
d samp


g op
rat
o
s ar
 
ot co
s
d
r
d 

 th
 graph.
Most of th
 

puts to th
 computat
o
 graph has stat
c shap
, s

c
 th
y ar
 mod

 


ghts a
d buff
rs, a
d 


 
ot cha
g
 dur

g th
 

f
t
m
 of th
 mod

. O

y th
 

put 
ds a
d pos
t
o
 
ds hav
 symbo

c shap
s, 
.
. th
 shap
 ca
 cha
g
 from batch to batch. Ho

v
r, th
y 


 shar
 th
 sam
 symbo

c shap
s. That 
s to say, th
 o

y cha
g

g s
z
 to th
 computat
o
 graph, 
s th
 batch s
z
 (
umb
r of tok

s proc
ss
d 

 th
 curr

t for
ard pass).
Th
 att

t
o
 op
rat
o
 
s comp

cat
d, a
d 
t 

ds to 

t
ract 

th kv cach
s, 

th comp

cat
d shap
s. Fortu
at

y, th
 output of th
 att

t
o
 op
rat
o
 just shar
 th
 sam
 shap
 as th
 

put qu
ry of th
 att

t
o
 op
rat
o
. Th
r
for
, 

 
rap th
 
ho

 att

t
o
 op
rat
o
 

to a PyTorch custom op `torch.ops.v
m.u

f

d_att

t
o
_

th_output`, so that Dy
amo 


 
ot try to 

sp
ct a
y of th
 

t
r
a
 op
rat
o
s. Th
s 
ay, a
though att

t
o
 op
rat
o
 
s comp

cat
d, 

 ca
 st

 captur
 th
 mod

's computat
o
 graph as a fu
-graph, from Dy
amo's p
rsp
ct
v
.
Th
 computat
o
 graph 
s furth
r sp

t 

to p

c
s, by th
 `sp

tt

g_ops` (usua
y th
s 
s th
 att

t
o
 op
rat
o
). Th
r
for
, 

 th
 `~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/computat
o
_graph.py` f


, 

 ca
 s
 
ots of submodu

s, 
ach submodu

 
s a p

c
 of graph aft
r sp

tt

g:
- Att

t
o
 op
rat
o
 
ts

f 
s a submodu

.
- Th
 part of computat
o
 graph, from o

 att

t
o
 op
rat
o
 to th
 

xt att

t
o
 op
rat
o
, 
s a submodu

.
Ev
ry submodu

 ca
 b
 
d

t
f

d by 
ts 

d
x, a
d 


 b
 proc
ss
d 

d
v
dua
y.
## Computat
o
 Graph Comp

at
o

I
 th
 v
ry v
rbos
 
ogs, 

 ca
 a
so s
:
```co
so


DEBUG 03-07 03:52:37 [back

ds.py:134] stor
 th
 0-th graph for shap
 No

 from 

ductor v
a ha
d

 ('fp
gy
q3v3
zjzphd45
kf
pabggdbjpy
gr7tta4hj6up
sts

', '~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/

ductor_cach
/

/c

zrk3
ttdqatuz
o
ajy
v
o3
vjcs2vfd
dz
zoz
3z
3
y.py')
DEBUG 03-07 03:52:39 [back

ds.py:134] stor
 th
 1-th graph for shap
 No

 from 

ductor v
a ha
d

 ('f7fm
odmf3h3by5
u2c4zar
oxbg4
yt
r3ujdd2jph
4pospfd', '~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/

ductor_cach
/
y/c
yfzx
dfsj7
ha
u
s2mca2omqka4r7mgc
d
f6xfjh645

6k2.py')
...
DEBUG 03-07 03:52:45 [back

ds.py:134] stor
 th
 15-th graph for shap
 No

 from 

ductor v
a ha
d

 ('f7fm
odmf3h3by5
u2c4zar
oxbg4
yt
r3ujdd2jph
4pospfd', '~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/

ductor_cach
/
y/c
yfzx
dfsj7
ha
u
s2mca2omqka4r7mgc
d
f6xfjh645

6k2.py')
DEBUG 03-07 03:52:45 [back

ds.py:134] stor
 th
 16-th graph for shap
 No

 from 

ductor v
a ha
d

 ('fvj3cco
7m34f3d
r4
tmu55mmu
44
5xym
hrj


sy
sk7q6jy', '~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/

ductor_cach
/tf/ctfftkg
j7b4
cttq5cymx6c

372uoauupq
6
dsvp
ucavqcjc.py')
```
Th
s m
a
s th
 f
rst p

c
 of computat
o
 graph (

th shap
 `No

` for symbo

c shap
) 
s comp


d by I
ductor (

th a k
y `fp
gy
q3v3
zjzphd45
kf
pabggdbjpy
gr7tta4hj6up
sts

`). Th
 comp


d k
r


 
s stor
d 

  `~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/

ductor_cach
/

/c

zrk3
ttdqatuz
o
ajy
v
o3
vjcs2vfd
dz
zoz
3z
3
y.py`. You ca
 op

 th
 f


 to s
 
hat 
s th
 cod
 I
ductor f

a
y ru
s.
O

 mor
 d
ta

: you ca
 s
 that th
 1-th graph a
d th
 15-th graph hav
 th
 sam
 k
y, 
h


 th
 0-th graph a
d th
 16-th graph ar
 d
ff
r

t. Th
s 
s 
xp
ct
d, s

c
 

 sp

t th
 graph by th
 att

t
o
 op, 

 g
t 3 u

qu
 subgraphs:
- th
 f
rst 
ay
r b
for
 att

t
o

- 
v
ry m
dd

 
ay
r, from o

 att

t
o
 op
rat
o
 to th
 

xt att

t
o
 op
rat
o

- th
 f

a
 
ay
r aft
r att

t
o

If 

 a
r
ady hav
 th
 cach
 d
r
ctory (
.g. ru
 th
 sam
 cod
 for th
 s
co
d t
m
), 

 


 s
 th
 fo
o


g 
ogs:
```co
so


DEBUG 03-07 04:00:45 [back

ds.py:86] D
r
ct
y 
oad th
 0-th graph for shap
 No

 from 

ductor v
a ha
d

 ('fp
gy
q3v3
zjzphd45
kf
pabggdbjpy
gr7tta4hj6up
sts

', '~/.cach
/v
m/torch_comp


_cach
/1517964802/ra
k_0_0/

ductor_cach
/

/c

zrk3
ttdqatuz
o
ajy
v
o3
vjcs2vfd
dz
zoz
3z
3
y.py')
```
Th
s t
m
, I
ductor comp

at
o
 
s comp

t

y bypass
d, a
d 

 


 
oad from d
sk to r
ad th
 comp

at
o
 art
fact 

 g
t from th
 
ast t
m
.
Th
 abov
 
xamp

 just us
s I
ductor to comp


 for a g


ra
 shap
 (
.
. symbo

c shap
). W
 ca
 a
so us
 I
ductor to comp


 for som
 of th
 sp
c
f
c shap
s, for 
xamp

:
```bash
v
m s
rv
 m
ta-
ama/L
ama-3.2-1B \
  --comp

at
o
_co
f
g '{"comp


_s
z
s": [1, 2, 4, 8]}'
```
Th

 
t 


 a
so comp


 a sp
c
f
c k
r


 just for batch s
z
 `1, 2, 4, 8`. At th
s t
m
, a
 of th
 shap
s 

 th
 computat
o
 graph ar
 stat
c a
d k
o

, a
d 

 


 tur
 o
 auto-tu


g to tu

 for max p
rforma
c
. Th
s ca
 b
 s
o
 
h

 you ru
 
t for th
 f
rst t
m
, but th
 

xt t
m
 you ru
 
t, 

 ca
 d
r
ct
y bypass th
 tu


g a
d ru
 th
 tu

d k
r


.
Wh

 a
 th
 shap
s ar
 k
o

, `torch.comp


` ca
 compar
 d
ff
r

t co
f
gs, a
d oft

 f

d som
 b
tt
r co
f
gs to ru
 th
 k
r


. For 
xamp

, 

 ca
 s
 th
 fo
o


g 
og:
??? co
so

 "Logs"
    ```
    AUTOTUNE mm(8x2048, 2048x3072)
      tr
to
_mm_4 0.0130 ms 100.0% ACC_TYPE='t
.f
oat32', ALLOW_TF32=Fa
s
, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=No

, EVEN_K=Tru
, GROUP_M=8, 
um_stag
s=5, 
um_
arps=2
      tr
to
_mm_8 0.0134 ms 97.4% ACC_TYPE='t
.f
oat32', ALLOW_TF32=Fa
s
, BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=No

, EVEN_K=Tru
, GROUP_M=8, 
um_stag
s=5, 
um_
arps=4
      tr
to
_mm_12 0.0148 ms 87.7% ACC_TYPE='t
.f
oat32', ALLOW_TF32=Fa
s
, BLOCK_K=128, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=No

, EVEN_K=Tru
, GROUP_M=8, 
um_stag
s=4, 
um_
arps=4
      mm 0.0160 ms 81.6%
      tr
to
_mm_16 0.0165 ms 78.7% ACC_TYPE='t
.f
oat32', ALLOW_TF32=Fa
s
, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=No

, EVEN_K=Tru
, GROUP_M=8, 
um_stag
s=5, 
um_
arps=8
      tr
to
_mm_3 0.0199 ms 65.4% ACC_TYPE='t
.f
oat32', ALLOW_TF32=Fa
s
, BLOCK_K=32, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=No

, EVEN_K=Tru
, GROUP_M=8, 
um_stag
s=5, 
um_
arps=2
      tr
to
_mm_1 0.0203 ms 64.2% ACC_TYPE='t
.f
oat32', ALLOW_TF32=Fa
s
, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=No

, EVEN_K=Tru
, GROUP_M=8, 
um_stag
s=2, 
um_
arps=2
      tr
to
_mm_7 0.0203 ms 64.1% ACC_TYPE='t
.f
oat32', ALLOW_TF32=Fa
s
, BLOCK_K=64, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=No

, EVEN_K=Tru
, GROUP_M=8, 
um_stag
s=3, 
um_
arps=4
      tr
to
_mm_2 0.0208 ms 62.5% ACC_TYPE='t
.f
oat32', ALLOW_TF32=Fa
s
, BLOCK_K=32, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=No

, EVEN_K=Tru
, GROUP_M=8, 
um_stag
s=5, 
um_
arps=4
      tr
to
_mm_11 0.0215 ms 60.5% ACC_TYPE='t
.f
oat32', ALLOW_TF32=Fa
s
, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=No

, EVEN_K=Tru
, GROUP_M=8, 
um_stag
s=3, 
um_
arps=4
    S

g

Proc
ss AUTOTUNE b

chmark

g tak
s 2.0428 s
co
ds a
d 7.5727 s
co
ds pr
comp



g
    ```
It m
a
s, for a matr
x mu
t
p

cat
o
 

th shap
 `8x2048x3072`, `torch.comp


` tr

s tr
to
 t
mp
at
 

th var
ous co
f
gs, a
d 
t 
s much fast
r tha
 th
 d
fau
t cod
 (
h
ch d
spatch
s to cub
as 

brary).
U
fortu
at

y, b
caus
 auto-tu


g tak
s qu
t
 a 
o
g t
m
 (from s
co
ds to m

ut
s, d
p

d

g o
 th
 mod

 s
z
 a
d th
 batch s
z
), 
v

 though 
t ca
 b
 cach
d for 
at
r us
, for th
 sak
 of us
r-fr


d



ss, 

 tur
 
t off by d
fau
t. If you 
a
t to hav
 max p
rforma
c
, 
t 
s r
comm

d
d to try 
t, by comp



g sp
c
f
c shap
s.
## Cudagraph Captur

vLLM's V1 arch
t
ctur
 us
s p

c


s
 cudagraph that a

g
s 

th th
 p

c


s
 comp

at
o
. Th
 fu
 computat
o
 graph 
s sp

t as m

t
o

d abov
, a
d 

 o

y captur
 th
 cudagraph for th
 p

c
 of graph b
t


 att

t
o
 op
rat
o
s (

c
ud

g th
 f
rst graph b
for
 a
y att

t
o
 op
rat
o
, a
d th
 
ast graph aft
r a
 th
 att

t
o
 op
rat
o
). Th
s 
s bas
d o
 a commo
 obs
rvat
o
: computat
o
 b
t


 att

t
o
s ar
 usua
y tok

-

s
 a
d 
asy to d
a
 

th for cudagraph; 
h


 th
 att

t
o
 op
rat
o
 
s 
o
-tr
v
a
 to b
 cudagraph compat
b

. Thus, by ru


g th
 att

t
o
 op
rat
o
 

 
ag
r mod
 
h


 th
 r
st op
rat
o
s 

 cudagraph, 

 k
p th
 f

x
b


ty of th
 att

t
o
 op
rat
o
.
Th
 p

c


s
 cudagraph a
so has f


-gra


d m
mory ma
ag
m

t. Th
 purpos
 
s to o

y 
xc
ud
 th
 att

t
o
 k
r


 from cudagraph, 
h


 k
p

g a
 th
 r
st modu

s a
d th
 m
mory a
ocat
o
 op
rat
o
s 

 th
 cudagraph. Th
s 
s 
hy th
 att

t
o
 op
rat
o
 

 V1 has th
 output t

sor as th
 

put of th
 att

t
o
.
Th
 cudagraphs ar
 captur
d a
d ma
ag
d by th
 comp


r back

d, a
d r
p
ay
d 
h

 th
 batch s
z
 has corr
spo
d

g cudagraph captur
d. Th
 ca

r of th
 mod

 (mod

 ru

r) o

y 

ds to mak
 sur
 
t ma
ag
s th
 

put buff
rs corr
ct
y. A
 of th
 

t
rm
d
at
 buff
rs ar
 ma
ag
d automat
ca
y by th
 comp


r back

d.
By d
fau
t, vLLM 


 try to d
t
rm


 a s
t of s
z
s to captur
 cudagraph. You ca
 a
so ov
rr
d
 
t us

g th
 co
f
g `cudagraph_captur
_s
z
s`:
```bash
v
m s
rv
 m
ta-
ama/L
ama-3.2-1B \
  --comp

at
o
-co
f
g '{"cudagraph_captur
_s
z
s": [1, 2, 4, 8]}'
```
Th

 
t 


 o

y captur
 cudagraph for th
 sp
c
f

d s
z
s. It ca
 b
 us
fu
 to hav
 f


-gra


d co
tro
 ov
r th
 cudagraph captur
.
### Fu
 Cudagraph captur

It 
s poss
b

 to 

c
ud
 att

t
o
 as part of th
 cudagraph 
f us

g a
 att

t
o
 back

d that 
s cudagraph compat
b

. Th
s ca
 
mprov
 p
rforma
c
 

 som
 cas
s such as d
cod
 sp
d for sma

r mod

s or MOEs. S
 [CUDA Graphs](cuda_graphs.md) for mor
 d
ta

s.
