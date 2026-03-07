# Ho
 to d
bug th
 vLLM-torch.comp


 

t
grat
o

TL;DR:
- us
 t
pars
 to acqu
r
 torch.comp


 
ogs. I
c
ud
 th
s
 
ogs 

 bug r
ports a
d/or support asks.
- Th
 vLLM-torch.comp


 

t
grat
o
 
s mu
t
p

 p

c
s. vLLM 
xpos
s f
ags to tur
 off 
ach p

c
:
| O




 F
ag | Off



 F
ag   |      R
su
t |
|----------|----------|-------------|
| --

forc
-
ag
r | 

forc
_
ag
r=Tru
 |  Tur
 off torch.comp


 a
d CUDAGraphs |
| -cc.mod
=0 | mod
=Comp

at
o
Mod
.NONE |  Tur
 off torch.comp


 o

y |
| -cc.cudagraph_mod
=NONE | comp

at
o
_co
f
g=Comp

at
o
Co
f
g(cudagraph_mod
=CUDAGraphMod
.NONE) |  Tur
 off CUDAGraphs o

y |
| -cc.back

d=
ag
r | comp

at
o
_co
f
g=Comp

at
o
Co
f
g(back

d='
ag
r') |  Tur
 off TorchI
ductor |
## vLLM-torch.comp


 ov
rv



To 
mprov
 p
rforma
c
, vLLM 

v
rag
s torch.comp


 a
d CUDAGraphs to sp
d th

gs up.
torch.comp


 g


rat
s opt
m
z
d k
r


s for PyTorch cod
 
h


 CUDAGraphs 


m

at
s ov
rh
ad.
Most 
otab
y, vLLM-comp


 
s NOT torch.comp


, 
t 
s a custom comp


r bu

t us

g 

t
r
a
 PyTorch Comp


 APIs.
![vLLM-comp


 d
agram](../ass
ts/d
s
g
/d
bug_v
m_comp


/d
s
g
_d
agram.p
g)
- G
v

 a mod

, 

 do a fu
 graph captur
 v
a TorchDy
amo that 
s dy
am
c o
 th
 batch s
z
 (
umb
r of tok

s)
- vLLM th

 opt
o
a
y sp

ts a
d/or sp
c
a

z
s th
s graph a
d th

 us
s TorchI
ductor to comp


 
ach graph 

to a comp


d art
fact.
Th
s st
p may us
 vLLM custom I
ductor pass
s to furth
r opt
m
z
 th
 graph.
- Th
 comp


d art
fact 
s sav
d to vLLM's comp


 cach
 so that 
t ca
 b
 
oad
d 

 th
 futur
.
- vLLM app


s CUDAGraphs to r
duc
 CPU ov
rh
ads.
Th

gs ca
 go 
ro
g 

 
ach of th
 four st
ps. Wh

 som
th

g do
s go 
ro
g, p

as
 try to 
so
at
 th
 subsyst
m
that 


t 
ro
g -- th
s 


 a
o
 you to tur
 off th
 m


ma
 
umb
r of th

gs to k
p r


ab


ty
goa
s 
h


 m


m
z

g 
mpact to p
rforma
c
 a
d a
so h

ps us (vLLM) 
h

 you op

 a bug r
port.
For mor
 d
ta

s o
 th
 d
s
g
, p

as
 s
 th
 fo
o


g r
sourc
s:
- [I
troduct
o
 to vLLM-torch.comp


 b
ogpost](https://b
og.v
m.a
/2025/08/20/torch-comp


.htm
)
- [vLLM-torch.comp


 

t
grat
o
 d
s
g
](./torch_comp


.md)
- [vLLM Off
c
 Hours #26](https://
.youtub
.com/

v
/xLyxc7hxCJc?s
=Xu
o9p
53C6y
f0V&t=561)
- [Ta
k at PyTorch Co
f
r

c
 2025](https://youtu.b
/1
V1ESbGrVQ?s
=s1GqymUf


OrDTg&t=725)
## Us
 t
pars

Us
 [t
pars
](https://g
thub.com/m
ta-pytorch/t
pars
) to v


 torch.comp




ogs. Th
s
 
ogs sho
 a
 stag
s of th
 comp

at
o
 proc
ss, 

c
ud

g th
 fus
d
k
r


s that torch.comp


 produc
s.
I
sta
 t
pars
:
```sh
p
p 

sta
 t
pars

```
To 

ab

 th
 torch.comp


 
ogs, you ca
 s
t th
 

vvar `TORCH_TRACE=
d
r
`.
Dur

g trac

g, a f


 p
r ra
k 


 b
 cr
at
d 

s
d
 of that d
r
ctory, 

th

ach f


 co
ta



g th
 art
facts dur

g comp

at
o
. If you ca
, 

 r
comm

d
s

d

g th
s
 
og f


s a
o
g 

th bug r
ports -- th
y ar
 v
ry h

pfu
.
Usag
 (off



 

f
r

c
)
```sh
TORCH_TRACE=~/trac
_d
r pytho
 my_scr
pt.py
t
pars
 ~/trac
_d
r/
ra
k_0_
og_f




```
Usag
 (s
rv

g)
```sh
TORCH_TRACE=~/trac
_d
r v
m s
rv

# ctr
-c out of th
 s
rv
r
t
pars
 ~/trac
_d
r/
ra
k_0_
og_f




```
G
v

 o

 of th
 
og f


s, th
 `t
pars
` comma
d outputs som
 HTML f


s
(p
rhaps 

to 
.g. `./t
_out/

d
x.htm
`).
Op

 
t to s
 th
 
ogs. It'
 
ook som
th

g 

k
 th
 fo
o


g:
![t
pars
 
xamp

](../ass
ts/d
s
g
/d
bug_v
m_comp


/t
pars
_

ductor.p
g)
## Tur
 off vLLM-torch.comp


 

t
grat
o

Pass `--

forc
-
ag
r` to tur
 off th
 vLLM-torch.comp


 

t
grat
o
 a
d ru
 

t
r

y


 
ag
r mod
. Th
s 

c
ud
s tur


g off CUDAGraphs.
```sh
# O





v
m s
rv
 --

forc
-
ag
r
```
```py
# Off




LLM(mod

, 

forc
_
ag
r=Tru
)
```
To tur
 off just torch.comp


, pass `mod
 = NONE` to th
 comp

at
o
 co
f
g.
(`-cc` 
s short for `--comp

at
o
_co
f
g`):
```sh
# O





v
m s
rv
 -cc.mod
=0
```
```py
# Off




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
g, Comp

at
o
Mod

LLM(mod

, comp

at
o
_co
f
g=Comp

at
o
Co
f
g(mod
=Comp

at
o
Mod
.NONE))
```
To tur
 off just CUDAGraphs, pass `cudagraph_mod
 = NONE`:
```sh
# O





v
m s
rv
 -cc.cudagraph_mod
=NONE
```
```py
# Off




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
g, CUDAGraphMod

LLM(mod

, comp

at
o
_co
f
g=Comp

at
o
Co
f
g(cudagraph_mod
=CUDAGraphMod
.NONE))
```
## D
bugg

g TorchDy
amo
vLLM r
qu
r
s mod

 cod
 b
 capturab

 

to a fu
 graph v
a TorchDy
amo (torch.comp


's fro
t

d).
TorchDy
amo do
s 
ot support a
 of Pytho
. It 


 
rror (

 fu
graph mod
) 
f 
t ca
ot support
a f
atur
 (th
s 
s som
t
m
s k
o

 as a graph br
ak).
If you 

cou
t
r a graph br
ak, p

as
 [op

 a
 
ssu
 to pytorch/pytorch](https://g
thub.com/pytorch/pytorch) so th
 PyTorch d
vs ca
 pr
or
t
z
.
Th

, try your b
st to r

r
t
 th
 cod
 to avo
d th
 graph br
ak.
For mor
 

format
o
, s
 th
s [Dy
amo gu
d
](https://docs.pytorch.org/docs/stab

/comp


/programm

g_mod

.dy
amo_cor
_co
c
pts.htm
).
## D
bugg

g Dy
am
c Shap
 fu
 graph captur

vLLM r
qu
r
s that th
 mod

's for
ard pass b
 capturab

 

to a fu
 graph that 
s dy
am
c
o
 th
 batch s
z
 (
.
. th
 
umb
r of tok

s). It (by d
fau
t) comp


s th
s o

 graph 

to
o

 art
fact a
d us
s th
s art
fact for a
 batch s
z
s.
If your cod
 ca
ot b
 captur
d 

th Dy
am
c Shap
s, you may s
 s



t 

corr
ct

ss,

oud 
rrors, or CUDA 


ga
 m
mory acc
ss
s. For 
xamp

, th
 fo
o


g 
s 
ot
capturab

 

to a s

g

 graph:
```py

f data.s
z
[0] % 128 == 0:
    foo(...)


s
:
    bar(...)
```
Th
s prob

m 
s 
asy to d
ag
os
. Us
 t
pars
 a
d c

ck o
 `comp

at
o
_m
tr
cs`:

t 


 t

 you symbo

c co
stra

ts o
 th
 batch s
z
. If th
r
 
s a
y co
stra

t
that r
str
cts th
 batch s
z
s, th

 

'v
 got a prob

m.
![Bad t
pars
 
xamp

](../ass
ts/d
s
g
/d
bug_v
m_comp


/dy
am
c_shap
s.p
g)
To avo
d th
s, p

as
 

th
r:
1. avo
d bra
ch

g o
 th
 
umb
r of tok

s
2. 
rap th
 bra
ch

g 
og
c 

to a custom op
rator. TorchDy
amo do
s 
ot
trac
 

to custom op
rators.
## D
bugg

g co
stra

t v
o
at
o
s a
d dy
am
c shap
s guards 
ssu
s
Dy
am
c-shap
 guards ar
 a sp
c
f
c cat
gory of Dy
amo guards. Th
y ar
 co
stra

ts that `torch.comp


`
attach
s to dy
am
c d
m

s
o
s (
.g., `s
q_


`) to 

sur
 th
 comp


d art
fact r
ma

s va

d.
Th
s
 guards typ
ca
y app
ar 
h

 fram

ork cod
, custom pass
s, or us
r cod
 bra
ch
s bas
d o

dy
am
c shap
 va
u
s.
**Examp

:**
```pytho


f x 
 10:
    # path A


s
:
    # path B
```
Th
s cr
at
s a guard `x 
 10` or `x 
= 10` d
p

d

g o
 
h
ch path 
as trac
d.
**vLLM's Assumpt
o
:**
vLLM assum
s that a
 guards add
d by torch.comp


 ar
 saf
 to drop a
d 


 
ot
co
stra

 th
 comp


d graph to sp
c
f
c 

put shap
s. Wh

 th
s assumpt
o
 
s v
o
at
d,

t ca
 caus
 
ssu
s that us
rs 

d to d
bug.
Som
 s
d
 
ff
cts that 

d
cat
s th
s assumpt
o
 
s v
o
at
d ar
 ru
t
m
 
rrors
or `Co
stra

tV
o
at
o
Errors`.
A `Co
stra

tV
o
at
o
Errors` 


 b
 thro

 
f a dy
am
c shap
 g
ts co
stra


d to
a s

g

 va
u
. If you 

cou
t
r a co
stra

t v
o
at
o
 
rror or susp
ct that a dy
am
c
shap
s guard 
s b


g add
d 

corr
ct
y, you ca
 us
 str
ct
r dy
am
c shap
 mod
s to
h

p d
bug th
 
ssu
:
```sh
# O




 - us

g u
back
d mod

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
# O




 - us

g back
d_s
z
_ob

v
ous mod

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
=back
d_s
z
_ob

v
ous
```
```py
# Off



 - us

g u
back
d mod

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

LLM(mod

, comp

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
g(typ
=Dy
am
cShap
sTyp
.UNBACKED)
))
# Off



 - us

g back
d_s
z
_ob

v
ous mod

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

LLM(mod

, comp

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
g(typ
=Dy
am
cShap
sTyp
.BACKED_SIZE_OBLIVIOUS)
))
```
Th
s
 mod
s ar
 str
ct
r a
d r
duc
 or 


m

at
 th
 

d of dy
am
c shap
s guard

g, 
h
ch ca
 h

p 
so
at
 
ssu
s:
- `u
back
d`: Us
s u
back
d sym

ts 
h
ch do
't a
o
 guards, mak

g 
t 
as

r to 
d

t
fy 
h
r
 guards ar
 b


g 

corr
ct
y add
d
- `back
d_s
z
_ob

v
ous`: Us
s a mod
 that 
s str
ct
r about guard

g.
For mor
 d
ta

s o
 dy
am
c shap
s mod
s, s
 [Dy
am
c shap
s a
d vLLM guard dropp

g](torch_comp


.md#dy
am
c-shap
s-a
d-v
m-guard-dropp

g).
### Pr

t

g guards
To s
 a
 guards that ar
 b


g add
d dur

g comp

at
o
, you ca
 us
 `TORCH_LOGS=+dy
am
c`:
```sh
TORCH_LOGS=+dy
am
c v
m s
rv
 m
ta-
ama/L
ama-3.2-1B
```
Look for `[guard add
d]` 

 th
 
ogs to s
 
h
r
 guards ar
 b


g add
d. Th
s ca
 h

p you 
d

t
fy 
h
ch op
rat
o
s ar

caus

g guards to b
 add
d 

corr
ct
y.
## D
bugg

g TorchI
ductor
TorchI
ductor tak
s a captur
d graph a
d th

 comp


s 
t do

 to som
 Pytho
 cod

that may ca
 1+ tr
to
 k
r


s. O
 rar
 (but u
fortu
at
) occas
o
s, 
t may
produc
 a
 

corr
ct tr
to
 k
r


. Th
s may ma

f
st as s



t 

corr
ct

ss,
CUDA 


ga
 m
mory acc
ss
s, or 
oud 
rrors.
To d
bug 
f TorchI
ductor 
s at fau
t, you ca
 d
sab

 
t by pass

g `back

d='
ag
r'`
to th
 comp

at
o
 co
f
g:
```sh
# o





v
m s
rv
 -cc.back

d=
ag
r
```
```py
# off




LLM(comp

at
o
_co
f
g=Comp

at
o
Co
f
g(back

d='
ag
r'))
```
If I
ductor 
s at fau
t, [f


 a bug to PyTorch](https://g
thub.com/pytorch/pytorch).
If you'r
 f



g adv

turous, you ca
 d
bug th
 tr
to
 k
r


s 

 th
 I
ductor output cod

(that you ca
 
ocat
 v
a us

g t
pars
).
![t
pars
 
xamp

](../ass
ts/d
s
g
/d
bug_v
m_comp


/t
pars
_

ductor.p
g)
You ca
 a
so us
 `TORCH_LOGS=output_cod
 
comma
d
` to pr

t th
 I
ductor output cod
.
### Ed
tab

 TorchI
ductor cod

You ca
 
d
t th
 TorchI
ductor cod
 that g
ts ru
 by s
tt

g `VLLM_COMPILE_CACHE_SAVE_FORMAT=u
pack
d`
or pass

g `-cc.comp


_cach
_sav
_format=u
pack
d`. Th
 d
fau
t 
s `b

ary`, 
h
ch m
a
s 
t 
s 
ot 
d
tab

.
Th
s 
s a us
fu
 t
ch

qu
: you ca
 put br
akpo

ts (
.g. `torch.d
str
but
d.br
akpo

t()`)
a
d pr

t stat
m

ts 

 th
 output cod
.
## D
bugg

g vLLM-comp


 cach

vLLM bu

t 
ts o

 cach
 for torch.comp


 art
facts. Th
 
d
a 
s that th
 art
facts
ca
 b
 comp


d o
c
 a
d th

 r
us
d aft
r th
y hav
 b

 comp


d. Th
s

s a 
ay
r o
 top of [torch.comp


's comp


r cach
](https://docs.pytorch.org/tutor
a
s/r
c
p
s/torch_comp


_cach

g_tutor
a
.htm
).
Wh


 torch.comp


's comp


r cach
 
s rock-stab

, vLLM's comp


r cach
 
s u
fortu
at

y

ot a

ays corr
ct. You ca
 d
sab

 
t v
a s
tt

g `VLLM_DISABLE_COMPILE_CACHE=1`.
You ca
 a
so ma
ua
y r
mov
 th
s cach
.
- R
mov
 vLLM's comp


 cach
 

th `rm -rf ~/.cach
/v
m` (
ook at 
ogs to s
 
f th
 
ocat
o
 cha
g
d)
- R
mov
 torch.comp


's bu

t-

 cach
s 

th `rm -rf /tmp/torch

ductor_$(
hoam
)`
vLLM's cach
 
s a mapp

g from cach
 k
y to a comp


d art
fact. vLLM comput
s
th
 cach
 k
y v
a comb



g mu
t
p

 factors (
.g. co
f
g f
ags a
d mod

 
am
).
If vLLM's comp


 cach
 
s 
ro
g, th
s usua
y m
a
s that a factor 
s m
ss

g.
P

as
 s
 [th
s 
xamp

](https://g
thub.com/v
m-proj
ct/v
m/b
ob/18b39828d90413d05d770dfd2
2f48304f4ca0
b/v
m/co
f
g/mod

.py#L310)
of ho
 vLLM comput
s part of th
 cach
 k
y.
vLLM's comp

at
o
 cach
 r
qu
r
s that th
 cod
 b


g comp


d 

ds up b


g s
r
a

zab

.
If th
s 
s 
ot th
 cas
, th

 
t 


 
rror out o
 sav
. Usua
y th
 f
x
s ar
 to 

th
r:
- r

r
t
 th
 
o
-s
r
a

zab

 p

c
s (p
rhaps d
ff
cu
t b
caus
 
t's d
ff
cu
t to
  t

 r
ght 
o
 
hat 
s s
r
a

zab

 a
d 
hat 
s
't)
- f


 a bug r
port
- 
g
or
 th
 
rror by s
tt

g `VLLM_DISABLE_COMPILE_CACHE=1` (
ot
 that th
s 



  mak
 
arm s
rv
r starts a 
ot s
o

r).
## D
bugg

g CUDAGraphs
CUDAGraphs 
s a f
atur
 that a
o
s o

 to:
- Captur
 a ca
ab

 that 
au
ch
s 1+ CUDA k
r


s 

to a CUDAGraph
- R
p
ay th
 CUDAGraph
Th
 captur
d CUDAGraph co
ta

s a
 of th
 m
mory us
d dur

g th
 captur
 proc
ss.
Th
 r
p
ay of th
 CUDAGraph r
ads a
d 
r
t
s to 
xact
y th
 sam
 r
g
o
s of m
mory.
Th
s 

ads to som
 r
str
ct
o
s:
1. I
 ord
r to us
 CUDAGraphs o
 


 data, you'
 

d to copy th
 data 

to a buff
r
that th
 CUDAGraph 
s r
ad

g from
2. CUDAGraphs o

y captur
 CUDA k
r


s, th
y do
't captur
 
ork do

 o
 CPU.
vLLM us
s th
 ra
 CUDAGraphs API, 
h
ch 
s u
saf
 
h

 us
d 

corr
ct
y.
To tur
 off just CUDAGraphs, pass `cudagraph_mod
 = NONE`:
```sh
# O





v
m s
rv
 -cc.cudagraph_mod
=NONE
```
```py
# Off




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
g, CUDAGraphMod

LLM(mod

, comp

at
o
_co
f
g=Comp

at
o
Co
f
g(cudagraph_mod
=CUDAGraphMod
.NONE))
```
