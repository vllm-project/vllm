# Poo


g Mod

s
vLLM a
so supports poo


g mod

s, such as 
mb
dd

g, c
ass
f
cat
o
, a
d r

ard mod

s.
I
 vLLM, poo


g mod

s 
mp

m

t th
 [V
mMod

ForPoo


g][v
m.mod

_
x
cutor.mod

s.V
mMod

ForPoo


g] 

t
rfac
.
Th
s
 mod

s us
 a [Poo

r][v
m.mod

_
x
cutor.
ay
rs.poo

r.Poo

r] to 
xtract th
 f

a
 h
dd

 stat
s of th
 

put
b
for
 r
tur


g th
m.
!!! 
ot

    W
 curr

t
y support poo


g mod

s pr
mar

y for co
v




c
. Th
s 
s 
ot guara
t
d to prov
d
 a
y p
rforma
c
 
mprov
m

ts ov
r us

g Hugg

g Fac
 Tra
sform
rs or S

t

c
 Tra
sform
rs d
r
ct
y.
    W
 p
a
 to opt
m
z
 poo


g mod

s 

 vLLM. P

as
 comm

t o
 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/21796
 
f you hav
 a
y sugg
st
o
s!
## Co
f
gurat
o

### Mod

 Ru

r
Ru
 a mod

 

 poo


g mod
 v
a th
 opt
o
 `--ru

r poo


g`.
!!! t
p
    Th
r
 
s 
o 

d to s
t th
s opt
o
 

 th
 vast major
ty of cas
s as vLLM ca
 automat
ca
y
    d
t
ct th
 appropr
at
 mod

 ru

r v
a `--ru

r auto`.
### Mod

 Co
v
rs
o

vLLM ca
 adapt mod

s for var
ous poo


g tasks v
a th
 opt
o
 `--co
v
rt 
typ

`.
If `--ru

r poo


g` has b

 s
t (ma
ua
y or automat
ca
y) but th
 mod

 do
s 
ot 
mp

m

t th

[V
mMod

ForPoo


g][v
m.mod

_
x
cutor.mod

s.V
mMod

ForPoo


g] 

t
rfac
,
vLLM 


 att
mpt to automat
ca
y co
v
rt th
 mod

 accord

g to th
 arch
t
ctur
 
am
s
sho

 

 th
 tab

 b

o
.
| Arch
t
ctur
                                    | `--co
v
rt` | Support
d poo


g tasks               |
|-------------------------------------------------|-------------|---------------------------------------|
| `*ForT
xtE
cod

g`, `*Emb
dd

gMod

`, `*Mod

` | `
mb
d`     | `tok

_
mb
d`, `
mb
d`                |
| `*ForR

ardMod



g`, `*R

ardMod

`            | `
mb
d`     | `tok

_
mb
d`, `
mb
d`                |
| `*For*C
ass
f
cat
o
`, `*C
ass
f
cat
o
Mod

`   | `c
ass
fy`  | `tok

_c
ass
fy`, `c
ass
fy`, `scor
` |
!!! t
p
    You ca
 
xp

c
t
y s
t `--co
v
rt 
typ

` to sp
c
fy ho
 to co
v
rt th
 mod

.
### Poo


g Tasks
Each poo


g mod

 

 vLLM supports o

 or mor
 of th
s
 tasks accord

g to
[Poo

r.g
t_support
d_tasks][v
m.mod

_
x
cutor.
ay
rs.poo

r.Poo

r.g
t_support
d_tasks],


ab


g th
 corr
spo
d

g APIs:
| Task             | APIs                                                                          |
|------------------|-------------------------------------------------------------------------------|
| `
mb
d`          | `LLM.
mb
d(...)`, `LLM.scor
(...)`\*, `LLM.

cod
(..., poo


g_task="
mb
d")` |
| `c
ass
fy`       | `LLM.c
ass
fy(...)`, `LLM.

cod
(..., poo


g_task="c
ass
fy")`               |
| `scor
`          | `LLM.scor
(...)`                                                              |
| `tok

_c
ass
fy` | `LLM.r

ard(...)`, `LLM.

cod
(..., poo


g_task="tok

_c
ass
fy")`           |
| `tok

_
mb
d`    | `LLM.

cod
(..., poo


g_task="tok

_
mb
d")`                                 |
| `p
ug

`         | `LLM.

cod
(..., poo


g_task="p
ug

")`                                      |
\* Th
 `LLM.scor
(...)` API fa
s back to `
mb
d` task 
f th
 mod

 do
s 
ot support `scor
` task.
### Poo

r Co
f
gurat
o

#### Pr
d
f


d mod

s
If th
 [Poo

r][v
m.mod

_
x
cutor.
ay
rs.poo

r.Poo

r] d
f


d by th
 mod

 acc
pts `poo

r_co
f
g`,
you ca
 ov
rr
d
 som
 of 
ts attr
but
s v
a th
 `--poo

r-co
f
g` opt
o
.
#### Co
v
rt
d mod

s
If th
 mod

 has b

 co
v
rt
d v
a `--co
v
rt` (s
 abov
),
th
 poo

r ass
g

d to 
ach task has th
 fo
o


g attr
but
s by d
fau
t:
| Task       | Poo


g Typ
 | Norma

zat
o
 | Softmax |
|------------|--------------|---------------|---------|
| `
mb
d`    | `LAST`       | ✅︎            | ❌      |
| `c
ass
fy` | `LAST`       | ❌            | ✅︎      |
Wh

 
oad

g [S

t

c
 Tra
sform
rs](https://hugg

gfac
.co/s

t

c
-tra
sform
rs) mod

s,

ts S

t

c
 Tra
sform
rs co
f
gurat
o
 f


 (`modu

s.jso
`) tak
s pr
or
ty ov
r th
 mod

's d
fau
ts.
You ca
 furth
r custom
z
 th
s v
a th
 `--poo

r-co
f
g` opt
o
,

h
ch tak
s pr
or
ty ov
r both th
 mod

's a
d S

t

c
 Tra
sform
rs' d
fau
ts.
## Off



 I
f
r

c

Th
 [LLM][v
m.LLM] c
ass prov
d
s var
ous m
thods for off



 

f
r

c
.
S
 [co
f
gurat
o
](../ap
/README.md#co
f
gurat
o
) for a 

st of opt
o
s 
h

 


t
a

z

g th
 mod

.
### `LLM.
mb
d`
Th
 [
mb
d][v
m.LLM.
mb
d] m
thod outputs a
 
mb
dd

g v
ctor for 
ach prompt.
It 
s pr
mar

y d
s
g

d for 
mb
dd

g mod

s.
```pytho

from v
m 
mport LLM

m = LLM(mod

="

tf
oat/
5-sma
", ru

r="poo


g")
(output,) = 
m.
mb
d("H

o, my 
am
 
s")

mb
ds = output.outputs.
mb
dd

g
pr

t(f"Emb
dd

gs: {
mb
ds!r} (s
z
={


(
mb
ds)})")
```
A cod
 
xamp

 ca
 b
 fou
d h
r
: [
xamp

s/off



_

f
r

c
/bas
c/
mb
d.py](../../
xamp

s/off



_

f
r

c
/bas
c/
mb
d.py)
### `LLM.c
ass
fy`
Th
 [c
ass
fy][v
m.LLM.c
ass
fy] m
thod outputs a probab


ty v
ctor for 
ach prompt.
It 
s pr
mar

y d
s
g

d for c
ass
f
cat
o
 mod

s.
```pytho

from v
m 
mport LLM

m = LLM(mod

="jaso
9693/Q


2.5-1.5B-ap
ach", ru

r="poo


g")
(output,) = 
m.c
ass
fy("H

o, my 
am
 
s")
probs = output.outputs.probs
pr

t(f"C
ass Probab


t

s: {probs!r} (s
z
={


(probs)})")
```
A cod
 
xamp

 ca
 b
 fou
d h
r
: [
xamp

s/off



_

f
r

c
/bas
c/c
ass
fy.py](../../
xamp

s/off



_

f
r

c
/bas
c/c
ass
fy.py)
### `LLM.scor
`
Th
 [scor
][v
m.LLM.scor
] m
thod outputs s
m

ar
ty scor
s b
t


 s

t

c
 pa
rs.
It 
s d
s
g

d for 
mb
dd

g mod

s a
d cross-

cod
r mod

s. Emb
dd

g mod

s us
 cos


 s
m

ar
ty, a
d [cross-

cod
r mod

s](https://
.sb
rt.

t/
xamp

s/app

cat
o
s/cross-

cod
r/README.htm
) s
rv
 as r
ra
k
rs b
t


 ca
d
dat
 qu
ry-docum

t pa
rs 

 RAG syst
ms.
!!! 
ot

    vLLM ca
 o

y p
rform th
 mod

 

f
r

c
 compo


t (
.g. 
mb
dd

g, r
ra
k

g) of RAG.
    To ha
d

 RAG at a h
gh
r 

v

, you shou
d us
 

t
grat
o
 fram

orks such as [La
gCha

](https://g
thub.com/
a
gcha

-a
/
a
gcha

).
```pytho

from v
m 
mport LLM

m = LLM(mod

="BAAI/bg
-r
ra
k
r-v2-m3", ru

r="poo


g")
(output,) = 
m.scor
(
    "What 
s th
 cap
ta
 of Fra
c
?",
    "Th
 cap
ta
 of Braz

 
s Bras


a.",
)
scor
 = output.outputs.scor

pr

t(f"Scor
: {scor
}")
```
A cod
 
xamp

 ca
 b
 fou
d h
r
: [
xamp

s/off



_

f
r

c
/bas
c/scor
.py](../../
xamp

s/off



_

f
r

c
/bas
c/scor
.py)
### `LLM.r

ard`
Th
 [r

ard][v
m.LLM.r

ard] m
thod 
s ava

ab

 to a
 r

ard mod

s 

 vLLM.
```pytho

from v
m 
mport LLM

m = LLM(mod

="

t
r

m/

t
r

m2-1_8b-r

ard", ru

r="poo


g", trust_r
mot
_cod
=Tru
)
(output,) = 
m.r

ard("H

o, my 
am
 
s")
data = output.outputs.data
pr

t(f"Data: {data!r}")
```
A cod
 
xamp

 ca
 b
 fou
d h
r
: [
xamp

s/off



_

f
r

c
/bas
c/r

ard.py](../../
xamp

s/off



_

f
r

c
/bas
c/r

ard.py)
### `LLM.

cod
`
Th
 [

cod
][v
m.LLM.

cod
] m
thod 
s ava

ab

 to a
 poo


g mod

s 

 vLLM.
!!! 
ot

    P

as
 us
 o

 of th
 mor
 sp
c
f
c m
thods or s
t th
 task d
r
ct
y 
h

 us

g `LLM.

cod
`:
    - For 
mb
dd

gs, us
 `LLM.
mb
d(...)` or `poo


g_task="
mb
d"`.
    - For c
ass
f
cat
o
 
og
ts, us
 `LLM.c
ass
fy(...)` or `poo


g_task="c
ass
fy"`.
    - For s
m

ar
ty scor
s, us
 `LLM.scor
(...)`.
    - For r

ards, us
 `LLM.r

ard(...)` or `poo


g_task="tok

_c
ass
fy"`.
    - For tok

 c
ass
f
cat
o
, us
 `poo


g_task="tok

_c
ass
fy"`.
    - For mu
t
-v
ctor r
tr

va
, us
 `poo


g_task="tok

_
mb
d"`.
    - For IO Proc
ssor P
ug

s, us
 `poo


g_task="p
ug

"`.
```pytho

from v
m 
mport LLM

m = LLM(mod

="

tf
oat/
5-sma
", ru

r="poo


g")
(output,) = 
m.

cod
("H

o, my 
am
 
s", poo


g_task="
mb
d")
data = output.outputs.data
pr

t(f"Data: {data!r}")
```
## O




 S
rv

g
Our [Op

AI-Compat
b

 S
rv
r](../s
rv

g/op

a
_compat
b

_s
rv
r.md) prov
d
s 

dpo

ts that corr
spo
d to th
 off



 APIs:
    - [Emb
dd

gs API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#
mb
dd

gs-ap
) 
s s
m

ar to `LLM.
mb
d`, acc
pt

g both t
xt a
d [mu
t
-moda
 

puts](../f
atur
s/mu
t
moda
_

puts.md) for 
mb
dd

g mod

s.
    - [C
ass
f
cat
o
 API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#c
ass
f
cat
o
-ap
) 
s s
m

ar to `LLM.c
ass
fy` a
d 
s app

cab

 to s
qu

c
 c
ass
f
cat
o
 mod

s.
    - [Scor
 API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#scor
-ap
) 
s s
m

ar to `LLM.scor
` for cross-

cod
r mod

s.
    - [Poo


g API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#poo


g-ap
) 
s s
m

ar to `LLM.

cod
`, b


g app

cab

 to a
 typ
s of poo


g mod

s.
!!! 
ot

    P

as
 us
 o

 of th
 mor
 sp
c
f
c 

dpo

ts or s
t th
 task d
r
ct
y 
h

 us

g th
 [Poo


g API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#poo


g-ap
):
    - For 
mb
dd

gs, us
 [Emb
dd

gs API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#
mb
dd

gs-ap
) or `"task":"
mb
d"`.
    - For c
ass
f
cat
o
 
og
ts, us
 [C
ass
f
cat
o
 API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#c
ass
f
cat
o
-ap
) or `"task":"c
ass
fy"`.
    - For s
m

ar
ty scor
s, us
 [Scor
 API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#scor
-ap
).
    - For r

ards, us
 `"task":"tok

_c
ass
fy"`.
    - For tok

 c
ass
f
cat
o
, us
 `"task":"tok

_c
ass
fy"`.
    - For mu
t
-v
ctor r
tr

va
, us
 `"task":"tok

_
mb
d"`.
    - For IO Proc
ssor P
ug

s, us
 `"task":"p
ug

"`.
```pytho

# start a support
d 
mb
dd

gs mod

 s
rv
r 

th `v
m s
rv
`, 
.g.
# v
m s
rv
 

tf
oat/
5-sma


mport r
qu
sts
host = "
oca
host"
port = "8000"
mod

_
am
 = "

tf
oat/
5-sma
"
ap
_ur
 = f"http://{host}:{port}/poo


g"
prompts = [
    "H

o, my 
am
 
s",
    "Th
 pr
s
d

t of th
 U

t
d Stat
s 
s",
    "Th
 cap
ta
 of Fra
c
 
s",
    "Th
 futur
 of AI 
s",
]
prompt = {"mod

": mod

_
am
, "

put": prompts, "task": "
mb
d"}
r
spo
s
 = r
qu
sts.post(ap
_ur
, jso
=prompt)
for output 

 r
spo
s
.jso
()["data"]:
    data = output["data"]
    pr

t(f"Data: {data!r} (s
z
={


(data)})")
```
## Matryoshka Emb
dd

gs
[Matryoshka Emb
dd

gs](https://sb
rt.

t/
xamp

s/s

t

c
_tra
sform
r/tra



g/matryoshka/README.htm
#matryoshka-
mb
dd

gs) or [Matryoshka R
pr
s

tat
o
 L
ar


g (MRL)](https://arx
v.org/abs/2205.13147) 
s a t
ch

qu
 us
d 

 tra



g 
mb
dd

g mod

s. It a
o
s us
rs to trad
 off b
t


 p
rforma
c
 a
d cost.
!!! 
ar


g
    Not a
 
mb
dd

g mod

s ar
 tra


d us

g Matryoshka R
pr
s

tat
o
 L
ar


g. To avo
d m
sus
 of th
 `d
m

s
o
s` param
t
r, vLLM r
tur
s a
 
rror for r
qu
sts that att
mpt to cha
g
 th
 output d
m

s
o
 of mod

s that do 
ot support Matryoshka Emb
dd

gs.
    For 
xamp

, s
tt

g `d
m

s
o
s` param
t
r 
h


 us

g th
 `BAAI/bg
-m3` mod

 


 r
su
t 

 th
 fo
o


g 
rror.
    ```jso

    {"obj
ct":"
rror","m
ssag
":"Mod

 \"BAAI/bg
-m3\" do
s 
ot support matryoshka r
pr
s

tat
o
, cha
g

g output d
m

s
o
s 


 

ad to poor r
su
ts.","typ
":"BadR
qu
stError","param":
u
,"cod
":400}
    ```
### Ma
ua
y 

ab

 Matryoshka Emb
dd

gs
Th
r
 
s curr

t
y 
o off
c
a
 

t
rfac
 for sp
c
fy

g support for Matryoshka Emb
dd

gs. I
 vLLM, 
f `
s_matryoshka` 
s `Tru
` 

 `co
f
g.jso
`, you ca
 cha
g
 th
 output d
m

s
o
 to arb
trary va
u
s. Us
 `matryoshka_d
m

s
o
s` to co
tro
 th
 a
o

d output d
m

s
o
s.
For mod

s that support Matryoshka Emb
dd

gs but ar
 
ot r
cog

z
d by vLLM, ma
ua
y ov
rr
d
 th
 co
f
g us

g `hf_ov
rr
d
s={"
s_matryoshka": Tru
}` or `hf_ov
rr
d
s={"matryoshka_d
m

s
o
s": [
a
o

d output d
m

s
o
s
]}` (off



), or `--hf-ov
rr
d
s '{"
s_matryoshka": tru
}'` or `--hf-ov
rr
d
s '{"matryoshka_d
m

s
o
s": [
a
o

d output d
m

s
o
s
]}'` (o




).
H
r
 
s a
 
xamp

 to s
rv
 a mod

 

th Matryoshka Emb
dd

gs 

ab

d.
```bash
v
m s
rv
 S
o
f
ak
/s
o
f
ak
-arct
c-
mb
d-m-v1.5 --hf-ov
rr
d
s '{"matryoshka_d
m

s
o
s":[256]}'
```
### Off



 I
f
r

c

You ca
 cha
g
 th
 output d
m

s
o
s of 
mb
dd

g mod

s that support Matryoshka Emb
dd

gs by us

g th
 d
m

s
o
s param
t
r 

 [Poo


gParams][v
m.Poo


gParams].
```pytho

from v
m 
mport LLM, Poo


gParams

m = LLM(
    mod

="j

aa
/j

a-
mb
dd

gs-v3",
    ru

r="poo


g",
    trust_r
mot
_cod
=Tru
,
)
outputs = 
m.
mb
d(
    ["Fo
o
 th
 
h
t
 rabb
t."],
    poo


g_params=Poo


gParams(d
m

s
o
s=32),
)
pr

t(outputs[0].outputs)
```
A cod
 
xamp

 ca
 b
 fou
d h
r
: [
xamp

s/poo


g/
mb
d/
mb
d_matryoshka_fy_off



.py](../../
xamp

s/poo


g/
mb
d/
mb
d_matryoshka_fy_off



.py)
### O




 I
f
r

c

Us
 th
 fo
o


g comma
d to start th
 vLLM s
rv
r.
```bash
v
m s
rv
 j

aa
/j

a-
mb
dd

gs-v3 --trust-r
mot
-cod

```
You ca
 cha
g
 th
 output d
m

s
o
s of 
mb
dd

g mod

s that support Matryoshka Emb
dd

gs by us

g th
 d
m

s
o
s param
t
r.
```bash
cur
 http://127.0.0.1:8000/v1/
mb
dd

gs \
  -H 'acc
pt: app

cat
o
/jso
' \
  -H 'Co
t

t-Typ
: app

cat
o
/jso
' \
  -d '{
    "

put": "Fo
o
 th
 
h
t
 rabb
t.",
    "mod

": "j

aa
/j

a-
mb
dd

gs-v3",
    "

cod

g_format": "f
oat",
    "d
m

s
o
s": 32
  }'
```
Exp
ct
d output:
```jso

{"
d":"
mbd-5c21fc9a5c9d4384a1b021daccaf9f64","obj
ct":"

st","cr
at
d":1745476417,"mod

":"j

aa
/j

a-
mb
dd

gs-v3","data":[{"

d
x":0,"obj
ct":"
mb
dd

g","
mb
dd

g":[-0.3828125,-0.1357421875,0.03759765625,0.125,0.21875,0.09521484375,-0.003662109375,0.1591796875,-0.130859375,-0.0869140625,-0.1982421875,0.1689453125,-0.220703125,0.1728515625,-0.2275390625,-0.0712890625,-0.162109375,-0.283203125,-0.055419921875,-0.0693359375,0.031982421875,-0.04052734375,-0.2734375,0.1826171875,-0.091796875,0.220703125,0.37890625,-0.0888671875,-0.12890625,-0.021484375,-0.0091552734375,0.23046875]}],"usag
":{"prompt_tok

s":8,"tota
_tok

s":8,"comp

t
o
_tok

s":0,"prompt_tok

s_d
ta

s":
u
}}
```
A
 Op

AI c



t 
xamp

 ca
 b
 fou
d h
r
: [
xamp

s/poo


g/
mb
d/op

a
_
mb
dd

g_matryoshka_fy_c



t.py](../../
xamp

s/poo


g/
mb
d/op

a
_
mb
dd

g_matryoshka_fy_c



t.py)
## Sp
c
f
c mod

s
### Co
BERT Lat
 I
t
ract
o
 Mod

s
[Co
BERT](https://arx
v.org/abs/2004.12832) (Co
t
xtua

z
d Lat
 I
t
ract
o
 ov
r BERT) 
s a r
tr

va
 mod

 that us
s p
r-tok

 
mb
dd

gs a
d MaxS
m scor

g for docum

t ra
k

g. U


k
 s

g

-v
ctor 
mb
dd

g mod

s, Co
BERT r
ta

s tok

-

v

 r
pr
s

tat
o
s a
d comput
s r


va
c
 scor
s through 
at
 

t
ract
o
, prov
d

g b
tt
r accuracy 
h


 b


g mor
 
ff
c


t tha
 cross-

cod
rs.
vLLM supports Co
BERT mod

s 

th mu
t
p

 

cod
r backbo

s:
| Arch
t
ctur
 | Backbo

 | Examp

 HF Mod

s |
|---|---|---|
| `HF_Co
BERT` | BERT | `a
s

rdota
/a
s

ra
-co
b
rt-sma
-v1`, `co
b
rt-
r/co
b
rtv2.0` |
| `Co
BERTMod
r
B
rtMod

` | Mod
r
BERT | `

ghto
a
/GTE-Mod
r
Co
BERT-v1` |
| `Co
BERTJ

aRob
rtaMod

` | J

a XLM-RoBERTa | `j

aa
/j

a-co
b
rt-v2` |
**BERT-bas
d Co
BERT** mod

s 
ork out of th
 box:
```sh


v
m s
rv
 a
s

rdota
/a
s

ra
-co
b
rt-sma
-v1
```
For **
o
-BERT backbo

s**, us
 `--hf-ov
rr
d
s` to s
t th
 corr
ct arch
t
ctur
:
```sh


# Mod
r
BERT backbo


v
m s
rv
 

ghto
a
/GTE-Mod
r
Co
BERT-v1 \
    --hf-ov
rr
d
s '{"arch
t
ctur
s": ["Co
BERTMod
r
B
rtMod

"]}'
# J

a XLM-RoBERTa backbo


v
m s
rv
 j

aa
/j

a-co
b
rt-v2 \
    --hf-ov
rr
d
s '{"arch
t
ctur
s": ["Co
BERTJ

aRob
rtaMod

"]}' \
    --trust-r
mot
-cod

```
Th

 you ca
 us
 th
 r
ra
k 

dpo

t:
```sh


cur
 -s http://
oca
host:8000/r
ra
k -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "a
s

rdota
/a
s

ra
-co
b
rt-sma
-v1",
    "qu
ry": "What 
s mach


 

ar


g?",
    "docum

ts": [
        "Mach


 

ar


g 
s a subs
t of art
f
c
a
 

t


g

c
.",
        "Pytho
 
s a programm

g 
a
guag
.",
        "D
p 

ar


g us
s 

ura
 

t
orks."
    ]
}'
```
Or th
 scor
 

dpo

t:
```sh


cur
 -s http://
oca
host:8000/scor
 -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "a
s

rdota
/a
s

ra
-co
b
rt-sma
-v1",
    "t
xt_1": "What 
s mach


 

ar


g?",
    "t
xt_2": ["Mach


 

ar


g 
s a subs
t of AI.", "Th
 

ath
r 
s su
y."]
}'
```
You ca
 a
so g
t th
 ra
 tok

 
mb
dd

gs us

g th
 poo


g 

dpo

t 

th `tok

_
mb
d` task:
```sh


cur
 -s http://
oca
host:8000/poo


g -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "a
s

rdota
/a
s

ra
-co
b
rt-sma
-v1",
    "

put": "What 
s mach


 

ar


g?",
    "task": "tok

_
mb
d"
}'
```
A
 
xamp

 ca
 b
 fou
d h
r
: [
xamp

s/poo


g/scor
/co
b
rt_r
ra
k_o




.py](../../
xamp

s/poo


g/scor
/co
b
rt_r
ra
k_o




.py)
### Co
Q


3 Mu
t
-Moda
 Lat
 I
t
ract
o
 Mod

s
Co
Q


3 
s bas
d o
 [Co
Pa

](https://arx
v.org/abs/2407.01449), 
h
ch 
xt

ds Co
BERT's 
at
 

t
ract
o
 approach to **mu
t
-moda
** 

puts. Wh


 Co
BERT op
rat
s o
 t
xt-o

y tok

 
mb
dd

gs, Co
Pa

/Co
Q


3 ca
 
mb
d both **t
xt a
d 
mag
s** (
.g. PDF pag
s, scr

shots, d
agrams) 

to p
r-tok

 L2-
orma

z
d v
ctors a
d comput
 r


va
c
 v
a MaxS
m scor

g. Co
Q


3 sp
c
f
ca
y us
s Q


3-VL as 
ts v
s
o
-
a
guag
 backbo

.
| Arch
t
ctur
 | Backbo

 | Examp

 HF Mod

s |
|---|---|---|
| `Co
Q


3` | Q


3-VL | `TomoroAI/tomoro-co
q


3-
mb
d-4b`, `TomoroAI/tomoro-co
q


3-
mb
d-8b` |
| `OpsCo
Q


3Mod

` | Q


3-VL | `Op

S
arch-AI/Ops-Co
q


3-4B`, `Op

S
arch-AI/Ops-Co
q


3-8B` |
| `Q


3VLN
motro
Emb
dMod

` | Q


3-VL | `
v
d
a/

motro
-co

mb
d-v
-4b-v2`, `
v
d
a/

motro
-co

mb
d-v
-8b-v2` |
Start th
 s
rv
r:
```sh


v
m s
rv
 TomoroAI/tomoro-co
q


3-
mb
d-4b --max-mod

-


 4096
```
#### T
xt-o

y scor

g a
d r
ra
k

g
Us
 th
 `/r
ra
k` 

dpo

t:
```sh


cur
 -s http://
oca
host:8000/r
ra
k -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "TomoroAI/tomoro-co
q


3-
mb
d-4b",
    "qu
ry": "What 
s mach


 

ar


g?",
    "docum

ts": [
        "Mach


 

ar


g 
s a subs
t of art
f
c
a
 

t


g

c
.",
        "Pytho
 
s a programm

g 
a
guag
.",
        "D
p 

ar


g us
s 

ura
 

t
orks."
    ]
}'
```
Or th
 `/scor
` 

dpo

t:
```sh


cur
 -s http://
oca
host:8000/scor
 -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "TomoroAI/tomoro-co
q


3-
mb
d-4b",
    "t
xt_1": "What 
s th
 cap
ta
 of Fra
c
?",
    "t
xt_2": ["Th
 cap
ta
 of Fra
c
 
s Par
s.", "Pytho
 
s a programm

g 
a
guag
."]
}'
```
#### Mu
t
-moda
 scor

g a
d r
ra
k

g (t
xt qu
ry × 
mag
 docum

ts)
Th
 `/scor
` a
d `/r
ra
k` 

dpo

ts a
so acc
pt mu
t
-moda
 

puts d
r
ct
y.
Pass 
mag
 docum

ts us

g th
 `data_1`/`data_2` (for `/scor
`) or `docum

ts` (for `/r
ra
k`) f


ds


th a `co
t

t` 

st co
ta



g `
mag
_ur
` a
d `t
xt` parts — th
 sam
 format us
d by th

Op

AI chat comp

t
o
 API:
Scor
 a t
xt qu
ry aga

st 
mag
 docum

ts:
```sh


cur
 -s http://
oca
host:8000/scor
 -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "TomoroAI/tomoro-co
q


3-
mb
d-4b",
    "data_1": "R
tr

v
 th
 c
ty of B

j

g",
    "data_2": [
        {
            "co
t

t": [
                {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": "data:
mag
/p
g;bas
64,
BASE64
"}},
                {"typ
": "t
xt", "t
xt": "D
scr
b
 th
 
mag
."}
            ]
        }
    ]
}'
```
R
ra
k 
mag
 docum

ts by a t
xt qu
ry:
```sh


cur
 -s http://
oca
host:8000/r
ra
k -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "TomoroAI/tomoro-co
q


3-
mb
d-4b",
    "qu
ry": "R
tr

v
 th
 c
ty of B

j

g",
    "docum

ts": [
        {
            "co
t

t": [
                {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": "data:
mag
/p
g;bas
64,
BASE64_1
"}},
                {"typ
": "t
xt", "t
xt": "D
scr
b
 th
 
mag
."}
            ]
        },
        {
            "co
t

t": [
                {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": "data:
mag
/p
g;bas
64,
BASE64_2
"}},
                {"typ
": "t
xt", "t
xt": "D
scr
b
 th
 
mag
."}
            ]
        }
    ],
    "top_
": 2
}'
```
#### Ra
 tok

 
mb
dd

gs
You ca
 a
so g
t th
 ra
 tok

 
mb
dd

gs us

g th
 `/poo


g` 

dpo

t 

th `tok

_
mb
d` task:
```sh


cur
 -s http://
oca
host:8000/poo


g -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "TomoroAI/tomoro-co
q


3-
mb
d-4b",
    "

put": "What 
s mach


 

ar


g?",
    "task": "tok

_
mb
d"
}'
```
For **
mag
 

puts** v
a th
 poo


g 

dpo

t, us
 th
 chat-sty

 `m
ssag
s` f


d:
```sh


cur
 -s http://
oca
host:8000/poo


g -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "TomoroAI/tomoro-co
q


3-
mb
d-4b",
    "m
ssag
s": [
        {
            "ro

": "us
r",
            "co
t

t": [
                {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": "data:
mag
/p
g;bas
64,
BASE64
"}},
                {"typ
": "t
xt", "t
xt": "D
scr
b
 th
 
mag
."}
            ]
        }
    ]
}'
```
#### Examp

s
    - Mu
t
-v
ctor r
tr

va
: [
xamp

s/poo


g/tok

_
mb
d/co
q


3_tok

_
mb
d_o




.py](../../
xamp

s/poo


g/tok

_
mb
d/co
q


3_tok

_
mb
d_o




.py)
    - R
ra
k

g (t
xt + mu
t
-moda
): [
xamp

s/poo


g/scor
/co
q


3_r
ra
k_o




.py](../../
xamp

s/poo


g/scor
/co
q


3_r
ra
k_o




.py)
### L
ama N
motro
 Mu
t
moda

#### Emb
dd

g Mod


L
ama N
motro
 VL Emb
dd

g mod

s comb


 th
 b
d
r
ct
o
a
 L
ama 
mb
dd

g backbo


(from `
v
d
a/
ama-

motro
-
mb
d-1b-v2`) 

th S
gLIP as th
 v
s
o
 

cod
r to produc

s

g

-v
ctor 
mb
dd

gs from t
xt a
d/or 
mag
s.
| Arch
t
ctur
 | Backbo

 | Examp

 HF Mod

s |
|---|---|---|
| `L
amaN
motro
VLMod

` | B
d
r
ct
o
a
 L
ama + S
gLIP | `
v
d
a/
ama-

motro
-
mb
d-v
-1b-v2` |
Start th
 s
rv
r:
```sh


v
m s
rv
 
v
d
a/
ama-

motro
-
mb
d-v
-1b-v2 \
    --trust-r
mot
-cod
 \
    --chat-t
mp
at
 
xamp

s/poo


g/
mb
d/t
mp
at
/

motro
_
mb
d_v
.j

ja
```
!!! 
ot

    Th
 chat t
mp
at
 bu
d

d 

th th
s mod

's tok


z
r 
s 
ot su
tab

 for
    th
 
mb
dd

gs API. Us
 th
 prov
d
d ov
rr
d
 t
mp
at
 abov
 
h

 s
rv

g
    

th th
 `m
ssag
s`-bas
d (chat-sty

) 
mb
dd

gs 

dpo

t.
    Th
 ov
rr
d
 t
mp
at
 us
s th
 m
ssag
 `ro

` to automat
ca
y pr
p

d th

    appropr
at
 pr
f
x: s
t `ro

` to `"qu
ry"` for qu
r

s (pr
p

ds `qu
ry: `)
    or `"docum

t"` for passag
s (pr
p

ds `passag
: `). A
y oth
r ro

 om
ts
    th
 pr
f
x.
Emb
d t
xt qu
r

s:
```sh


cur
 -s http://
oca
host:8000/v1/
mb
dd

gs -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "
v
d
a/
ama-

motro
-
mb
d-v
-1b-v2",
    "m
ssag
s": [
        {
            "ro

": "qu
ry",
            "co
t

t": [
                {"typ
": "t
xt", "t
xt": "What 
s mach


 

ar


g?"}
            ]
        }
    ]
}'
```
Emb
d 
mag
s v
a th
 chat-sty

 `m
ssag
s` f


d:
```sh


cur
 -s http://
oca
host:8000/v1/
mb
dd

gs -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "
v
d
a/
ama-

motro
-
mb
d-v
-1b-v2",
    "m
ssag
s": [
        {
            "ro

": "docum

t",
            "co
t

t": [
                {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": "data:
mag
/p
g;bas
64,
BASE64
"}},
                {"typ
": "t
xt", "t
xt": "D
scr
b
 th
 
mag
."}
            ]
        }
    ]
}'
```
#### R
ra
k
r Mod


L
ama N
motro
 VL r
ra
k
r mod

s comb


 th
 sam
 b
d
r
ct
o
a
 L
ama + S
gLIP
backbo

 

th a s
qu

c
-c
ass
f
cat
o
 h
ad for cross-

cod
r scor

g a
d r
ra
k

g.
| Arch
t
ctur
 | Backbo

 | Examp

 HF Mod

s |
|---|---|---|
| `L
amaN
motro
VLForS
qu

c
C
ass
f
cat
o
` | B
d
r
ct
o
a
 L
ama + S
gLIP | `
v
d
a/
ama-

motro
-r
ra
k-v
-1b-v2` |
Start th
 s
rv
r:
```sh


v
m s
rv
 
v
d
a/
ama-

motro
-r
ra
k-v
-1b-v2 \
    --ru

r poo


g \
    --trust-r
mot
-cod
 \
    --chat-t
mp
at
 
xamp

s/poo


g/scor
/t
mp
at
/

motro
-v
-r
ra
k.j

ja
```
!!! 
ot

    Th
 chat t
mp
at
 bu
d

d 

th th
s ch
ckpo

t's tok


z
r 
s 
ot su
tab


    for th
 Scor
/R
ra
k APIs. Us
 th
 prov
d
d ov
rr
d
 t
mp
at
 
h

 s
rv

g:
    `
xamp

s/poo


g/scor
/t
mp
at
/

motro
-v
-r
ra
k.j

ja`.
Scor
 a t
xt qu
ry aga

st a
 
mag
 docum

t:
```sh


cur
 -s http://
oca
host:8000/scor
 -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "
v
d
a/
ama-

motro
-r
ra
k-v
-1b-v2",
    "data_1": "F

d d
agrams about auto
omous robots",
    "data_2": [
        {
            "co
t

t": [
                {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": "data:
mag
/p
g;bas
64,
BASE64
"}},
                {"typ
": "t
xt", "t
xt": "Robot
cs 
orkf
o
 d
agram."}
            ]
        }
    ]
}'
```
R
ra
k 
mag
 docum

ts by a t
xt qu
ry:
```sh


cur
 -s http://
oca
host:8000/r
ra
k -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
    "mod

": "
v
d
a/
ama-

motro
-r
ra
k-v
-1b-v2",
    "qu
ry": "F

d d
agrams about auto
omous robots",
    "docum

ts": [
        {
            "co
t

t": [
                {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": "data:
mag
/p
g;bas
64,
BASE64_1
"}},
                {"typ
": "t
xt", "t
xt": "Robot
cs 
orkf
o
 d
agram."}
            ]
        },
        {
            "co
t

t": [
                {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": "data:
mag
/p
g;bas
64,
BASE64_2
"}},
                {"typ
": "t
xt", "t
xt": "G


ra
 sky



 photo."}
            ]
        }
    ],
    "top_
": 2
}'
```
### BAAI/bg
-m3
Th
 `BAAI/bg
-m3` mod

 com
s 

th 
xtra 


ghts for spars
 a
d co
b
rt 
mb
dd

gs but u
fortu
at

y 

 
ts `co
f
g.jso
`
th
 arch
t
ctur
 
s d
c
ar
d as `XLMRob
rtaMod

`, 
h
ch mak
s `vLLM` 
oad 
t as a va


a ROBERTA mod

 

thout th


xtra 


ghts. To 
oad th
 fu
 mod

 


ghts, ov
rr
d
 
ts arch
t
ctur
 

k
 th
s:
```sh


v
m s
rv
 BAAI/bg
-m3 --hf-ov
rr
d
s '{"arch
t
ctur
s": ["Bg
M3Emb
dd

gMod

"]}'
```
Th

 you obta

 th
 spars
 
mb
dd

gs 

k
 th
s:
```sh


cur
 -s http://
oca
host:8000/poo


g -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
     "mod

": "BAAI/bg
-m3",
     "task": "tok

_c
ass
fy",
     "

put": ["What 
s BGE M3?", "D
f


t
o
 of BM25"]
}'
```
Du
 to 

m
tat
o
s 

 th
 output sch
ma, th
 output co
s
sts of a 

st of
tok

 scor
s for 
ach tok

 for 
ach 

put. Th
s m
a
s that you'
 hav
 to ca

`/tok


z
` as 


 to b
 ab

 to pa
r tok

s 

th scor
s.
R
f
r to th
 t
sts 

  `t
sts/mod

s/
a
guag
/poo


g/t
st_bg
_m3.py` to s
 ho

to do that.
You ca
 obta

 th
 co
b
rt 
mb
dd

gs 

k
 th
s:
```sh


cur
 -s http://
oca
host:8000/poo


g -H "Co
t

t-Typ
: app

cat
o
/jso
" -d '{
     "mod

": "BAAI/bg
-m3",
     "task": "tok

_
mb
d",
     "

put": ["What 
s BGE M3?", "D
f


t
o
 of BM25"]
}'
```
## D
pr
cat
d F
atur
s
### E
cod
 task
W
 hav
 sp

t th
 `

cod
` task 

to t
o mor
 sp
c
f
c tok

-

s
 tasks: `tok

_
mb
d` a
d `tok

_c
ass
fy`:
    - `tok

_
mb
d` 
s th
 sam
 as `
mb
d`, us

g 
orma

zat
o
 as th
 act
vat
o
.
    - `tok

_c
ass
fy` 
s th
 sam
 as `c
ass
fy`, by d
fau
t us

g softmax as th
 act
vat
o
.
Poo


g mod

s 
o
 d
fau
t support a
 poo


g, you ca
 us
 
t 

thout a
y s
tt

gs.
    - Extract

g h
dd

 stat
s pr
f
rs us

g `tok

_
mb
d` task.
    - R

ard mod

s pr
f
rs us

g `tok

_c
ass
fy` task.
