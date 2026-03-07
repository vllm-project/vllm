# G


rat
v
 Mod

s
vLLM prov
d
s f
rst-c
ass support for g


rat
v
 mod

s, 
h
ch cov
rs most of LLMs.
I
 vLLM, g


rat
v
 mod

s 
mp

m

t th
 [V
mMod

ForT
xtG


rat
o
][v
m.mod

_
x
cutor.mod

s.V
mMod

ForT
xtG


rat
o
] 

t
rfac
.
Bas
d o
 th
 f

a
 h
dd

 stat
s of th
 

put, th
s
 mod

s output 
og probab


t

s of th
 tok

s to g


rat
,

h
ch ar
 th

 pass
d through [Samp

r][v
m.v1.samp

.samp

r.Samp

r] to obta

 th
 f

a
 t
xt.
## Co
f
gurat
o

### Mod

 Ru

r (`--ru

r`)
Ru
 a mod

 

 g


rat
o
 mod
 v
a th
 opt
o
 `--ru

r g


rat
`.
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
 mod

 ru

r to us
 v
a `--ru

r auto`.
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
### `LLM.g


rat
`
Th
 [g


rat
][v
m.LLM.g


rat
] m
thod 
s ava

ab

 to a
 g


rat
v
 mod

s 

 vLLM.
It 
s s
m

ar to [
ts cou
t
rpart 

 HF Tra
sform
rs](https://hugg

gfac
.co/docs/tra
sform
rs/ma

/

/ma

_c
ass
s/t
xt_g


rat
o
#tra
sform
rs.G


rat
o
M
x

.g


rat
),

xc
pt that tok


zat
o
 a
d d
tok


zat
o
 ar
 a
so p
rform
d automat
ca
y.
```pytho

from v
m 
mport LLM

m = LLM(mod

="fac
book/opt-125m")
outputs = 
m.g


rat
("H

o, my 
am
 
s")
for output 

 outputs:
    prompt = output.prompt
    g


rat
d_t
xt = output.outputs[0].t
xt
    pr

t(f"Prompt: {prompt!r}, G


rat
d t
xt: {g


rat
d_t
xt!r}")
```
You ca
 opt
o
a
y co
tro
 th
 
a
guag
 g


rat
o
 by pass

g [Samp


gParams][v
m.Samp


gParams].
For 
xamp

, you ca
 us
 gr
dy samp


g by s
tt

g `t
mp
ratur
=0`:
```pytho

from v
m 
mport LLM, Samp


gParams

m = LLM(mod

="fac
book/opt-125m")
params = Samp


gParams(t
mp
ratur
=0)
outputs = 
m.g


rat
("H

o, my 
am
 
s", params)
for output 

 outputs:
    prompt = output.prompt
    g


rat
d_t
xt = output.outputs[0].t
xt
    pr

t(f"Prompt: {prompt!r}, G


rat
d t
xt: {g


rat
d_t
xt!r}")
```
!!! 
mporta
t
    By d
fau
t, vLLM 


 us
 samp


g param
t
rs r
comm

d
d by mod

 cr
ator by app
y

g th
 `g


rat
o
_co
f
g.jso
` from th
 hugg

gfac
 mod

 r
pos
tory 
f 
t 
x
sts. I
 most cas
s, th
s 


 prov
d
 you 

th th
 b
st r
su
ts by d
fau
t 
f [Samp


gParams][v
m.Samp


gParams] 
s 
ot sp
c
f

d.
    Ho

v
r, 
f vLLM's d
fau
t samp


g param
t
rs ar
 pr
f
rr
d, p

as
 pass `g


rat
o
_co
f
g="v
m"` 
h

 cr
at

g th
 [LLM][v
m.LLM] 

sta
c
.
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
c/bas
c.py](../../
xamp

s/off



_

f
r

c
/bas
c/bas
c.py)
### `LLM.b
am_s
arch`
Th
 [b
am_s
arch][v
m.LLM.b
am_s
arch] m
thod 
mp

m

ts [b
am s
arch](https://hugg

gfac
.co/docs/tra
sform
rs/

/g


rat
o
_strat
g

s#b
am-s
arch) o
 top of [g


rat
][v
m.LLM.g


rat
].
For 
xamp

, to s
arch us

g 5 b
ams a
d output at most 50 tok

s:
```pytho

from v
m 
mport LLM
from v
m.samp


g_params 
mport B
amS
archParams

m = LLM(mod

="fac
book/opt-125m")
params = B
amS
archParams(b
am_

dth=5, max_tok

s=50)
outputs = 
m.b
am_s
arch([{"prompt": "H

o, my 
am
 
s "}], params)
for output 

 outputs:
    g


rat
d_t
xt = output.s
qu

c
s[0].t
xt
    pr

t(f"G


rat
d t
xt: {g


rat
d_t
xt!r}")
```
### `LLM.chat`
Th
 [chat][v
m.LLM.chat] m
thod 
mp

m

ts chat fu
ct
o
a

ty o
 top of [g


rat
][v
m.LLM.g


rat
].
I
 part
cu
ar, 
t acc
pts 

put s
m

ar to [Op

AI Chat Comp

t
o
s API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/chat)
a
d automat
ca
y app


s th
 mod

's [chat t
mp
at
](https://hugg

gfac
.co/docs/tra
sform
rs/

/chat_t
mp
at

g) to format th
 prompt.
!!! 
mporta
t
    I
 g


ra
, o

y 

struct
o
-tu

d mod

s hav
 a chat t
mp
at
.
    Bas
 mod

s may p
rform poor
y as th
y ar
 
ot tra


d to r
spo
d to th
 chat co
v
rsat
o
.
??? cod

    ```pytho

    from v
m 
mport LLM
    
m = LLM(mod

="m
ta-
ama/M
ta-L
ama-3-8B-I
struct")
    co
v
rsat
o
 = [
        {
            "ro

": "syst
m",
            "co
t

t": "You ar
 a h

pfu
 ass
sta
t",
        },
        {
            "ro

": "us
r",
            "co
t

t": "H

o",
        },
        {
            "ro

": "ass
sta
t",
            "co
t

t": "H

o! Ho
 ca
 I ass
st you today?",
        },
        {
            "ro

": "us
r",
            "co
t

t": "Wr
t
 a
 
ssay about th
 
mporta
c
 of h
gh
r 
ducat
o
.",
        },
    ]
    outputs = 
m.chat(co
v
rsat
o
)
    for output 

 outputs:
        prompt = output.prompt
        g


rat
d_t
xt = output.outputs[0].t
xt
        pr

t(f"Prompt: {prompt!r}, G


rat
d t
xt: {g


rat
d_t
xt!r}")
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
c/chat.py](../../
xamp

s/off



_

f
r

c
/bas
c/chat.py)
If th
 mod

 do
s
't hav
 a chat t
mp
at
 or you 
a
t to sp
c
fy a
oth
r o

,
you ca
 
xp

c
t
y pass a chat t
mp
at
:
```pytho

from v
m.

trypo

ts.chat_ut

s 
mport 
oad_chat_t
mp
at

# You ca
 f

d a 

st of 
x
st

g chat t
mp
at
s u
d
r `
xamp

s/`
custom_t
mp
at
 = 
oad_chat_t
mp
at
(chat_t
mp
at
="
path_to_t
mp
at

")
pr

t("Load
d chat t
mp
at
:", custom_t
mp
at
)
outputs = 
m.chat(co
v
rsat
o
, chat_t
mp
at
=custom_t
mp
at
)
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
- [Comp

t
o
s API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#comp

t
o
s-ap
) 
s s
m

ar to `LLM.g


rat
` but o

y acc
pts t
xt.
- [Chat API](../s
rv

g/op

a
_compat
b

_s
rv
r.md#chat-ap
)  
s s
m

ar to `LLM.chat`, acc
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

puts.md) for mod

s 

th a chat t
mp
at
.
