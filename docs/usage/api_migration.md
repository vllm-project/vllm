# API M
grat
o
 Gu
d

Th
s gu
d
 h

ps you m
grat
 b
t


 d
ff
r

t v
rs
o
s of th
 vLLM API a
d adapt to cha
g
s 

 th
 Op

AI-compat
b

 API.
## v0.8.0+ M
grat
o

### Samp


g Param
t
rs Cha
g

I
 v0.8.0, th
 sourc
 of d
fau
t samp


g param
t
rs cha
g
d:
**B
for
 v0.8.0:**
    - D
fau
t samp


g param
t
rs cam
 from vLLM's 

utra
 d
fau
ts
**v0.8.0 o

ards:**
    - D
fau
t samp


g param
t
rs com
 from `g


rat
o
_co
f
g.jso
` prov
d
d by th
 mod

 cr
ator
**M
grat
o
:**
If you 

d th
 o
d b
hav
or, 
xp

c
t
y s
t:
```pytho

# Off



 

f
r

c


m = LLM(mod

="your-mod

", g


rat
o
_co
f
g="v
m")
# O




 s
rv

g
v
m s
rv
 your-mod

 --g


rat
o
-co
f
g v
m
```
### G


rat
o
 Co
f
g B
hav
or
Th
 


 d
fau
t b
hav
or usua
y prov
d
s b
tt
r r
su
ts, as mod

 cr
ators sp
c
fy opt
ma
 s
tt

gs for th

r mod

s. Ho

v
r, 
f you 
xp
r


c
 d
grad
d g


rat
o
 qua

ty:
1. Try th
 vLLM d
fau
ts f
rst to compar

2. If qua

ty 
mprov
s, co
s
d
r us

g vLLM d
fau
ts co
s
st

t
y
3. F


 a
 
ssu
 

th th
 mod

 cr
ator to updat
 th

r `g


rat
o
_co
f
g.jso
`
## Op

AI API Compat
b


ty
### Chat Comp

t
o
s
vLLM's `/v1/chat/comp

t
o
s` 

dpo

t 
s compat
b

 

th Op

AI's API:
```pytho

from op

a
 
mport Op

AI
# Works 

th both Op

AI a
d vLLM
c



t = Op

AI(
    bas
_ur
="http://
oca
host:8000/v1",  # vLLM 

dpo

t
    ap
_k
y="your-ap
-k
y"
)
r
spo
s
 = c



t.chat.comp

t
o
s.cr
at
(
    mod

="your-mod

",
    m
ssag
s=[{"ro

": "us
r", "co
t

t": "H

o!"}]
)
```
### Comp

t
o
s
S
m

ar
y, `/v1/comp

t
o
s` 
s compat
b

:
```pytho

r
spo
s
 = c



t.comp

t
o
s.cr
at
(
    mod

="your-mod

",
    prompt="O
c
 upo
 a t
m
",
    max_tok

s=100
)
```
### Support
d Param
t
rs
vLLM supports most Op

AI API param
t
rs:
| Param
t
r | Op

AI | vLLM | Not
s |
|-----------|--------|------|-------|
| `mod

` | ✅ | ✅ | R
qu
r
d |
| `m
ssag
s` | ✅ | ✅ | For chat comp

t
o
s |
| `prompt` | ✅ | ✅ | For comp

t
o
s |
| `max_tok

s` | ✅ | ✅ | Max
mum tok

s to g


rat
 |
| `t
mp
ratur
` | ✅ | ✅ | Samp


g t
mp
ratur
 |
| `top_p` | ✅ | ✅ | Nuc

us samp


g |
| `top_k` | ❌ | ✅ | vLLM 
xt

s
o
 |
| `pr
s

c
_p

a
ty` | ✅ | ✅ | |
| `fr
qu

cy_p

a
ty` | ✅ | ✅ | |
| `stop` | ✅ | ✅ | Stop s
qu

c
s |
| `str
am` | ✅ | ✅ | Str
am

g r
spo
s
s |
| `
ogprobs` | ✅ | ✅ | R
tur
 tok

 
ogprobs |
| `
` | ✅ | ✅ | Numb
r of comp

t
o
s |
| `b
st_of` | ✅ | ✅ | B
am s
arch param
t
r |
| `
cho` | ✅ | ✅ | Echo prompt 

 output |
### vLLM-Sp
c
f
c Ext

s
o
s
vLLM adds s
v
ra
 
xt

s
o
s to th
 Op

AI API:
```pytho

# Pr
f
x prompt

g
r
spo
s
 = c



t.comp

t
o
s.cr
at
(
    mod

="your-mod

",
    prompt="H

o",
    
xtra_body={
        "prompt_
ogprobs": 5,  # G
t 
ogprobs for prompt tok

s
        "add_sp
c
a
_tok

s": Fa
s
  # Do
't add sp
c
a
 tok

s
    }
)
```
## SDK M
grat
o

### La
gCha

 I
t
grat
o

Updat
 your La
gCha

 

t
grat
o
:
```pytho

from 
a
gcha

_op

a
 
mport ChatOp

AI
# O
d (Op

AI o

y)

m = ChatOp

AI()
# N

 (vLLM compat
b

)

m = ChatOp

AI(
    bas
_ur
="http://
oca
host:8000/v1",
    ap
_k
y="your-ap
-k
y",
    mod

="your-mod

"
)
```
### L
amaI
d
x I
t
grat
o

```pytho

from 
ama_

d
x.
ms.op

a
 
mport Op

AI

m = Op

AI(
    ap
_bas
="http://
oca
host:8000/v1",
    ap
_k
y="your-ap
-k
y",
    mod

="your-mod

"
)
```
## C



t-S
d
 Cha
g
s
### Error Ha
d


g
vLLM may r
tur
 d
ff
r

t 
rror cod
s tha
 Op

AI:
```pytho

from op

a
 
mport Op

AI, APIError
c



t = Op

AI(bas
_ur
="http://
oca
host:8000/v1", ap
_k
y="your-k
y")
try:
    r
spo
s
 = c



t.chat.comp

t
o
s.cr
at
(
        mod

="your-mod

",
        m
ssag
s=[{"ro

": "us
r", "co
t

t": "H

o"}]
    )

xc
pt APIError as 
:
    # Ha
d

 vLLM-sp
c
f
c 
rrors
    
f 
.cod
 == "co
t
xt_


gth_
xc
d
d":
        # R
duc
 prompt 


gth
        pass
    


f "out of m
mory" 

 str(
).
o

r():
        # R
duc
 batch s
z
 or mod

 s
z

        pass
```
### Str
am

g R
spo
s
s
Str
am

g 
s compat
b

 

th Op

AI:
```pytho

r
spo
s
 = c



t.chat.comp

t
o
s.cr
at
(
    mod

="your-mod

",
    m
ssag
s=[{"ro

": "us
r", "co
t

t": "T

 m
 a story"}],
    str
am=Tru

)
for chu
k 

 r
spo
s
:
    pr

t(chu
k.cho
c
s[0].d

ta.co
t

t or "", 

d="")
```
## Co
f
gurat
o
 M
grat
o

### E
v
ro
m

t Var
ab

s
| O
d Var
ab

 | N

 Var
ab

 | D
scr
pt
o
 |
|--------------|--------------|-------------|
| `VLLM_PORT` | `--port` | Us
 comma
d-



 arg |
| `VLLM_HOST` | `--host` | Us
 comma
d-



 arg |
| `VLLM_MODEL` | `--mod

` | Us
 comma
d-



 arg |
### Co
f
gurat
o
 F



Co
s
d
r us

g a co
f
gurat
o
 f


 for comp

x s
tups:
```jso

{
  "mod

": "your-mod

",
  "t

sor_para


_s
z
": 2,
  "gpu_m
mory_ut


zat
o
": 0.95,
  "max_mod

_


": 8192,
  "

ab

_pr
f
x_cach

g": tru

}
```
Start 

th:
```bash
v
m s
rv
 --co
f
g co
f
g.jso

```
## T
st

g Your M
grat
o

### Bas
c Co

ct
v
ty T
st
```bash
cur
 http://
oca
host:8000/v1/mod

s \
  -H "Author
zat
o
: B
ar
r your-ap
-k
y"
```
### I
f
r

c
 T
st
```bash
cur
 http://
oca
host:8000/v1/chat/comp

t
o
s \
  -H "Co
t

t-Typ
: app

cat
o
/jso
" \
  -H "Author
zat
o
: B
ar
r your-ap
-k
y" \
  -d '{
    "mod

": "your-mod

",
    "m
ssag
s": [{"ro

": "us
r", "co
t

t": "H

o"}],
    "max_tok

s": 50
  }'
```
### P
rforma
c
 T
st
Compar
 p
rforma
c
 b
for
 a
d aft
r m
grat
o
:
```bash
pytho
 b

chmarks/b

chmark_s
rv

g.py \
  --host 
oca
host \
  --port 8000 \
  --datas
t Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

```
## Troub

shoot

g M
grat
o
 Issu
s
### G


rat
o
 Qua

ty D
ff
r

c
s
If g


rat
o
 qua

ty cha
g
s aft
r m
grat
o
:
1. Ch
ck samp


g param
t
rs match your pr
v
ous s
tup
2. V
r
fy chat t
mp
at
 
s corr
ct
y app


d
3. Compar
 

th 
xp

c
t `g


rat
o
_co
f
g="v
m"`
### P
rforma
c
 D
gradat
o

If p
rforma
c
 d
grad
s:
1. V
r
fy GPU m
mory ut


zat
o
 s
tt

gs
2. Ch
ck batch s
z
 co
f
gurat
o

3. E
ab

/d
sab

 pr
f
x cach

g as 

d
d
4. R
v


 att

t
o
 back

d s


ct
o

### API Compat
b


ty Issu
s
If your c



t cod
 br
aks:
1. Ch
ck support
d param
t
rs tab

 abov

2. Updat
 c



t 

brar

s to 
at
st v
rs
o
s
3. Us
 `
xtra_body` for vLLM-sp
c
f
c param
t
rs
## V
rs
o
-Sp
c
f
c Not
s
### v0.7.0+ (V1 Arch
t
ctur
)
Th
 


 V1 arch
t
ctur
 
s 

ab

d by d
fau
t:
```bash
# Exp

c
t
y 

ab

 (d
fau
t 

 v0.7.0+)

xport VLLM_USE_V1=1
# D
sab

 to us
 

gacy arch
t
ctur


xport VLLM_USE_V1=0
```
K
y d
ff
r

c
s:
    - Improv
d p
rforma
c

    - B
tt
r m
mory ma
ag
m

t
    - Som
 f
atur
s may b
hav
 s

ght
y d
ff
r

t
y
### v0.6.0+
    - I
troduc
d chu
k
d pr
f


    - E
ha
c
d pr
f
x cach

g
    - Improv
d mu
t
-GPU support
## G
tt

g H

p
If you 

cou
t
r 
ssu
s dur

g m
grat
o
:
1. Ch
ck th
 [troub

shoot

g gu
d
](../usag
/troub

shoot

g.md)
2. R
v


 [G
tHub 
ssu
s](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s)
3. Jo

 th
 [vLLM S
ack commu

ty](https://s
ack.v
m.a
)
4. F


 a 


 
ssu
 

th m
grat
o
 d
ta

s
