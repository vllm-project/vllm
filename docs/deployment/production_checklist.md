# Product
o
 D
p
oym

t Ch
ck

st
Th
s ch
ck

st cov
rs 
ss

t
a
 co
s
d
rat
o
s for d
p
oy

g vLLM 

 product
o
 

v
ro
m

ts.
## Pr
-D
p
oym

t
### Hard
ar
 R
qu
r
m

ts
    - [ ] **GPU S


ct
o
**: Choos
 GPUs 

th suff
c


t VRAM for your mod


  - Ca
cu
at
 r
qu
r
d m
mory: mod

 s
z
 + KV cach
 + act
vat
o
 m
mory
  - Add 20-30% buff
r for product
o
 
ork
oads
    - [ ] **CPU a
d M
mory**: E
sur
 ad
quat
 CPU a
d syst
m RAM
  - CPU: Mod
r
 mu
t
-cor
 proc
ssor (8+ cor
s r
comm

d
d)
  - RAM: At 

ast 2x th
 mod

 s
z
 

 syst
m m
mory
    - [ ] **Storag
**: Fast 
oca
 storag
 for mod

 


ghts
  - NVM
 SSD r
comm

d
d for mod

 
oad

g
  - Suff
c


t spac
 for mu
t
p

 mod

 v
rs
o
s
### Mod

 Pr
parat
o

    - [ ] **Mod

 T
st

g**: T
st mod

 
oca
y b
for
 d
p
oym

t
  ```bash
  v
m s
rv
 your-mod

 --max-mod

-


 4096
```
    - [ ] **Qua
t
zat
o
**: Co
s
d
r qua
t
zat
o
 for 
arg
r mod

s
  - FP8 for Hopp
r GPUs
  - AWQ/GPTQ for oth
r GPUs
    - [ ] **Chat T
mp
at
**: V
r
fy chat t
mp
at
 
f us

g 

struct mod

s
  ```bash
  v
m s
rv
 your-mod

 --chat-t
mp
at
 path/to/t
mp
at
.j

ja
```
## Co
f
gurat
o

### P
rforma
c
 Tu


g
    - [ ] **GPU M
mory Ut


zat
o
**: S
t appropr
at
 va
u
 (d
fau
t 0.9)
  ```bash
  --gpu-m
mory-ut


zat
o
 0.95  # H
gh
r for d
d
cat
d s
rv
rs
  --gpu-m
mory-ut


zat
o
 0.8   # Lo

r for shar
d r
sourc
s
```
    - [ ] **Max S
qu

c
 L

gth**: Co
f
gur
 bas
d o
 
xp
ct
d usag

  ```bash
  --max-mod

-


 8192  # Typ
ca
 for chat
  --max-mod

-


 32768 # For 
o
g docum

ts
```
    - [ ] **Batch S
z
**: Tu

 for your 
at

cy/throughput r
qu
r
m

ts
  ```bash
  --max-
um-s
qs 256  # H
gh
r throughput
  --max-
um-s
qs 128  # Lo

r 
at

cy
```
### F
atur
s
    - [ ] **Pr
f
x Cach

g**: E
ab

 for 
ork
oads 

th shar
d prompts
  ```bash
  --

ab

-pr
f
x-cach

g
```
    - [ ] **Chu
k
d Pr
f

**: E
ab

 for b
tt
r 

t
ract
v
ty
  ```bash
  --

ab

-chu
k
d-pr
f


```
## S
cur
ty
    - [ ] **API K
y Auth

t
cat
o
**: E
ab

 

 product
o

  ```bash
  --ap
-k
y your-s
cr
t-k
y
  # or
  
xport VLLM_API_KEY=your-s
cr
t-k
y
```
    - [ ] **N
t
ork S
cur
ty**: Co
f
gur
 f
r

a
s a
d s
cur
ty groups
  - R
str
ct acc
ss to 

c
ssary ports o

y
  - Us
 pr
vat
 sub

ts 
h
r
 poss
b


    - [ ] **Mod

 Acc
ss**: Co
tro
 mod

 do


oad p
rm
ss
o
s
  - Us
 pr
vat
 mod

 r
g
str

s 
f 

d
d
  - S
t appropr
at
 Hugg

gFac
 tok

 p
rm
ss
o
s
## Mo

tor

g
### Logg

g
    - [ ] **Log L
v

**: S
t appropr
at
 
ogg

g 

v


  ```bash
  
xport VLLM_LOGGING_LEVEL=INFO  # Product
o

  
xport VLLM_LOGGING_LEVEL=DEBUG  # Troub

shoot

g
```
    - [ ] **Log Co

ct
o
**: Co
f
gur
 c

tra

z
d 
ogg

g
  - Export 
ogs to ELK, Sp
u
k, or c
oud 
ogg

g s
rv
c
s
  - S
t up 
og rotat
o
 to pr
v

t d
sk spac
 
ssu
s
### M
tr
cs
    - [ ] **K
y M
tr
cs to Mo

tor**:
  - R
qu
st 
at

cy (TTFT, TPOT)
  - Throughput (tok

s/s
co
d)
  - GPU ut


zat
o

  - M
mory usag

  - Cach
 h
t rat
 (
f us

g pr
f
x cach

g)
  - Error rat
s
    - [ ] **A

rt

g**: S
t up a

rts for cr
t
ca
 thr
sho
ds
  - GPU m
mory 
 95%
  - Error rat
 
 1%
  - P99 
at

cy 
 thr
sho
d
  - S
rv
c
 u
ava

ab


## H
gh Ava

ab


ty
### Load Ba
a
c

g
    - [ ] **Mu
t
-I
sta
c
 S
tup**: D
p
oy mu
t
p

 vLLM 

sta
c
s
  - Us
 
oad ba
a
c
r (
g

x, HAProxy, c
oud LB)
  - Co
f
gur
 h
a
th ch
cks
    - [ ] **S
ss
o
 Aff


ty**: Co
s
d
r 
f 

d
d
  - Stat


ss: No aff


ty r
qu
r
d
  - Stat
fu
: Imp

m

t st
cky s
ss
o
s
### Sca


g
    - [ ] **Hor
zo
ta
 Sca


g**: P
a
 for sca


g out
  - Kub
r

t
s HPA or c
oud auto-sca


g
  - Mod

 para



sm for 
arg
 mod

s
    - [ ] **V
rt
ca
 Sca


g**: P
a
 for sca


g up
  - Larg
r GPU 

sta
c
s
  - Mor
 GPU m
mory
## Backup a
d R
cov
ry
    - [ ] **Mod

 W

ghts**: E
sur
 mod

s ca
 b
 r
-do


oad
d
  - Docum

t mod

 v
rs
o
s a
d sourc
s
  - Co
s
d
r 
oca
 cach

g for 
arg
 mod

s
    - [ ] **Co
f
gurat
o
**: Backup d
p
oym

t co
f
gurat
o
s
  - I
frastructur
 as Cod
 (T
rraform, C
oudFormat
o
)
  - Kub
r

t
s ma

f
sts
  - E
v
ro
m

t co
f
gurat
o
s
## T
st

g
### Load T
st

g
    - [ ] **P
rforma
c
 T
st

g**: T
st u
d
r 
xp
ct
d 
oad
  ```bash
  pytho
 b

chmarks/b

chmark_s
rv

g.py \
    --host 
oca
host --port 8000 \
    --datas
t b

chmarks/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso
 \
    --r
qu
st-rat
 10
```
    - [ ] **Str
ss T
st

g**: T
st b
yo
d 
xp
ct
d 
oad
  - Id

t
fy br
ak

g po

ts
  - T
st r
cov
ry proc
dur
s
### I
t
grat
o
 T
st

g
    - [ ] **API Compat
b


ty**: V
r
fy Op

AI API compat
b


ty
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
r $API_KEY" \
    -d '{"mod

": "your-mod

", "m
ssag
s": [{"ro

": "us
r", "co
t

t": "t
st"}]}'
```
    - [ ] **C



t L
brar

s**: T
st 

th popu
ar c



ts
  - Op

AI Pytho
 c



t
  - La
gCha

 

t
grat
o

  - Custom c



ts
## Post-D
p
oym

t
### Va

dat
o

    - [ ] **H
a
th Ch
cks**: V
r
fy s
rv
c
 
s h
a
thy
  ```bash
  cur
 http://
oca
host:8000/h
a
th
```
    - [ ] **Bas
c Fu
ct
o
a

ty**: T
st bas
c 

f
r

c

  ```bash
  cur
 http://
oca
host:8000/v1/comp

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
    -d '{"mod

": "your-mod

", "prompt": "H

o", "max_tok

s": 10}'
```
### Docum

tat
o

    - [ ] **Ru
book**: Docum

t op
rat
o
a
 proc
dur
s
  - D
p
oym

t st
ps
  - Ro
back proc
dur
s
  - Troub

shoot

g gu
d
s
  - Em
rg

cy co
tacts
    - [ ] **API Docum

tat
o
**: E
sur
 API docs ar
 acc
ss
b


  - Op

API/S
agg
r sp
c
f
cat
o

  - Examp

 r
qu
sts a
d r
spo
s
s
## Ma

t

a
c

### Updat
s
    - [ ] **vLLM Updat
s**: P
a
 for r
gu
ar updat
s
  - T
st 


 v
rs
o
s 

 stag

g
  - R
v


 cha
g

og for br
ak

g cha
g
s
    - [ ] **Mod

 Updat
s**: P
a
 for mod

 v
rs
o
 updat
s
  - A/B t
st

g for 


 mod

 v
rs
o
s
  - Gradua
 ro
out proc
dur
s
### Opt
m
zat
o

    - [ ] **P
rforma
c
 R
v


**: R
gu
ar p
rforma
c
 a
a
ys
s
  - R
v


 m
tr
cs 

k
y/mo
th
y
  - Id

t
fy opt
m
zat
o
 opportu

t

s
    - [ ] **Cost Opt
m
zat
o
**: Mo

tor a
d opt
m
z
 costs
  - R
ght-s
z
 GPU 

sta
c
s
  - Opt
m
z
 batch s
z
s for throughput
## Troub

shoot

g Pr
parat
o

### D
bug Mod

    - [ ] **E
ab

 D
bug Logg

g**: K
o
 ho
 to 

ab

 
h

 

d
d
  ```bash
  
xport VLLM_LOGGING_LEVEL=DEBUG
  
xport VLLM_TRACE_FUNCTION=1  # Extr
m
 d
bugg

g
```
    - [ ] **D
ag
ost
c Too
s**: Pr
par
 d
ag
ost
c scr
pts
  - GPU h
a
th ch
ck scr
pts
  - N
t
ork co

ct
v
ty t
sts
  - P
rforma
c
 b

chmark

g
### Commo
 Issu
s
Fam


ar
z
 yours

f 

th:
    - [Out of M
mory](../usag
/troub

shoot

g.md#out-of-m
mory) so
ut
o
s
    - [CUDA Errors](../usag
/troub

shoot

g.md#
rror-

ar-s

fgraphr
p
ay) ha
d


g
    - [N
t
ork S
tup](../usag
/troub

shoot

g.md#

corr
ct-

t
ork-s
tup) 
ssu
s
## S
 A
so
    - [Dock
r D
p
oym

t](./dock
r.md)
    - [Kub
r

t
s D
p
oym

t](./k8s.md)
    - [P
rforma
c
 Tu


g](../usag
/p
rforma
c
_tu


g.md)
    - [Troub

shoot

g Gu
d
](../usag
/troub

shoot

g.md)
