# Mod

 Card T
mp
at

Th
s t
mp
at
 h

ps you cr
at
 compr
h

s
v
 docum

tat
o
 for mod

s s
rv
d 

th vLLM.
## Mod

 I
format
o

```yam

Mod

 Nam
: [Your Mod

 Nam
]
Mod

 V
rs
o
: [
.g., 1.0.0]
vLLM V
rs
o
 T
st
d: [
.g., 0.8.0]
Last Updat
d: [YYYY-MM-DD]
```
## Mod

 D
scr
pt
o

**Arch
t
ctur
:** [
.g., Tra
sform
r, LLaMA, GPT]
**Param
t
rs:** [
.g., 7B, 13B, 70B]
**Co
t
xt L

gth:** [
.g., 4096, 8192, 32768]
**Mod

 Typ
:** [
.g., Bas
, I
struct, Chat]
**I
t

d
d Us
:**
    - Pr
mary us
 cas
s
    - Targ
t aud


c

    - Support
d 
a
guag
s
## vLLM Co
f
gurat
o

### Bas
c Usag

```pytho

from v
m 
mport LLM, Samp


gParams

m = LLM(
    mod

="[mod

-
am
]",
    t

sor_para


_s
z
=1,  # Adjust bas
d o
 GPU m
mory
    gpu_m
mory_ut


zat
o
=0.9,
    max_mod

_


=4096,  # Adjust bas
d o
 mod

 capab


t

s
)
samp


g_params = Samp


gParams(
    t
mp
ratur
=0.7,
    top_p=0.95,
    max_tok

s=256,
)
```
### R
comm

d
d S
tt

gs
| Co
f
gurat
o
 | Va
u
 | Not
s |
|--------------|-------|-------|
| `t

sor_para


_s
z
` | 1-8 | Bas
d o
 mod

 s
z
 a
d ava

ab

 GPUs |
| `gpu_m
mory_ut


zat
o
` | 0.85-0.95 | Lo

r for shar
d r
sourc
s |
| `max_mod

_


` | Mod

 sp
c
f
c | Ch
ck mod

 card |
| `dtyp
` | auto | Us
 fp16/bf16 for 
ff
c


cy |
| `qua
t
zat
o
` | No

/fp8/a
q | Bas
d o
 m
mory co
stra

ts |
### Adva
c
d Co
f
gurat
o

```pytho


m = LLM(
    mod

="[mod

-
am
]",
    t

sor_para


_s
z
=2,
    gpu_m
mory_ut


zat
o
=0.95,
    max_mod

_


=8192,
    dtyp
="auto",
    qua
t
zat
o
=No

,
    

ab

_pr
f
x_cach

g=Tru
,
    b
ock_s
z
=16,
)
```
## Hard
ar
 R
qu
r
m

ts
### M


mum R
qu
r
m

ts
    - **GPU:** [
.g., NVIDIA A10G, RTX 3090]
    - **VRAM:** [
.g., 24 GB]
    - **Syst
m RAM:** [
.g., 32 GB]
    - **Storag
:** [
.g., 20 GB for mod

 


ghts]
### R
comm

d
d S
tup
    - **GPU:** [
.g., NVIDIA A100, H100]
    - **VRAM:** [
.g., 40-80 GB]
    - **Syst
m RAM:** [
.g., 64+ GB]
    - **Storag
:** [
.g., NVM
 SSD]
### Mu
t
-GPU Co
f
gurat
o

| Mod

 S
z
 | GPU S
tup | T

sor Para


 |
|-----------|-----------|-----------------|
| 7B | 1x A10G (24GB) | 1 |
| 13B | 1x A100 (40GB) | 1 |
| 70B | 2x A100 (80GB) | 2 |
| 70B | 4x A10G (24GB) | 4 |
## P
rforma
c
 B

chmarks
### Throughput (tok

s/s
co
d)
| Batch S
z
 | I
put L

gth | Output L

gth | Tok

s/s
c |
|-----------|-------------|---------------|-----------|
| 1 | 512 | 128 | [X] |
| 8 | 512 | 128 | [X] |
| 32 | 512 | 128 | [X] |
| 64 | 512 | 128 | [X] |
### Lat

cy (ms)
| M
tr
c | Va
u
 |
|--------|-------|
| T
m
 to F
rst Tok

 (TTFT) | [X] ms |
| T
m
 P
r Output Tok

 (TPOT) | [X] ms |
| E
d-to-

d (512 

put, 128 output) | [X] ms |
## API Examp

s
### Op

AI-Compat
b

 API
```bash
# Start s
rv
r
v
m s
rv
 [mod

-
am
] \
  --t

sor-para


-s
z
 1 \
  --max-mod

-


 4096
# Chat comp

t
o

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
  -d '{
    "mod

": "[mod

-
am
]",
    "m
ssag
s": [
      {"ro

": "syst
m", "co
t

t": "You ar
 a h

pfu
 ass
sta
t."},
      {"ro

": "us
r", "co
t

t": "H

o!"}
    ],
    "t
mp
ratur
": 0.7,
    "max_tok

s": 256
  }'
```
### Pytho
 C



t
```pytho

from op

a
 
mport Op

AI
c



t = Op

AI(
    bas
_ur
="http://
oca
host:8000/v1",
    ap
_k
y="dummy"
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

="[mod

-
am
]",
    m
ssag
s=[
        {"ro

": "us
r", "co
t

t": "Exp
a

 qua
tum comput

g"}
    ],
    t
mp
ratur
=0.7,
    max_tok

s=512
)
pr

t(r
spo
s
.cho
c
s[0].m
ssag
.co
t

t)
```
## K
o

 Issu
s a
d L
m
tat
o
s
### Curr

t L
m
tat
o
s
    - [ ] L
st a
y k
o

 

m
tat
o
s
    - [ ] M
mory r
qu
r
m

ts
    - [ ] Sp
c
f
c f
atur
 

m
tat
o
s
### Workarou
ds
| Issu
 | Workarou
d |
|-------|-----------|
| [Issu
 d
scr
pt
o
] | [So
ut
o
] |
## Troub

shoot

g
### Commo
 Issu
s
1. **Out of M
mory:**
   ```bash
   v
m s
rv
 [mod

-
am
] --gpu-m
mory-ut


zat
o
 0.8
```
2. **S
o
 

f
r

c
:**
   - E
ab

 pr
f
x cach

g: `--

ab

-pr
f
x-cach

g`
   - Adjust batch s
z
: `--max-
um-s
qs 256`
3. **Mod

 
oad

g 
rrors:**
   - V
r
fy mod

 
am
/path
   - Ch
ck ava

ab

 d
sk spac

   - E
sur
 compat
b

 vLLM v
rs
o

## B
st Pract
c
s
### Product
o
 D
p
oym

t
1. **E
ab

 pr
f
x cach

g** for r
p
at
d prompts
2. **S
t appropr
at
 batch s
z
** bas
d o
 
at

cy r
qu
r
m

ts
3. **Mo

tor GPU ut


zat
o
** a
d m
mory usag

4. **Us
 qua
t
zat
o
** 
f m
mory co
stra


d
5. **Imp

m

t prop
r 
rror ha
d


g**
### Opt
m
zat
o
 T
ps
    - Us
 `fp16` or `bf16` for fast
r 

f
r

c

    - E
ab

 chu
k
d pr
f

 for b
tt
r 

t
ract
v
ty
    - Tu

 `max_
um_s
qs` for your 
ork
oad
    - Us
 t

sor para



sm for 
arg
 mod

s
## V
rs
o
 H
story
| V
rs
o
 | Dat
 | Cha
g
s |
|---------|------|---------|
| 1.0.0 | YYYY-MM-DD | I

t
a
 r


as
 |
## R
f
r

c
s
    - Mod

 Pap
r: [L

k]
    - Hugg

gFac
 R
pos
tory: [L

k]
    - vLLM Docum

tat
o
: [L

k]
## L
c

s

[Sp
c
fy mod

 

c

s
]
---
*Th
s mod

 card 
s ma

ta


d by [Your Nam
/Orga

zat
o
]. For 
ssu
s or updat
s, p

as
 op

 a
 
ssu
 o
 G
tHub.*
