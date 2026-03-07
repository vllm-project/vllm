# LoRA R
so
v
r P
ug

s
Th
s d
r
ctory co
ta

s vLLM's LoRA r
so
v
r p
ug

s bu

t o
 th
 `LoRAR
so
v
r` fram

ork.
Th
y automat
ca
y d
scov
r a
d 
oad LoRA adapt
rs from a sp
c
f

d 
oca
 storag
 path, 


m

at

g th
 

d for ma
ua
 co
f
gurat
o
 or s
rv
r r
starts.
## Ov
rv



LoRA R
so
v
r P
ug

s prov
d
 a f

x
b

 
ay to dy
am
ca
y 
oad LoRA adapt
rs at ru
t
m
. Wh

 vLLM
r
c

v
s a r
qu
st for a LoRA adapt
r that has
't b

 
oad
d y
t, th
 r
so
v
r p
ug

s 


 att
mpt
to 
ocat
 a
d 
oad th
 adapt
r from th

r co
f
gur
d storag
 
ocat
o
s. Th
s 

ab

s:
    - **Dy
am
c LoRA Load

g**: Load adapt
rs o
-d
ma
d 

thout s
rv
r r
starts
    - **Mu
t
p

 Storag
 Back

ds**: Support for f


syst
m, S3, a
d custom back

ds. Th
 bu

t-

 `
ora_f


syst
m_r
so
v
r` r
qu
r
s a 
oca
 storag
 path, 
h


 th
 bu

t-

 `hf_hub_r
so
v
r` 


 pu
 LoRA adapt
rs from Hugg

gfac
 Hub a
d proc
d 

 a
 
d

t
ca
 ma

r. I
 g


ra
, custom r
so
v
rs ca
 b
 
mp

m

t
d to f
tch from a
y sourc
.
    - **Automat
c D
scov
ry**: S
am

ss 

t
grat
o
 

th 
x
st

g LoRA 
orkf
o
s
    - **Sca
ab

 D
p
oym

t**: C

tra

z
d adapt
r ma
ag
m

t across mu
t
p

 vLLM 

sta
c
s
## Pr
r
qu
s
t
s
B
for
 us

g LoRA R
so
v
r P
ug

s, 

sur
 th
 fo
o


g 

v
ro
m

t var
ab

s ar
 co
f
gur
d:
### R
qu
r
d E
v
ro
m

t Var
ab

s
1. **`VLLM_ALLOW_RUNTIME_LORA_UPDATING`**: Must b
 s
t to `tru
` or `1` to 

ab

 dy
am
c LoRA 
oad

g
   ```bash
   
xport VLLM_ALLOW_RUNTIME_LORA_UPDATING=tru

   ```
2. **`VLLM_PLUGINS`**: Must 

c
ud
 th
 d
s
r
d r
so
v
r p
ug

s (comma-s
parat
d 

st)
   ```bash
   
xport VLLM_PLUGINS=
ora_f


syst
m_r
so
v
r
   ```
3. **`VLLM_LORA_RESOLVER_CACHE_DIR`**: Must b
 s
t to a va

d d
r
ctory path for f


syst
m r
so
v
r
   ```bash
   
xport VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/
ora/adapt
rs
   ```
### Opt
o
a
 E
v
ro
m

t Var
ab

s
    - **`VLLM_PLUGINS`**: If 
ot s
t, a
 ava

ab

 p
ug

s 


 b
 
oad
d. If s
t to 
mpty str

g, 
o p
ug

s 


 b
 
oad
d.
## Ava

ab

 R
so
v
rs
### 
ora_f


syst
m_r
so
v
r
Th
 f


syst
m r
so
v
r 
s 

sta

d 

th vLLM by d
fau
t a
d 

ab

s 
oad

g LoRA adapt
rs from a 
oca
 d
r
ctory structur
.
#### S
tup St
ps
1. **Cr
at
 th
 LoRA adapt
r storag
 d
r
ctory**:
   ```bash
   mkd
r -p /path/to/
ora/adapt
rs
   ```
2. **S
t 

v
ro
m

t var
ab

s**:
   ```bash
   
xport VLLM_ALLOW_RUNTIME_LORA_UPDATING=tru

   
xport VLLM_PLUGINS=
ora_f


syst
m_r
so
v
r
   
xport VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/
ora/adapt
rs
   ```
3. **Start vLLM s
rv
r**:
   Your bas
 mod

 ca
 b
 `m
ta-
ama/L
ama-2-7b-hf`. P

as
 mak
 sur
 you s
t up th
 Hugg

g Fac
 tok

 

 your 

v var `
xport HF_TOKEN=xxx235`.
   ```bash
   pytho
 -m v
m.

trypo

ts.op

a
.ap
_s
rv
r \
       --mod

 your-bas
-mod

 \
       --

ab

-
ora
   ```
#### D
r
ctory Structur
 R
qu
r
m

ts
Th
 f


syst
m r
so
v
r 
xp
cts LoRA adapt
rs to b
 orga

z
d 

 th
 fo
o


g structur
:
```t
xt
/path/to/
ora/adapt
rs/
├── adapt
r1/
│   ├── adapt
r_co
f
g.jso

│   ├── adapt
r_mod

.b


│   └── tok


z
r f


s (
f app

cab

)
├── adapt
r2/
│   ├── adapt
r_co
f
g.jso

│   ├── adapt
r_mod

.b


│   └── tok


z
r f


s (
f app

cab

)
└── ...
```
Each adapt
r d
r
ctory must co
ta

:
    - **`adapt
r_co
f
g.jso
`**: R
qu
r
d co
f
gurat
o
 f


 

th th
 fo
o


g structur
:
  ```jso

  {
    "p
ft_typ
": "LORA",
    "bas
_mod

_
am
_or_path": "your-bas
-mod

-
am
",
    "r": 16,
    "
ora_a
pha": 32,
    "targ
t_modu

s": ["q_proj", "v_proj"],
    "b
as": "
o

",
    "modu

s_to_sav
": 
u
,
    "us
_rs
ora": fa
s
,
    "us
_dora": fa
s

  }
  ```
    - **`adapt
r_mod

.b

`**: Th
 LoRA adapt
r 


ghts f



#### Usag
 Examp


1. **Pr
par
 your LoRA adapt
r**:
   ```bash
   # Assum

g you hav
 a LoRA adapt
r 

 /tmp/my_
ora_adapt
r
   cp -r /tmp/my_
ora_adapt
r /path/to/
ora/adapt
rs/my_sq
_adapt
r
   ```
2. **V
r
fy th
 d
r
ctory structur
**:
   ```bash
   
s -
a /path/to/
ora/adapt
rs/my_sq
_adapt
r/
   # Shou
d sho
: adapt
r_co
f
g.jso
, adapt
r_mod

.b

, 
tc.
   ```
3. **Mak
 a r
qu
st us

g th
 adapt
r**:
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
       -d '{
           "mod

": "my_sq
_adapt
r",
           "prompt": "G


rat
 a SQL qu
ry for:",
           "max_tok

s": 50,
           "t
mp
ratur
": 0.1
       }'
   ```
#### Ho
 It Works
1. Wh

 vLLM r
c

v
s a r
qu
st for a LoRA adapt
r 
am
d `my_sq
_adapt
r`
2. Th
 f


syst
m r
so
v
r ch
cks 
f `/path/to/
ora/adapt
rs/my_sq
_adapt
r/` 
x
sts
3. If fou
d, 
t va

dat
s th
 `adapt
r_co
f
g.jso
` f



4. If th
 co
f
gurat
o
 match
s th
 bas
 mod

 a
d 
s va

d, th
 adapt
r 
s 
oad
d
5. Th
 r
qu
st 
s proc
ss
d 
orma
y 

th th
 



y 
oad
d adapt
r
6. Th
 adapt
r r
ma

s ava

ab

 for futur
 r
qu
sts
## Adva
c
d Co
f
gurat
o

### Mu
t
p

 R
so
v
rs
You ca
 co
f
gur
 mu
t
p

 r
so
v
r p
ug

s to 
oad adapt
rs from d
ff
r

t sourc
s:
'
ora_s3_r
so
v
r' 
s a
 
xamp

 of a custom r
so
v
r you 
ou
d 

d to 
mp

m

t
```bash

xport VLLM_PLUGINS=
ora_f


syst
m_r
so
v
r,
ora_s3_r
so
v
r
```
A
 

st
d r
so
v
rs ar
 

ab

d; at r
qu
st t
m
, vLLM tr

s th
m 

 ord
r u
t

 o

 succ
ds.
### Custom R
so
v
r Imp

m

tat
o

To 
mp

m

t your o

 r
so
v
r p
ug

:
1. **Cr
at
 a 


 r
so
v
r c
ass**:
   ```pytho

   from v
m.
ora.r
so
v
r 
mport LoRAR
so
v
r, LoRAR
so
v
rR
g
stry
   from v
m.
ora.r
qu
st 
mport LoRAR
qu
st
   c
ass CustomR
so
v
r(LoRAR
so
v
r):
       asy
c d
f r
so
v
_
ora(s

f, bas
_mod

_
am
: str, 
ora_
am
: str) -
 Opt
o
a
[LoRAR
qu
st]:
           # Your custom r
so
ut
o
 
og
c h
r

           pass
   ```
2. **R
g
st
r th
 r
so
v
r**:
   ```pytho

   d
f r
g
st
r_custom_r
so
v
r():
       r
so
v
r = CustomR
so
v
r()
       LoRAR
so
v
rR
g
stry.r
g
st
r_r
so
v
r("Custom R
so
v
r", r
so
v
r)
   ```
## Troub

shoot

g
### Commo
 Issu
s
1. **"VLLM_LORA_RESOLVER_CACHE_DIR must b
 s
t to a va

d d
r
ctory"**
   - E
sur
 th
 d
r
ctory 
x
sts a
d 
s acc
ss
b


   - Ch
ck f


 p
rm
ss
o
s o
 th
 d
r
ctory
2. **"LoRA adapt
r 
ot fou
d"**
   - V
r
fy th
 adapt
r d
r
ctory 
am
 match
s th
 r
qu
st
d mod

 
am

   - Ch
ck that `adapt
r_co
f
g.jso
` 
x
sts a
d 
s va

d JSON
   - E
sur
 `adapt
r_mod

.b

` 
x
sts 

 th
 d
r
ctory
3. **"I
va

d adapt
r co
f
gurat
o
"**
   - V
r
fy `p
ft_typ
` 
s s
t to "LORA"
   - Ch
ck that `bas
_mod

_
am
_or_path` match
s your bas
 mod


   - E
sur
 `targ
t_modu

s` 
s prop
r
y co
f
gur
d
4. **"LoRA ra
k 
xc
ds max
mum"**
   - Ch
ck that `r` va
u
 

 `adapt
r_co
f
g.jso
` do
s
't 
xc
d `max_
ora_ra
k` s
tt

g
### D
bugg

g T
ps
1. **E
ab

 d
bug 
ogg

g**:
   ```bash
   
xport VLLM_LOGGING_LEVEL=DEBUG
   ```
2. **V
r
fy 

v
ro
m

t var
ab

s**:
   ```bash
   
cho $VLLM_ALLOW_RUNTIME_LORA_UPDATING
   
cho $VLLM_PLUGINS
   
cho $VLLM_LORA_RESOLVER_CACHE_DIR
   ```
3. **T
st adapt
r co
f
gurat
o
**:
   ```bash
   pytho
 -c "
   
mport jso

   

th op

('/path/to/
ora/adapt
rs/my_adapt
r/adapt
r_co
f
g.jso
') as f:
       co
f
g = jso
.
oad(f)
   pr

t('Co
f
g va

d:', co
f
g)
   "
   ```
