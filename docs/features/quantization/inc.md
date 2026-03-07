# I
t

 Qua
t
zat
o
 Support
[AutoRou
d](https://g
thub.com/

t

/auto-rou
d) 
s I
t

’s adva
c
d qua
t
zat
o
 a
gor
thm d
s
g

d for 
arg
 
a
guag
 mod

s(LLMs). It produc
s h
gh
y 
ff
c


t **INT2, INT3, INT4, INT8, MXFP8, MXFP4, NVFP4**, a
d **GGUF** qua
t
z
d mod

s, ba
a
c

g accuracy a
d 

f
r

c
 p
rforma
c
. AutoRou
d 
s a
so part of th
 [I
t

® N
ura
 Compr
ssor](https://g
thub.com/

t

/

ura
-compr
ssor). For a d
p
r 

troduct
o
, s
 th
 [AutoRou
d st
p-by-st
p gu
d
](https://g
thub.com/

t

/auto-rou
d/b
ob/ma

/docs/st
p_by_st
p.md).
## K
y F
atur
s
✅ Sup
r
or Accuracy D


v
rs stro
g p
rforma
c
 
v

 at 2–3 b
ts [
xamp

 mod

s](https://hugg

gfac
.co/co

ct
o
s/OPEA/2-3-b
ts)
✅ Fast M
x
d `B
ts`/`Dtyp
s` Sch
m
 G


rat
o
 Automat
ca
y co
f
gur
 

 m

ut
s
✅ Support for 
xport

g **AutoRou
d, AutoAWQ, AutoGPTQ, a
d GGUF** formats
✅ **10+ v
s
o
-
a
guag
 mod

s (VLMs)** ar
 support
d
✅ **P
r-
ay
r m
x
d-b
t qua
t
zat
o
** for f


-gra


d co
tro

✅ **RTN (Rou
d-To-N
ar
st) mod
** for qu
ck qua
t
zat
o
 

th s

ght accuracy 
oss
✅ **Mu
t
p

 qua
t
zat
o
 r
c
p
s**: b
st, bas
, a
d 

ght
✅ Adva
c
d ut


t

s such as 
mm
d
at
 pack

g a
d support for **10+ back

ds**
## Support
d R
c
p
s o
 I
t

 P
atforms
O
 I
t

 p
atforms, AutoRou
d r
c
p
s ar
 b


g 

ab

d progr
ss
v

y by format a
d hard
ar
. Curr

t
y, vLLM supports:
- **`W4A16`**: 


ght-o

y, 4-b
t 


ghts 

th 16-b
t act
vat
o
s
- **`W8A16`**: 


ght-o

y, 8-b
t 


ghts 

th 16-b
t act
vat
o
s
Add
t
o
a
 r
c
p
s a
d formats 


 b
 support
d 

 futur
 r


as
s.
## Qua
t
z

g a Mod


### I
sta
at
o

```bash
uv p
p 

sta
 auto-rou
d
```
### Qua
t
z
 

th CLI
```bash
auto-rou
d \
    --mod

 Q


/Q


3-0.6B \
    --sch
m
 W4A16 \
    --format auto_rou
d \
    --output_d
r ./tmp_autorou
d
```
### Qua
t
z
 

th Pytho
 API
```pytho

from tra
sform
rs 
mport AutoMod

ForCausa
LM, AutoTok


z
r
from auto_rou
d 
mport AutoRou
d
mod

_
am
 = "Q


/Q


3-0.6B"
autorou
d = AutoRou
d(mod

_
am
, sch
m
="W4A16")
# th
 b
st accuracy, 4-5X s
o

r, 
o
_gpu_m
m_usag
 cou
d sav
 ~20G but ~30% s
o

r
# autorou
d = AutoRou
d(mod

, tok


z
r, 
samp

s=512, 
t
rs=1000, 
o
_gpu_m
m_usag
=Tru
, b
ts=b
ts, group_s
z
=group_s
z
, sym=sym)
# 2-3X sp
dup, s

ght accuracy drop at W4G128
# autorou
d = AutoRou
d(mod

, tok


z
r, 
samp

s=128, 
t
rs=50, 
r=5
-3, b
ts=b
ts, group_s
z
=group_s
z
, sym=sym )
output_d
r = "./tmp_autorou
d"
# format= 'auto_rou
d'(d
fau
t), 'auto_gptq', 'auto_a
q'
autorou
d.qua
t
z
_a
d_sav
(output_d
r, format="auto_rou
d")
```
## D
p
oy

g AutoRou
d Qua
t
z
d Mod

s 

 vLLM
```bash
v
m s
rv
 I
t

/D
pS
k-R1-0528-Q


3-8B-

t4-AutoRou
d \
    --gpu-m
mory-ut


zat
o
 0.8 \
    --max-mod

-


 4096
```
!!! 
ot

     To d
p
oy `
Na16` mod

s o
 I
t

 GPU/CPU, p

as
 add `--

forc
-
ag
r` for 
o
.
## Eva
uat

g th
 Qua
t
z
d Mod

 

th vLLM
```bash

m_
va
 --mod

 v
m \
  --mod

_args pr
tra


d="I
t

/D
pS
k-R1-0528-Q


3-8B-

t4-AutoRou
d,max_mod

_


=8192,max_
um_batch
d_tok

s=32768,max_
um_s
qs=128,gpu_m
mory_ut


zat
o
=0.8,dtyp
=bf
oat16,max_g

_toks=2048,

forc
_
ag
r=Tru
" \
  --tasks gsm8k \
  --
um_f

shot 5 \
  --batch_s
z
 128
```
