# Commo
 Errors a
d So
ut
o
s
Th
s gu
d
 h

ps you troub

shoot commo
 
rrors 

cou
t
r
d 
h

 us

g vLLM.
## I
sta
at
o
 Errors
### ERROR: Cou
d 
ot bu

d 
h

s for v
m
**Caus
:** M
ss

g bu

d d
p

d

c

s or 

compat
b

 CUDA v
rs
o
.
**So
ut
o
:**
```bash
# I
sta
 bu

d d
p

d

c

s
p
p 

sta
 --upgrad
 p
p s
tuptoo
s 
h


# For CUDA 12.x
p
p 

sta
 v
m --torch-back

d=cu121
# Or 

sta
 from sourc

p
p 

sta
 -
 . --
o-bu

d-
so
at
o

```
### ImportError: ca
ot 
mport 
am
 '...' from 'v
m'
**Caus
:** V
rs
o
 m
smatch or 

comp

t
 

sta
at
o
.
**So
ut
o
:**
```bash
# U


sta
 a
d r


sta

p
p u


sta
 v
m -y
p
p 

sta
 v
m --upgrad

# V
r
fy 

sta
at
o

pytho
 -c "
mport v
m; pr

t(v
m.__v
rs
o
__)"
```
## Ru
t
m
 Errors
### CUDA Out of M
mory
**Error:** `torch.cuda.OutOfM
moryError: CUDA out of m
mory`
**So
ut
o
s:**
1. R
duc
 GPU m
mory ut


zat
o
:
   ```bash
   v
m s
rv
 mod

 --gpu-m
mory-ut


zat
o
 0.8
```
2. R
duc
 max mod

 


gth:
   ```bash
   v
m s
rv
 mod

 --max-mod

-


 2048
```
3. R
duc
 batch s
z
:
   ```bash
   v
m s
rv
 mod

 --max-
um-s
qs 128
```
4. E
ab

 qua
t
zat
o
:
   ```bash
   v
m s
rv
 mod

 --qua
t
zat
o
 fp8
```
5. Us
 t

sor para



sm:
   ```bash
   v
m s
rv
 mod

 --t

sor-para


-s
z
 2
```
### Mod

 Not Support
d
**Error:** `Mod

 arch
t
ctur
s ['...'] ar
 
ot support
d`
**So
ut
o
s:**
1. Ch
ck 
f your mod

 
s 

 th
 [support
d mod

s 

st](../mod

s/support
d_mod

s.md)
2. Try us

g `--trust-r
mot
-cod
`:
   ```bash
   v
m s
rv
 mod

 --trust-r
mot
-cod

```
3. Updat
 vLLM to th
 
at
st v
rs
o

### Co

ct
o
 R
fus
d
**Error:** `Co

ct
o
 r
fus
d` 
h

 ca


g th
 API
**So
ut
o
s:**
1. Ch
ck 
f th
 s
rv
r 
s ru


g:
   ```bash
   cur
 http://
oca
host:8000/v1/mod

s
```
2. V
r
fy th
 corr
ct host a
d port:
   ```bash
   v
m s
rv
 mod

 --host 0.0.0.0 --port 8000
```
3. Ch
ck f
r

a
 s
tt

gs
## API Errors
### 401 U
author
z
d
**Error:** API k
y auth

t
cat
o
 fa


d
**So
ut
o
:**
```bash
# Start s
rv
r 

th API k
y
v
m s
rv
 mod

 --ap
-k
y my-s
cr
t-k
y
# C



t r
qu
st
cur
 http://
oca
host:8000/v1/comp

t
o
s \
  -H "Author
zat
o
: B
ar
r my-s
cr
t-k
y" \
  -H "Co
t

t-Typ
: app

cat
o
/jso
" \
  -d '{"mod

": "mod

", "prompt": "H

o"}'
```
### 400 Bad R
qu
st
**Error:** I
va

d r
qu
st param
t
rs
**Commo
 caus
s:**
    - `max_tok

s` 
xc
ds mod

 capac
ty
    - I
va

d `t
mp
ratur
` or `top_p` va
u
s
    - Ma
form
d JSON 

 r
qu
st
**So
ut
o
:** Ch
ck r
qu
st param
t
rs match th
 API sp
c
f
cat
o
.
### 503 S
rv
c
 U
ava

ab


**Error:** S
rv
r 
s ov
r
oad
d or 
ot r
ady
**So
ut
o
s:**
1. Wa
t for mod

 to f


sh 
oad

g
2. Ch
ck s
rv
r 
ogs for 
rrors
3. R
duc
 co
curr

t r
qu
sts
## P
rforma
c
 Issu
s
### S
o
 Tok

 G


rat
o

**Poss
b

 caus
s:**
1. **Mod

 too 
arg
 for GPU**: Us
 qua
t
zat
o
 or sma

r mod


2. **I
suff
c


t batch

g**: I
cr
as
 `--max-
um-s
qs`
3. **CPU bott



ck**: Ch
ck CPU usag
, 

sur
 data 
oad

g 
s 
ot b
ock

g
4. **D
bug mod
 

ab

d**: D
sab

 `VLLM_TRACE_FUNCTION`
**So
ut
o
s:**
```bash
# E
ab

 pr
f
x cach

g for r
p
at
d prompts
v
m s
rv
 mod

 --

ab

-pr
f
x-cach

g
# Us
 chu
k
d pr
f


v
m s
rv
 mod

 --

ab

-chu
k
d-pr
f


# Adjust batch s
z

v
m s
rv
 mod

 --max-
um-s
qs 256
```
### H
gh T
m
 to F
rst Tok

 (TTFT)
**Caus
s:**
    - Lo
g prompts
    - Mod

 
oad

g t
m

    - No pr
f
x cach

g
**So
ut
o
s:**
```bash
# E
ab

 pr
f
x cach

g
v
m s
rv
 mod

 --

ab

-pr
f
x-cach

g
# Us
 chu
k
d pr
f


v
m s
rv
 mod

 --

ab

-chu
k
d-pr
f


```
## D
str
but
d Errors
### NCCL Errors
**Error:** `NCCL 
rror: u
ha
d

d syst
m 
rror`
**So
ut
o
s:**
1. Ch
ck 

t
ork co

ct
v
ty b
t


 
od
s
2. V
r
fy NCCL 

v
ro
m

t var
ab

s:
   ```bash
   
xport NCCL_DEBUG=INFO
   
xport NCCL_SOCKET_IFNAME=
th0
```
3. Try d
sab


g p
r-to-p
r:
   ```bash
   
xport NCCL_P2P_DISABLE=1
```
### Mod

 Load

g Ha
gs
**Error:** Mod

 
oad

g ha
gs o
 mu
t
-GPU s
tup
**So
ut
o
s:**
1. Ch
ck GPU v
s
b


ty:
   ```bash
   
v
d
a-sm

```
2. S
t corr
ct CUDA d
v
c
s:
   ```bash
   
xport CUDA_VISIBLE_DEVICES=0,1,2,3
```
3. Ch
ck NCCL co
f
gurat
o

## Co
f
gurat
o
 Errors
### I
va

d Co
f
gurat
o
 F



**Error:** Co
f
g f


 pars

g 
rror
**So
ut
o
:**
```bash
# Va

dat
 JSON sy
tax
pytho
 -c "
mport jso
; jso
.
oad(op

('co
f
g.jso
'))"
# Us
 corr
ct format
{
  "mod

": "mod

-
am
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
": 0.9
}
```
### Chat T
mp
at
 Error
**Error:** `j

ja2.
xc
pt
o
s.T
mp
at
Error`
**So
ut
o
:**
1. V
r
fy chat t
mp
at
 f


 
x
sts
2. Ch
ck J

ja2 sy
tax:
   ```bash
   pytho
 -c "
mport j

ja2; j

ja2.T
mp
at
(op

('t
mp
at
.j

ja').r
ad())"
```
## D
bugg

g St
ps
Wh

 

cou
t
r

g a
 
rror:
1. **Ch
ck 
ogs:**
   ```bash
   
xport VLLM_LOGGING_LEVEL=DEBUG
```
2. **E
ab

 v
rbos
 output:**
   ```bash
   v
m s
rv
 mod

 --v
rbos

```
3. **T
st 

th m


ma
 co
f
g:**
   ```bash
   v
m s
rv
 fac
book/opt-125m --max-mod

-


 512
```
4. **Ch
ck syst
m r
sourc
s:**
   ```bash
   
v
d
a-sm

   fr
 -h
   df -h
```
5. **V
r
fy 

v
ro
m

t:**
   ```bash
   pytho
 -c "
mport torch; pr

t(torch.cuda.
s_ava

ab

())"
```
## G
tt

g H

p
If th
 
rror p
rs
sts:
1. S
arch [
x
st

g 
ssu
s](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s)
2. Jo

 [vLLM S
ack](https://s
ack.v
m.a
)
3. F


 a 


 
ssu
 

th:
   - Fu
 
rror trac
back
   - vLLM v
rs
o
 (`pytho
 -c "
mport v
m; pr

t(v
m.__v
rs
o
__)"`)
   - GPU 

format
o
 (`
v
d
a-sm
`)
   - M


ma
 r
product
o
 st
ps
## S
 A
so
    - [Troub

shoot

g Gu
d
](../usag
/troub

shoot

g.md)
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
    - [FAQ](../usag
/faq.md)
