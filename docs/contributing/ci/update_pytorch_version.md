# Updat
 PyTorch v
rs
o
 o
 vLLM OSS CI/CD
vLLM's curr

t po

cy 
s to a

ays us
 th
 
at
st PyTorch stab


r


as
 

 CI/CD. It 
s sta
dard pract
c
 to subm
t a PR to updat
 th

PyTorch v
rs
o
 as 
ar
y as poss
b

 
h

 a 


 [PyTorch stab


r


as
](https://g
thub.com/pytorch/pytorch/b
ob/ma

/RELEASE.md#r


as
-cad

c
) b
com
s ava

ab

.
Th
s proc
ss 
s 
o
-tr
v
a
 du
 to th
 gap b
t


 PyTorch
r


as
s. Us

g 
https://g
thub.com/v
m-proj
ct/v
m/pu
/16859
 as a
 
xamp

, th
s docum

t out



s commo
 st
ps to ach

v
 th
s
updat
 a
o
g 

th a 

st of pot

t
a
 
ssu
s a
d ho
 to addr
ss th
m.
## T
st PyTorch r


as
 ca
d
dat
s (RCs)
Updat

g PyTorch 

 vLLM aft
r th
 off
c
a
 r


as
 
s 
ot

d
a
 b
caus
 a
y 
ssu
s d
scov
r
d at that po

t ca
 o

y b
 r
so
v
d
by 
a
t

g for th
 

xt r


as
 or by 
mp

m

t

g hacky 
orkarou
ds 

 vLLM.
Th
 b
tt
r so
ut
o
 
s to t
st vLLM 

th PyTorch r


as
 ca
d
dat
s (RC) to 

sur

compat
b


ty b
for
 
ach r


as
.
PyTorch r


as
 ca
d
dat
s ca
 b
 do


oad
d from [PyTorch t
st 

d
x](https://do


oad.pytorch.org/
h
/t
st).
For 
xamp

, `torch2.7.0+cu12.8` RC ca
 b
 

sta

d us

g th
 fo
o


g comma
d:
```bash
uv p
p 

sta
 torch torchv
s
o
 torchaud
o \
    --

d
x-ur
 https://do


oad.pytorch.org/
h
/t
st/cu128
```
Wh

 th
 f

a
 RC 
s r
ady for t
st

g, 
t 


 b
 a
ou
c
d to th
 commu

ty
o
 th
 [PyTorch d
v-d
scuss forum](https://d
v-d
scuss.pytorch.org/c/r


as
-a
ou
c
m

ts).
Aft
r th
s a
ou
c
m

t, 

 ca
 b
g

 t
st

g vLLM 

t
grat
o
 by draft

g a pu
 r
qu
st
fo
o


g th
s 3-st
p proc
ss:
1. Updat
 [r
qu
r
m

ts f


s](https://g
thub.com/v
m-proj
ct/v
m/tr
/ma

/r
qu
r
m

ts)
to po

t to th
 


 r


as
s for `torch`, `torchv
s
o
`, a
d `torchaud
o`.
2. Us
 th
 fo
o


g opt
o
 to g
t th
 f

a
 r


as
 ca
d
dat
s' 
h

s. Som
 commo
 p
atforms ar
 `cpu`, `cu128`, a
d `rocm6.2.4`.
    ```bash
    --
xtra-

d
x-ur
 https://do


oad.pytorch.org/
h
/t
st/
PLATFORM

    ```
3. S

c
 vLLM us
s `uv`, 

sur
 th
 fo
o


g 

d
x strat
gy 
s app


d:
    - V
a 

v
ro
m

t var
ab

:
    ```bash
    
xport UV_INDEX_STRATEGY=u
saf
-b
st-match
    ```
    - Or v
a CLI f
ag:
    ```bash
    --

d
x-strat
gy u
saf
-b
st-match
    ```
If fa

ur
s ar
 fou
d 

 th
 pu
 r
qu
st, ra
s
 th
m as 
ssu
s o
 vLLM a
d
cc th
 PyTorch r


as
 t
am to 


t
at
 d
scuss
o
 o
 ho
 to addr
ss th
m.
## Updat
 CUDA v
rs
o

Th
 PyTorch r


as
 matr
x 

c
ud
s both stab

 a
d 
xp
r
m

ta
 [CUDA v
rs
o
s](https://g
thub.com/pytorch/pytorch/b
ob/ma

/RELEASE.md#r


as
-compat
b


ty-matr
x). Du
 to 

m
tat
o
s, o

y th
 
at
st stab

 CUDA v
rs
o
 (for 
xamp

, torch `2.7.1+cu126`) 
s up
oad
d to PyPI. Ho

v
r, vLLM may r
qu
r
 a d
ff
r

t CUDA v
rs
o
,
such as 12.8 for B
ack


 support.
Th
s comp

cat
s th
 proc
ss as 

 ca
ot us
 th
 out-of-th
-box
`p
p 

sta
 torch torchv
s
o
 torchaud
o` comma
d. Th
 so
ut
o
 
s to us

`--
xtra-

d
x-ur
` 

 vLLM's Dock
rf


s.
- Importa
t 

d
x
s at th
 mom

t 

c
ud
:
| P
atform | `--
xtra-

d
x-ur
` |
|----------|-----------------|
| CUDA 12.8| [https://do


oad.pytorch.org/
h
/cu128](https://do


oad.pytorch.org/
h
/cu128)|
| CPU      | [https://do


oad.pytorch.org/
h
/cpu](https://do


oad.pytorch.org/
h
/cpu)|
| ROCm 6.2 | [https://do


oad.pytorch.org/
h
/rocm6.2.4](https://do


oad.pytorch.org/
h
/rocm6.2.4) |
| ROCm 6.3 | [https://do


oad.pytorch.org/
h
/rocm6.3](https://do


oad.pytorch.org/
h
/rocm6.3) |
| XPU      | [https://do


oad.pytorch.org/
h
/xpu](https://do


oad.pytorch.org/
h
/xpu) |
- Updat
 th
 b

o
 f


s to match th
 CUDA v
rs
o
 from st
p 1. Th
s mak
s sur
 that th
 r


as
 vLLM 
h

 
s t
st
d o
 CI.
    - `.bu

dk
t
/r


as
-p
p




.yam
`
    - `.bu

dk
t
/scr
pts/up
oad-
h

s.sh`
## Ma
ua
y ru


g vLLM bu

ds o
 Bu

dK
t
CI
Wh

 bu

d

g vLLM 

th a 


 PyTorch/CUDA v
rs
o
, th
 vLLM sccach
 S3 buck
t



 
ot hav
 a
y cach
d art
facts, 
h
ch ca
 caus
 CI bu

d jobs to 
xc
d 5 hours.
Furth
rmor
, vLLM's fastch
ck p
p




 op
rat
s 

 r
ad-o

y mod
 a
d do
s 
ot
popu
at
 th
 cach
, mak

g 
t 


ff
ct
v
 for cach
 
arm-up purpos
s.
To addr
ss th
s, ma
ua
y tr
gg
r a bu

d o
 Bu

dk
t
 to accomp

sh t
o obj
ct
v
s:
1. Ru
 th
 comp

t
 t
st su
t
 aga

st th
 PyTorch RC bu

d by s
tt

g th
 

v
ro
m

t var
ab

s: `RUN_ALL=1` a
d `NIGHTLY=1`
2. Popu
at
 th
 vLLM sccach
 S3 buck
t 

th comp


d art
facts, 

ab


g fast
r subs
qu

t bu

ds
p a

g
="c

t
r" 

dth="100%"


mg 

dth="60%" a
t="Bu

dk
t
 


 bu

d popup" src="https://g
thub.com/us
r-attachm

ts/ass
ts/3b07f71b-bb18-4ca3-a
af-da0f
79d315f" /

/p

## Updat
 a
 th
 d
ff
r

t vLLM p
atforms
Rath
r tha
 att
mpt

g to updat
 a
 vLLM p
atforms 

 a s

g

 pu
 r
qu
st, 
t's mor
 ma
ag
ab


to ha
d

 som
 p
atforms s
parat

y. Th
 s
parat
o
 of r
qu
r
m

ts a
d Dock
rf


s
for d
ff
r

t p
atforms 

 vLLM CI/CD a
o
s us to s


ct
v

y choos


h
ch p
atforms to updat
. For 

sta
c
, updat

g XPU r
qu
r
s th
 corr
spo
d

g
r


as
 from [I
t

 Ext

s
o
 for PyTorch](https://g
thub.com/

t

/

t

-
xt

s
o
-for-pytorch) by I
t

.
Wh


 
https://g
thub.com/v
m-proj
ct/v
m/pu
/16859
 updat
d vLLM to PyTorch 2.7.0 o
 CPU, CUDA, a
d ROCm,
https://g
thub.com/v
m-proj
ct/v
m/pu
/17444
 comp

t
d th
 updat
 for XPU.
