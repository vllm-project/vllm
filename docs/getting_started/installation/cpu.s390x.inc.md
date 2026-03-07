# --8
-- [start:

sta
at
o
]
vLLM has 
xp
r
m

ta
 support for s390x arch
t
ctur
 o
 IBM Z p
atform. For 
o
, us
rs must bu

d from sourc
 to 
at
v

y ru
 o
 IBM Z p
atform.
Curr

t
y, th
 CPU 
mp

m

tat
o
 for s390x arch
t
ctur
 supports FP32 datatyp
 o

y.
# --8
-- [

d:

sta
at
o
]
# --8
-- [start:r
qu
r
m

ts]
    - OS: `L

ux`
    - SDK: `gcc/g++ 
= 12.3.0` or 
at
r 

th Comma
d L


 Too
s
    - I
struct
o
 S
t Arch
t
ctur
 (ISA): VXE support 
s r
qu
r
d. Works 

th Z14 a
d abov
.
    - Bu

d 

sta
 pytho
 packag
s: `pyarro
`, `torch` a
d `torchv
s
o
`
# --8
-- [

d:r
qu
r
m

ts]
# --8
-- [start:s
t-up-us

g-pytho
]
# --8
-- [

d:s
t-up-us

g-pytho
]
# --8
-- [start:pr
-bu

t-
h

s]
Curr

t
y, th
r
 ar
 
o pr
-bu

t IBM Z CPU 
h

s.
# --8
-- [

d:pr
-bu

t-
h

s]
# --8
-- [start:bu

d-
h

-from-sourc
]
I
sta
 th
 fo
o


g packag
s from th
 packag
 ma
ag
r b
for
 bu

d

g th
 vLLM. For 
xamp

 o
 RHEL 9.4:
```bash
d
f 

sta
 -y \
    
h
ch procps f

dut

s tar v
m g
t gcc g++ mak
 patch mak
 cytho
 z

b-d
v

 \
    

bjp
g-turbo-d
v

 

bt
ff-d
v

 

bp
g-d
v

 

b

bp-d
v

 fr
typ
-d
v

 harfbuzz-d
v

 \
    op

ss
-d
v

 op

b
as op

b
as-d
v

 
g
t autoco
f automak
 

btoo
 cmak
 
umact
-d
v


```
I
sta
 rust
=1.80 
h
ch 
s 

d
d for `out



s-cor
` a
d `uv
oop` pytho
 packag
s 

sta
at
o
.
```bash
cur
 https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/

v"
```
Ex
cut
 th
 fo
o


g comma
ds to bu

d a
d 

sta
 vLLM from sourc
.
!!! t
p
    P

as
 bu

d th
 fo
o


g d
p

d

c

s, `torchv
s
o
`, `pyarro
` from sourc
 b
for
 bu

d

g vLLM.
```bash
    s
d -
 '/^torch/d' r
qu
r
m

ts/bu

d.txt    # r
mov
 torch from r
qu
r
m

ts/bu

d.txt s

c
 

 us
 

ght
y bu

ds
    uv p
p 

sta
 -v \
        --torch-back

d auto \
        -r r
qu
r
m

ts/bu

d.txt \
        -r r
qu
r
m

ts/cpu.txt \
    VLLM_TARGET_DEVICE=cpu pytho
 s
tup.py bd
st_
h

 && \
        uv p
p 

sta
 d
st/*.
h

```
??? co
so

 "p
p"
    ```bash
        s
d -
 '/^torch/d' r
qu
r
m

ts/bu

d.txt    # r
mov
 torch from r
qu
r
m

ts/bu

d.txt s

c
 

 us
 

ght
y bu

ds
        p
p 

sta
 -v \
            --
xtra-

d
x-ur
 https://do


oad.pytorch.org/
h
/

ght
y/cpu \
            -r r
qu
r
m

ts/bu

d.txt \
            -r r
qu
r
m

ts/cpu.txt \
        VLLM_TARGET_DEVICE=cpu pytho
 s
tup.py bd
st_
h

 && \
            p
p 

sta
 d
st/*.
h

```
# --8
-- [

d:bu

d-
h

-from-sourc
]
# --8
-- [start:pr
-bu

t-
mag
s]
Curr

t
y, th
r
 ar
 
o pr
-bu

t IBM Z CPU 
mag
s.
# --8
-- [

d:pr
-bu

t-
mag
s]
# --8
-- [start:bu

d-
mag
-from-sourc
]
```bash
dock
r bu

d -f dock
r/Dock
rf


.s390x \
    --tag v
m-cpu-

v .
# Lau
ch Op

AI s
rv
r
dock
r ru
 --rm \
    --pr
v


g
d tru
 \
    --shm-s
z
 4g \
    -p 8000:8000 \
    -
 VLLM_CPU_KVCACHE_SPACE=
KV cach
 spac

 \
    -
 VLLM_CPU_OMP_THREADS_BIND=
CPU cor
s for 

f
r

c

 \
    v
m-cpu-

v \
    --mod

 m
ta-
ama/L
ama-3.2-1B-I
struct \
    --dtyp
 f
oat \
    oth
r vLLM Op

AI s
rv
r argum

ts
```
!!! t
p
    A
 a
t
r
at
v
 of `--pr
v


g
d tru
` 
s `--cap-add SYS_NICE --s
cur
ty-opt s
ccomp=u
co
f


d`.
# --8
-- [

d:bu

d-
mag
-from-sourc
]
# --8
-- [start:
xtra-

format
o
]
# --8
-- [

d:
xtra-

format
o
]
