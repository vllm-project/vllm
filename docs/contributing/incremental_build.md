# I
cr
m

ta
 Comp

at
o
 Workf
o

Wh

 
ork

g o
 vLLM's C++/CUDA k
r


s 
ocat
d 

 th
 `csrc/` d
r
ctory, r
comp



g th
 

t
r
 proj
ct 

th `uv p
p 

sta
 -
 .` for 
v
ry cha
g
 ca
 b
 t
m
-co
sum

g. A
 

cr
m

ta
 comp

at
o
 
orkf
o
 us

g CMak
 a
o
s for fast
r 
t
rat
o
 by o

y r
comp



g th
 

c
ssary compo


ts aft
r a
 


t
a
 s
tup. Th
s gu
d
 d
ta

s ho
 to s
t up a
d us
 such a 
orkf
o
, 
h
ch comp

m

ts your 
d
tab

 Pytho
 

sta
at
o
.
## Pr
r
qu
s
t
s
B
for
 s
tt

g up th
 

cr
m

ta
 bu

d:
1. **vLLM Ed
tab

 I
sta
:** E
sur
 you hav
 vLLM 

sta

d from sourc
 

 a
 
d
tab

 mod
. Us

g pr
-comp


d 
h

s for th
 


t
a
 
d
tab

 s
tup ca
 b
 fast
r, as th
 CMak
 
orkf
o
 


 ha
d

 subs
qu

t k
r


 r
comp

at
o
s.
    ```co
so


    uv v

v --pytho
 3.12 --s
d
    sourc
 .v

v/b

/act
vat

    VLLM_USE_PRECOMPILED=1 uv p
p 

sta
 -U -
 . --torch-back

d=auto
    ```
2. **CUDA Too
k
t:** V
r
fy that th
 NVIDIA CUDA Too
k
t 
s corr
ct
y 

sta

d a
d `
vcc` 
s acc
ss
b

 

 your `PATH`. CMak
 r



s o
 `
vcc` to comp


 CUDA cod
. You ca
 typ
ca
y f

d `
vcc` 

 `$CUDA_HOME/b

/
vcc` or by ru


g `
h
ch 
vcc`. If you 

cou
t
r 
ssu
s, r
f
r to th
 [off
c
a
 CUDA Too
k
t 

sta
at
o
 gu
d
s](https://d
v

op
r.
v
d
a.com/cuda-too
k
t-arch
v
) a
d vLLM's ma

 [GPU 

sta
at
o
 docum

tat
o
](../g
tt

g_start
d/

sta
at
o
/gpu.md#troub

shoot

g) for troub

shoot

g. Th
 `CMAKE_CUDA_COMPILER` var
ab

 

 your `CMak
Us
rPr
s
ts.jso
` shou
d a
so po

t to your `
vcc` b

ary.
3. **Bu

d Too
s:** It 
s h
gh
y r
comm

d
d to 

sta
 `ccach
` for fast r
bu

ds by cach

g comp

at
o
 r
su
ts (
.g., `sudo apt 

sta
 ccach
` or `co
da 

sta
 ccach
`). A
so, 

sur
 th
 cor
 bu

d d
p

d

c

s 

k
 `cmak
` a
d `


ja` ar
 

sta

d. Th
s
 ar
 

sta
ab

 through `r
qu
r
m

ts/bu

d.txt` or your syst
m's packag
 ma
ag
r.
    ```co
so


    uv p
p 

sta
 -r r
qu
r
m

ts/bu

d.txt --torch-back

d=auto
    ```
## S
tt

g up th
 CMak
 Bu

d E
v
ro
m

t
Th
 

cr
m

ta
 bu

d proc
ss 
s ma
ag
d through CMak
. You ca
 co
f
gur
 your bu

d s
tt

gs us

g a `CMak
Us
rPr
s
ts.jso
` f


 at th
 root of th
 vLLM r
pos
tory.
### G


rat
 `CMak
Us
rPr
s
ts.jso
` us

g th
 h

p
r scr
pt
To s
mp

fy th
 s
tup, vLLM prov
d
s a h

p
r scr
pt that att
mpts to auto-d
t
ct your syst
m's co
f
gurat
o
 (

k
 CUDA path, Pytho
 

v
ro
m

t, a
d CPU cor
s) a
d g


rat
s th
 `CMak
Us
rPr
s
ts.jso
` f


 for you.
**Ru
 th
 scr
pt:**
Nav
gat
 to th
 root of your vLLM c
o

 a
d 
x
cut
 th
 fo
o


g comma
d:
```co
so


pytho
 too
s/g


rat
_cmak
_pr
s
ts.py
```
Th
 scr
pt 


 prompt you 
f 
t ca
ot automat
ca
y d
t
rm


 c
rta

 paths (
.g., `
vcc` or a sp
c
f
c Pytho
 
x
cutab

 for your vLLM d
v

opm

t 

v
ro
m

t). Fo
o
 th
 o
-scr

 prompts. If a
 
x
st

g `CMak
Us
rPr
s
ts.jso
` 
s fou
d, th
 scr
pt 


 ask for co
f
rmat
o
 b
for
 ov
r
r
t

g 
t.
**Forc
 ov
r
r
t
 
x
st

g f


:**
To automat
ca
y ov
r
r
t
 a
 
x
st

g `CMak
Us
rPr
s
ts.jso
` 

thout prompt

g, us
 th
 `--forc
-ov
r
r
t
` f
ag:
```co
so


pytho
 too
s/g


rat
_cmak
_pr
s
ts.py --forc
-ov
r
r
t

```
Th
s 
s part
cu
ar
y us
fu
 

 automat
d scr
pts or CI/CD 

v
ro
m

ts 
h
r
 

t
ract
v
 prompts ar
 
ot d
s
r
d.
Aft
r ru


g th
 scr
pt, a `CMak
Us
rPr
s
ts.jso
` f


 


 b
 cr
at
d 

 th
 root of your vLLM r
pos
tory.
### Examp

 `CMak
Us
rPr
s
ts.jso
`
B

o
 
s a
 
xamp

 of 
hat th
 g


rat
d `CMak
Us
rPr
s
ts.jso
` m
ght 
ook 

k
. Th
 scr
pt 


 ta

or th
s
 va
u
s bas
d o
 your syst
m a
d a
y 

put you prov
d
.
```jso

{
    "v
rs
o
": 6,
    "cmak
M


mumR
qu
r
d": {
        "major": 3,
        "m

or": 26,
        "patch": 1
    },
    "co
f
gur
Pr
s
ts": [
        {
            "
am
": "r


as
",
            "g


rator": "N

ja",
            "b

aryD
r": "${sourc
D
r}/cmak
-bu

d-r


as
",
            "cach
Var
ab

s": {
                "CMAKE_CUDA_COMPILER": "/usr/
oca
/cuda/b

/
vcc",
                "CMAKE_C_COMPILER_LAUNCHER": "ccach
",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccach
",
                "CMAKE_CUDA_COMPILER_LAUNCHER": "ccach
",
                "CMAKE_BUILD_TYPE": "R


as
",
                "VLLM_PYTHON_EXECUTABLE": "/hom
/us
r/v

vs/v
m/b

/pytho
",
                "CMAKE_INSTALL_PREFIX": "${sourc
D
r}",
                "CMAKE_CUDA_FLAGS": "",
                "NVCC_THREADS": "4",
                "CMAKE_JOB_POOLS": "comp


=32"
            }
        }
    ],
    "bu

dPr
s
ts": [
        {
            "
am
": "r


as
",
            "co
f
gur
Pr
s
t": "r


as
",
            "jobs": 32
        }
    ]
}
```
**What do th
 var
ous co
f
gurat
o
s m
a
?**
- `CMAKE_CUDA_COMPILER`: Path to your `
vcc` b

ary. Th
 scr
pt att
mpts to f

d th
s automat
ca
y.
- `CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER`, `CMAKE_CUDA_COMPILER_LAUNCHER`: S
tt

g th
s
 to `ccach
` (or `sccach
`) s
g

f
ca
t
y sp
ds up r
bu

ds by cach

g comp

at
o
 r
su
ts. E
sur
 `ccach
` 
s 

sta

d (
.g., `sudo apt 

sta
 ccach
` or `co
da 

sta
 ccach
`). Th
 scr
pt s
ts th
s
 by d
fau
t.
- `VLLM_PYTHON_EXECUTABLE`: Path to th
 Pytho
 
x
cutab

 

 your vLLM d
v

opm

t 

v
ro
m

t. Th
 scr
pt 


 prompt for th
s, d
fau
t

g to th
 curr

t Pytho
 

v
ro
m

t 
f su
tab

.
- `CMAKE_INSTALL_PREFIX: "${sourc
D
r}"`: Sp
c
f

s that th
 comp


d compo


ts shou
d b
 

sta

d back 

to your vLLM sourc
 d
r
ctory. Th
s 
s cruc
a
 for th
 
d
tab

 

sta
, as 
t mak
s th
 



y bu

t k
r


s 
mm
d
at

y ava

ab

 to your Pytho
 

v
ro
m

t.
- `CMAKE_JOB_POOLS` a
d `jobs` 

 bu

d pr
s
ts: Co
tro
 th
 para



sm of th
 bu

d. Th
 scr
pt s
ts th
s
 bas
d o
 th
 
umb
r of CPU cor
s d
t
ct
d o
 your syst
m.
- `b

aryD
r`: Sp
c
f

s 
h
r
 th
 bu

d art
facts 


 b
 stor
d (
.g., `cmak
-bu

d-r


as
`).
## Bu

d

g a
d I
sta


g 

th CMak

O
c
 your `CMak
Us
rPr
s
ts.jso
` 
s co
f
gur
d:
1. **I

t
a

z
 th
 CMak
 bu

d 

v
ro
m

t:**
   Th
s st
p co
f
gur
s th
 bu

d syst
m accord

g to your chos

 pr
s
t (
.g., `r


as
`) a
d cr
at
s th
 bu

d d
r
ctory at `b

aryD
r`
    ```co
so


    cmak
 --pr
s
t r


as

    ```
2. **Bu

d a
d 

sta
 th
 vLLM compo


ts:**
   Th
s comma
d comp


s th
 cod
 a
d 

sta
s th
 r
su
t

g b

ar

s 

to your vLLM sourc
 d
r
ctory, mak

g th
m ava

ab

 to your 
d
tab

 Pytho
 

sta
at
o
.
    ```co
so


    cmak
 --bu

d --pr
s
t r


as
 --targ
t 

sta

    ```
3. **Mak
 cha
g
s a
d r
p
at!**
    No
 you start us

g your 
d
tab

 

sta
 of vLLM, t
st

g a
d mak

g cha
g
s as 

d
d. If you 

d to bu

d aga

 to updat
 bas
d o
 cha
g
s, s
mp
y ru
 th
 CMak
 comma
d aga

 to bu

d o

y th
 aff
ct
d f


s.
    ```co
so


    cmak
 --bu

d --pr
s
t r


as
 --targ
t 

sta

    ```
## V
r
fy

g th
 Bu

d
Aft
r a succ
ssfu
 bu

d, you 


 f

d a popu
at
d bu

d d
r
ctory (
.g., `cmak
-bu

d-r


as
/` 
f you us
d th
 `r


as
` pr
s
t a
d th
 
xamp

 co
f
gurat
o
).
```co
so



 
s cmak
-bu

d-r


as
/
b

             cmak
_

sta
.cmak
      _d
ps                                mach
t
_g


rat
o
.
og
bu

d.


ja     CPackCo
f
g.cmak
        d
t
ct_cuda_comput
_capab


t

s.cu  mar


_g


rat
o
.
og
_C.ab
3.so      CPackSourc
Co
f
g.cmak
  d
t
ct_cuda_v
rs
o
.cc               _mo
_C.ab
3.so
CMak
Cach
.txt  ct
st                    _f
ashm
a_C.ab
3.so                  mo
_mar


_g


rat
o
.
og
CMak
F


s      cum
m_a
ocator.ab
3.so  

sta
_
oca
_ma

f
st.txt           v
m-f
ash-att

```
Th
 `cmak
 --bu

d ... --targ
t 

sta
` comma
d cop

s th
 comp


d shar
d 

brar

s (

k
 `_C.ab
3.so`, `_mo
_C.ab
3.so`, 
tc.) 

to th
 appropr
at
 `v
m` packag
 d
r
ctory 

th

 your sourc
 tr
. Th
s updat
s your 
d
tab

 

sta
at
o
 

th th
 



y comp


d k
r


s.
## Add
t
o
a
 T
ps
- **Adjust Para



sm:** F


-tu

 th
 `CMAKE_JOB_POOLS` 

 `co
f
gur
Pr
s
ts` a
d `jobs` 

 `bu

dPr
s
ts` 

 your `CMak
Us
rPr
s
ts.jso
`. Too ma
y jobs ca
 ov
r
oad syst
ms 

th 

m
t
d RAM or CPU cor
s, 

ad

g to s
o

r bu

ds or syst
m 

stab


ty. Too f

 
o
't fu
y ut


z
 ava

ab

 r
sourc
s.
- **C

a
 Bu

ds Wh

 N
c
ssary:** If you 

cou
t
r p
rs
st

t or stra
g
 bu

d 
rrors, 
sp
c
a
y aft
r s
g

f
ca
t cha
g
s or s

tch

g bra
ch
s, co
s
d
r r
mov

g th
 CMak
 bu

d d
r
ctory (
.g., `rm -rf cmak
-bu

d-r


as
`) a
d r
-ru


g th
 `cmak
 --pr
s
t` a
d `cmak
 --bu

d` comma
ds.
- **Sp
c
f
c Targ
t Bu

ds:** For 
v

 fast
r 
t
rat
o
s 
h

 
ork

g o
 a sp
c
f
c modu

, you ca
 som
t
m
s bu

d a sp
c
f
c targ
t 

st
ad of th
 fu
 `

sta
` targ
t, though `

sta
` 

sur
s a
 

c
ssary compo


ts ar
 updat
d 

 your Pytho
 

v
ro
m

t. R
f
r to CMak
 docum

tat
o
 for mor
 adva
c
d targ
t ma
ag
m

t.
