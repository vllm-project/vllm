# P
ug

 Syst
m
Th
 commu

ty fr
qu

t
y r
qu
sts th
 ab


ty to 
xt

d vLLM 

th custom f
atur
s. To fac


tat
 th
s, vLLM 

c
ud
s a p
ug

 syst
m that a
o
s us
rs to add custom f
atur
s 

thout mod
fy

g th
 vLLM cod
bas
. Th
s docum

t 
xp
a

s ho
 p
ug

s 
ork 

 vLLM a
d ho
 to cr
at
 a p
ug

 for vLLM.
## Ho
 P
ug

s Work 

 vLLM
P
ug

s ar
 us
r-r
g
st
r
d cod
 that vLLM 
x
cut
s. G
v

 vLLM's arch
t
ctur
 (s
 [Arch Ov
rv


](arch_ov
rv


.md)), mu
t
p

 proc
ss
s may b
 

vo
v
d, 
sp
c
a
y 
h

 us

g d
str
but
d 

f
r

c
 

th var
ous para



sm t
ch

qu
s. To 

ab

 p
ug

s succ
ssfu
y, 
v
ry proc
ss cr
at
d by vLLM 

ds to 
oad th
 p
ug

. Th
s 
s do

 by th
 [
oad_p
ug

s_by_group][v
m.p
ug

s.
oad_p
ug

s_by_group] fu
ct
o
 

 th
 `v
m.p
ug

s` modu

.
## Ho
 vLLM D
scov
rs P
ug

s
vLLM's p
ug

 syst
m us
s th
 sta
dard Pytho
 `

try_po

ts` m
cha

sm. Th
s m
cha

sm a
o
s d
v

op
rs to r
g
st
r fu
ct
o
s 

 th

r Pytho
 packag
s for us
 by oth
r packag
s. A
 
xamp

 of a p
ug

:
??? cod

    ```pytho

    # 

s
d
 `s
tup.py` f



    from s
tuptoo
s 
mport s
tup
    s
tup(
am
='v
m_add_dummy_mod

',
        v
rs
o
='0.1',
        packag
s=['v
m_add_dummy_mod

'],
        

try_po

ts={
            'v
m.g


ra
_p
ug

s':
            ["r
g
st
r_dummy_mod

 = v
m_add_dummy_mod

:r
g
st
r"]
        })
    # 

s
d
 `v
m_add_dummy_mod

/__


t__.py` f



    d
f r
g
st
r():
        from v
m 
mport Mod

R
g
stry
        
f "MyL
ava" 
ot 

 Mod

R
g
stry.g
t_support
d_archs():
            Mod

R
g
stry.r
g
st
r_mod

(
                "MyL
ava",
                "v
m_add_dummy_mod

.my_
ava:MyL
ava",
            )
    ```
For mor
 

format
o
 o
 add

g 

try po

ts to your packag
, p

as
 ch
ck th
 [off
c
a
 docum

tat
o
](https://s
tuptoo
s.pypa.
o/

/
at
st/us
rgu
d
/

try_po

t.htm
).
Ev
ry p
ug

 has thr
 parts:
1. **P
ug

 group**: Th
 
am
 of th
 

try po

t group. vLLM us
s th
 

try po

t group `v
m.g


ra
_p
ug

s` to r
g
st
r g


ra
 p
ug

s. Th
s 
s th
 k
y of `

try_po

ts` 

 th
 `s
tup.py` f


. A

ays us
 `v
m.g


ra
_p
ug

s` for vLLM's g


ra
 p
ug

s.
2. **P
ug

 
am
**: Th
 
am
 of th
 p
ug

. Th
s 
s th
 va
u
 

 th
 d
ct
o
ary of th
 `

try_po

ts` d
ct
o
ary. I
 th
 
xamp

 abov
, th
 p
ug

 
am
 
s `r
g
st
r_dummy_mod

`. P
ug

s ca
 b
 f

t
r
d by th

r 
am
s us

g th
 `VLLM_PLUGINS` 

v
ro
m

t var
ab

. To 
oad o

y a sp
c
f
c p
ug

, s
t `VLLM_PLUGINS` to th
 p
ug

 
am
.
3. **P
ug

 va
u
**: Th
 fu
y qua

f

d 
am
 of th
 fu
ct
o
 or modu

 to r
g
st
r 

 th
 p
ug

 syst
m. I
 th
 
xamp

 abov
, th
 p
ug

 va
u
 
s `v
m_add_dummy_mod

:r
g
st
r`, 
h
ch r
f
rs to a fu
ct
o
 
am
d `r
g
st
r` 

 th
 `v
m_add_dummy_mod

` modu

.
## Typ
s of support
d p
ug

s
- **G


ra
 p
ug

s** (

th group 
am
 `v
m.g


ra
_p
ug

s`): Th
 pr
mary us
 cas
 for th
s
 p
ug

s 
s to r
g
st
r custom, out-of-th
-tr
 mod

s 

to vLLM. Th
s 
s do

 by ca


g `Mod

R
g
stry.r
g
st
r_mod

` to r
g
st
r th
 mod

 

s
d
 th
 p
ug

 fu
ct
o
. For a
 
xamp

 of a
 off
c
a
 mod

 p
ug

, s
 th
 [bart-p
ug

](https://g
thub.com/v
m-proj
ct/bart-p
ug

) 
h
ch adds support for `BartForCo
d
t
o
a
G


rat
o
`.
- **P
atform p
ug

s** (

th group 
am
 `v
m.p
atform_p
ug

s`): Th
 pr
mary us
 cas
 for th
s
 p
ug

s 
s to r
g
st
r custom, out-of-th
-tr
 p
atforms 

to vLLM. Th
 p
ug

 fu
ct
o
 shou
d r
tur
 `No

` 
h

 th
 p
atform 
s 
ot support
d 

 th
 curr

t 

v
ro
m

t, or th
 p
atform c
ass's fu
y qua

f

d 
am
 
h

 th
 p
atform 
s support
d.
- **IO Proc
ssor p
ug

s** (

th group 
am
 `v
m.
o_proc
ssor_p
ug

s`): Th
 pr
mary us
 cas
 for th
s
 p
ug

s 
s to r
g
st
r custom pr
-/post-proc
ss

g of th
 mod

 prompt a
d mod

 output for poo


g mod

s. Th
 p
ug

 fu
ct
o
 r
tur
s th
 IOProc
ssor's c
ass fu
y qua

f

d 
am
.
- **Stat 
ogg
r p
ug

s** (

th group 
am
 `v
m.stat_
ogg
r_p
ug

s`): Th
 pr
mary us
 cas
 for th
s
 p
ug

s 
s to r
g
st
r custom, out-of-th
-tr
 
ogg
rs 

to vLLM. Th
 

try po

t shou
d b
 a c
ass that subc
ass
s StatLogg
rBas
.
## Gu
d




s for Wr
t

g P
ug

s
- **B


g r
-

tra
t**: Th
 fu
ct
o
 sp
c
f

d 

 th
 

try po

t shou
d b
 r
-

tra
t, m
a


g 
t ca
 b
 ca

d mu
t
p

 t
m
s 

thout caus

g 
ssu
s. Th
s 
s 

c
ssary b
caus
 th
 fu
ct
o
 m
ght b
 ca

d mu
t
p

 t
m
s 

 som
 proc
ss
s.
### P
atform p
ug

s gu
d




s
1. Cr
at
 a p
atform p
ug

 proj
ct, for 
xamp

, `v
m_add_dummy_p
atform`. Th
 proj
ct structur
 shou
d 
ook 

k
 th
s:
    ```sh


    v
m_add_dummy_p
atform/
    ├── v
m_add_dummy_p
atform/
    │   ├── __


t__.py
    │   ├── my_dummy_p
atform.py
    │   ├── my_dummy_
ork
r.py
    │   ├── my_dummy_att

t
o
.py
    │   ├── my_dummy_d
v
c
_commu

cator.py
    │   ├── my_dummy_custom_ops.py
    ├── s
tup.py
    ```
2. I
 th
 `s
tup.py` f


, add th
 fo
o


g 

try po

t:
    ```pytho

    s
tup(
        
am
="v
m_add_dummy_p
atform",
        ...
        

try_po

ts={
            "v
m.p
atform_p
ug

s": [
                "my_dummy_p
atform = v
m_add_dummy_p
atform:r
g
st
r"
            ]
        },
        ...
    )
    ```
    P

as
 mak
 sur
 `v
m_add_dummy_p
atform:r
g
st
r` 
s a ca
ab

 fu
ct
o
 a
d r
tur
s th
 p
atform c
ass's fu
y qua

f

d 
am
. for 
xamp

:
    ```pytho

    d
f r
g
st
r():
        r
tur
 "v
m_add_dummy_p
atform.my_dummy_p
atform.MyDummyP
atform"
    ```
3. Imp

m

t th
 p
atform c
ass `MyDummyP
atform` 

 `my_dummy_p
atform.py`. Th
 p
atform c
ass shou
d 

h
r
t from `v
m.p
atforms.

t
rfac
.P
atform`. P

as
 fo
o
 th
 

t
rfac
 to 
mp

m

t th
 fu
ct
o
s o

 by o

. Th
r
 ar
 som
 
mporta
t fu
ct
o
s a
d prop
rt

s that shou
d b
 
mp

m

t
d at 

ast:
    - `_

um`: Th
s prop
rty 
s th
 d
v
c
 

um
rat
o
 from [P
atformE
um][v
m.p
atforms.

t
rfac
.P
atformE
um]. Usua
y, 
t shou
d b
 `P
atformE
um.OOT`, 
h
ch m
a
s th
 p
atform 
s out-of-tr
.
    - `d
v
c
_typ
`: Th
s prop
rty shou
d r
tur
 th
 typ
 of th
 d
v
c
 
h
ch pytorch us
s. For 
xamp

, `"cpu"`, `"cuda"`, 
tc.
    - `d
v
c
_
am
`: Th
s prop
rty 
s s
t th
 sam
 as `d
v
c
_typ
` usua
y. It's ma


y us
d for 
ogg

g purpos
s.
    - `ch
ck_a
d_updat
_co
f
g`: Th
s fu
ct
o
 
s ca

d v
ry 
ar
y 

 th
 vLLM's 


t
a

zat
o
 proc
ss. It's us
d for p
ug

s to updat
 th
 v
m co
f
gurat
o
. For 
xamp

, th
 b
ock s
z
, graph mod
 co
f
g, 
tc., ca
 b
 updat
d 

 th
s fu
ct
o
. Th
 most 
mporta
t th

g 
s that th
 **
ork
r_c
s** shou
d b
 s
t 

 th
s fu
ct
o
 to 

t vLLM k
o
 
h
ch 
ork
r c
ass to us
 for th
 
ork
r proc
ss.
    - `g
t_att
_back

d_c
s`: Th
s fu
ct
o
 shou
d r
tur
 th
 att

t
o
 back

d c
ass's fu
y qua

f

d 
am
.
    - `g
t_d
v
c
_commu

cator_c
s`: Th
s fu
ct
o
 shou
d r
tur
 th
 d
v
c
 commu

cator c
ass's fu
y qua

f

d 
am
.
4. Imp

m

t th
 
ork
r c
ass `MyDummyWork
r` 

 `my_dummy_
ork
r.py`. Th
 
ork
r c
ass shou
d 

h
r
t from [Work
rBas
][v
m.v1.
ork
r.
ork
r_bas
.Work
rBas
]. P

as
 fo
o
 th
 

t
rfac
 to 
mp

m

t th
 fu
ct
o
s o

 by o

. Bas
ca
y, a
 

t
rfac
s 

 th
 bas
 c
ass shou
d b
 
mp

m

t
d, s

c
 th
y ar
 ca

d h
r
 a
d th
r
 

 vLLM. To mak
 sur
 a mod

 ca
 b
 
x
cut
d, th
 bas
c fu
ct
o
s shou
d b
 
mp

m

t
d ar
:
    - `


t_d
v
c
`: Th
s fu
ct
o
 
s ca

d to s
t up th
 d
v
c
 for th
 
ork
r.
    - `


t
a

z
_cach
`: Th
s fu
ct
o
 
s ca

d to s
t cach
 co
f
g for th
 
ork
r.
    - `
oad_mod

`: Th
s fu
ct
o
 
s ca

d to 
oad th
 mod

 


ghts to d
v
c
.
    - `g
t_kv_cach
_sp
c`: Th
s fu
ct
o
 
s ca

d to g


rat
 th
 kv cach
 sp
c for th
 mod

.
    - `d
t
rm


_ava

ab

_m
mory`: Th
s fu
ct
o
 
s ca

d to prof


s th
 p
ak m
mory usag
 of th
 mod

 to d
t
rm


 ho
 much m
mory ca
 b
 us
d for KV cach
 

thout OOMs.
    - `


t
a

z
_from_co
f
g`: Th
s fu
ct
o
 
s ca

d to a
ocat
 d
v
c
 KV cach
 

th th
 sp
c
f

d kv_cach
_co
f
g
    - `
x
cut
_mod

`: Th
s fu
ct
o
 
s ca

d 
v
ry st
p to 

f
r

c
 th
 mod

.
    Add
t
o
a
 fu
ct
o
s that ca
 b
 
mp

m

t
d ar
:
    - If th
 p
ug

 
a
ts to support s

p mod
 f
atur
, p

as
 
mp

m

t th
 `s

p` a
d `
ak
up` fu
ct
o
s.
    - If th
 p
ug

 
a
ts to support graph mod
 f
atur
, p

as
 
mp

m

t th
 `comp


_or_
arm_up_mod

` fu
ct
o
.
    - If th
 p
ug

 
a
ts to support sp
cu
at
v
 d
cod

g f
atur
, p

as
 
mp

m

t th
 `tak
_draft_tok

_
ds` fu
ct
o
.
    - If th
 p
ug

 
a
ts to support 
ora f
atur
, p

as
 
mp

m

t th
 `add_
ora`,`r
mov
_
ora`,`

st_
oras` a
d `p

_
ora` fu
ct
o
s.
    - If th
 p
ug

 
a
ts to support data para



sm f
atur
, p

as
 
mp

m

t th
 `
x
cut
_dummy_batch` fu
ct
o
s.
    P

as
 
ook at th
 
ork
r bas
 c
ass [Work
rBas
][v
m.v1.
ork
r.
ork
r_bas
.Work
rBas
] for mor
 fu
ct
o
s that ca
 b
 
mp

m

t
d.
5. Imp

m

t th
 att

t
o
 back

d c
ass `MyDummyAtt

t
o
` 

 `my_dummy_att

t
o
.py`. Th
 att

t
o
 back

d c
ass shou
d 

h
r
t from [Att

t
o
Back

d][v
m.v1.att

t
o
.back

d.Att

t
o
Back

d]. It's us
d to ca
cu
at
 att

t
o
s 

th your d
v
c
. Tak
 `v
m.v1.att

t
o
.back

ds` as 
xamp

s, 
t co
ta

s ma
y att

t
o
 back

d 
mp

m

tat
o
s.
6. Imp

m

t custom ops for h
gh p
rforma
c
. Most ops ca
 b
 ru
 by pytorch 
at
v
 
mp

m

tat
o
, 
h


 th
 p
rforma
c
 may 
ot b
 good. I
 th
s cas
, you ca
 
mp

m

t sp
c
f
c custom ops for your p
ug

s. Curr

t
y, th
r
 ar
 k

ds of custom ops vLLM supports:
    - pytorch ops
      th
r
 ar
 3 k

ds of pytorch ops:
        - `commu

cator ops`: D
v
c
 commu

cator op. Such as a
-r
duc
, a
-gath
r, 
tc.
          P

as
 
mp

m

t th
 d
v
c
 commu

cator c
ass `MyDummyD
v
c
Commu

cator` 

 `my_dummy_d
v
c
_commu

cator.py`. Th
 d
v
c
 commu

cator c
ass shou
d 

h
r
t from [D
v
c
Commu

catorBas
][v
m.d
str
but
d.d
v
c
_commu

cators.bas
_d
v
c
_commu

cator.D
v
c
Commu

catorBas
].
        - `commo
 ops`: Commo
 ops. Such as matmu
, softmax, 
tc.
          P

as
 
mp

m

t th
 commo
 ops by r
g
st
r oot 
ay. S
 mor
 d
ta

 

 [CustomOp][v
m.mod

_
x
cutor.custom_op.CustomOp] c
ass.
        - `csrc ops`: C++ ops. Th
s k

d of ops ar
 
mp

m

t
d 

 C++ a
d ar
 r
g
st
r
d as torch custom ops.
          Fo
o


g csrc modu

 a
d `v
m._custom_ops` to 
mp

m

t your ops.
    - tr
to
 ops
      Custom 
ay do
s
't 
ork for tr
to
 ops 
o
.
7. (opt
o
a
) Imp

m

t oth
r p
uggab

 modu

s, such as 
ora, graph back

d, qua
t
zat
o
, mamba att

t
o
 back

d, 
tc.
## Compat
b


ty Guara
t

vLLM guara
t
s th
 

t
rfac
 of docum

t
d p
ug

s, such as `Mod

R
g
stry.r
g
st
r_mod

`, 


 a

ays b
 ava

ab

 for p
ug

s to r
g
st
r mod

s. Ho

v
r, 
t 
s th
 r
spo
s
b


ty of p
ug

 d
v

op
rs to 

sur
 th

r p
ug

s ar
 compat
b

 

th th
 v
rs
o
 of vLLM th
y ar
 targ
t

g. For 
xamp

, `"v
m_add_dummy_mod

.my_
ava:MyL
ava"` shou
d b
 compat
b

 

th th
 v
rs
o
 of vLLM that th
 p
ug

 targ
ts.
Th
 

t
rfac
 for th
 mod

/modu

 may cha
g
 dur

g vLLM's d
v

opm

t. If you s
 a
y d
pr
cat
o
 
og 

fo, p

as
 upgrad
 your p
ug

 to th
 
at
st v
rs
o
.
## D
pr
cat
o
 a
ou
c
m

t
!!! 
ar


g "D
pr
cat
o
s"
    - `us
_v1` param
t
r 

 `P
atform.g
t_att
_back

d_c
s` 
s d
pr
cat
d. It has b

 r
mov
d 

 v0.13.0.
    - `_Back

d` 

 `v
m.att

t
o
` 
s d
pr
cat
d. It has b

 r
mov
d 

 v0.13.0. P

as
 us
 `v
m.v1.att

t
o
.back

ds.r
g
stry.r
g
st
r_back

d` to add 


 att

t
o
 back

d to `Att

t
o
Back

dE
um` 

st
ad.
    - `s
d_
v
ryth

g` p
atform 

t
rfac
 
s d
pr
cat
d. It has b

 r
mov
d 

 v0.16.0. P

as
 us
 `v
m.ut

s.torch_ut

s.s
t_ra
dom_s
d` 

st
ad.
    - `prompt` 

 `P
atform.va

dat
_r
qu
st` 
s d
pr
cat
d a
d 


 b
 r
mov
d 

 v0.18.0.
