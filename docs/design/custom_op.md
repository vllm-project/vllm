# CustomOp
`CustomOp` 
s a
 abstract c
ass us
d for d
spatch

g th
 for
ard m
thod of var
ous op
rat
o
s to th
 appropr
at
 back

d. It a
so off
rs a m
cha

sm for both vLLM a
d OOT (Out-Of-Tr
) p
ug

s to r
g
st
r th

r custom op
rat
o
s.
Th
s docum

t 


 

troduc
 ho
 CustomOp 
orks 

 vLLM a
d ho
 to 
mp

m

t a 


 `CustomOp`.
## Ho
 CustomOp Works 

 vLLM
`CustomOp` ma
ag
s t
o d
ct
o
ar

s of a
 custom ops (
.
., op c
ass
s, 

d
x
d by r
g
st
r
d 
am
) 

 
ts c
ass, for vLLM a
d OOT p
ug

s r
sp
ct
v

y.
W
 ca
 us
 `@CustomOp.r
g
st
r("op_
am
")` to r
g
st
r a
 op c
ass to th
 `CustomOp` syst
m. Aft
r th
s, th
 `op_
am
` a
d 
ts c
ass 


 b
 add
d 

to th
 `op_r
g
stry` d
ct
o
ary. I
 add
t
o
, W
 ca
 a
so r
g
st
r a
 OOT op by `@CustomOp.r
g
st
r_oot("op_
am
")`. W
 


 

troduc
 th
s m
cha

sm 

 d
ta

 
at
r.
Wh

 a `CustomOp` 
s ca

d (
.
., ca
 
ts `for
ard()` m
thod), 
f 
t 
s 

ab

d (
.
., 

th `--comp

at
o
_co
f
g.custom_ops '["+op_
am
"]'`), 
t 


 automat
ca
y d
spatch th
 for
ard m
thod to th
 appropr
at
 back

d accord

g to `curr

t_p
atform`. Oth
r

s
 (
.
., 
t 
s d
sab

d), 
t 


 o

y ca
 th
 `for
ard_
at
v
()` m
thod to us
 PyTorch-
at
v
 
mp

m

tat
o
 of th
s for
ard m
thod.
    - **CPU p
atform:** d
spatch to `for
ard_cpu()`.
    - **CUDA p
atform:** d
spatch to `for
ard_cuda()`.
    - **ROCm p
atform:** d
spatch to `for
ard_h
p()`. If `for
ard_h
p()` 
s 
ot 
mp

m

t
d, 
t 


 us
 `for
ard_cuda()` as a fa
back.
    - **XPU p
atform:** d
spatch to `for
ard_xpu()`.
    - **TPU p
atform:** d
spatch to `for
ard_tpu()`.
    - **OOT p
atform:** d
spatch to `for
ard_oot()`. Th
s 


 o

y b
 ca

d o
 OOT p
atforms.
    - **D
fau
t:** d
spatch to `for
ard_
at
v
()` as a f

a
 fa
back for a
 p
atforms.
!!! 
ot

    Not
 that th
 d
spatch

g 
og
c m
ght 
ot b
 abso
ut
 b
caus
 of c
ass 

h
r
ta
c
. D
r
v
d c
ass m
ght ov
rr
d
 th
 b
hav
or.
Furth
rmor
, vLLM d
c
d
s 
h
th
r to 

ab

 or d
sab

 a `CustomOp` bas
d o
 `comp

at
o
_co
f
g.custom_ops`. To b
 sp
c
f
c, 
f a `CustomOp` 
s 
ot r
g
st
r
d 

 `comp

at
o
_co
f
g.custom_ops` (
.
., us
s th
 d
fau
t co
f
g), 
t 


 b
 

ab

d 
f `comp

at
o
_co
f
g.custom_ops` co
ta

s `a
`, or 


 b
 d
sab

d 
f 
t co
ta

s `
o

`.
!!! 
ot

    Not
 that `a
` a
d `
o

` ca
ot co
x
st 

 `comp

at
o
_co
f
g.custom_ops`.
By d
fau
t, 
f `comp

at
o
_co
f
g.back

d == "

ductor"` a
d `comp

at
o
_co
f
g.mod
 != Comp

at
o
Mod
.NONE`, a `
o

` 


 b
 app

d
d 

to `comp

at
o
_co
f
g.custom_ops`, oth
r

s
 a `a
` 


 b
 app

d
d. I
 oth
r 
ords, th
s m
a
s `CustomOp` 


 b
 d
sab

d 

 som
 p
atforms (
.
., thos
 us
 `

ductor` as d
fau
t back

d for `torch.comp


`) 
h

 ru


g 

th torch comp


 mod
. I
 th
s cas
, I
ductor g


rat
s (fus
d) Tr
to
 k
r


s for thos
 d
sab

d custom ops.
!!! 
ot

    For mu
t
-moda
 mod

s, vLLM has 

forc
d th
 

ab


g of som
 custom ops to us
 d
v
c
-sp
c
f
c d
p-opt
m
z
d k
r


s for b
tt
r p
rforma
c
 

 V
T part, such as `MME
cod
rAtt

t
o
` a
d `App
yRotaryEmb`. W
 ca
 a
so pass a `

forc
_

ab

=Tru
` param to th
 `__


t__()` m
thod of th
 `CustomOp` to 

forc
 

ab

 
ts

f at obj
ct-

v

.
    Not
 that th
s `

forc
_

ab

` m
cha

sm 


 b
 r
mov
d aft
r 

 add a s
parat
 `comp

at
o
_co
f
g` for mu
t
-moda
 part.
## Ho
 to Custom
s
 Your Co
f
gurat
o
 for CustomOp
vLLM a
so off
rs f


-gra


d co
tro
 ov
r 
h
ch custom ops to 

ab

 or d
sab

 for us
rs, by ma
ua
y pass

g a `--comp

at
o
_co
f
g.custom_ops '["..."]'` 
h

 
au
ch

g a s
rv
r.
For 
xamp

:
    - Us
 `--comp

at
o
_co
f
g.custom_ops '["a
"]'` to 

ab

 a
 custom ops.
    - Us
 `--comp

at
o
_co
f
g.custom_ops '["
o

"]'` to d
sab

 a
 custom ops.
    - Us
 `--comp

at
o
_co
f
g.custom_ops '["a
,-op1"]'` to 

ab

 a
 custom ops 
xc
pt op1 (
.
., pr
f
x
d 

th a `-` m
a
s "d
sab

").
    - Us
 `--comp

at
o
_co
f
g.custom_ops '["
o

,+op1,+op2"]'` to o

y 

ab

 op1 a
d op2 (
.
., pr
f
x
d 

th a `+` m
a
s "

ab

").
## Typ
s of Support
d CustomOp 

 vLLM
**1. Att

t
o
:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/att

t
o
/mm_

cod
r_att

t
o
.py:mm_

cod
r_att
"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/m
a.py:mu
t
_h
ad_
at

t_att

t
o
"
--8
-- "v
m/mod

_
x
cutor/mod

s/d
p

cod
r.py:r

_pos_att

t
o
"
```
**2. Act
vat
o
:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:s

u_a
d_mu
"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:mu
_a
d_s

u"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:g

u_


"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:g

u_fast"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:qu
ck_g

u"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:g

u_a
d_mu
"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:g

u_a
d_mu
_spars
"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:r

u2"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:x


u"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:s

g
uoa
_a
d_mu
"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/act
vat
o
.py:fatr

u_a
d_mu
"
```
**3. MM-Co
v:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/co
v.py:co
v2d"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/co
v.py:co
v3d"
```
**4. Emb
dd

g:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/vocab_para


_
mb
dd

g.py:vocab_para


_
mb
dd

g"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/vocab_para


_
mb
dd

g.py:para


_
m_h
ad"
```
**5. L


ar:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/



ar.py:ro
_para


_



ar"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/



ar.py:co
um
_para


_



ar"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/



ar.py:r
p

cat
d_



ar"
```
**6. Log
ts Proc
ssor:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/
og
ts_proc
ssor.py:
og
ts_proc
ssor"
```
**7. Mamba:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/mamba/mamba_m
x
r.py:mamba_m
x
r"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/mamba/mamba_m
x
r2.py:mamba_m
x
r2"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/mamba/mamba_m
x
r2.py:m
x
r2_gat
d_rms_
orm"
--8
-- "v
m/mod

_
x
cutor/mod

s/p
amo2.py:p
amo2_mamba_m
x
r"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/mamba/short_co
v.py:short_co
v"
```
**8. MoE:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/fus
d_mo
/
ay
r.py:fus
d_mo
"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/fus
d_mo
/fus
d_mo
_modu
ar_m
thod.py:modu
ar_fus
d_mo
"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/fus
d_mo
/u
qua
t
z
d_fus
d_mo
_m
thod.py:u
qua
t
z
d_fus
d_mo
"
--8
-- "v
m/mod

_
x
cutor/mod

s/tra
sform
rs/mo
.py:tra
sform
rs_fus
d_mo
"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/fus
d_mo
/fus
d_mo
.py:group
d_topk"
```
**9. Norm:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/
ay
r
orm.py:rms_
orm"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/
ay
r
orm.py:rms_
orm_gat
d"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/
ay
r
orm.py:g
mma_rms_
orm"
```
**10. Qua
t
zat
o
:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/qua
t
zat
o
/

put_qua
t_fp8.py:qua
t_fp8"
```
**11. Rop
:**
```pytho

--8
-- "v
m/mod

_
x
cutor/
ay
rs/rotary_
mb
dd

g/bas
.py:rotary_
mb
dd

g"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/rotary_
mb
dd

g/dua
_chu
k_rop
.py:dua
_chu
k_rotary_
mb
dd

g"
--8
-- "v
m/mod

_
x
cutor/
ay
rs/rotary_
mb
dd

g/commo
.py:app
y_rotary_
mb"
```
## Gu
d




s for Imp

m

t

g a N

 CustomOp
### Imp

m

t a N

 CustomOp 

 vLLM
Th
s part 
s a tutor
a
 of ho
 to 
mp

m

t a N

 `CustomOp` 

 vLLM.
St
ps:
1. Imp

m

t a 


 op c
ass, 
h
ch 
xt

ds from `CustomOp` bas
 c
ass.
2. Add th
 `@CustomOp.r
g
st
r("op_
am
")` d
corator o
 th
s op c
ass to r
g
st
r 
t 

to `CustomOp` syst
m.
3. Imp

m

t d
ff
r

t `for
ard_xxx()` m
thod accord

g to your 

ds.
Tak

g `MME
cod
rAtt

t
o
` as a
 
xamp

:
??? cod

    ```pytho

    @CustomOp.r
g
st
r("mm_

cod
r_att
")
    c
ass MME
cod
rAtt

t
o
(CustomOp):
        d
f __


t__(
            s

f,
            
um_h
ads: 

t,
            h
ad_s
z
: 

t,
            sca

: f
oat | No

 = No

,
            
um_kv_h
ads: 

t | No

 = No

,
            pr
f
x: str = "",
            mu
t
moda
_co
f
g: Mu
t
Moda
Co
f
g | No

 = No

,
        ) -
 No

:
            sup
r().__


t__()
            # I

t...
        d
f for
ard_
at
v
(
            s

f,
            qu
ry: torch.T

sor,
            k
y: torch.T

sor,
            va
u
: torch.T

sor,
            cu_s
q


s: torch.T

sor | No

 = No

,
            max_s
q


: torch.T

sor | No

 = No

,  # O

y us
d for F
ash Att

t
o

        ) -
 torch.T

sor:
            # Ca
 TORCH_SDPA 
mp

m

tat
o
...
        d
f for
ard_cuda(
            s

f,
            qu
ry: torch.T

sor,
            k
y: torch.T

sor,
            va
u
: torch.T

sor,
            cu_s
q


s: torch.T

sor | No

 = No

,
            max_s
q


: torch.T

sor | No

 = No

,  # O

y us
d for F
ash Att

t
o

        ) -
 torch.T

sor:
            # Ca
 FA or TORCH_SDPA 
mp

m

tat
o
...
        d
f for
ard_cpu(
            s

f,
            qu
ry: torch.T

sor,
            k
y: torch.T

sor,
            va
u
: torch.T

sor,
            cu_s
q


s: torch.T

sor | No

 = No

,
            max_s
q


: torch.T

sor | No

 = No

,  # O

y us
d for F
ash Att

t
o

        ) -
 torch.T

sor:
            # Ca
 TORCH_SDPA 
mp

m

tat
o
...
        d
f for
ard_xpu(
            s

f,
            qu
ry: torch.T

sor,
            k
y: torch.T

sor,
            va
u
: torch.T

sor,
            cu_s
q


s: torch.T

sor | No

 = No

,
            max_s
q


: torch.T

sor | No

 = No

,  # O

y us
d for F
ash Att

t
o

        ) -
 torch.T

sor:
            # Ca
 FA 
mp

m

tat
o
...
        d
f for
ard_tpu(
            s

f,
            qu
ry: torch.T

sor,
            k
y: torch.T

sor,
            va
u
: torch.T

sor,
            cu_s
q


s: torch.T

sor | No

 = No

,
            max_s
q


: torch.T

sor | No

 = No

,  # O

y us
d for F
ash Att

t
o

        ) -
 torch.T

sor:
            # Ca
 PALLAS 
mp

m

tat
o
...
```
### R
g
st
r a N

 CustomOp 

 OOT D
v
c
 P
ug

s
Curr

t
y, tha
ks to [vLLM's hard
ar
-p
ug

 m
cha

sm](./p
ug

_syst
m.md), th
r
 ar
 var
ous OOT d
v
c
 p
ug

s 
m
rg

g out to 

ab

 vLLM s
am

ss
y ru
s o
 d
ff
r

t hard
ar
s. You ca
 a
so f

d mor
 d
ta

s about th
s m
cha

sm at [I
troduc

g vLLM Hard
ar
 P
ug

, B
st Pract
c
 from Asc

d NPU](https://b
og.v
m.a
/2025/05/12/hard
ar
-p
ug

.htm
).
    - **Off
c
a
 d
v
c
 p
ug

s:** [v
m-asc

d](https://g
thub.com/v
m-proj
ct/v
m-asc

d) (for Hua


 Asc

d NPU), [v
m-spyr
](https://g
thub.com/v
m-proj
ct/v
m-spyr
)
(for Spyr
), [v
m-gaud
](https://g
thub.com/v
m-proj
ct/v
m-gaud
) (for I
t

 Gaud
), [v
m-

uro
](https://g
thub.com/v
m-proj
ct/v
m-

uro
) (for AWS N
uro
), [v
m-m
ta](https://g
thub.com/v
m-proj
ct/v
m-m
ta
) (for App

 S


co
), 
tc.
    - **No
-off
c
a
 d
v
c
 p
ug

s:** [v
m-m
tax](https://g
thub.com/M
taX-MACA/vLLM-m
tax) (for M
taX GPU), [v
m-ku

u
](https://g
thub.com/ba
du/vLLM-Ku

u
) (for Ba
du Ku

u
 XPU), 
tc.
I
 th
s cas
, `CustomOp` ca
 

ab

 th
s
 hard
ar
 ma
ufactur
rs to s
am

ss
y r
p
ac
 vLLM's op
rat
o
s 

th th

r d
p-opt
m
z
d k
r


s for sp
c
f
c d
v
c
s at ru
t
m
, by just r
g
st
r

g a
 OOT `CustomOp` a
d 
mp

m

t

g th
 `for
ard_oot()` m
thod.
No
, th
s part 


 sho
 you ho
 to r
g
st
r a
 OOT `CustomOp` for a d
v
c
 p
ug

.
Tak

g `MME
cod
rAtt

t
o
` as a
 
xamp

:
1. Imp

m

t a `CustomMME
cod
rAtt

t
o
` c
ass 
h
ch 
xt

ds from `MME
cod
rAtt

t
o
` a
d 
mp

m

t 
ts `for
ard_oot()` m
thod.
2. R
g
st
r your `CustomMME
cod
rAtt

t
o
` 

to vLLM to r
p
ac
 `MME
cod
rAtt

t
o
`.
??? cod

    ```pytho

    from v
m.mod

_
x
cutor.
ay
rs.att

t
o
 
mport MME
cod
rAtt

t
o

    from v
m.mod

_
x
cutor.custom_op 
mport CustomOp
    @CustomOp.r
g
st
r_oot("MME
cod
rAtt

t
o
")
    c
ass CustomMME
cod
rAtt

t
o
(MME
cod
rAtt

t
o
):
        d
f __


t__(...):
            sup
r().__


t__(...)
        d
f for
ard_oot(...):
            # Ca
 opt
m
z
d d
v
c
-sp
c
f
c k
r


s.
            ...
```
I
 th
s cas
, a 


 
t
m `{"MME
cod
rAtt

t
o
": CustomMME
cod
rAtt

t
o
}` 


 b
 add
d 

to `op_r
g
stry_oot`. Wh

 


t
a

z

g a `MME
cod
rAtt

t
o
` op obj
ct, 
f th
 c
ass 
am
 (
.
., `MME
cod
rAtt

t
o
`) 
s co
ta


d 

 th
 k
ys of `op_r
g
stry_oot`, vLLM 


 r
p
ac
 
t 

th our r
g
st
r
d c
ass (
.
., `CustomMME
cod
rAtt

t
o
`) a
d 

sta
t
at
 
t.
Aft
r that, 
h

 th
s `MME
cod
rAtt

t
o
` op 
s ca

d, your `for
ard_oot()` 


 b
 ca

d 
f 
t 
s 

ab

d. Thus, you 


 g
t 
xp
ct
d p
rforma
c
 o
 your hard
ar
s 

thout d
r
ct
y mod
fy vLLM.
I
 add
t
o
, you ca
 a
so r
g
st
r a
 your `CustomOp` at o

 p
ac
 for b
tt
r ma
ag
m

t.
??? cod

    ```pytho

    from v
m.mod

_
x
cutor.custom_op 
mport CustomOp
    REGISTERED_CUSTOM_OPS = {
        "CustomOP1": YourCustomOp1,
        "CustomOP2": YourCustomOp2,
        "CustomOP3": YourCustomOp3,
    }
    for op_
am
, op_c
s 

 REGISTERED_CUSTOM_OPS.
t
ms():
        CustomOp.r
g
st
r_oot(_d
corat
d_op_c
s=op_c
s, 
am
=op_
am
)
```
