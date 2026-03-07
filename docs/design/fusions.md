# Fus
o
 torch.comp


 pass
s
vLLM app


s a s
t of k
r


/op
rator fus
o
s at comp


 t
m
 (v
a custom [`torch.comp


`](torch_comp


.md) I
ductor pass
s)
to s
parat
 opt
m
zat
o
s from mod

 d
f


t
o
s a
d avo
d br
ak

g 
ay
r abstract
o
s 

 mod

 cod
.
Th
s
 fus
o
s ar
 co
tro

d by f


ds 

 [`PassCo
f
g`][v
m.co
f
g.comp

at
o
.PassCo
f
g] a
d ar
 automat
ca
y 

ab

d
at appropr
at
 [opt
m
zat
o
 

v

s](opt
m
zat
o
_

v

s.md).
## Qu
ck R
f
r

c

Th
 tab

 b

o
 maps 
ach fus
o
 to 
ts co
tro


g f
ag/co
f
g k
ob, th

op
rat
o
s 
t fus
s, 
hat 

v

 

ab

s 
t by d
fau
t, a
d a
 

d
cat
v
 sp
dup.
Th
 Fu
graph co
um
 

d
cat
s 
h
th
r th
 fus
o
 r
qu
r
s th
 

t
r
 mod

 graph to b

v
s
b

 (

th
r v
a I
ductor part
t
o
 or `sp

tt

g_ops=[]`),
a
d th
 
ast co
um
 

d
cat
s 
h
th
r th
 fus
o
 act
vat
s for a
 `
um_tok

s`
or just o
 th
 
o
 or h
gh 

d.
!!! 

fo
    Sp
dup d
p

ds h
av

y o
 th
 
xact mod

, batch s
z
, a
d hard
ar
.
    If tu


g p
rforma
c
 by ha
d, a

ays b

chmark your 
xact us
-cas
 

th a
d 

thout th
 fus
o
 to v
r
fy th
 
mpact.
| Fus
o
                                                                         | `PassCo
f
g` f
ag            | Fus
d op
rat
o
s                               | D
fau
t at                     | E2E Sp
dup        | Fu
graph | `
um_tok

s` |
|--------------------------------------------------------------------------------|------------------------------|------------------------------------------------|--------------------------------|--------------------|-----------|--------------|
| [A
R
duc
 + RMSNorm](#a
r
duc
--rms
orm-fus
_a
r
duc
_rms)                  | `fus
_a
r
duc
_rms`         | A
-r
duc
 → RMSNorm (+r
s
dua
_add) (→ qua
t) | O2 (Hopp
r/B
ack


 + TP 
 1) | 5-20%              | No        | Lo
          |
| [Att

t
o
 + Qua
t](#att

t
o
--qua
t
zat
o
-fus
_att
_qua
t)                  | `fus
_att
_qua
t`            | Att

t
o
 output → FP8/NVFP4 qua
t             | Off by d
fau
t                 | 3-7%               | Y
s       | A

ays       |
| [RoPE + KV-Cach
 Updat
](#rop
--kv-cach
-updat
-fus
_rop
_kvcach
)             | `fus
_rop
_kvcach
`          | Rotary 
mb
dd

g → KV cach
 
r
t
              | O1 (ROCm/AITER o

y)           | TBD                | No        | Lo
          |
| [QK Norm + RoPE](#qk-
orm--rop
-

ab

_qk_
orm_rop
_fus
o
)                    | `

ab

_qk_
orm_rop
_fus
o
` | Q/K RMSNorm → rotary 
mb
dd

g                 | Off by d
fau
t                 | 2-3%               | No        | Lo
          |
| [S
qu

c
 Para



sm](#s
qu

c
-para



sm-

ab

_sp)                        | `

ab

_sp`                  | A
R
duc
 → R
duc
Scatt
r + A
Gath
r          | Off by d
fau
t                 | Pr
r
q for Asy
cTP | Y
s       | H
gh         |
| [Asy
cTP GEMM + co

ct
v
](#asy
ctp-g
mm--co

ct
v
-ov
r
ap-fus
_g
mm_comms) | `fus
_g
mm_comms`            | GEMM → r
duc
-scatt
r / a
-gath
r → GEMM      | Off by d
fau
t                 | 7-10%              | Y
s       | H
gh         |
| [RMSNorm + Qua
t](#rms
orm--qua
t
zat
o
-fus
_
orm_qua
t)                      | `fus
_
orm_qua
t`            | RMSNorm (+r
s
dua
 add) → FP8/FP4 qua
t        | O1 (co
d
t
o
a
)               | 1-4%               | No        | A

ays       |
| [S
LU+Mu
 + Qua
t](#s

umu
--qua
t
zat
o
-fus
_act_qua
t)                      | `fus
_act_qua
t`             | S
LU+Mu
 act
vat
o
 → FP8/FP4 qua
t            | O1 (co
d
t
o
a
)               | 1-4%               | No        | A

ays       |
| [RMSNorm + Padd

g](#rms
orm--padd

g-fus
_act_padd

g)                        | `fus
_act_padd

g`           | R
s
dua
 add + RMSNorm → padd

g               | O1 (ROCm/AITER o

y)           | TBD                | No        | A

ays       |
## Support Matr
x
Th
 tab

 b

o
 

sts th
 qua
t
zat
o
 sch
m
s support
d by 
ach fus
o
 o
 
ach p
atform.
**—** m
a
s th
 fus
o
 
s 
ot ava

ab

 o
 that p
atform. Th
 
at
st a
d 

-progr
ss 
ork 
s ava

ab

 

 th
 track

g 
ssu
:
[#36066](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/36066)
| Fus
o
                       | SM100 (B
ack


)                        | SM90 (Hopp
r)                            | SM89 (Ada)                               | SM80 (Amp
r
) | ROCm                                     |
|------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|---------------|------------------------------------------|
| `fus
_a
r
duc
_rms`         | FP16/BF16, FP8 stat
c, NVFP4             | FP16/BF16, FP8 stat
c                    | —                                        | —             | —                                        |
| `fus
_att
_qua
t`\*          | FP8 stat
c\*, NVFP4\*                    | FP8 stat
c\*                             | FP8 stat
c\*                             | —             | FP8 stat
c\*                             |
| `fus
_rop
_kvcach
`          | —                                        | —                                        | —                                        | —             | FP16/BF16                                |
| `

ab

_qk_
orm_rop
_fus
o
` | FP16/BF16                                | FP16/BF16                                | FP16/BF16†                               | FP16/BF16†    | —                                        |
| `

ab

_sp`                  | FP16/BF16, FP8 stat
c†                   | FP16/BF16, FP8 stat
c                    | FP16/BF16†                               | FP16/BF16†    | —                                        |
| `fus
_g
mm_comms`            | FP16/BF16, FP8 stat
c†                   | FP16/BF16, FP8 stat
c                    | FP16/BF16†                               | FP16/BF16†    | —                                        |
| `fus
_
orm_qua
t`            | FP8 stat
c, FP8 p
r-tok

, FP8 p
r-group | FP8 stat
c, FP8 p
r-tok

, FP8 p
r-group | FP8 stat
c, FP8 p
r-tok

, FP8 p
r-group | —             | FP8 stat
c, FP8 p
r-tok

, FP8 p
r-group |
| `fus
_act_qua
t`             | FP8 stat
c, NVFP4                        | FP8 stat
c                               | FP8 stat
c                               | —             | FP8 p
r-group                            |
| `fus
_act_padd

g`           | —                                        | —                                        | —                                        | —             | FP16/BF16                                |
\* `fus
_att
_qua
t` support d
p

ds o
 th
 att

t
o
 back

d 

 us
; 
ot a
 back

ds support
fus
d qua
t
zat
o
 output. S
 th
 [`fus
_att
_qua
t` s
ct
o
](#att

t
o
--qua
t
zat
o
-fus
_att
_qua
t)
for p
r-back

d d
ta

s.
† `

ab

_sp` a
d `fus
_g
mm_comms` ar
 o

y autoco
f
gur
d for SM90 today;
oth
r arch
t
ctur
s support r
qu
r
s s
tt

g `PassCo
f
g.sp_m

_tok

_
um` 
xp

c
t
y.
SM100 support a
so r
qu
r
s s
tt

g `VLLM_DISABLED_KERNELS=F
ashI
f
rFP8Sca

dMML


arK
r


`.
## E
ab


g / D
sab


g Fus
o
s
Fus
o
s ar
 
xpos
d through `PassCo
f
g`, 
h
ch 
s 

st
d 

s
d
 `Comp

at
o
Co
f
g`:
```pytho

from v
m 
mport LLM
from v
m.co
f
g 
mport Comp

at
o
Co
f
g, PassCo
f
g

m = LLM(
    mod

="...",
    opt
m
zat
o
_

v

=2, # D
fau
t opt
m
zat
o
 

v


    comp

at
o
_co
f
g=Comp

at
o
Co
f
g(
        pass_co
f
g=PassCo
f
g(
            fus
_
orm_qua
t=Tru
,
            fus
_act_qua
t=Tru
,
            fus
_a
r
duc
_rms=Fa
s
,  # d
sab

 a sp
c
f
c fus
o

        )
    ),
)
```
Fus
o
s ca
 a
so b
 

ab

d us

g comma
d-



 f
ags 

th a
y `v
m ...` comma
d:
```bash
# E
ab

 O2 d
fau
ts, but tur
 off a
r
duc
 fus
o

v
m s
rv
 m
ta-
ama/L
ama-3.1-8B-I
struct -O2 -cc.pass_co
f
g.fus
_a
r
duc
_rms=Fa
s

# Th
 abov
 
s 
qu
va


t to th
 mor
 v
rbos
:
v
m s
rv
 m
ta-
ama/L
ama-3.1-8B-I
struct -O2 --comp

at
o
-co
f
g '{"pass_co
f
g": {"fus
_a
r
duc
_rms": fa
s
}}'
# Sam
 sy
tax 

 oth
r comma
ds, 
.g. v
m b

ch:
v
m b

ch 
at

cy --mod

=m
ta-
ama/L
ama-3.1-8B-I
struct -O2 -cc.pass_co
f
g.fus
_a
r
duc
_rms=Fa
s

```
F


ds s
t 
xp

c
t
y by th
 us
r a

ays tak
 pr
c
d

c
 ov
r opt
m
zat
o
-

v

 d
fau
ts.
## Fus
o
 D
ta

s
### A
R
duc
 + RMSNorm (`fus
_a
r
duc
_rms`)
!!! 
ar


g
    TP+DP a
d TP+PP comb

at
o
s ar
 curr

t
y brok


    ([#34458](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/34458) a
d
    [#35426](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/35426)).
    O

y support
d o
 NVIDIA Hopp
r (SM90) a
d B
ack


 (SM100) 

th F
ashI
f
r 

sta

d.
**What 
t fus
s.** Fus
s th
 t

sor-para


 a
-r
duc
 co

ct
v
 

th th
 subs
qu

t r
s
dua
 add,
RMSNorm, a
d opt
o
a
y a qua
t
zat
o
 st
p 

to a s

g

 F
ashI
f
r / TRT-LLM commu

cat
o
 k
r


.
Th
s fus
o
 
s o

y prof
tab

 for sma
 `
um_tok

s`,
so th
 fus
o
 
s o

y p
rform
d 

 th
 
o

r comp


d ra
g
.
Patt
r
s cov
r
d:
    - `A
R
duc
 → RMSNorm(+r
s
dua
_add)`: CUDA sm90+ 

th F
ashI
f
r
    - `A
R
duc
 → RMSNorm(+r
s
dua
_add) → FP8 stat
c qua
t`: CUDA sm90+ 

th F
ashI
f
r
    - `A
R
duc
 → RMSNorm(+r
s
dua
_add) → NVFP4 dy
am
c qua
t`: CUDA sm100+ 

th F
ashI
f
r
Th
 max
mum t

sor s
z
 b

o
 
h
ch th
 fus
d k
r


 
s us
d 
s hard
ar
-d
p

d

t (64 MB for TP=2
o
 SM90/SM100) a
d co
f
gurab

 v
a `PassCo
f
g.f
_a
r
duc
_fus
o
_max_s
z
_mb`.
**Cod
 
ocat
o
s.**
    - Pass: [`v
m/comp

at
o
/pass
s/fus
o
/a
r
duc
_rms_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/a
r
duc
_rms_fus
o
.py)
    - F
ashI
f
r a
-r
duc
: [`v
m/d
str
but
d/d
v
c
_commu

cators/f
ash

f
r_a
_r
duc
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/d
str
but
d/d
v
c
_commu

cators/f
ash

f
r_a
_r
duc
.py)
    - B

chmark: [`b

chmarks/k
r


s/b

chmark_fus
d_co

ct
v
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/b

chmarks/k
r


s/b

chmark_fus
d_co

ct
v
.py)
### Att

t
o
 + Qua
t
zat
o
 (`fus
_att
_qua
t`)
!!! 

fo
    `fus
_att
_qua
t` 
s curr

t
y 
ot 

ab

d at a
y opt
m
zat
o
 

v

 by d
fau
t a
d must b
 s
t
    
xp

c
t
y. It r
qu
r
s th
 fu
 mod

 graph to b
 v
s
b

 (I
ductor part
t
o
 or `sp

tt

g_ops=[]`).
**What 
t fus
s.** Fus
s th
 att

t
o
 output qua
t
zat
o
 d
r
ct
y aft
r th
 att

t
o
 computat
o
,



m

at

g a fu
-pr
c
s
o
 m
mory rou
d-tr
p of th
 att

t
o
 output. Patt
r
s cov
r
d:
`Att

t
o
 → FP8 stat
c qua
t`:
    - `TRITON_ATTN`: CUDA, ROCm
    - `FLASHINFER`: CUDA sm100+ 

th F
ashI
f
r 

sta

d
    - `ROCM_ATTN`: ROCm
    - `ROCM_AITER_UNIFIED_ATTN`: ROCm 

th AITER
`Att

t
o
 → NVFP4 dy
am
c qua
t`:
    - `FLASHINFER`: CUDA sm100+ 

th F
ashI
f
r 

sta

d
Oth
r att

t
o
 back

ds do 
ot support fus
d output qua
t
zat
o
 y
t.
**Cod
 
ocat
o
s.**
    - Pass: [`v
m/comp

at
o
/pass
s/fus
o
/att
_qua
t_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/att
_qua
t_fus
o
.py)
    - Att

t
o
 back

ds: [`v
m/v1/att

t
o
/back

ds/`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/v1/att

t
o
/back

ds/)
### RoPE + KV-Cach
 Updat
 (`fus
_rop
_kvcach
`)
!!! 

fo
    ROCm/AITER-o

y. Not ava

ab

 o
 NVIDIA CUDA or CPU. Th
 fus
o
 
s o

y 

ab

d for
    `
um_tok

s ≤ 256` by d
fau
t du
 to AITER fus
d k
r


 p
rforma
c
 
ssu
s.
    Th
s thr
sho
d 
s co
f
gurab

 v
a `PassCo
f
g.rop
_kvcach
_fus
o
_max_tok

_
um`.
**What 
t fus
s.** Fus
s th
 rotary pos
t
o
a
 
mb
dd

g k
r


 

th th
 KV-cach
 scatt
r/
r
t
 

to
a s

g

 k
r


, avo
d

g s
parat
 r
ads a
d 
r
t
s of th
 k
y a
d va
u
 t

sors.
R
qu
r
s: AMD ROCm 

th AITER 

ab

d, th
 `rotary_
mb
dd

g` custom op act
v
 (automat
c),
a
d th
 `kv_cach
` updat
 op v
s
b

 

 th
 graph: 

th
r by us

g I
ductor graph part
t
o

or r
mov
d from `sp

tt

g_ops`.
If th
s
 co
d
t
o
s ar
 s
t, th
 fus
o
 
s 

ab

d automat
ca
y for opt
m
zat
o
 

v

 O1 a
d abov
.
**Cod
 
ocat
o
s.**
    - Pass: [`v
m/comp

at
o
/pass
s/fus
o
/rop
_kvcach
_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/rop
_kvcach
_fus
o
.py)
### S
qu

c
 Para



sm (`

ab

_sp`)
**What 
t fus
s.** R
p
ac
s a
-r
duc
 co

ct
v
s 

th r
duc
-scatt
r + 
oca
 RMSNorm + a
-gath
r,
sp

tt

g th
 s
qu

c
 d
m

s
o
 across TP ra
ks. Th
s r
structur
s th
 graph so th
 subs
qu

t Asy
cTP
pass ca
 fus
 th
 r
duc
-scatt
r / a
-gath
r 

th th
 surrou
d

g GEMMs.
S
qu

c
 Para



sm 
ts

f do
s 
ot d
r
ct
y 
mprov
 p
rforma
c
; 
t 
s a pr
r
qu
s
t
 for th

Asy
cTP pass (`fus
_g
mm_comms`). SP 
s o

y app


d abov
 a m


mum tok

 thr
sho
d that 
s
autoco
f
gur
d bas
d o
 d
v
c
 capab


ty a
d mod

 `h
dd

_s
z
`. Curr

t
y o

y act
v
 o

H100/SM90 for mod

s 

th `h
dd

_s
z
 
= 8192`. Th
 thr
sho
d 
s co
f
gurab

 v
a
`PassCo
f
g.sp_m

_tok

_
um`.
Th
 g


ra
 tra
sformat
o
:
```t
xt
I
put → A
R
duc
 → RMSNorm → Output
b
com
s:
I
put → R
duc
Scatt
r → 
oca
 RMSNorm → A
Gath
r → Output
```
Patt
r
s cov
r
d:
    - F
rst b
ock: `A
R
duc
 → RMSNorm` → `R
duc
Scatt
r → RMSNorm → A
Gath
r`
    - M
dd

 b
ocks: `A
R
duc
 → fus
d_add_RMSNorm` → `R
duc
Scatt
r → fus
d_add_RMSNorm → A
Gath
r`
    - Both 

th opt
o
a
 `→ FP8 stat
c qua
t` suff
x
R
qu
r
s: `us
_

ductor_graph_part
t
o
=Tru
` **or** p

c


s
 comp

at
o
 

th stat
c s
z
s
d
v
s
b

 by `t

sor_para


_s
z
`.
Support
d hard
ar
: O

y t
st
d o
 NVIDIA CUDA, poss
b
y 
orks o
 ROCm. FP8 a
-gath
r r
qu
r
s sm90+.
**Cod
 
ocat
o
s.**
    - Pass: [`v
m/comp

at
o
/pass
s/fus
o
/s
qu

c
_para



sm.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/s
qu

c
_para



sm.py)
### Asy
cTP GEMM + Co

ct
v
 Ov
r
ap (`fus
_g
mm_comms`)
!!! 

fo
    R
qu
r
s `

ab

_sp=Tru
` (

ab

d automat
ca
y). Th
s pass 
s a 
o-op 
f S
qu

c
 Para



sm has 
ot b

 app


d.
**What 
t fus
s.** Aft
r S
qu

c
 Para



sm tra
sforms th
 graph, fus
s GEMM k
r


s 

th th

surrou
d

g r
duc
-scatt
r (output proj
ct
o
) a
d a
-gath
r (

put proj
ct
o
) us

g
`torch.ops.symm_m
m` symm
tr
c-m
mory pr
m
t
v
s, ov
r
app

g commu

cat
o
 a
d computat
o
.
Th
s ov
r
ap 
s o

y prof
tab

 for 
arg
 `
um_tok

s`, so th
 fus
o
 (a
d pr
c
d

g SP)

s o

y p
rform
d 

 th
 h
gh
r comp


d ra
g
 abov
 `PassCo
f
g.sp_m

_tok

_
um`.
Patt
r
s cov
r
d:
    - `GEMM → r
duc
-scatt
r` → `fus
d_matmu
_r
duc
_scatt
r`
    - `a
-gath
r → GEMM` → `a
_gath
r_matmu
`
    - FP8 sca

d var
a
ts of both patt
r
s
Support
d hard
ar
: NVIDIA CUDA 

th symm
tr
c-m
mory (`torch.d
str
but
d._symm
tr
c_m
mory`) support.
O
 B200, patt
r
-match

g fp8 F
ashI
f
r sca

d MM 
s 
ot support
d, so 
t must b
 d
sab

d
([#27893](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/27893))
```sh


VLLM_DISABLED_KERNELS=F
ashI
f
rFP8Sca

dMML


arK
r


 ...
```
**Cod
 
ocat
o
s.**
    - Pass: [`v
m/comp

at
o
/pass
s/fus
o
/co

ct
v
_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/co

ct
v
_fus
o
.py)
    - S
qu

c
 para



sm pass: [`v
m/comp

at
o
/pass
s/fus
o
/s
qu

c
_para



sm.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/s
qu

c
_para



sm.py)
### QK Norm + RoPE (`

ab

_qk_
orm_rop
_fus
o
`)
!!! 

fo
    O

y app

cab

 to mod

s that app
y p
r-h
ad RMSNorm to Q a
d K b
for
 rotary pos
t
o
a

    
mb
dd

g (
.g. Q


). Not 

ab

d by d
fau
t at a
y opt
m
zat
o
 

v

 du
 to p
rf 
ssu
s o
 H100:
    [#34391](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/34391)
**What 
t fus
s.** Fus
s th
 s
qu

c
: sp

t QKV → r
shap
 → Q/K RMSNorm → r
shap
 → rotary

mb
dd

g 

to a s

g

 `fus
d_qk_
orm_rop
` CUDA k
r


.
```t
xt
# U
fus
d:
q, k, v = sp

t(qkv)
q_
orm = rms_
orm(q.v


(h
ads))
k_
orm = rms_
orm(k.v


(kv_h
ads))
q_rop
, k_rop
 = rotary_
mb
dd

g(q_
orm, k_
orm, ...)
# Fus
d:
fus
d_qk_
orm_rop
(qkv, ...)
```
Support
d hard
ar
: CUDA (sm80+) o

y, t
st
d o

y o
 sm90 a
d sm100.
**Cod
 
ocat
o
s.**
    - Pass: [`v
m/comp

at
o
/pass
s/fus
o
/qk_
orm_rop
_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/qk_
orm_rop
_fus
o
.py)
    - CUDA k
r


: [`csrc/ops.h`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/csrc/ops.h) (`fus
d_qk_
orm_rop
`)
### RMSNorm + Qua
t
zat
o
 (`fus
_
orm_qua
t`)
!!! 
ar


g
    O
 NVIDIA, I
ductor actua
y g


rat
s a fast
r fus
d k
r


 tha
 our custom CUDA k
r


.
    H

c
, th
s fus
o
 
s o

y 

ab

d 
h

 

th
r `rms_
orm` or `qua
t_fp8` 
s us

g a custom k
r


.
**What 
t fus
s.** Comb


s th
 custom `rms_
orm` / `fus
d_add_rms_
orm`
op
rat
o
s 

th subs
qu

t qua
t
zat
o
 

to a s

g

 fus
d k
r


,



m

at

g a
 

t
rm
d
at
 r
ad/
r
t
 of th
 fu
-pr
c
s
o
 act
vat
o
 t

sor.
T
o var
a
ts ar
 fus
d:
    - *P
a

 RMSNorm + qua
t*: `rms_
orm(x) → qua
t_fp8(y)`
    - *Fus
d-add RMSNorm + qua
t*: `fus
d_add_rms_
orm(x, r
s
dua
) → qua
t_fp8(y)` — a
so updat
s th
 r
s
dua
 

-p
ac
.
Not
 that AITER fus
o
s ar
 curr

t
y 

 a s
parat
 pass 

 `v
m.comp

at
o
.pass
s.fus
o
.rocm_a
t
r_fus
o
`.
Support
d qua
t
zat
o
 sch
m
/hard
ar
 comb

at
o
s:
    - FP8 stat
c p
r-t

sor: CUDA & HIP k
r



    - FP8 dy
am
c p
r-tok

: CUDA & HIP k
r


, AITER
    - FP8 dy
am
c p
r-tok

-group (128/64): CUDA & HIP k
r


, AITER
**Cod
 
ocat
o
s.**
    - Pass: [`v
m/comp

at
o
/pass
s/fus
o
/rms_qua
t_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/rms_qua
t_fus
o
.py)
    - ROCm AITER pass: [`v
m/comp

at
o
/pass
s/fus
o
/rocm_a
t
r_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/rocm_a
t
r_fus
o
.py)
    - CUDA/HIP k
r


s: [`csrc/
ay
r
orm_qua
t_k
r


s.cu`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/csrc/
ay
r
orm_qua
t_k
r


s.cu)
### S
LU+Mu
 + Qua
t
zat
o
 (`fus
_act_qua
t`)
!!! 
ar


g
    Sam
 as `fus
_
orm_qua
t`: o
 NVIDIA, I
ductor g


rat
s a fast
r fus
d k
r


 tha
 our custom ops.
    Th
s fus
o
 
s o

y 

ab

d 
h

 

th
r `s

u_a
d_mu
` or `qua
t_fp8` ar
 us

g a custom k
r


,
    or for NVFP4-qua
t
z
d mod

s (
h
r
 FP4 qua
t 
s a

ays a custom op).
**What 
t fus
s.** Fus
s th
 `s

u_a
d_mu
` gat
-up proj
ct
o
 act
vat
o
 

th subs
qu

t qua
t
zat
o
 

to a s

g

 k
r


,
avo
d

g mat
r
a

zat
o
 of th
 fu
-pr
c
s
o
 post-act
vat
o
 t

sor.
Not
 that AITER fus
o
s ar
 

 a s
parat
 pass 

 `v
m.comp

at
o
.pass
s.fus
o
.rocm_a
t
r_fus
o
`.
Support
d qua
t
zat
o
 sch
m
/hard
ar
 comb

at
o
s:
    - FP8 stat
c p
r-t

sor: CUDA & HIP k
r



    - NVFP4 dy
am
c: CUDA sm100+ o

y 

th F
ashI
f
r
    - FP8 p
r-tok

-group (128): ROCm AITER o

y
**Cod
 
ocat
o
s.**
    - Pass: [`v
m/comp

at
o
/pass
s/fus
o
/act_qua
t_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/act_qua
t_fus
o
.py)
    - ROCm AITER pass: [`v
m/comp

at
o
/pass
s/fus
o
/rocm_a
t
r_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/rocm_a
t
r_fus
o
.py)
    - CUDA/HIP k
r


s: [`csrc/qua
t
zat
o
/`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/csrc/qua
t
zat
o
/)
### RMSNorm + Padd

g (`fus
_act_padd

g`)
!!! 

fo
    ROCm/AITER-o

y. Targ
t
d at GPT-OSS mod

s.
**What 
t fus
s.** Fus
s a r
s
dua
 add + RMSNorm 

th a subs
qu

t padd

g op
rat
o
 that pads
th
 h
dd

 d
m

s
o
 to a mu
t
p

 r
qu
r
d by do

str
am AITER Tr
to
 GEMM k
r


s.
R
qu
r
s: AMD ROCm 

th AITER RMSNorm 

ab

d. E
ab

d by d
fau
t 

 opt
m
zat
o
 

v

 O1 a
d abov


h

 th
 h
dd

 s
z
 
s 2880 a
d AITER Tr
to
 GEMMs *
ot* 

ab

d.
**Cod
 
ocat
o
s.**
    - Pass: [`v
m/comp

at
o
/pass
s/fus
o
/rocm_a
t
r_fus
o
.py`](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/comp

at
o
/pass
s/fus
o
/rocm_a
t
r_fus
o
.py) (`RocmA
t
rTr
to
AddRMSNormPadFus
o
Pass`)
## S
 A
so
    - [Opt
m
zat
o
 L
v

s](opt
m
zat
o
_

v

s.md) — h
gh-

v

 pr
s
ts that s
t
  fus
o
 d
fau
ts.
    - [torch.comp


 

 vLLM](torch_comp


.md) — ho
 th
 I
ductor pass p
p





  
orks.
    - [Att

t
o
 Back

ds](att

t
o
_back

ds.md) — att

t
o
-sp
c
f
c k
r



  s


ct
o
.
