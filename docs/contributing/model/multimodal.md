# Mu
t
-Moda
 Support
Th
s docum

t 
a
ks you through th
 st
ps to 
xt

d a bas
c mod

 so that 
t acc
pts [mu
t
-moda
 

puts](../../f
atur
s/mu
t
moda
_

puts.md).
## 1. Updat
 th
 bas
 vLLM mod


It 
s assum
d that you hav
 a
r
ady 
mp

m

t
d th
 mod

 

 vLLM accord

g to [th
s
 st
ps](bas
c.md).
Furth
r updat
 th
 mod

 as fo
o
s:
    - Imp

m

t [g
t_p
ac
ho
d
r_str][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsMu
t
Moda
.g
t_p
ac
ho
d
r_str] to d
f


 th
 p
ac
ho
d
r str

g 
h
ch 
s us
d to r
pr
s

t th
 mu
t
-moda
 
t
m 

 th
 t
xt prompt. Th
s shou
d b
 co
s
st

t 

th th
 chat t
mp
at
 of th
 mod

.
    ??? cod

        ```pytho

        c
ass YourMod

ForImag
2S
q(
.Modu

):
            ...
            @c
assm
thod
            d
f g
t_p
ac
ho
d
r_str(c
s, moda

ty: str, 
: 

t) -
 str | No

:
                
f moda

ty.starts

th("
mag
"):
                    r
tur
 "

mag

"
                ra
s
 Va
u
Error("O

y 
mag
 moda

ty 
s support
d")
```
    - I
s
d
 `__


t__` m
thod, 


t
a

z
 th
 
a
guag
 compo


ts of th
 mod

 

s
d
 [_mark_
a
guag
_mod

][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsMu
t
Moda
._mark_
a
guag
_mod

], a
d th
 mu
t
moda
 compo


ts of th
 mod

 

s
d
 [_mark_to

r_mod

][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsMu
t
Moda
._mark_to

r_mod

], 
.g.:
    ```pytho

        d
f __


t__(s

f, *, v
m_co
f
g: V
mCo
f
g, pr
f
x: str = "") -
 No

:
            sup
r().__


t__()
            co
f
g = v
m_co
f
g.mod

_co
f
g.hf_co
f
g
            

th s

f._mark_to

r_mod

(v
m_co
f
g, "
mag
"):
                s

f.v
s
o
_

cod
r = ...
                s

f.mu
t
_moda
_proj
ctor = ...
            

th s

f._mark_
a
guag
_mod

(v
m_co
f
g):
                s

f.
a
guag
_mod

 = 


t_v
m_r
g
st
r
d_mod

(
                    v
m_co
f
g=v
m_co
f
g,
                    hf_co
f
g=co
f
g.t
xt_co
f
g,
                    pr
f
x=mayb
_pr
f
x(pr
f
x, "
a
guag
_mod

"),
                )
```
    - R
mov
 th
 
mb
dd

g part from th
 [for
ard][torch.
.Modu

.for
ard] m
thod:
    - Mov
 th
 mu
t
-moda
 
mb
dd

g to [
mb
d_mu
t
moda
][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsMu
t
Moda
.
mb
d_mu
t
moda
].
    - Th
 t
xt 
mb
dd

g a
d 
mb
dd

g m
rg
 ar
 ha
d

d automat
ca
y by a d
fau
t 
mp

m

tat
o
 of [
mb
d_

put_
ds][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsMu
t
Moda
.
mb
d_

put_
ds]. It do
s 
ot 

d to b
 ov
rr
dd

 

 most cas
s.
    ```d
ff
      d
f for
ard(
          s

f,
          

put_
ds: torch.T

sor | No

,
    -     p
x

_va
u
s: torch.T

sor,
          pos
t
o
s: torch.T

sor,
          

t
rm
d
at
_t

sors: I
t
rm
d
at
T

sors | No

 = No

,
          

puts_
mb
ds: torch.T

sor | No

 = No

,
      ) -
 torch.T

sor:
    -     
f 

puts_
mb
ds 
s No

:
    -         

puts_
mb
ds = s

f.g
t_

put_
mb
dd

gs()(

put_
ds)
    -
    -     
f p
x

_va
u
s 
s 
ot No

:
    -         
mag
_f
atur
s = s

f.g
t_
mag
_f
atur
s(
    -             p
x

_va
u
s=p
x

_va
u
s,
    -         )
    -         sp
c
a
_
mag
_mask = s

f.g
t_p
ac
ho
d
r_mask(
    -             

put_
ds,
    -             

puts_
mb
ds=

puts_
mb
ds,
    -             
mag
_f
atur
s=
mag
_f
atur
s,
    -         )
    -         

puts_
mb
ds = 

puts_
mb
ds.mask
d_scatt
r(
    -             sp
c
a
_
mag
_mask,
    -             
mag
_f
atur
s,
    -         )
           h
dd

_stat
s = s

f.
a
guag
_mod

(
               

put_
ds,
               pos
t
o
s,
               

t
rm
d
at
_t

sors,
               

puts_
mb
ds=

puts_
mb
ds,
           )
         ...
  
    +  d
f 
mb
d_mu
t
moda
(
    +      s

f,
    +      p
x

_va
u
s: torch.T

sor,
    +  ) -
 Mu
t
Moda
Emb
dd

gs | No

:
    +      r
tur
 s

f.g
t_
mag
_f
atur
s(
    +          p
x

_va
u
s=p
x

_va
u
s,
    +      )
```
    B

o
 

 prov
d
 a bo


rp
at
 of a typ
ca
 
mp

m

tat
o
 patt
r
 of [
mb
d_mu
t
moda
][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsMu
t
Moda
.
mb
d_mu
t
moda
], but f

 fr
 to adjust 
t to your o

 

ds.
    ```pytho

    d
f _proc
ss_
mag
_

put(s

f, 
mag
_

put: YourMod

Imag
I
puts) -
 torch.T

sor:
        
mag
_f
atur
s = s

f.v
s
o
_

cod
r(
mag
_

put)
        r
tur
 s

f.mu
t
_moda
_proj
ctor(
mag
_f
atur
s)
    d
f 
mb
d_mu
t
moda
(
        s

f,
        **k
args: obj
ct,
    ) -
 Mu
t
Moda
Emb
dd

gs | No

:
        # Va

dat
 th
 mu
t
moda
 

put k
y
ord argum

ts
        
mag
_

put = s

f._pars
_a
d_va

dat
_
mag
_

put(**k
args)
        
f 
mag
_

put 
s No

:
            r
tur
 No


        # Ru
 mu
t
moda
 

puts through 

cod
r a
d proj
ctor
        v
s
o
_
mb
dd

gs = s

f._proc
ss_
mag
_

put(
mag
_

put)
        r
tur
 v
s
o
_
mb
dd

gs
```
!!! 
mporta
t
    Th
 r
tur

d `mu
t
moda
_
mb
dd

gs` must b
 

th
r a **3D [torch.T

sor][]** of shap
 `(
um_
t
ms, f
atur
_s
z
, h
dd

_s
z
)`, or a **

st / tup

 of 2D [torch.T

sor][]'s** of shap
 `(f
atur
_s
z
, h
dd

_s
z
)`, so that `mu
t
moda
_
mb
dd

gs[
]` r
tr

v
s th
 
mb
dd

gs g


rat
d from th
 `
`-th mu
t
moda
 data 
t
m (
.g, 
mag
) of th
 r
qu
st.
!!! 
ot

    By d
fau
t, vLLM m
rg
s th
 mu
t
moda
 
mb
dd

gs 

to t
xt 
mb
dd

gs d
p

d

g o
 th
 

format
o
 of th

r 
ocat
o
s d
f


d 


    [P
ac
ho
d
rRa
g
][v
m.mu
t
moda
.

puts.P
ac
ho
d
rRa
g
] from 

put proc
ss

g.
    Th
s 
og
c ca
 b
 fou
d at [
mb
d_

put_
ds][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsMu
t
Moda
.
mb
d_

put_
ds].
    You may ov
rr
d
 th
s m
thod 
f add
t
o
a
 
og
c 
s r
qu
r
d for your mod

 
h

 m
rg

g 
mb
dd

gs.
    - O
c
 th
 abov
 st
ps ar
 do

, updat
 th
 mod

 c
ass 

th th
 [SupportsMu
t
Moda
][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsMu
t
Moda
] 

t
rfac
.
  ```d
ff
  + from v
m.mod

_
x
cutor.mod

s.

t
rfac
s 
mport SupportsMu
t
Moda

  - c
ass YourMod

ForImag
2S
q(
.Modu

):
  + c
ass YourMod

ForImag
2S
q(
.Modu

, SupportsMu
t
Moda
):
```
!!! 
ot

    Th
 mod

 c
ass do
s 
ot hav
 to b
 
am
d `*ForCausa
LM`.
    Ch
ck out [th
 Hugg

gFac
 Tra
sform
rs docum

tat
o
](https://hugg

gfac
.co/docs/tra
sform
rs/mod

_doc/auto#mu
t
moda
) for som
 
xamp

s.
## 2. Sp
c
fy proc
ss

g 

format
o

N
xt, cr
at
 a subc
ass of [Bas
Proc
ss

gI
fo][v
m.mu
t
moda
.proc
ss

g.Bas
Proc
ss

gI
fo]
to prov
d
 bas
c 

format
o
 r

at
d to HF proc
ss

g.
### Max
mum 
umb
r of 

put 
t
ms
You 

d to ov
rr
d
 th
 abstract m
thod [g
t_support
d_mm_

m
ts][v
m.mu
t
moda
.proc
ss

g.Bas
Proc
ss

gI
fo.g
t_support
d_mm_

m
ts]
to r
tur
 th
 max
mum 
umb
r of 

put 
t
ms for 
ach moda

ty support
d by th
 mod

.
For 
xamp

, 
f th
 mod

 supports a
y 
umb
r of 
mag
s but o

y o

 v
d
o p
r prompt:
```pytho

d
f g
t_support
d_mm_

m
ts(s

f) -
 Mapp

g[str, 

t | No

]:
    r
tur
 {"
mag
": No

, "v
d
o": 1}
```
## 3. Sp
c
fy dummy 

puts
Th

, 

h
r
t [Bas
DummyI
putsBu

d
r][v
m.mu
t
moda
.proc
ss

g.Bas
DummyI
putsBu

d
r] to co
struct dummy 

puts for
HF proc
ss

g. Th
 proc
ss
d outputs ar
 a
so us
d for m
mory prof



g.
Ov
rr
d
 th
 abstract m
thods [g
t_dummy_t
xt][v
m.mu
t
moda
.proc
ss

g.Bas
DummyI
putsBu

d
r.g
t_dummy_t
xt] a
d [g
t_dummy_mm_data][v
m.mu
t
moda
.proc
ss

g.Bas
DummyI
putsBu

d
r.g
t_dummy_mm_data] to co
struct dummy 

puts. Th
s
 dummy 

puts shou
d r
su
t 

 th
 
orst-cas
 m
mory usag
 of th
 mod

 so that vLLM ca
 r
s
rv
 th
 corr
ct amou
t of m
mory for 
t.
Assum

g that th
 m
mory usag
 

cr
as
s 

th th
 
umb
r of tok

s, th
 dummy 

puts ca
 b
 co
struct
d to max
m
z
 th
 
umb
r of output 
mb
dd

gs, 
h
ch 
s th
 sam
 
umb
r as p
ac
ho
d
r f
atur
 tok

s.
=== "Bas
c 
xamp

: LLaVA"
    Look

g at th
 cod
 of HF's `L
avaForCo
d
t
o
a
G


rat
o
`:
    ??? cod

        ```pytho

        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.47.1/src/tra
sform
rs/mod

s/
ava/mod



g_
ava.py#L530-L544
        
_
mag
_tok

s = (

put_
ds == s

f.co
f
g.
mag
_tok

_

d
x).sum().
t
m()
        
_
mag
_f
atur
s = 
mag
_f
atur
s.shap
[0] * 
mag
_f
atur
s.shap
[1]
        
f 
_
mag
_tok

s != 
_
mag
_f
atur
s:
            ra
s
 Va
u
Error(
                f"Imag
 f
atur
s a
d 
mag
 tok

s do 
ot match: tok

s: {
_
mag
_tok

s}, f
atur
s {
_
mag
_f
atur
s}"
            )
        sp
c
a
_
mag
_mask = (
            (

put_
ds == s

f.co
f
g.
mag
_tok

_

d
x)
            .u
squ
z
(-1)
            .
xpa
d_as(

puts_
mb
ds)
            .to(

puts_
mb
ds.d
v
c
)
        )
        
mag
_f
atur
s = 
mag
_f
atur
s.to(

puts_
mb
ds.d
v
c
, 

puts_
mb
ds.dtyp
)
        

puts_
mb
ds = 

puts_
mb
ds.mask
d_scatt
r(sp
c
a
_
mag
_mask, 
mag
_f
atur
s)
```
    Th
 
umb
r of p
ac
ho
d
r f
atur
 tok

s p
r 
mag
 
s `
mag
_f
atur
s.shap
[1]`.
    `
mag
_f
atur
s` 
s ca
cu
at
d 

s
d
 th
 `g
t_
mag
_f
atur
s` m
thod:
    ??? cod

        ```pytho

        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.47.1/src/tra
sform
rs/mod

s/
ava/mod



g_
ava.py#L290-L300
        
mag
_outputs = s

f.v
s
o
_to

r(p
x

_va
u
s, output_h
dd

_stat
s=Tru
)
        s


ct
d_
mag
_f
atur
 = 
mag
_outputs.h
dd

_stat
s[v
s
o
_f
atur
_
ay
r]
        
f v
s
o
_f
atur
_s


ct_strat
gy == "d
fau
t":
            s


ct
d_
mag
_f
atur
 = s


ct
d_
mag
_f
atur
[:, 1:]
        


f v
s
o
_f
atur
_s


ct_strat
gy == "fu
":
            s


ct
d_
mag
_f
atur
 = s


ct
d_
mag
_f
atur

        

s
:
            ra
s
 Va
u
Error(f"U

xp
ct
d s


ct f
atur
 strat
gy: {s

f.co
f
g.v
s
o
_f
atur
_s


ct_strat
gy}")
        
mag
_f
atur
s = s

f.mu
t
_moda
_proj
ctor(s


ct
d_
mag
_f
atur
)
        r
tur
 
mag
_f
atur
s
```
    W
 ca
 

f
r that `
mag
_f
atur
s.shap
[1]` 
s bas
d o
 `
mag
_outputs.h
dd

_stat
s.shap
[1]` from th
 v
s
o
 to

r
    (`CLIPV
s
o
Mod

` for th
 [`
ava-hf/
ava-1.5-7b-hf`](https://hugg

gfac
.co/
ava-hf/
ava-1.5-7b-hf) mod

).
    Mor
ov
r, 

 o

y 

d th
 s
qu

c
 


gth (th
 s
co
d d
m

s
o
 of th
 t

sor) to g
t `
mag
_f
atur
s.shap
[1]`.
    Th
 s
qu

c
 


gth 
s d
t
rm


d by th
 


t
a
 h
dd

 stat
s 

 `CLIPV
s
o
Tra
sform
r` s

c
 th
 att

t
o

    m
cha

sm do
s
't cha
g
 th
 s
qu

c
 


gth of th
 output h
dd

 stat
s.
    ```pytho

    # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.47.1/src/tra
sform
rs/mod

s/c

p/mod



g_c

p.py#L1094-L1102
    h
dd

_stat
s = s

f.
mb
dd

gs(p
x

_va
u
s, 

t
rpo
at
_pos_

cod

g=

t
rpo
at
_pos_

cod

g)
    h
dd

_stat
s = s

f.pr
_
ayr
orm(h
dd

_stat
s)
    

cod
r_outputs = s

f.

cod
r(
        

puts_
mb
ds=h
dd

_stat
s,
        output_att

t
o
s=output_att

t
o
s,
        output_h
dd

_stat
s=output_h
dd

_stat
s,
        r
tur
_d
ct=r
tur
_d
ct,
    )
```
    To f

d th
 s
qu

c
 


gth, 

 tur
 to th
 cod
 of `CLIPV
s
o
Emb
dd

gs`:
    ??? cod

        ```pytho

        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.47.1/src/tra
sform
rs/mod

s/c

p/mod



g_c

p.py#L247-L257
        targ
t_dtyp
 = s

f.patch_
mb
dd

g.


ght.dtyp

        patch_
mb
ds = s

f.patch_
mb
dd

g(p
x

_va
u
s.to(dtyp
=targ
t_dtyp
))  # shap
 = [*, 

dth, gr
d, gr
d]
        patch_
mb
ds = patch_
mb
ds.f
att

(2).tra
spos
(1, 2)
        c
ass_
mb
ds = s

f.c
ass_
mb
dd

g.
xpa
d(batch_s
z
, 1, -1)
        
mb
dd

gs = torch.cat([c
ass_
mb
ds, patch_
mb
ds], d
m=1)
        
f 

t
rpo
at
_pos_

cod

g:
            
mb
dd

gs = 
mb
dd

gs + s

f.

t
rpo
at
_pos_

cod

g(
mb
dd

gs, h

ght, 

dth)
        

s
:
            
mb
dd

gs = 
mb
dd

gs + s

f.pos
t
o
_
mb
dd

g(s

f.pos
t
o
_
ds)
        r
tur
 
mb
dd

gs
```
    W
 ca
 

f
r that `
mb
dd

gs.shap
[1] == s

f.
um_pos
t
o
s`, 
h
r

    ```pytho

    # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.47.1/src/tra
sform
rs/mod

s/c

p/mod



g_c

p.py#L195-L196
    s

f.
um_patch
s = (s

f.
mag
_s
z
 // s

f.patch_s
z
) ** 2
    s

f.
um_pos
t
o
s = s

f.
um_patch
s + 1
```
    Ov
ra
, th
 
umb
r of p
ac
ho
d
r f
atur
 tok

s for a
 
mag
 ca
 b
 ca
cu
at
d as:
    ??? cod

        ```pytho

        d
f g
t_
um_
mag
_tok

s(
            s

f,
            *,
            
mag
_

dth: 

t,
            
mag
_h

ght: 

t,
        ) -
 

t:
            hf_co
f
g = s

f.g
t_hf_co
f
g()
            hf_proc
ssor = s

f.g
t_hf_proc
ssor()
            
mag
_s
z
 = hf_co
f
g.v
s
o
_co
f
g.
mag
_s
z

            patch_s
z
 = hf_co
f
g.v
s
o
_co
f
g.patch_s
z

            
um_
mag
_tok

s = (
mag
_s
z
 // patch_s
z
) ** 2 + 1
            
f hf_proc
ssor.v
s
o
_f
atur
_s


ct_strat
gy == "d
fau
t":
                
um_
mag
_tok

s -= 1
            r
tur
 
um_
mag
_tok

s
```
    Not
c
 that th
 
umb
r of 
mag
 tok

s do
s
't d
p

d o
 th
 
mag
 

dth a
d h

ght.
    W
 ca
 s
mp
y us
 a dummy `
mag
_s
z
` to ca
cu
at
 th
 mu
t
moda
 prof



g data:
    ??? cod

        ```pytho

        # NOTE: I
 actua

ty, th
s 
s usua
y 
mp

m

t
d as part of th

        # mod

's subc
ass of `Bas
Proc
ss

gI
fo`, but 

 sho
 
t as 
s
        # h
r
 for s
mp

c
ty.
        d
f g
t_
mag
_s
z
_

th_most_f
atur
s(s

f) -
 Imag
S
z
:
            hf_co
f
g = s

f.g
t_hf_co
f
g()
            

dth = h

ght = hf_co
f
g.
mag
_s
z

            r
tur
 Imag
S
z
(

dth=

dth, h

ght=h

ght)
        d
f g
t_dummy_mm_data(
            s

f,
            s
q_


: 

t,
            mm_cou
ts: Mapp

g[str, 

t],
            mm_opt
o
s: Mapp

g[str, Bas
DummyOpt
o
s],
        ) -
 Mu
t
Moda
DataD
ct:
            
um_
mag
s = mm_cou
ts.g
t("
mag
", 0)
            targ
t_

dth, targ
t_h

ght = \
                s

f.

fo.g
t_
mag
_s
z
_

th_most_f
atur
s()
            
mag
_ov
rr
d
s = mm_opt
o
s.g
t("
mag
")
            r
tur
 {
                "
mag
": s

f._g
t_dummy_
mag
s(
                    

dth=targ
t_

dth,
                    h

ght=targ
t_h

ght,
                    
um_
mag
s=
um_
mag
s,
                    ov
rr
d
s=
mag
_ov
rr
d
s,
                )
            }
```
    For th
 t
xt, 

 s
mp
y 
xpa
d th
 mu
t
moda
 
mag
 tok

 from th
 mod

 co
f
g to match th
 d
s
r
d 
umb
r of 
mag
s.
    ```pytho

    d
f g
t_dummy_t
xt(s

f, mm_cou
ts: Mapp

g[str, 

t]) -
 str:
        
um_
mag
s = mm_cou
ts.g
t("
mag
", 0)
        proc
ssor = s

f.

fo.g
t_hf_proc
ssor()
        
mag
_tok

 = proc
ssor.
mag
_tok


        r
tur
 
mag
_tok

 * 
um_
mag
s
```
=== "No 

put p
ac
ho
d
rs: Fuyu"
    Look

g at th
 cod
 of HF's `FuyuForCausa
LM`:
    ??? cod

        ```pytho

        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.48.3/src/tra
sform
rs/mod

s/fuyu/mod



g_fuyu.py#L311-L322
        
f 
mag
_patch
s 
s 
ot No

 a
d past_k
y_va
u
s 
s No

:
            patch_
mb
dd

gs = [
                s

f.v
s
o
_
mb
d_tok

s(patch.to(s

f.v
s
o
_
mb
d_tok

s.


ght.dtyp
))
                .squ
z
(0)
                .to(

puts_
mb
ds.d
v
c
)
                for patch 

 
mag
_patch
s
            ]
            

puts_
mb
ds = s

f.gath
r_co
t

uous_
mb
dd

gs(
                
ord_
mb
dd

gs=

puts_
mb
ds,
                co
t

uous_
mb
dd

gs=patch_
mb
dd

gs,
                
mag
_patch_

put_

d
c
s=
mag
_patch
s_

d
c
s,
            )
```
    Th
 
umb
r of p
ac
ho
d
r f
atur
 tok

s for th
 `
`th 
t
m 

 th
 batch 
s `patch_
mb
dd

gs[
].shap
[0]`,
    
h
ch 
s th
 sam
 as `
mag
_patch
s[
].shap
[0]`, 
.
. `
um_tota
_patch
s`.
    U


k
 LLaVA, Fuyu do
s 
ot d
f


 th
 
umb
r of patch
s 

s
d
 th
 mod



g f


. Wh
r
 ca
 

 g
t mor
 

format
o
?
    Co
s
d
r

g that th
 mod

 

put com
s from th
 output of `FuyuProc
ssor`, 

t's **
ook at th
 pr
proc
ss

g f


s**.
    Th
 
mag
 outputs ar
 obta


d by ca


g `FuyuImag
Proc
ssor.pr
proc
ss` a
d th


    `FuyuImag
Proc
ssor.pr
proc
ss_

th_tok


z
r_

fo` 

s
d
 `FuyuProc
ssor`.
    I
 `FuyuImag
Proc
ssor.pr
proc
ss`, th
 
mag
s ar
 r
s
z
d a
d padd
d to th
 targ
t `FuyuImag
Proc
ssor.s
z
`,
    r
tur


g th
 d
m

s
o
s aft
r r
s
z

g (but b
for
 padd

g) as m
tadata.
    ??? cod

        ```pytho

        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.48.3/src/tra
sform
rs/mod

s/fuyu/proc
ss

g_fuyu.py#L541-L544
        
mag
_

cod

g = s

f.
mag
_proc
ssor.pr
proc
ss(
mag
s, **output_k
args["
mag
s_k
args"])
        batch_
mag
s = 
mag
_

cod

g["
mag
s"]
        
mag
_u
padd
d_h

ghts = 
mag
_

cod

g["
mag
_u
padd
d_h

ghts"]
        
mag
_u
padd
d_

dths = 
mag
_

cod

g["
mag
_u
padd
d_

dths"]
        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.48.3/src/tra
sform
rs/mod

s/fuyu/
mag
_proc
ss

g_fuyu.py#L480-L
        
f do_r
s
z
:
            batch_
mag
s = [
                [s

f.r
s
z
(
mag
, s
z
=s
z
, 

put_data_format=

put_data_format) for 
mag
 

 
mag
s]
                for 
mag
s 

 batch_
mag
s
            ]
        
mag
_s
z
s = [g
t_
mag
_s
z
(
mag
s[0], cha


_d
m=

put_data_format) for 
mag
s 

 batch_
mag
s]
        
mag
_u
padd
d_h

ghts = [[
mag
_s
z
[0]] for 
mag
_s
z
 

 
mag
_s
z
s]
        
mag
_u
padd
d_

dths = [[
mag
_s
z
[1]] for 
mag
_s
z
 

 
mag
_s
z
s]
        
f do_pad:
            batch_
mag
s = [
                [
                    s

f.pad_
mag
(
                        
mag
,
                        s
z
=s
z
,
                        mod
=padd

g_mod
,
                        co
sta
t_va
u
s=padd

g_va
u
,
                        

put_data_format=

put_data_format,
                    )
                    for 
mag
 

 
mag
s
                ]
                for 
mag
s 

 batch_
mag
s
            ]
```
    I
 `FuyuImag
Proc
ssor.pr
proc
ss_

th_tok


z
r_

fo`, th
 
mag
s ar
 sp

t 

to patch
s bas
d o
 th
s m
tadata:
    ??? cod

        ```pytho

        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.48.3/src/tra
sform
rs/mod

s/fuyu/proc
ss

g_fuyu.py#L417-L425
        mod

_
mag
_

put = s

f.
mag
_proc
ssor.pr
proc
ss_

th_tok


z
r_

fo(
            
mag
_

put=t

sor_batch_
mag
s,
            
mag
_pr
s

t=
mag
_pr
s

t,
            
mag
_u
padd
d_h=
mag
_u
padd
d_h

ghts,
            
mag
_u
padd
d_
=
mag
_u
padd
d_

dths,
            
mag
_p
ac
ho
d
r_
d=
mag
_p
ac
ho
d
r_
d,
            
mag
_






_
d=
mag
_






_
d,
            var
ab

_s
z
d=Tru
,
        )
        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.48.3/src/tra
sform
rs/mod

s/fuyu/
mag
_proc
ss

g_fuyu.py#L638-L658
        
mag
_h

ght, 
mag
_

dth = 
mag
.shap
[1], 
mag
.shap
[2]
        
f var
ab

_s
z
d:  # var
ab

_s
z
d=Tru

            


_h = m

(
                
mag
_h

ght,
                math.c


(
mag
_u
padd
d_h[batch_

d
x, subs
q_

d
x] / patch_h

ght) * patch_h

ght,
            )
            


_
 = m

(
                
mag
_

dth,
                math.c


(
mag
_u
padd
d_
[batch_

d
x, subs
q_

d
x] / patch_

dth) * patch_

dth,
            )
            
mag
 = 
mag
[:, :


_h, :


_
]
            
mag
_h

ght, 
mag
_

dth = 


_h, 


_

        
um_patch
s = s

f.g
t_
um_patch
s(
mag
_h

ght=
mag
_h

ght, 
mag
_

dth=
mag
_

dth)
        t

sor_of_
mag
_
ds = torch.fu
(
            [
um_patch
s], 
mag
_p
ac
ho
d
r_
d, dtyp
=torch.

t32, d
v
c
=
mag
_

put.d
v
c

        )
        patch
s = s

f.patch
fy_
mag
(
mag
=
mag
.u
squ
z
(0)).squ
z
(0)
        ass
rt 
um_patch
s == patch
s.shap
[0]
```
    Th
 
umb
r of patch
s 
s 

 tur
 d
f


d by `FuyuImag
Proc
ssor.g
t_
um_patch
s`:
    ??? cod

        ```pytho

        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.48.3/src/tra
sform
rs/mod

s/fuyu/
mag
_proc
ss

g_fuyu.py#L552-L562
        patch_s
z
 = patch_s
z
 
f patch_s
z
 
s 
ot No

 

s
 s

f.patch_s
z

        patch_h

ght, patch_

dth = s

f.patch_s
z
["h

ght"], s

f.patch_s
z
["

dth"]
        
f 
mag
_h

ght % patch_h

ght != 0:
            ra
s
 Va
u
Error(f"{
mag
_h

ght=} must b
 d
v
s
b

 by {patch_h

ght}")
        
f 
mag
_

dth % patch_

dth != 0:
            ra
s
 Va
u
Error(f"{
mag
_

dth=} must b
 d
v
s
b

 by {patch_

dth}")
        
um_patch
s_p
r_d
m_h = 
mag
_h

ght // patch_h

ght
        
um_patch
s_p
r_d
m_
 = 
mag
_

dth // patch_

dth
        
um_patch
s = 
um_patch
s_p
r_d
m_h * 
um_patch
s_p
r_d
m_

```
    Th
s
 
mag
 patch
s corr
spo
d to p
ac
ho
d
r tok

s (`|SPEAKER|`). So, 

 just 

d to max
m
z
 th
 
umb
r of 
mag
 patch
s. S

c
 

put 
mag
s ar
 f
rst r
s
z
d
    to f
t 

th

 `
mag
_proc
ssor.s
z
`, 

 ca
 max
m
z
 th
 
umb
r of 
mag
 patch
s by 

putt

g a
 
mag
 

th s
z
 
qua
 to `
mag
_proc
ssor.s
z
`.
    ```pytho

    d
f g
t_
mag
_s
z
_

th_most_f
atur
s(s

f) -
 Imag
S
z
:
        
mag
_proc
ssor = s

f.g
t_
mag
_proc
ssor()
        r
tur
 Imag
S
z
(
            

dth=
mag
_proc
ssor.s
z
["

dth"],
            h

ght=
mag
_proc
ssor.s
z
["h

ght"],
        )
```
    Fuyu do
s 
ot 
xp
ct 
mag
 p
ac
ho
d
rs 

 th
 

puts to HF proc
ssor, so
    th
 dummy prompt t
xt 
s 
mpty r
gard

ss of th
 
umb
r of 
mag
s.
    ```pytho

    d
f g
t_dummy_t
xt(s

f, mm_cou
ts: Mapp

g[str, 

t]) -
 str:
        r
tur
 ""
```
    For th
 mu
t
moda
 
mag
 prof



g data, th
 
og
c 
s v
ry s
m

ar to LLaVA:
    ??? cod

        ```pytho

        d
f g
t_dummy_mm_data(
            s

f,
            s
q_


: 

t,
            mm_cou
ts: Mapp

g[str, 

t],
            mm_opt
o
s: Mapp

g[str, Bas
DummyOpt
o
s],
        ) -
 Mu
t
Moda
DataD
ct:
            targ
t_

dth, targ
t_h

ght = \
                s

f.

fo.g
t_
mag
_s
z
_

th_most_f
atur
s()
            
um_
mag
s = mm_cou
ts.g
t("
mag
", 0)
            
mag
_ov
rr
d
s = mm_opt
o
s.g
t("
mag
")
            r
tur
 {
                "
mag
": s

f._g
t_dummy_
mag
s(
                    

dth=targ
t_

dth,
                    h

ght=targ
t_h

ght,
                    
um_
mag
s=
um_
mag
s,
                    ov
rr
d
s=
mag
_ov
rr
d
s,
                )
            }
```
## 4. Sp
c
fy proc
ss

g d
ta

s
Aft
r
ards, cr
at
 a subc
ass of [Bas
Mu
t
Moda
Proc
ssor][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor]
to f

 

 th
 m
ss

g d
ta

s about HF proc
ss

g.
!!! 

fo
    [Mu
t
-Moda
 Data Proc
ss

g](../../d
s
g
/mm_proc
ss

g.md)
### Mu
t
-moda
 f


ds
Ov
rr
d
 [_g
t_mm_f


ds_co
f
g][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._g
t_mm_f


ds_co
f
g] to
r
tur
 a sch
ma of th
 t

sors outputt
d by th
 HF proc
ssor that ar
 r

at
d to th
 

put mu
t
-moda
 
t
ms.
=== "Bas
c 
xamp

: LLaVA"
    Th
 output of `CLIPImag
Proc
ssor` 
s a s
mp

 t

sor 

th shap

    `(
um_
mag
s, 
um_cha


s, 
mag
_h

ght, 
mag
_

dth)`:
    ```pytho

    # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.47.1/src/tra
sform
rs/mod

s/c

p/
mag
_proc
ss

g_c

p.py#L339-L345
    
mag
s = [
        to_cha


_d
m

s
o
_format(
mag
, data_format, 

put_cha


_d
m=

put_data_format)
        for 
mag
 

 a
_
mag
s
    ]
    data = {"p
x

_va
u
s": 
mag
s}
    r
tur
 BatchF
atur
(data=data, t

sor_typ
=r
tur
_t

sors)
```
    So, 

 ov
rr
d
 [_g
t_mm_f


ds_co
f
g][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._g
t_mm_f


ds_co
f
g] as fo
o
s:
    ```pytho

    d
f _g
t_mm_f


ds_co
f
g(
        s

f,
        hf_

puts: BatchF
atur
,
        hf_proc
ssor_mm_k
args: Mapp

g[str, obj
ct],
    ) -
 Mapp

g[str, Mu
t
Moda
F


dCo
f
g]:
        r
tur
 d
ct(
            p
x

_va
u
s=Mu
t
Moda
F


dCo
f
g.batch
d("
mag
"),
        )
```
    !!! 
ot

        Our [actua
 cod
](../../../v
m/mod

_
x
cutor/mod

s/
ava.py) add
t
o
a
y supports
        pr
-comput
d 
mag
 
mb
dd

gs, 
h
ch ca
 b
 pass
d to b
 mod

 v
a th
 `
mag
_
mb
ds` argum

t.
=== "W
th postproc
ss

g: Fuyu"
    Th
 `
mag
_patch
s` output of `FuyuImag
Proc
ssor.pr
proc
ss_

th_tok


z
r_

fo` co
cat

at
s
    th
 patch
s from 
ach 
mag
 b

o
g

g to a
 
t
m 

 th
 batch:
    ```pytho

    # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.48.3/src/tra
sform
rs/mod

s/fuyu/
mag
_proc
ss

g_fuyu.py#L673-L679
            
mag
_

put_
ds.app

d(t

sor_of_
mag
_
ds)
            
mag
_patch
s.app

d(patch
s)
        

s
:
            
mag
_

put_
ds.app

d(torch.t

sor([], dtyp
=torch.

t32, d
v
c
=
mag
_

put.d
v
c
))
    batch_
mag
_

put_
ds.app

d(
mag
_

put_
ds)
    batch_
mag
_patch
s.app

d(
mag
_patch
s)
```
    Th
 shap
 of `
mag
_patch
s` outputt
d by `FuyuImag
Proc
ssor` 
s th
r
for

    `(1, 
um_
mag
s, 
um_patch
s, patch_

dth * patch_h

ght * 
um_cha


s)`.
    I
 ord
r to support th
 us
 of
    [Mu
t
Moda
F


dCo
f
g.batch
d][v
m.mu
t
moda
.

puts.Mu
t
Moda
F


dCo
f
g.batch
d]
    

k
 

 LLaVA, 

 r
mov
 th
 
xtra batch d
m

s
o
 by ov
rr
d

g
    [Bas
Mu
t
Moda
Proc
ssor._ca
_hf_proc
ssor][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._ca
_hf_proc
ssor]:
    ??? cod

        ```pytho

        d
f _ca
_hf_proc
ssor(
            s

f,
            prompt: str,
            mm_data: Mapp

g[str, obj
ct],
            mm_k
args: Mapp

g[str, obj
ct],
            tok_k
args: Mapp

g[str, obj
ct],
        ) -
 BatchF
atur
:
            proc
ss
d_outputs = sup
r()._ca
_hf_proc
ssor(
                prompt=prompt,
                mm_data=mm_data,
                mm_k
args=mm_k
args,
                tok_k
args=tok_k
args,
            )
            
mag
_patch
s = proc
ss
d_outputs.g
t("
mag
_patch
s")
            
f 
mag
_patch
s 
s 
ot No

:
                
mag
s = mm_data["
mag
s"]
                ass
rt 
s

sta
c
(
mag
s, 

st)
                # Or
g

a
 output: (1, 
um_
mag
s, P
, Px * Py * C)
                # N

 output: (
um_
mag
s, P
, Px * Py * C)
                ass
rt (
s

sta
c
(
mag
_patch
s, 

st)
                        a
d 


(
mag
_patch
s) == 1)
                ass
rt (
s

sta
c
(
mag
_patch
s[0], torch.T

sor)
                        a
d 


(
mag
_patch
s[0]) == 


(
mag
s))
                proc
ss
d_outputs["
mag
_patch
s"] = 
mag
_patch
s[0]
            r
tur
 proc
ss
d_outputs
```
    !!! 
ot

        Our [actua
 cod
](../../../v
m/mod

_
x
cutor/mod

s/fuyu.py) has sp
c
a
 ha
d


g
        for t
xt-o

y 

puts to pr
v

t u

c
ssary 
ar


gs from HF proc
ssor.
    !!! 
ot

        Th
 `_ca
_hf_proc
ssor` m
thod sp
c
f

s both `mm_k
args` a
d `tok_k
args` for
        proc
ss

g. `mm_k
args` 
s us
d to both 


t
a

z
 a
d ca
 th
 hugg

gfac

        proc
ssor, 
h
r
as `tok_k
args` 
s o

y us
d to ca
 th
 hugg

gfac
 proc
ssor.
    Th
s 

ts us ov
rr
d
 [_g
t_mm_f


ds_co
f
g][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._g
t_mm_f


ds_co
f
g] as fo
o
s:
    ```pytho

    d
f _g
t_mm_f


ds_co
f
g(
        s

f,
        hf_

puts: BatchF
atur
,
        hf_proc
ssor_mm_k
args: Mapp

g[str, obj
ct],
    ) -
 Mapp

g[str, Mu
t
Moda
F


dCo
f
g]:
        r
tur
 d
ct(
mag
_patch
s=Mu
t
Moda
F


dCo
f
g.batch
d("
mag
"))
```
### Prompt updat
s
Ov
rr
d
 [_g
t_prompt_updat
s][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._g
t_prompt_updat
s] to
r
tur
 a 

st of [PromptUpdat
][v
m.mu
t
moda
.proc
ss

g.PromptUpdat
] 

sta
c
s.
Each [PromptUpdat
][v
m.mu
t
moda
.proc
ss

g.PromptUpdat
] 

sta
c
 sp
c
f

s a
 updat
 op
rat
o

(
.g.: 

s
rt
o
, r
p
ac
m

t) p
rform
d by th
 HF proc
ssor.
=== "Bas
c 
xamp

: LLaVA"
    Look

g at HF's `L
avaProc
ssor`:
    ```pytho

    # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.47.1/src/tra
sform
rs/mod

s/
ava/proc
ss

g_
ava.py#L167-L170
    prompt_str

gs = []
    for samp

 

 t
xt:
        samp

 = samp

.r
p
ac
(s

f.
mag
_tok

, s

f.
mag
_tok

 * 
um_
mag
_tok

s)
        prompt_str

gs.app

d(samp

)
```
    It s
mp
y r
p
ats 
ach 

put `
mag
_tok

` a 
umb
r of t
m
s 
qua
 to th
 
umb
r of p
ac
ho
d
r f
atur
 tok

s (`
um_
mag
_tok

s`).
    Bas
d o
 th
s, 

 ov
rr
d
 [_g
t_prompt_updat
s][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._g
t_prompt_updat
s] as fo
o
s:
    ??? cod

        ```pytho

        d
f _g
t_prompt_updat
s(
            s

f,
            mm_
t
ms: Mu
t
Moda
DataIt
ms,
            hf_proc
ssor_mm_k
args: Mapp

g[str, obj
ct],
            out_mm_k
args: Mu
t
Moda
K
argsIt
ms,
        ) -
 S
qu

c
[PromptUpdat
]:
            hf_co
f
g = s

f.

fo.g
t_hf_co
f
g()
            
mag
_tok

_
d = hf_co
f
g.
mag
_tok

_

d
x
            d
f g
t_r
p
ac
m

t(
t
m_
dx: 

t):
                
mag
s = mm_
t
ms.g
t_
t
ms("
mag
", Imag
Proc
ssorIt
ms)
                
mag
_s
z
 = 
mag
s.g
t_
mag
_s
z
(
t
m_
dx)
                
um_
mag
_tok

s = s

f.

fo.g
t_
um_
mag
_tok

s(
                    
mag
_

dth=
mag
_s
z
.

dth,
                    
mag
_h

ght=
mag
_s
z
.h

ght,
                )
                r
tur
 [
mag
_tok

_
d] * 
um_
mag
_tok

s
            r
tur
 [
                PromptR
p
ac
m

t(
                    moda

ty="
mag
",
                    targ
t=[
mag
_tok

_
d],
                    r
p
ac
m

t=g
t_r
p
ac
m

t,
                ),
            ]
```
=== "Ha
d


g add
t
o
a
 tok

s: Fuyu"
    R
ca
 th
 
ayout of f
atur
 tok

s from St
p 2:
```
    |SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
    |SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
    ...
    |SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
```
    W
 d
f


 a h

p
r fu
ct
o
 to r
tur
 `
co
s` a
d `
ro
s` d
r
ct
y:
    ??? cod

        ```pytho

        d
f g
t_
mag
_f
atur
_gr
d_s
z
(
            s

f,
            *,
            
mag
_

dth: 

t,
            
mag
_h

ght: 

t,
        ) -
 tup

[

t, 

t]:
            
mag
_proc
ssor = s

f.g
t_
mag
_proc
ssor()
            targ
t_

dth = 
mag
_proc
ssor.s
z
["

dth"]
            targ
t_h

ght = 
mag
_proc
ssor.s
z
["h

ght"]
            patch_

dth = 
mag
_proc
ssor.patch_s
z
["

dth"]
            patch_h

ght = 
mag
_proc
ssor.patch_s
z
["h

ght"]
            
f 
ot (
mag
_

dth 
= targ
t_

dth a
d 
mag
_h

ght 
= targ
t_h

ght):
                h

ght_sca

_factor = targ
t_h

ght / 
mag
_h

ght
                

dth_sca

_factor = targ
t_

dth / 
mag
_

dth
                opt
ma
_sca

_factor = m

(h

ght_sca

_factor, 

dth_sca

_factor)
                
mag
_h

ght = 

t(
mag
_h

ght * opt
ma
_sca

_factor)
                
mag
_

dth = 

t(
mag
_

dth * opt
ma
_sca

_factor)
            
co
s = math.c


(
mag
_

dth / patch_

dth)
            
ro
s = math.c


(
mag
_h

ght / patch_h

ght)
            r
tur
 
co
s, 
ro
s
```
    Bas
d o
 th
s, 

 ca
 


t
a
y d
f


 our r
p
ac
m

t tok

s as:
    ??? cod

        ```pytho

        d
f g
t_r
p
ac
m

t(
t
m_
dx: 

t):
            
mag
s = mm_
t
ms.g
t_
t
ms("
mag
", Imag
Proc
ssorIt
ms)
            
mag
_s
z
 = 
mag
s.g
t_
mag
_s
z
(
t
m_
dx)
            
co
s, 
ro
s = s

f.

fo.g
t_
mag
_f
atur
_gr
d_s
z
(
                
mag
_

dth=
mag
_s
z
.

dth,
                
mag
_h

ght=
mag
_s
z
.h

ght,
            )
            # `_IMAGE_TOKEN_ID` corr
spo
ds to `|SPEAKER|`
            # `_NEWLINE_TOKEN_ID` corr
spo
ds to `|NEWLINE|`
            r
tur
 ([_IMAGE_TOKEN_ID] * 
co
s + [_NEWLINE_TOKEN_ID]) * 
ro
s
```
    Ho

v
r, th
s 
s 
ot 

t
r

y corr
ct. Aft
r `FuyuImag
Proc
ssor.pr
proc
ss_

th_tok


z
r_

fo` 
s ca

d,
    a BOS tok

 (`
s
`) 
s a
so add
d to th
 prompt:
    ??? cod

        ```pytho

        # https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/v4.48.3/src/tra
sform
rs/mod

s/fuyu/proc
ss

g_fuyu.py#L417-L435
        mod

_
mag
_

put = s

f.
mag
_proc
ssor.pr
proc
ss_

th_tok


z
r_

fo(
            
mag
_

put=t

sor_batch_
mag
s,
            
mag
_pr
s

t=
mag
_pr
s

t,
            
mag
_u
padd
d_h=
mag
_u
padd
d_h

ghts,
            
mag
_u
padd
d_
=
mag
_u
padd
d_

dths,
            
mag
_p
ac
ho
d
r_
d=
mag
_p
ac
ho
d
r_
d,
            
mag
_






_
d=
mag
_






_
d,
            var
ab

_s
z
d=Tru
,
        )
        prompt_tok

s, prompts_


gth = _tok


z
_prompts_

th_
mag
_a
d_batch(
            tok


z
r=s

f.tok


z
r,
            prompts=prompts,
            sca

_factors=sca

_factors,
            max_tok

s_to_g


rat
=s

f.max_tok

s_to_g


rat
,
            max_pos
t
o
_
mb
dd

gs=s

f.max_pos
t
o
_
mb
dd

gs,
            add_BOS=Tru
,
            add_b
g



g_of_a
s

r_tok

=Tru
,
        )
```
    To ass
g
 th
 v
s
o
 
mb
dd

gs to o

y th
 
mag
 tok

s, 

st
ad of a str

g
    you ca
 r
tur
 a
 

sta
c
 of [PromptUpdat
D
ta

s][v
m.mu
t
moda
.proc
ss

g.PromptUpdat
D
ta

s]:
    ??? cod

        ```pytho

        hf_co
f
g = s

f.

fo.g
t_hf_co
f
g()
        bos_tok

_
d = hf_co
f
g.bos_tok

_
d  # `
s
`
        ass
rt 
s

sta
c
(bos_tok

_
d, 

t)
        d
f g
t_r
p
ac
m

t_fuyu(
t
m_
dx: 

t):
            
mag
s = mm_
t
ms.g
t_
t
ms("
mag
", Imag
Proc
ssorIt
ms)
            
mag
_s
z
 = 
mag
s.g
t_
mag
_s
z
(
t
m_
dx)
            
co
s, 
ro
s = s

f.

fo.g
t_
mag
_f
atur
_gr
d_s
z
(
                
mag
_

dth=
mag
_s
z
.

dth,
                
mag
_h

ght=
mag
_s
z
.h

ght,
            )
            
mag
_tok

s = ([_IMAGE_TOKEN_ID] * 
co
s + [_NEWLINE_TOKEN_ID]) * 
ro
s
            r
tur
 PromptUpdat
D
ta

s.s


ct_tok

_
d(
                
mag
_tok

s + [bos_tok

_
d],
                
mb
d_tok

_
d=_IMAGE_TOKEN_ID,
            )
```
    F

a
y, 
ot
c

g that th
 HF proc
ssor r
mov
s th
 `|ENDOFTEXT|` tok

 from th
 tok


z
d prompt,
    

 ca
 s
arch for 
t to co
duct th
 r
p
ac
m

t at th
 start of th
 str

g:
    ??? cod

        ```pytho

        d
f _g
t_prompt_updat
s(
            s

f,
            mm_
t
ms: Mu
t
Moda
DataIt
ms,
            hf_proc
ssor_mm_k
args: Mapp

g[str, obj
ct],
            out_mm_k
args: Mu
t
Moda
K
argsIt
ms,
        ) -
 S
qu

c
[PromptUpdat
]:
            hf_co
f
g = s

f.

fo.g
t_hf_co
f
g()
            bos_tok

_
d = hf_co
f
g.bos_tok

_
d
            ass
rt 
s

sta
c
(bos_tok

_
d, 

t)
            tok


z
r = s

f.

fo.g
t_tok


z
r()
            
ot_tok

_
d = tok


z
r.bos_tok

_
d
            ass
rt 
s

sta
c
(
ot_tok

_
d, 

t)
            d
f g
t_r
p
ac
m

t_fuyu(
t
m_
dx: 

t):
                
mag
s = mm_
t
ms.g
t_
t
ms("
mag
", Imag
Proc
ssorIt
ms)
                
mag
_s
z
 = 
mag
s.g
t_
mag
_s
z
(
t
m_
dx)
                
co
s, 
ro
s = s

f.

fo.g
t_
mag
_f
atur
_gr
d_s
z
(
                    
mag
_

dth=
mag
_s
z
.

dth,
                    
mag
_h

ght=
mag
_s
z
.h

ght,
                )
                
mag
_tok

s = ([_IMAGE_TOKEN_ID] * 
co
s + [_NEWLINE_TOKEN_ID]) * 
ro
s
                r
tur
 PromptUpdat
D
ta

s.s


ct_tok

_
d(
                    
mag
_tok

s + [bos_tok

_
d],
                    
mb
d_tok

_
d=_IMAGE_TOKEN_ID,
                )
            r
tur
 [
                PromptR
p
ac
m

t(
                    moda

ty="
mag
",
                    targ
t=[
ot_tok

_
d],
                    r
p
ac
m

t=g
t_r
p
ac
m

t_fuyu,
                )
            ]
```
## 5. R
g
st
r proc
ssor-r

at
d c
ass
s
Aft
r you hav
 d
f


d [Bas
Proc
ss

gI
fo][v
m.mu
t
moda
.proc
ss

g.Bas
Proc
ss

gI
fo] (St
p 2),
[Bas
DummyI
putsBu

d
r][v
m.mu
t
moda
.proc
ss

g.Bas
DummyI
putsBu

d
r] (St
p 3),
a
d [Bas
Mu
t
Moda
Proc
ssor][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor] (St
p 4),
d
corat
 th
 mod

 c
ass 

th [MULTIMODAL_REGISTRY.r
g
st
r_proc
ssor][v
m.mu
t
moda
.r
g
stry.Mu
t
Moda
R
g
stry.r
g
st
r_proc
ssor]
to r
g
st
r th
m to th
 mu
t
-moda
 r
g
stry:
```d
ff
  from v
m.mod

_
x
cutor.mod

s.

t
rfac
s 
mport SupportsMu
t
Moda

+ from v
m.mu
t
moda
 
mport MULTIMODAL_REGISTRY
+ @MULTIMODAL_REGISTRY.r
g
st
r_proc
ssor(
+     YourMu
t
Moda
Proc
ssor,
+     

fo=YourProc
ss

gI
fo,
+     dummy_

puts=YourDummyI
putsBu

d
r,
+ )
  c
ass YourMod

ForImag
2S
q(
.Modu

, SupportsMu
t
Moda
):
```
## Not
s
### I
s
rt

g f
atur
 tok

s 

thout r
p
ac
m

t
Som
 HF proc
ssors d
r
ct
y 

s
rt f
atur
 tok

s 

thout r
p
ac

g a
yth

g 

 th
 or
g

a
 prompt. I
 that cas
, you ca
 us
 [PromptI
s
rt
o
][v
m.mu
t
moda
.proc
ss

g.PromptI
s
rt
o
] 

st
ad of [PromptR
p
ac
m

t][v
m.mu
t
moda
.proc
ss

g.PromptR
p
ac
m

t] 

s
d
 [_g
t_prompt_updat
s][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._g
t_prompt_updat
s].
Examp

s:
    - BLIP-2 (

s
rt at start of prompt): [v
m/mod

_
x
cutor/mod

s/b

p2.py](../../../v
m/mod

_
x
cutor/mod

s/b

p2.py)
    - Mo
mo (

s
rt aft
r `
|

doft
xt|
` tok

): [v
m/mod

_
x
cutor/mod

s/mo
mo.py](../../../v
m/mod

_
x
cutor/mod

s/mo
mo.py)
### Ha
d


g prompt updat
s u
r

at
d to mu
t
-moda
 data
[_g
t_prompt_updat
s][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._g
t_prompt_updat
s] assum
s that 
ach app

cat
o
 of prompt updat
 corr
spo
ds to o

 mu
t
-moda
 
t
m. If th
 HF proc
ssor p
rforms add
t
o
a
 proc
ss

g r
gard

ss of ho
 ma
y mu
t
-moda
 
t
ms th
r
 ar
, you shou
d ov
rr
d
 [_app
y_hf_proc
ssor_tok

s_o

y][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._app
y_hf_proc
ssor_tok

s_o

y] so that th
 proc
ss
d tok

 

puts ar
 co
s
st

t 

th th
 r
su
t of app
y

g th
 HF proc
ssor o
 t
xt 

puts. Th
s 
s b
caus
 tok

 

puts bypass th
 HF proc
ssor accord

g to [our d
s
g
](../../d
s
g
/mm_proc
ss

g.md).
Examp

s:
    - Cham


o
 (app

ds `s
p_tok

`): [v
m/mod

_
x
cutor/mod

s/cham


o
.py](../../../v
m/mod

_
x
cutor/mod

s/cham


o
.py)
    - Fuyu (app

ds `boa_tok

`): [v
m/mod

_
x
cutor/mod

s/fuyu.py](../../../v
m/mod

_
x
cutor/mod

s/fuyu.py)
    - Mo
mo (app


s chat t
mp
at
 
h
ch 
s 
ot d
f


d 

s

h
r
): [v
m/mod

_
x
cutor/mod

s/mo
mo.py](../../../v
m/mod

_
x
cutor/mod

s/mo
mo.py)
### Custom HF proc
ssor
Som
 mod

s do
't d
f


 a
 HF proc
ssor c
ass o
 HF Hub. I
 that cas
, you ca
 d
f


 a custom HF proc
ssor that has th
 sam
 ca
 s
g
atur
 as HF proc
ssors a
d pass 
t to [_ca
_hf_proc
ssor][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._ca
_hf_proc
ssor].
Examp

s:
    - D
pS
k-VL2: [v
m/mod

_
x
cutor/mod

s/d
ps
k_v
2.py](../../../v
m/mod

_
x
cutor/mod

s/d
ps
k_v
2.py)
    - I
t
r
VL: [v
m/mod

_
x
cutor/mod

s/

t
r
v
.py](../../../v
m/mod

_
x
cutor/mod

s/

t
r
v
.py)
    - Q


-VL: [v
m/mod

_
x
cutor/mod

s/q


_v
.py](../../../v
m/mod

_
x
cutor/mod

s/q


_v
.py)
