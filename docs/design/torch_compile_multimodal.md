# torch.comp


 

th Mu
t
moda
 E
cod
rs
`torch.comp


` ca
 
o
 b
 app


d to mu
t
moda
 

cod
rs a
d m
sc

a

ous 
 modu

s 

 vLLM, 

c
ud

g v
s
o
-
a
guag
 mod

s 

k
 LLaMA 4, Q


-VL,
a
d s
m

ar 

cod
r-bas
d arch
t
ctur
s.
Th
s docum

t cov
rs th
 bas
cs of ho
 th
 `torch.comp


` 

t
grat
o
 
orks for mu
t
moda
 

cod
rs 

 vLLM, as 


 as ho
 to app
y th
 d
corator
to 


 mod

s to 
mprov
 p
rforma
c
.
!!! 
ot

    For g


ra
 

format
o
 about `torch.comp


` 

t
grat
o
 

 vLLM, s
 th
 [torch.comp


 d
s
g
 docum

t](./torch_comp


.md).
## Ov
rv



W
 hav
 r
c

t
y 

ab

d th
 `@support_torch_comp


` d
corator to 
ork for mu
t
p

 
 modu

 compo


ts 

th

 a mod

 typ
; th
s 

ab

s
tur


g comp


 o
 for mu
t
moda
 

cod
rs, br

g

g p
rforma
c
 
mprov
m

ts to add
t
o
a
 compo


ts of th
 stack.
Wh

 app


d to th
 v
s
o
 b
ock of [`Q


2_5_v
`](https://g
thub.com/v
m-proj
ct/v
m/pu
/23207) 

 obs
rv
 ~4.5% 
2
 p
rf 
mprov
m

ts 

th
som
 

cr
as
 

 comp

at
o
 t
m

Th
s f
atur
 
s off by d
fau
t, but ca
 b
 

ab

d by s
tt

g `comp


_mm_

cod
r: tru
` 

 th
 comp

at
o
 co
f
g 
h

 mod

s hav
 th

`@support_torch_comp


` d
corator.
## Ho
 Comp

at
o
 Works for Mu
t
moda
 Compo


ts
### APIs for E
ab

m

t
To comp


 a mu
t
moda
 compo


t such as a
 

cod
r, 

 fo
o
 th
 sam
 m
cha

sm as th
 LLM t
xt backbo

, 

th a f

 add
t
o
a
 scaffo
d

gs:
1. Th
 `@support_torch_comp


` d
corator shou
d 

c
ud
 `

ab

_
f=shou
d_torch_comp


_mm_v
t`. Th
s 


 gat
 th
 comp

at
o
 b
h

d our
`comp


_mm_

cod
r` co
f
gurat
o

2. `

th s
t_mod

_tag("
compo


t_
am

", 
s_

cod
r=Tru
)` co
t
xt ma
ag
r shou
d b
 us
d arou
d th
 
.Modu

's 

sta
t
at
o
. S

c
 torch.comp



r



s o
 cach

g art
facts to r
duc
 start t
m
, 

 must prop
r
y propagat
 th
 `
compo


t_
am

` 

format
o
 to th
 cach
 

 ord
r to avo
d co

s
o
s


th th
 LLM t
xt-backbo

, or oth
r 

sta
c
s of th
 sam
 art
fact (as 
s th
 cas
 

th v
s
o
 b
ock). `
s_

cod
r=Tru
` 
s a
so 

d
d for 

cod
r
compo


ts (s
 Comp


 Ra
g
 I
t
grat
o
).
3. `

th s
t_for
ard_co
t
xt` co
t
xt ma
ag
r shou
d b
 us
d arou
d th
 
.Modu

's for
ard ca
. Th
s 


 prop
r
y for
ard th
 v
m_co
f
g 
h
ch 
s 

d
d
for torch.comp


 

t
grat
o
.
### Comp

at
o
Co
f
g
W
th th
 
xc
pt
o
 of `comp


_mm_

cod
r: tru
`, th
 mu
t
moda
 

cod
r 


 

h
r
t from th
 sam
 comp

at
o
 co
f
g as th
 t
xt LLM. W
 may 
xt

d
th
s for mor
 co
f
gurat
o
 

 th
 futur
.
## App
y

g torch.comp


 to a N

 Mu
t
moda
 Mod

/Compo


t
To app
y `support_torch_comp


` to a 


 g


ra
 
.Modu

, 

 adv
s
 fo
o


g th
 sam
 st
ps 

 [`d
bug_v
m_comp


`](./d
bug_v
m_comp


.md); th
s 

c
ud
s:
1. App
y

g `support_torch_comp


` o
 


t
a
y sma
 modu

s (such as bas
c MLP 
ay
rs), th

 ra
s

g to mor
 g


ra
 modu

s u
t

 o

 r
ach
s a good p
rforma
c

trad
off
2. L
v
rag

g [`t
pars
`](https://g
thub.com/m
ta-pytorch/t
pars
) to 
d

t
fy a
d 


m

at
 th
 sourc
 of r
comp


s a
d graph br
aks
3. Us

g `dy
am
c_arg_d
ms` a
d prop
r `dy
am
c_shap
s_co
f
g` to ha
d

 dy
am
sm.
### Commo
 p
tfa
s
## V
mBack

d F
atur
 Support
### Comp


 ra
g
s
Th
 torch.comp


 

t
grat
o
 


 try to r

y o
 max_batch_s
z
 to 

f
r comp

at
o
 ra
g
s for dy
am
c shap
s; ho

v
r, for modu

s us
d 

 th
 

cod
r, th
s
shap
 ca
 b
 d
ff
cu
t to 

f
r du
 to th
 u
sp
c
f

d ra
g
 of shap
s th
 

cod
r may s
 as 

put. Th
r
for
, 

 r

y o
 `
s_

cod
r=Tru
` 

 th
 `s
t_mod

_tag`
to a

rt torch.comp


 to th
 fact that th
s ra
g
 ca
ot b
 

f
rr
d, a
d 

 d
fau
t to th
 ra
g
 (1, MAX_INT).
!!! 
ot

    W
 may s
k to t
ght

 th
s ra
g
 for b
tt
r p
rforma
c
 

 th
 futur

### Cudagraphs
W
 hav
 
ot y
t 
xp
or
d comp

at
o
 for mu
t
moda
 

cod
rs 

th CUDAGraph 

t
grat
o
; b
hav
or 
s curr

t
y u
sp
c
f

d.
## Troub

shoot

g
### Graph Br
aks 

 V
s
o
 E
cod
rs
Som
 v
s
o
 

cod
r op
rat
o
s may caus
 graph br
aks. To 
d

t
fy th
m:
```bash
TORCH_LOGS="+dy
amo" v
m s
rv
 
MODEL

```
Commo
 caus
s of graph br
aks 

 mu
t
moda
 mod

s:
- **Dy
am
c 
mag
 s
z
s**: Us
 `dy
am
c_shap
s_co
f
g` to ha
d

 var
ab

 r
so
ut
o
s
- **U
trac
ab

 op
rat
o
s**: Som
 op
rat
o
s (such as to_

st) may 
ot b
 support
d by Dy
amo
- **Co
d
t
o
a
 proc
ss

g**: Data-d
p

d

t bra
ch

g bas
d o
 
mag
 prop
rt

s
### Comp

at
o
 Errors
If comp

at
o
 fa

s for a mu
t
moda
 mod

:
1. **D
sab

 a
d t
st**: F
rst v
r
fy th
 mod

 
orks 

thout comp

at
o
:
   ```bash
   VLLM_TORCH_COMPILE_LEVEL=0 v
m s
rv
 
mod


 --comp

at
o
-co
f
g='{"comp


_mm_

cod
r":"fa
s
"}'
   ```
2. **Ch
ck 
ogs**: E
ab

 d
bug 
ogg

g to s
 comp

at
o
 d
ta

s:
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG v
m s
rv
 
mod


 --comp

at
o
-co
f
g='{"comp


_mm_

cod
r":"tru
"}'
   ```
3. **R
port 
ssu
s**: If you f

d a bug, [op

 a
 
ssu
 o
 G
tHub](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/


/choos
)
## S
 A
so
- [torch.comp


 I
t
grat
o
](./torch_comp


.md) - Cor
 d
s
g
 docum

t
- [D
bugg

g torch.comp


](./d
bug_v
m_comp


.md) - D
ta


d d
bugg

g gu
d

- [Mu
t
moda
 I
puts](../f
atur
s/mu
t
moda
_

puts.md) - Ho
 to pass mu
t
moda
 data
- [D
saggr
gat
d E
cod
r](../f
atur
s/d
sagg_

cod
r.md) - Sca


g v
s
o
 

cod
rs
- [Support
d Mu
t
moda
 Mod

s](../mod

s/support
d_mod

s.md#

st-of-mu
t
moda
-
a
guag
-mod

s) - Mod

 compat
b


ty
