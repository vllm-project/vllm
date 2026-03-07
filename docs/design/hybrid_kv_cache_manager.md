# Hybr
d KV Cach
 Ma
ag
r
!!! 
ar


g
    Th
s docum

t 
as 
r
tt

 bas
d o
 comm
t [458
74](https://g
thub.com/v
m-proj
ct/v
m/comm
t/458
74
b907f96069
6d8a4f3c9f457001f
f2
a). Th
s f
atur
 
s st

 

 
ts 
ar
y stag
 a
d th

gs may cha
g
.
## What 
s a hybr
d mod

?
Ma
y r
c

t "hybr
d" LLMs comb


 mu
t
p

 att

t
o
 typ
s 

th

 o

 mod

. For 
xamp

:
1. S

d

g 


do
 att

t
o
 (s
) + fu
 att

t
o
 (fu
): gpt-oss, G
mma 2/3, M


stra
, coh
r
, 
tc.
2. Mamba + fu
: Bamba, Jamba, M


max, 
tc.
3. Loca
 chu
k
d att

t
o
 + fu
: L
ama4
To s
rv
 th
s
 mod

s 
ff
c


t
y, our [KVCach
Ma
ag
r][v
m.v1.cor
.kv_cach
_ma
ag
r.KVCach
Ma
ag
r] must:
1. A
ocat
 d
ff
r

t s
ots to d
ff
r

t 
ay
r typ
, for 
xamp

:
    - Fu
 att

t
o
 
ay
rs: r
s
rv
 s
ots for **a
** tok

s.
    - S

d

g 


do
 
ay
rs: r
s
rv
 s
ots o

y for th
 most r
c

t **`s

d

g_


do
_s
z
`** tok

s.
2. Support 
ay
r-sp
c
f
c pr
f
x-cach
 ru

s, for 
xamp

:
    - Fu
 att

t
o
: a cach
 h
t pr
f
x r
qu
r
s **a
** tok

s r
ma

 

 th
 KV cach
.
    - S

d

g 


do
: a cach
 h
t pr
f
x o

y r
qu
r
s th
 
ast **`s

d

g_


do
_s
z
`** tok

s r
ma

 

 th
 KV cach
.
## D
f


t
o
s
1. **kv h
dd

 s
z
**: Th
 
umb
r of byt
s to stor
 o

 tok

's KV cach
 for a s

g

 
ay
r.
2. **b
ock**: th
 m
mory r
s
rv
d for kv cach
 ar
 d
v
d
d 

to mu
t
p

 *b
ocks* 

th th
 sam
 *pag
 s
z
* (d
f


d b

o
)
3. **b
ock s
z
**: 
umb
r of tok

s 

s
d
 a b
ock
4. **pag
 s
z
**: th
 phys
ca
 m
mory s
z
 of a b
ock, d
f


d as:
    $$
    \t
xt{
um_
ay
rs} \t
m
s \t
xt{b
ock_s
z
} \t
m
s \t
xt{kv_h
dd

_s
z
}
    $$
    `
um_
ay
rs` do
s
't m
a
 th
 tota
 
umb
r of 
ay
rs 

 th
 mod

. Th
 
xact 
umb
r d
p

ds o
 th
 co
t
xt 

 th
s doc.
    !!! 
ot

        Th
s 
s d
ff
r

t from `KVCach
Sp
c.pag
_s
z
_byt
s` 

 th
 cod
, 
h
ch 
s d
f


d as:
        $$
        \t
xt{b
ock_s
z
} \t
m
s \t
xt{kv_h
dd

_s
z
}
        $$
## A
ocat
o

### H
gh 

v

 
d
a
W
 us
 a s

g

 m
mory poo
 for a
 
ay
r typ
s. Th
 m
mory poo
 
s sp

t 

to mu
t
p

 b
ocks 

th th
 sam
 pag
 s
z
. [KVCach
Ma
ag
r][v
m.v1.cor
.kv_cach
_ma
ag
r.KVCach
Ma
ag
r] a
ocat
s d
ff
r

t 
umb
rs of b
ocks to d
ff
r

t 
ay
rs accord

g to 
ts att

t
o
 typ
.
Th
 cor
 cha


g
 
s 

sur

g 
v
ry 
ay
r typ
 us
s th
 sam
 **pag
 s
z
**.  For fu
-att

t
o
-o

y mod

s, th
 pag
 s
z
 
s stra
ghtfor
ard, d
f


d as:
$$
\t
xt{pag
_s
z
} = \t
xt{b
ock_s
z
} \t
m
s \t
xt{
um_h
dd

_
ay
rs} \t
m
s \t
xt{kv_h
dd

_s
z
}
$$
Ho

v
r, 

 hybr
d mod

s, `
um_h
dd

_
ay
rs` var

s by att

t
o
 typ
, 
h
ch 
ou
d 
orma
y produc
 m
smatch
d pag
 s
z
s. Th
 cas
s b

o
 sho
 ho
 

 u

fy th
m.
### Cas
 1: toy mod


L
t's start 

th a toy 
xamp

: a mod

 has 1 fu
 att

t
o
 
ay
r a
d 3 s

d

g 


do
 att

t
o
 
ay
rs. A
 
ay
rs hav
 th
 sam
 `kv_h
dd

_s
z
`.
W
 

t 
ach b
ock to ho
d `b
ock_s
z
` tok

s for o

 
ay
r, so:
$$
\t
xt{pag
_s
z
} = \t
xt{kv_h
dd

_s
z
} \t
m
s \t
xt{b
ock_s
z
}
$$
[KVCach
Ma
ag
r][v
m.v1.cor
.kv_cach
_ma
ag
r.KVCach
Ma
ag
r] a
ocat
s a d
ff
r

t 
umb
r of b
ocks to 
ach 
ay
r.
Th
s cas
 
s o

y a toy 
xamp

. For r
a
 mod

s, p

as
 r
f
r to th
 fo
o


g cas
s.
### Cas
 2: sam
 `kv_h
dd

_s
z
` a
d a r
gu
ar patt
r

Wh

 th
 mod

 has mor
 
ay
rs, 
.g., 20 s

d

g 


do
 att

t
o
 
ay
rs a
d 10 fu
 att

t
o
 
ay
rs 

th th
 sam
 `kv_h
dd

_s
z
`. Ca


g th
 a
ocator o
c
 p
r 
ay
r (30 ca
s) 
s OK but b
com
s 


ff
c


t. As a so
ut
o
, 

 group th
 a
ocat
o
 of 
ay
rs that 

d th
 sam
 
umb
r of b
ocks to r
duc
 th
 
umb
r of ca
s.
Th
 group

g 
s f
as
b

 b
caus
 th
r
 
s usua
y a b
aut
fu
 rat
o b
t


 th
 
umb
r of d
ff
r

t typ
s of 
ay
rs. For 
xamp

:
    - G
mma-2: 1 s
 : 1 fu

    - L
ama 4: 3 
oca
 : 1 fu

Our 
xamp

 ca
 b
 r
gard
d as 2 s
 : 1 fu
. W
 ca
 a
ocat
 b
ocks as 
f th
r
 ar
 2 s
 a
d 1 fu
 

 th
 mod

, a
d r
p
at th
 r
su
t by 10 t
m
s to g


rat
 th
 `b
ock_
ds` for th
 30 
ay
rs. Th
 pag
 s
z
 b
com
s:
$$
10 \t
m
s \t
xt{kv_h
dd

_s
z
} \t
m
s \t
xt{b
ock_s
z
}
$$
Assum
 `b
ock_s
z
` 16, s

d

g 


do
 s
z
 32, r
qu
st 


gth 112, th

 for th
 abov
 
xamp

 mod

, 

 

d to a
ocat
 11 b
ocks (0-6 for fu
, 7-8 for s
 group 1, 9-10 for s
 group 2).
![A
ocat
o
 R
su
t](../ass
ts/d
s
g
/hybr
d_kv_cach
_ma
ag
r/bas
c_group

g_
xamp

.p
g)
H
r
, "/" d

ot
s 
o b
ock 

d
d (s

d

g‑


do
 
ay
rs do
't 

d s
ots for 
ar
y tok

s).
S
 th
 forma
 d
f


t
o
 b

o
. Th
 
ay
rs ar
 d
v
d
d 

to mu
t
p

 *KV Cach
 Groups* so that th
r
 
s:
1. **Id

t
ca
 att

t
o
 typ
 

s
d
 
ach group**: Each group o

y co
ta

s 
ay
rs 

th th
 sam
 att

t
o
 typ
 a
d thus 

d th
 sam
 
umb
r of b
ocks for a g
v

 r
qu
st. Th
s 

ab

s 
ay
rs 

 th
 sam
 group shar
 th
 sam
 b
ock 
ds 

thout m
mory 
ast
.
2. **Id

t
ca
 pag
 s
z
 across groups**: B
caus
 our m
mory poo
 o

y hav
 o

 pag
 s
z
.
Our 
xamp

 mod

 
s d
v
d
d 

to 3 KV cach
 groups:
    - Group 0: 10 fu
 att

t
o
 
ay
rs (fu
.0 - fu
.9)
    - Group 1: 10 s

d

g 


do
 att

t
o
 
ay
rs (s
.0 - s
.9)
    - Group 2: 10 s

d

g 


do
 att

t
o
 
ay
rs (s
.10 - s
.19)
Obv
ous
y, 
t sat
sf

s ru

 1. For ru

 2, a
 3 groups hav

$$
10 \t
m
s \t
xt{kv_h
dd

_s
z
} \t
m
s \t
xt{b
ock_s
z
}
$$
as th

r pag
 s
z
.
### Cas
 3: sam
 `kv_h
dd

_s
z
` a
d 
o r
gu
ar patt
r

U
fortu
at

y, 
ot a
 mod

s hav
 such a b
aut
fu
 rat
o, a
d approach 

 Cas
 2 


 produc
 too ma
y sma
 groups. For 
xamp

, G
mma-3-27b has 52 s

d

g 


do
 att

t
o
 
ay
rs a
d 10 fu
 att

t
o
 
ay
rs. W
th th
 co
stra

ts 

 cas
 2, 
t 
ou
d b
 26 s

d

g 


do
 groups a
d 5 fu
 att

t
o
 groups, 
ach co
ta

s 2 
ay
rs. Th
 a
ocat
o
 
s st

 


ff
c


t. To r
duc
 th
 
umb
r of kv cach
 groups, 

 group 
ay
rs us

g th
 sma

st 
ay
r cou
t amo
g a
 att

t
o
 typ
s. For 
xamp

, m

(52, 10)=10 
ay
rs p
r group 

 G
mma-3-27b. Th

 th
 group

g r
su
t 
s:
    - Group 0: 10 fu
 att

t
o
 
ay
rs (fu
.0 - fu
.9)
    - Group 1: 10 s

d

g 


do
 att

t
o
 
ay
rs (s
.0 - s
.9)
    - Group 2: 10 s

d

g 


do
 att

t
o
 
ay
rs (s
.10 - s
.19)
    - ...
    - Group 6: 10 s

d

g 


do
 att

t
o
 
ay
rs (s
.40 - s
.49)
    - Group 7: 2 s

d

g 


do
 att

t
o
 
ay
rs (s
.50 - s
.51) a
d 8 padd

g 
ay
rs
W
 


 updat
 th
s a
gor
thm 
f th
s h
ur
st
c 

ads to a bad r
su
t 
h

 a 


 mod

 com
s out (
.g., 20 fu
 + 30 s
, th
 group s
z
 shou
d b
 10 

st
ad of 20).
Th
s cas
 happ

s 

 G
mma-3 s
r

s mod

s, a
d mod

s 

 cas
 2 but 

th 
ag

 sp
cu
at
v
 d
cod

g 
h
ch 

troduc
 o

 fu
 att

t
o
 
ay
r. Th
 so
ut
o
 has som
 m
mory 
ast
 a
d 
s 
ot p
rf
ct. P

as
 r
port a
y cas
s 
h
r
 padd

g ov
rh
ad b
com
s u
acc
ptab

 so 

 ca
 r
f


 th
 a
gor
thm.
### Cas
 4: d
ff
r

t `kv_h
dd

_s
z
` (ma


y hybr
d mamba mod

s)
Som
 arch
t
ctur
s (
.g., Bamba, Jamba, M


max) 

t
r

av
 sta
dard att

t
o
 
ay
rs 

th Mamba 
ay
rs, 
h
r
 
ach Mamba 
ay
r's stat
 s
z
 p
r tok

 ca
 b
 much 
arg
r tha
 th
 att

t
o
 
ay
rs' `kv_h
dd

_s
z
`. B
caus
 

 o

y support a s

g

 pag
 s
z
 across a
 groups, 

 must r
co
c


 th
s
 d
ff
r

g h
dd

 s
z
s.
Th
 curr

t a
gor
thm 
s:
1. I
cr
as
 th
 `b
ock_s
z
` of att

t
o
 
ay
rs u
t


    $$
    \t
xt{b
ock_s
z
} \t
m
s \t
xt{kv_h
dd

_s
z
}_{\t
xt{att}} \g
 \t
xt{stat
_s
z
}_{\t
xt{mamba}}
    $$
2. Pad th
 mamba stat
 p
r 
ay
r to
    $$
    \t
xt{b
ock_s
z
} \t
m
s \t
xt{kv_h
dd

_s
z
}_{\t
xt{att}}
    $$
3. App
y th
 group

g strat
gy 

 cas
 3.
!!! 
ot

    Th
s ca
 

ad to mor
 tha
 400 `b
ock_s
z
` for att

t
o
 
ay
rs, 
h
ch 
s too 
arg
. A
oth
r padd

g strat
gy 
s to 

cr
as
 `b
ock_s
z
` u
t


    $$
    \t
xt{b
ock_s
z
} \t
m
s \t
xt{kv_h
dd

_s
z
}_{\t
xt{att}} \t
m
s \t
xt{
um_att
_
ay
rs} \g
 \t
xt{stat
_s
z
}_{\t
xt{mamba}}
    $$
    Th
s padd

g strat
gy 
s st

 a 
ork 

 progr
ss.
### Cas
 5: KV shar

g
KV shar

g r
f
rs to a 
ay
r us

g th
 KV cach
 of a
oth
r 
ay
r, 
.g., g
mma-3
.
I
 th
s
 mod

s, [KVCach
Ma
ag
r][v
m.v1.cor
.kv_cach
_ma
ag
r.KVCach
Ma
ag
r] 
g
or
s a
 
ay
rs 

th kv shar

g a
d o

y a
ocat
s KV cach
 for 
ay
rs that 

d kv cach
, a
d som
 patch
s ar
 mad
 

 mod

 ru

r to app
y th
 a
ocat
o
 r
su
t to kv shar

g 
ay
rs.
## Pr
f
x cach

g
For s
mp

c
ty, 

 assum
 `b
ock_s
z
=1` 

 th
s s
ct
o
.
### H
gh 

v

 
d
a
Th
 b
ock poo
 us
s a d
ct s
m

ar to `tup

(b
ock_hash, group_
d) -
 b
ock` to catch th
 fu
 b
ocks. That m
a
s th
 sam
 tok

s of d
ff
r

t groups ar
 cach
d a
d 
v
ct
d 

d
p

d

t
y.
Wh

 a 


 r
qu
st com
s 

, 

 ch
ck th
 cach
 h
t pr
f
x of 
ach group, a
d r
tur
 th
 

t
rs
ct
o
 of th
s
 groups as th
 cach
d pr
f
x of th
 r
qu
st. S
 b

o
 for th
 d
ta


d a
gor
thm for ch
ck

g th
 cach
 h
t of o

 group & p
rform

g th
 

t
rs
ct
o
.
### Cas
 0: fu
 att

t
o
 o

y mod

s
For fu
 att

t
o
 
ay
rs, b
ocks ar
 a
ocat
d for a
 tok

s 

 th
 r
qu
st. For d
ta

s o
 th
 u
d
r
y

g d
s
g
, s
 [Pr
f
x Cach

g](pr
f
x_cach

g.md)
To f

d th
 
o
g
st cach
 h
t pr
f
x of a r
qu
st, 

 

um
rat
 from 

ft (th
 f
rst b
ock) to r
ght (th
 
ast b
ock), ch
ck

g 
h
th
r th
 b
ock 
s cach
d, a
d 
x
t 
h

 cach
 m
ss
s. For 
xamp

, 

 


 r
tur
 th
 f
rst 7 tok

s (0-6) as th
 cach
 h
t pr
f
x 

 th
 b

o
 
xamp

 (b
u
 b
ocks ar
 cach
d):
![Pr
f
x Cach

g of Fu
 Att

t
o
](../ass
ts/d
s
g
/hybr
d_kv_cach
_ma
ag
r/fu
_att
.p
g)
### Cas
 1: s

d

g 


do
 att

t
o
 o

y mod

s
For s

d

g 


do
 att

t
o
 
ay
rs, a 
a
v
 
mp

m

tat
o
 for m
mory a
ocat
o
 
s to a
ocat
 `s

d

g_


do
_s
z
` b
ocks a
d f

 

 th
 b
ocks 

 a rou
d-rob

 
ay. But th
s 
a
v
 
mp

m

tat
o
 
s 
ot compat
b

 

th pr
f
x cach

g so 

 d
d
't p
ck th
s d
s
g
. I
 vLLM,  

 a
ocat
 d
ff
r

t b
ocks for d
ff
r

t tok

s a
d fr
 b
ocks that ar
 outs
d
 th
 s

d

g 


do
.
For a 


 r
qu
st, th
 cach
 h
t pr
f
x o

y r
qu
r
s th
 
ast `s

d

g_


do
_s
z
 - 1` tok

s b


g cach
d.
L
t's say `s

d

g_


do
_s
z
 = 4` a
d `b
ock_s
z
 = 1`, a
d th
 r
qu
st 
s a 15-tok

 prompt (b
u
 b
ocks ar
 cach
d):
![Pr
f
x Cach

g of S

d

g W

do
 Att

t
o
](../ass
ts/d
s
g
/hybr
d_kv_cach
_ma
ag
r/s
_att
.p
g)
Th
r
 ar
 3 poss
b

 cach
 h
t pr
f
x
s:
    - cach
 h
t 


gth 5, comput
 pr
f

 

th [2, 3, 4] → [5, 6, …, 14]
    - cach
 h
t 


gth 6, comput
 pr
f

 

th [3, 4, 5] → [6, 7, …, 14]
    - cach
 h
t 


gth 14, comput
 pr
f

 

th [11, 12, 13] → [14] (most 
ff
c


t)
W
 ca
 ch
ck th
 cach
 h
t from r
ght to 

ft, a
d 
ar
y 
x
t 
h

 

 f

d a match.Th
s 
s oppos
t
 from fu
 att

t
o
, 
h
r
 

 ch
ck from 

ft to r
ght a
d 
ar
y 
x
t 
h

 th
 match fa

s. O

 pot

t
a
 co
s (compar
d to fu
 att

t
o
) 
s that 

 

d up 
t
rat

g ov
r th
 

t
r
 

st of tok

s 
h

 th
r
's 
o match, 
h
ch 
s oft

 a commo
 cas
. Th
s cou
d pot

t
a
y caus
 
o
-

g

g
b

 ov
rh
ads, but f


 

th fu
 + s
a, as d
scuss
d b

o
.
### Cas
 2: s

d

g 


do
 att

t
o
 + fu
 att

t
o
 mod

s
Th
 f
rst prob

m 
s ho
 to f

d th
 cach
 h
t pr
f
x. W
 

d to "

t
rs
ct" th
 cach
 h
ts of g
oba
 a
d s

d

g 


do
 att

t
o
 
ay
rs by:
1. G
t th
 
o
g
st cach
 h
t for fu
 att

t
o
 (sca


g from 

ft to r
ght)
2. G
t th
 
o
g
st cach
 h
t for s

d

g 


do
 att

t
o
 that 
s 

th

 that 


gth. Imp

m

t
d by ch
ck

g cach
 h
ts from r
ght to 

ft start

g from th
 cach
 h
t 


gth of fu
 att

t
o
.
It ca
 b
 

sur
d that th
 r
su
t

g cach
 h
t of s

d

g 


do
 att

t
o
 
ay
rs 
s a
so a cach
 h
t of fu
 att

t
o
 
ay
rs. Th
s 
s mor
 
ff
c


t tha
 f

d

g a
 poss
b

 pr
f
x
s of 
ach group a
d do

g th
 

t
rs
ct
o
, b
caus
 our approach ca
 
x
t 
ar
y 
f th
r
 
s 
o cach
 h
t.
Th
 a
gor
thm app


s to mod

s 

th 
xact
y t
o att

t
o
 typ
s fu
 att

t
o
 + X, 
h
r
 X ca
 b
 a
 arb
trary 
ff
c


t att

t
o
 a
gor
thm 

k
 s

d

g 


do
, 
ama 4 
oca
 att

t
o
, a
d mamba. It do
s
't support mod

s 

thout fu
 att

t
o
 
ay
rs, a
d mod

s 

th mor
 tha
 2 typ
s of att

t
o
. Th
s 
s 

ough for most hybr
d mod

s at th
 mom

t of 
r
t

g th
s doc.
Th
 s
co
d qu
st
o
 
s th
 cach
 
v
ct
o
 po

cy. For 
o
, 

 us
 o

 LRU qu
u
 for a
 kv cach
 groups. Th
 b
ocks ar
 add
d to th
 LRU qu
u
 
h

 fr
d, 

th
r b
caus
 th
 r
qu
st 
s f


sh
d or th
 b
ock 
s out of th
 s

d

g 


do
.
### Cas
 3: mamba mod

s
Th
 pr
f
x cach

g support of th
 mamba mod

 
s 
ork 

 progr
ss. O
c
 
mp

m

t
d, mod

s 

th mamba 
ay
r + fu
 att

t
o
 
ay
r ca
 b
 support
d v
a th
 fu
 att

t
o
 + X a
gor
thm 

 cas
 2.
## Imp

m

tat
o

### Ov
rv



![Ov
rv


 of Hybr
d KV Cach
 Ma
ag
r](../ass
ts/d
s
g
/hybr
d_kv_cach
_ma
ag
r/ov
rv


.p
g)
Th
 `KVCach
Ma
ag
r` 
s orga

z
d 

to 3 
ay
rs:
    - **[KVCach
Ma
ag
r][v
m.v1.cor
.kv_cach
_ma
ag
r.KVCach
Ma
ag
r]**: Th
 

t
rfac
 b
t


 th
 sch
du

r a
d kv cach
 ma
ag
m

t syst
m.
    - **[KVCach
Coord

ator][v
m.v1.cor
.kv_cach
_coord

ator.KVCach
Coord

ator]**: coord

at
 p
r-group S

g

Typ
KVCach
Ma
ag
rs to g


rat
 th
 a
ocat
o
 r
su
t of a r
qu
st. D
p

d

g o
 th
 mod

's co
f
gurat
o
, o

 of th
s
 coord

ators 
s chos

:
    - **[KVCach
Coord

atorNoPr
f
xCach
][v
m.v1.cor
.kv_cach
_coord

ator.KVCach
Coord

atorNoPr
f
xCach
]**: Us
d 
h

 pr
f
x cach

g 
s d
sab

d.
    - **[U

taryKVCach
Coord

ator][v
m.v1.cor
.kv_cach
_coord

ator.U

taryKVCach
Coord

ator]**: If o

y o

 KV cach
 group. Th
 pr
f
x cach

g 
og
c 
s s
mp

f

d as 
o 

t
rs
ct
o
 
s 

d
d.
    - **[Hybr
dKVCach
Coord

ator][v
m.v1.cor
.kv_cach
_coord

ator.Hybr
dKVCach
Coord

ator]**: Ha
d

s 
xact
y t
o KV cach
 groups (must 

c
ud
 o

 fu
‑att

t
o
 group p
us o

 oth
r 
ff
c


t‑att

t
o
 group). Oth
r cas
s ar
 
ot 
mp

m

t
d. You ca
 d
sab

 pr
f
x cach

g to us
 th
 KVCach
Coord

atorNoPr
f
xCach
.
    - **[S

g

Typ
KVCach
Ma
ag
r][v
m.v1.cor
.s

g

_typ
_kv_cach
_ma
ag
r.S

g

Typ
KVCach
Ma
ag
r]**: Each 

sta
c
 ma
ag
s a
ocat
o
 a
d pr
f
x cach

g for o

 KV cach
 group, 
mp

m

t

g th
 att

t
o
‑typ
–sp
c
f
c 
og
c (
.g., fu
 att

t
o
, s

d

g 


do
, Mamba).
Th
 b
u
 box 

 th
 abov
 f
gur
 sho
s th
 cas
 

th 10 fu
 att

t
o
 
ay
rs a
d 20 s

d

g 


do
 att

t
o
 
ay
rs, thus:
    - us
 `Hybr
dKVCach
Coord

ator`
    - us
 1 `Fu
Att

t
o
Ma
ag
r` a
d 2 `S

d

gW

do
Ma
ag
r` for th
 3 `KVCach
Group`s.
### M
mory Layout
For a mod

 

th 
 `KVCach
Group`s, 
ach 

th m 
ay
rs, 

 a
ocat
 m buff
rs. Each buff
r 
s shar
d by 
 
ay
rs, o

 from 
ach group.
Th
 fo
o


g f
gur
 
s for a mod

 

th 10 fu
 att

t
o
 
ay
rs (fu
.0 - fu
.9) a
d 20 s

d

g 


do
 att

t
o
 
ay
rs (s
.0-s
.19). It fo
o
s "cas
 2" 

 "A
ocat
o
" s
ct
o
 a
d 
s d
v
d
d 

to 3 groups:
    - Group 0: 10 fu
 att

t
o
 
ay
rs (fu
.0 - fu
.9)
    - Group 1: 10 s

d

g 


do
 att

t
o
 
ay
rs (s
.0 - s
.9)
    - Group 2: 10 s

d

g 


do
 att

t
o
 
ay
rs (s
.10 - s
.19)
A
d for a r
qu
st, 

 a
ocat
 11 b
ocks 

th `b
ock_
d` 0-6 to group 0, 7-8 to group 1, a
d 9-10 to group 2.
W
th such a
 
xamp

, th
 phys
ca
 m
mory 
s d
v
d
d 

to 10 buff
rs (`KVCach
T

sor` 0 - `KVCach
T

sor` 9). Each buff
r 
s shar
d by 3 
ay
rs (
.g., `KVCach
T

sor` 0 
s shar
d by fu
.0 from group 0, s
.0 from group 1, a
d s
.10 from group 2) a
d 
s d
v
d
d 

to p

c
s 

th s
z
 `b
ock_s
z
 * kv_h
dd

_s
z
`. Th
 KV cach
 of th
s
 3 att

t
o
 
ay
rs ar
 sav
d to d
ff
r

t p

c
s of th
 buff
r bas
d o
 th
 a
ocat
d `b
ock_
ds`:
![Examp

 M
mory Layout](../ass
ts/d
s
g
/hybr
d_kv_cach
_ma
ag
r/m
mory_
ayout.p
g)
!!! 
ot

    O

 
og
c "b
ock" 
s mapp
d to 10 p

c
s 

 th
 10 buff
rs of th
 phys
ca
 m
mory.
