# Automat
c Pr
f
x Cach

g
Pr
f
x cach

g kv-cach
 b
ocks 
s a popu
ar opt
m
zat
o
 

 LLM 

f
r

c
 to avo
d r
du
da
t prompt computat
o
s. Th
 cor
 
d
a 
s s
mp

 – 

 cach
 th
 kv-cach
 b
ocks of proc
ss
d r
qu
sts, a
d r
us
 th
s
 b
ocks 
h

 a 


 r
qu
st com
s 

 

th th
 sam
 pr
f
x as pr
v
ous r
qu
sts. S

c
 pr
f
x cach

g 
s a
most a fr
 
u
ch a
d 
o
’t cha
g
 mod

 outputs, 
t has b

 

d

y us
d by ma
y pub

c 

dpo

ts (
.g., Op

AI, A
throp
c, 
tc.) a
d most op

 sourc
 LLM 

f
r

c
 fram

orks (
.g., SGLa
g).
Wh


 th
r
 ar
 ma
y 
ays to 
mp

m

t pr
f
x cach

g, vLLM choos
s a hash-bas
d approach. Sp
c
f
ca
y, 

 hash 
ach kv-cach
 b
ock by th
 tok

s 

 th
 b
ock a
d th
 tok

s 

 th
 pr
f
x b
for
 th
 b
ock:
```t
xt
                    B
ock 1                  B
ock 2                  B
ock 3
         [A g

t

 br
z
 st
rr
d] [th
 

av
s as ch

dr

] [
augh
d 

 th
 d
sta
c
]
B
ock 1: |
--- b
ock tok

s ----
|
B
ock 2: |
------- pr
f
x ------
| |
--- b
ock tok

s ---
|
B
ock 3: |
------------------ pr
f
x --------------------
| |
--- b
ock tok

s ----
|
```
I
 th
 
xamp

 abov
, th
 KV cach
 

 th
 f
rst b
ock ca
 b
 u

qu

y 
d

t
f

d 

th th
 tok

 “A g

t

 br
z
 st
rr
d”. Th
 th
rd b
ock ca
 b
 u

qu

y 
d

t
f

d 

th th
 tok

s 

 th
 b
ock “
augh
d 

 th
 d
sta
c
”, a
o
g 

th th
 pr
f
x tok

s “A g

t

 br
z
 st
rr
d th
 

av
s as ch

dr

”. Th
r
for
, 

 ca
 bu

d th
 b
ock hash of `hash(tup

[compo


ts])`, 
h
r
 compo


ts ar
:
* Par

t hash va
u
: Th
 hash va
u
 of th
 par

t hash b
ock.
* B
ock tok

s: A tup

 of tok

s 

 th
s b
ock. Th
 r
aso
 to 

c
ud
 th
 
xact tok

s 
s to r
duc
 pot

t
a
 hash va
u
 co

s
o
.
* Extra hash
s: Oth
r va
u
s r
qu
r
d to mak
 th
s b
ock u

qu
, such as LoRA IDs, mu
t
-moda

ty 

put hash
s (s
 th
 
xamp

 b

o
), a
d cach
 sa
ts to 
so
at
 cach
s 

 mu
t
-t

a
t 

v
ro
m

ts.
!!! 
ot
 "Not
 1"
    W
 o

y cach
 fu
 b
ocks.
!!! 
ot
 "Not
 2"
    I
 pr
v
ous v
rs
o
s, th
 hash k
y 
as 
ot guara
t
d to b
 co

s
o
-fr
. As of v0.11, th
 d
fau
t hash

g a
gor
thm 
s `sha256`, 
h
ch addr
ss
s co

s
o
 r
sks.
    For `v
m s
rv
`, you ca
 co
tro
 th
 hash

g a
gor
thm v
a `--pr
f
x-cach

g-hash-a
go`:
    - `sha256` (d
fau
t): Us
s Pytho
's `p
ck

` for s
r
a

zat
o
. Hash
s may 
ot b
 r
produc
b

 across d
ff
r

t Pytho
 or vLLM v
rs
o
s.
    - `sha256_cbor`: Us
s `cbor2` for s
r
a

zat
o
, prov
d

g a r
produc
b

, cross-
a
guag
 compat
b

 hash. Th
s 
s r
comm

d
d for d
t
rm


st
c cach

g across 

v
ro
m

ts.
    - `xxhash`: `Us
s P
ck

 s
r
a

zat
o
 

th xxHash (128-b
t) for fast
r, 
o
-cryptograph
c hash

g. R
qu
r
s th
 opt
o
a
 `xxhash` packag
. IMPORTANT: Us
 of a hash

g a
gor
thm that 
s 
ot co
s
d
r
d cryptograph
ca
y s
cur
 th
or
t
ca
y 

cr
as
s th
 r
sk of hash co

s
o
s, 
h
ch ca
 caus
 u
d
f


d b
hav
or or 
v

 

ak pr
vat
 

format
o
 

 mu
t
-t

a
t 

v
ro
m

ts. Ev

 
f co

s
o
s ar
 st

 v
ry u


k

y, 
t 
s 
mporta
t to co
s
d
r your s
cur
ty r
sk to

ra
c
 aga

st th
 p
rforma
c
 b


f
ts b
for
 tur


g th
s o
.
    - `xxhash_cbor` comb


s ca
o

ca
 CBOR s
r
a

zat
o
 

th xxHash for r
produc
b

 hash

g. R
qu
r
s th
 opt
o
a
 `xxhash` packag
.
**A hash

g 
xamp

 

th mu
t
-moda

ty 

puts**
I
 th
s 
xamp

, 

 

ustrat
 ho
 pr
f
x cach

g 
orks 

th mu
t
-moda

ty 

puts (
.g., 
mag
s). Assum

g 

 hav
 a r
qu
st 

th th
 fo
o


g m
ssag
s:
```t
xt
m
ssag
s = [
    {"ro

": "us
r",
     "co
t

t": [
         {"typ
": "t
xt",
          "t
xt": "What's 

 th
s 
mag
?"
         },
         {"typ
": "
mag
_ur
",
          "
mag
_ur
": {"ur
": 
mag
_ur
},
         },
    ]},
]
```
It 


 b
com
 th
 fo
o


g prompt:
```t
xt
Prompt:
    
s
[INST]What's 

 th
s 
mag
?\
[IMG][/INST]
Tok


z
d prompt:
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]
Prompt 

th p
ac
ho
d
rs (
P
):
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 
P
, 
P
, ..., 
P
, 4]
```
As 

 ca
 s
, aft
r th
 tok


zat
o
, th
 `[IMG]` 


 b
 r
p
ac
d by a s
qu

c
 of p
ac
ho
d
r tok

s, a
d th
s
 p
ac
ho
d
rs 


 b
 r
p
ac
d by 
mag
 
mb
dd

gs dur

g pr
f

. Th
 cha


g
 for pr
f
x cach

g to support th
s cas
 
s 

 

d to d
ff
r

t
at
 
mag
s from th
 p
ac
ho
d
rs. To addr
ss th
s prob

m, 

 

cod
 th
 
mag
 hash g


rat
d by th
 fro
t

d 
mag
 proc
ssor. For 
xamp

, th
 hash of th
 b
ocks 

 th
 abov
 prompt 
ou
d b
 (assum

g b
ock s
z
 16, a
d 

 hav
 41 p
ac
ho
d
r tok

s):
```t
xt
B
ock 0
    Par

t hash: No


    Tok

 IDs: 1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 
p
, ..., 
p

    Extra hash: 

mag
 hash

B
ock 1
    Par

t hash: B
ock 0 hash
    Tok

 IDs: 
p
, ..., 
p

    Extra hash: 

mag
 hash

B
ock 2
    Par

t hash: B
ock 1 hash
    Tok

 IDs: 
p
, ..., 
p

    Extra hash: 

mag
 hash

B
ock 3
    Par

t hash: B
ock 2 hash
    Tok

 IDs: 
p
, ..., 
p
, 4
    Extra hash: 

mag
 hash

```
I
 th
 r
st of th
s docum

t, 

 f
rst 

troduc
 th
 data structur
 us
d for pr
f
x cach

g 

 vLLM v1, fo
o

d by th
 pr
f
x cach

g 
orkf
o
 of major KV cach
 op
rators (
.g., a
ocat
, app

d, fr
, 
v
ct
o
). F

a
y, 

 us
 a
 
xamp

 to 

ustrat
 th
 

d to 

d pr
f
x cach

g 
orkf
o
.
**Cach
 Iso
at
o
 for S
cur
ty**
To 
mprov
 pr
vacy 

 shar
d 

v
ro
m

ts, vLLM supports 
so
at

g pr
f
x cach
 r
us
 through opt
o
a
 p
r-r
qu
st sa
t

g. By 

c
ud

g a `cach
_sa
t` 

 th
 r
qu
st, th
s va
u
 
s 

j
ct
d 

to th
 hash of th
 f
rst b
ock, 

sur

g that o

y r
qu
sts 

th th
 sam
 sa
t ca
 r
us
 cach
d KV b
ocks. Th
s pr
v

ts t
m

g-bas
d attacks 
h
r
 a
 adv
rsary cou
d 

f
r cach
d co
t

t by obs
rv

g 
at

cy d
ff
r

c
s. Th
s off
rs prot
ct
o
 

thout comprom
s

g p
rforma
c
.
```jso

{
  "m
ssag
s": [
    {"ro

": "syst
m", "co
t

t": "You ar
 a h

pfu
 ass
sta
t."},
    {"ro

": "us
r", "co
t

t": "H
r
 
s a docum

t 

th d
ta

s about th
 
or
d s
r

s: ..."},
    {"ro

": "us
r", "co
t

t": "Who 
o
 th
 
or
d s
r

s 

 2020?"}
  ],
  "cach
_sa
t": "your-cach
-sa
t"
}
```
W
th th
s s
tup, cach
 shar

g 
s 

m
t
d to us
rs or r
qu
sts that 
xp

c
t
y agr
 o
 a commo
 sa
t, 

ab


g cach
 r
us
 

th

 a trust group 
h


 
so
at

g oth
rs.
## Data Structur

Th
 pr
f
x cach

g 

 vLLM v1 
s 
mp

m

t
d 

 th
 KV cach
 ma
ag
r. Th
 bas
c bu

d

g b
ock 
s th
 “B
ock” data c
ass (s
mp

f

d):
```pytho

c
ass KVCach
B
ock:
    # Th
 b
ock ID (
mmutab

)
    b
ock_
d: 

t
    # Th
 b
ock hash (


 b
 ass
g

d 
h

 th
 b
ock 
s fu
,
    # a
d 


 b
 r
s
t 
h

 th
 b
ock 
s 
v
ct
d).
    b
ock_hash: B
ockHash
    # Th
 
umb
r of r
qu
sts us

g th
s b
ock 
o
.
    r
f_c
t: 

t
    # Th
 po

t
rs to form a doub
y 


k
d 

st for th
 fr
 qu
u
.
    pr
v_fr
_b
ock: "KVCach
B
ock | No

" = No


    

xt_fr
_b
ock: "KVCach
B
ock | No

" = No


```
Th
r
 ar
 t
o d
s
g
 po

ts to h
gh

ght:
1. W
 a
ocat
 a
 KVCach
B
ock 
h

 


t
a

z

g th
 KV cach
 ma
ag
r to b
 a b
ock poo
. Th
s avo
ds Pytho
 obj
ct cr
at
o
 ov
rh
ads a
d ca
 
as

y track a
 b
ocks a
 th
 t
m
.
2. W
 

troduc
 doub
y 


k
d 

st po

t
rs d
r
ct
y 

 th
 KVCach
B
ock, so that 

 cou
d co
struct a fr
 qu
u
 d
r
ct
y. Th
s g
v
s us t
o b


f
ts:
    1. W
 cou
d hav
 O(1) comp

x
ty mov

g 


m

ts 

 th
 m
dd

 to th
 ta

.
    2. W
 cou
d avo
d 

troduc

g a
oth
r Pytho
 qu
u
 (
.g., `d
qu
`) 
h
ch has a 
rapp
r to th
 


m

ts.
As a r
su
t, 

 


 hav
 th
 fo
o


g compo


ts 
h

 th
 KV cach
 ma
ag
r 
s 


t
a

z
d:
![Compo


t Ov
rv


](../ass
ts/d
s
g
/pr
f
x_cach

g/ov
rv


.p
g)
* B
ock Poo
: A 

st of KVCach
B
ock.
* Fr
 B
ock Qu
u
: O

y stor
 th
 po

t
rs of h
ad a
d ta

 b
ocks for ma

pu
at
o
s.
* Cach
 b
ocks: Mapp

g from hash k
y to b
ock IDs.
* R
qu
st b
ocks: Mapp

g from r
qu
st ID to a
ocat
d b
ock IDs.
## Op
rat
o
s
### B
ock A
ocat
o

**N

 r
qu
st:** Workf
o
 for th
 sch
du

r to sch
du

 a 


 r
qu
st 

th KV cach
 b
ock a
ocat
o
:
1. Th
 sch
du

r ca
s `kv_cach
_ma
ag
r.g
t_comput
d_b
ocks()` to g
t a s
qu

c
 of b
ocks that hav
 a
r
ady b

 comput
d. Th
s 
s do

 by hash

g th
 prompt tok

s 

 th
 r
qu
st a
d 
ook

g up cach
 b
ocks.
2. Th
 sch
du

r ca
s `kv_cach
_ma
ag
r.a
ocat
_s
ots()`. It do
s th
 fo
o


g st
ps:
    1. Comput
 th
 
umb
r of 


 r
qu
r
d b
ocks, a
d r
tur
 
f th
r
 ar
 
o suff
c


t b
ocks to a
ocat
.
    2. “Touch” th
 comput
d b
ocks. It 

cr
as
s th
 r
f
r

c
 cou
t of th
 comput
d b
ock by o

, a
d r
mov
s th
 b
ock from th
 fr
 qu
u
 
f th
 b
ock 
as
’t us
d by oth
r r
qu
sts. Th
s 
s to avo
d th
s
 comput
d b
ocks b


g 
v
ct
d. S
 th
 
xamp

 

 th
 

xt s
ct
o
 for 

ustrat
o
.
    3. A
ocat
 


 b
ocks by popp

g th
 h
ads of th
 fr
 qu
u
. If th
 h
ad b
ock 
s a cach
d b
ock, th
s a
so “
v
cts” th
 b
ock so that 
o oth
r r
qu
sts ca
 r
us
 
t a
ymor
 from 
o
 o
.
    4. If a
 a
ocat
d b
ock 
s a
r
ady fu
 of tok

s, 

 
mm
d
at

y add 
t to th
 cach
 b
ock, so that th
 b
ock ca
 b
 r
us
d by oth
r r
qu
sts 

 th
 sam
 batch.
**Ru


g r
qu
st:** Workf
o
 for th
 sch
du

r to sch
du

 a ru


g r
qu
st 

th KV cach
 b
ock a
ocat
o
:
1. Th
 sch
du

r ca
s `kv_cach
_ma
ag
r.a
ocat
_s
ots()`. It do
s th
 fo
o


g st
ps:
    1. Comput
 th
 
umb
r of 


 r
qu
r
d b
ocks, a
d r
tur
 
f th
r
 ar
 
o suff
c


t b
ocks to a
ocat
.
    2. A
ocat
 


 b
ocks by popp

g th
 h
ads of th
 fr
 qu
u
. If th
 h
ad b
ock 
s a cach
d b
ock, th
s a
so “
v
cts” th
 b
ock so that 
o oth
r r
qu
sts ca
 r
us
 
t a
ymor
 from 
o
 o
.
    3. App

d tok

 IDs to th
 s
ots 

 
x
st

g b
ocks as 


 as th
 


 b
ocks. If a b
ock 
s fu
, 

 add 
t to th
 cach
 b
ock to cach
 
t.
**Dup

cat
d b
ocks**
Assum

g b
ock s
z
 
s 4 a
d you s

d a r
qu
st (R
qu
st 1\) 

th prompt ABCDEF a
d d
cod

g 


gth 3:
```t
xt
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]
T
m
 0:
  Tok

s: [A, B, C, D, E, F, G]
  B
ock Tab

: [0 (ABCD), 1 (EFG)]
  Cach
 B
ocks: 0
T
m
 1:
  Tok

s: [A, B, C, D, E, F, G, H]
  B
ock Tab

: [0 (ABCD), 1 (EFGH)]
  Cach
 B
ocks: 0, 1
T
m
 2:
  Tok

s: [A, B, C, D, E, F, G, H, I]
  B
ock Tab

: [0 (ABCD), 1 (EFGH), 2 (I)]
  Cach
 B
ocks: 0, 1
```
No
 b
ock 0 a
d b
ock 1 ar
 cach
d, a
d 

 s

d th
 sam
 r
qu
st aga

 (R
qu
st 2\) 

th gr
dy samp


g, so that 
t 


 produc
 
xact
y th
 sam
 outputs as th
 R
qu
st 1:
```t
xt
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]
T
m
 0:
  Tok

s: [A, B, C, D, E, F, G]
  B
ock Tab

: [0 (ABCD), 3 (EFG)]
  Cach
 B
ocks: 0, 1
T
m
 1:
  Tok

s: [A, B, C, D, E, F, G, H]
  B
ock Tab

: [0 (ABCD), 3 (EFGH)]
  Cach
 B
ocks: 0, 1, 3
```
As ca
 b
 s

, b
ock 3 
s a 


 fu
 b
ock a
d 
s cach
d. Ho

v
r, 
t 
s r
du
da
t as b
ock 1, m
a


g that 

 cach
d th
 sam
 b
ock t

c
. I
 v0, 
h

 d
t
ct

g b
ock 3 
s dup

cat
d, 

 fr
 b
ock 3 a
d 

t R
qu
st 2 us
 b
ock 1 

st
ad, so 
ts b
ock tab

 b
com
s `[0, 1]` 

 T
m
 1. Ho

v
r, th
 b
ock tab

 

 vLLM v1 
s app

d-o

y, m
a


g that cha
g

g th
 b
ock tab

 from `[0, 3]` to `[0, 1]` 
s 
ot a
o

d. As a r
su
t, 

 


 hav
 dup

cat
d b
ocks for th
 hash k
y E-H. Th
s dup

cat
o
 


 b
 


m

at
d 
h

 th
 r
qu
st 
s fr
d.
### Fr

Wh

 a r
qu
st 
s f


sh
d, 

 fr
 a
 
ts b
ocks 
f 
o oth
r r
qu
sts ar
 us

g th
m (r
f
r

c
 cou
t = 0). I
 th
s 
xamp

, 

 fr
 r
qu
st 1 a
d b
ock 2, 3, 4, 8 assoc
at
d 

th 
t. W
 ca
 s
 that th
 fr
d b
ocks ar
 add
d to th
 ta

 of th
 fr
 qu
u
 

 th
 *r
v
rs
* ord
r. Th
s 
s b
caus
 th
 
ast b
ock of a r
qu
st must hash mor
 tok

s a
d 
s 

ss 

k

y to b
 r
us
d by oth
r r
qu
sts. As a r
su
t, 
t shou
d b
 
v
ct
d f
rst.
![Fr
 qu
u
 aft
r a r
qu
st us fr
d](../ass
ts/d
s
g
/pr
f
x_cach

g/fr
.p
g)
### Ev
ct
o
 (LRU)
Wh

 th
 h
ad b
ock (

ast r
c

t
y us
d b
ock) of th
 fr
 qu
u
 
s cach
d, 

 hav
 to 
v
ct th
 b
ock to pr
v

t 
t from b


g us
d by oth
r r
qu
sts. Sp
c
f
ca
y, 
v
ct
o
 

vo
v
s th
 fo
o


g st
ps:
1. Pop th
 b
ock from th
 h
ad of th
 fr
 qu
u
. Th
s 
s th
 LRU b
ock to b
 
v
ct
d.
2. R
mov
 th
 b
ock ID from th
 cach
 b
ock.
3. R
mov
 th
 b
ock hash.
## Examp


I
 th
s 
xamp

, 

 assum
 th
 b
ock s
z
 
s 4 (
ach b
ock ca
 cach
 4 tok

s), a
d 

 hav
 10 b
ocks 

 th
 KV-cach
 ma
ag
r 

 tota
.
**T
m
 1: Th
 cach
 
s 
mpty a
d a 


 r
qu
st com
s 

.** W
 a
ocat
 4 b
ocks. 3 of th
m ar
 a
r
ady fu
 a
d cach
d. Th
 fourth b
ock 
s part
a
y fu
 

th 3 of 4 tok

s.
![Examp

 T
m
 1](../ass
ts/d
s
g
/pr
f
x_cach

g/
xamp

-t
m
-1.p
g)
**T
m
 2: R
qu
st 0 mak
s th
 b
ock 3 fu
 a
d asks for a 


 b
ock to k
p d
cod

g.** W
 cach
 b
ock 3 a
d a
ocat
 b
ock 4.
![Examp

 T
m
 2](../ass
ts/d
s
g
/pr
f
x_cach

g/
xamp

-t
m
-3.p
g)
**T
m
 3: R
qu
st 1 com
s 

 

th th
 14 prompt tok

s, 
h
r
 th
 f
rst 10 tok

s ar
 th
 sam
 as r
qu
st 0.** W
 ca
 s
 that o

y th
 f
rst 2 b
ocks (8 tok

s) h
t th
 cach
, b
caus
 th
 3rd b
ock o

y match
s 2 of 4 tok

s.
![Examp

 T
m
 3](../ass
ts/d
s
g
/pr
f
x_cach

g/
xamp

-t
m
-4.p
g)
**T
m
 4: R
qu
st 0 
s f


sh
d a
d fr
.** B
ocks 2, 3 a
d 4 ar
 add
d to th
 fr
 qu
u
 

 th
 r
v
rs
 ord
r (but b
ock 2 a
d 3 ar
 st

 cach
d). B
ock 0 a
d 1 ar
 
ot add
d to th
 fr
 qu
u
 b
caus
 th
y ar
 b


g us
d by R
qu
st 1.
![Examp

 T
m
 4](../ass
ts/d
s
g
/pr
f
x_cach

g/
xamp

-t
m
-5.p
g)
**T
m
 5: R
qu
st 1 
s f


sh
d a
d fr
.**
![Examp

 T
m
 5](../ass
ts/d
s
g
/pr
f
x_cach

g/
xamp

-t
m
-6.p
g)
**T
m
 6: R
qu
st 2 com
s 

 

th th
 29 prompt tok

s, 
h
r
 th
 f
rst 12 tok

s ar
 th
 sam
 as r
qu
st 0\.** Not
 that 
v

 th
 b
ock ord
r 

 th
 fr
 qu
u
 
as `7 - 8 - 9 - 4 - 3 - 2 - 6 - 5 - 1 - 0`, th
 cach
 h
t b
ocks (
.
., 0, 1, 2) ar
 touch
d a
d r
mov
d from th
 qu
u
 b
for
 a
ocat
o
, so th
 fr
 qu
u
 b
com
s `7 - 8 - 9 - 4 - 3 - 6 - 5`. As a r
su
t, th
 a
ocat
d b
ocks ar
 0 (cach
d), 1 (cach
d), 2 (cach
d), 7, 8, 9, 4, 3 (
v
ct
d).
![Examp

 T
m
 6](../ass
ts/d
s
g
/pr
f
x_cach

g/
xamp

-t
m
-7.p
g)
