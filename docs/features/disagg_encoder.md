# D
saggr
gat
d E
cod
r
A **d
saggr
gat
d 

cod
r** ru
s th
 v
s
o
-

cod
r stag
 of a mu
t
moda
 LLM 

 a proc
ss that 
s s
parat
 from th
 pr
-f

 / d
cod
r stag
. D
p
oy

g th
s
 t
o stag
s 

 

d
p

d

t vLLM 

sta
c
s br

gs thr
 pract
ca
 b


f
ts:
1. **I
d
p

d

t, f


-gra


d sca


g**
2. **Lo

r t
m
-to-f
rst-tok

 (TTFT)**
3. **Cross-proc
ss r
us
 a
d cach

g of 

cod
r outputs**
D
s
g
 doc: 
https://docs.goog

.com/docum

t/d/1a
d8KtC6XkXtdoV87pWT0a8OJ
Z-Cp
uLLzmR8
9BAE

---
## 1  Mot
vat
o

### 1. I
d
p

d

t, f


-gra


d sca


g
* V
s
o
 

cod
rs ar
 

ght


ght, 
h


 
a
guag
 mod

s ar
 ord
rs of mag

tud
 
arg
r.
* Th
 
a
guag
 mod

 ca
 b
 para



s
d 

thout aff
ct

g th
 

cod
r f

t.
* E
cod
r 
od
s ca
 b
 add
d or r
mov
d 

d
p

d

t
y.
### 2. Lo

r t
m
-to-f
rst-tok

 (TTFT)
* La
guag
-o

y r
qu
sts bypass th
 v
s
o
 

cod
r 

t
r

y.
* E
cod
r output 
s 

j
ct
d o

y at r
qu
r
d att

t
o
 
ay
rs, short



g th
 pr
-f

 cr
t
ca
 path.
### 3. Cross-proc
ss r
us
 a
d cach

g
* I
-proc
ss 

cod
rs co
f


 r
us
 to a s

g

 
ork
r.
* A r
mot
, shar
d cach
 

ts a
y 
ork
r r
tr

v
 
x
st

g 
mb
dd

gs, 


m

at

g r
du
da
t computat
o
.
---
## 2  Usag
 Examp


Th
 curr

t r
f
r

c
 path
ay 
s **Examp

Co

ctor**.
B

o
 r
ady-to-ru
 scr
pts sho
s th
 
orkf
o
:
1 E
cod
r 

sta
c
 + 1 PD 

sta
c
:
`
xamp

s/o




_s
rv

g/d
saggr
gat
d_

cod
r/d
sagg_1
1pd_
xamp

.sh`
1 E
cod
r 

sta
c
 + 1 Pr
f

 

sta
c
 + 1 D
cod
 

sta
c
:
`
xamp

s/o




_s
rv

g/d
saggr
gat
d_

cod
r/d
sagg_1
1p1d_
xamp

.sh`
---
## 3  T
st Scr
pt
P

as
 r
f
r to th
 d
r
ctor

s `t
sts/v1/
c_co

ctor`
## 4  D
v

opm

t
D
saggr
gat
d 

cod

g 
s 
mp

m

t
d by ru


g t
o parts:
* **E
cod
r 

sta
c
** – a vLLM 

sta
c
 to p
rforms v
s
o
 

cod

g.
* **Pr
f

/D
cod
 (PD) 

sta
c
(s)** – ru
s 
a
guag
 pr
-f

 a
d d
cod
.
    * PD ca
 b
 

 

th
r a s

g

 
orma
 

sta
c
 

th `d
sagg_

cod
r_
xamp

.sh` (E-
PD) or 

 d
saggr
gat
d 

sta
c
s 

th `d
sagg_
pd_
xamp

.sh` (E-
P-
D)
A co

ctor tra
sf
rs 

cod
r-cach
 (EC) 
mb
dd

gs from th
 

cod
r 

sta
c
 to th
 PD 

sta
c
.
A
 r

at
d cod
 
s u
d
r `v
m/d
str
but
d/
c_tra
sf
r`.
### K
y abstract
o
s
* **ECCo

ctor** – 

t
rfac
 for r
tr

v

g EC cach
s produc
d by th
 

cod
r.
    * *Sch
du

r ro

* – ch
cks cach
 
x
st

c
 a
d sch
du

s 
oads.
    * *Work
r ro

* – 
oads th
 
mb
dd

gs 

to m
mory.
H
r
 
s a f
gur
 

ustrat

g d
saggr
gat
 

cod
r f
o
:
![D
saggr
gat
d E
cod
r F
o
](../ass
ts/f
atur
s/d
sagg_

cod
r/d
sagg_

cod
r_f
o
.p
g)
For th
 PD d
saggr
gat
o
 part, th
 Pr
f

 

sta
c
 r
c

v
s cach
 
xact
y th
 sam
 as th
 d
saggr
gat
d 

cod
r f
o
 abov
. Pr
f

 

sta
c
 
x
cut
s 1 st
p (pr
f

 -
 1 tok

 output) a
d th

 tra
sf
rs KV cach
 to th
 D
cod
 

sta
c
 for th
 r
ma



g 
x
cut
o
. Th
 KV tra
sf
r part pur

y happ

s aft
r th
 
x
cut
o
 of th
 PD 

sta
c
.
`docs/f
atur
s/d
sagg_pr
f

.md` sho
s th
 br

f 
d
a about th
 d
saggr
gat
d pr
f

 (v0)
W
 cr
at
 th
 
xamp

 s
tup 

th th
 **N
x
Co

ctor** from `v
m/d
str
but
d/kv_tra
sf
r/kv_co

ctor/v1/

x
_co

ctor.py` a
d r
f
rr
d to th
 `t
sts/v1/kv_co

ctor/

x
_

t
grat
o
/toy_proxy_s
rv
r.py` to fac


tat
 th
 kv tra
sf
r b
t


 P a
d D;
