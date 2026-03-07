# vLLM-Proj
ct/Sp
cu
ators
![Us
r F
o
 L
ght](../../ass
ts/f
atur
s/sp
cu
at
v
_d
cod

g/sp
cu
ators-us
r-f
o
-

ght.svg#o

y-

ght)
![Us
r F
o
 Dark](../../ass
ts/f
atur
s/sp
cu
at
v
_d
cod

g/sp
cu
ators-us
r-f
o
-dark.svg#o

y-dark)
[Sp
cu
ators](https://docs.v
m.a
/proj
cts/sp
cu
ators/

/
at
st/) 
s a 

brary for acc


rat

g LLM 

f
r

c
 through sp
cu
at
v
 d
cod

g, prov
d

g 
ff
c


t draft mod

 tra



g that 

t
grat
s s
am

ss
y 

th vLLM to r
duc
 
at

cy a
d 
mprov
 throughput.
Sp
cu
ators prov
d
s th
 fo
o


g k
y f
atur
s:
    - **Off



 tra



g data g


rat
o
 us

g vLLM**: E
ab

 th
 g


rat
o
 of h
dd

 stat
s us

g vLLM. Data samp

s ar
 sav
d to d
sk a
d ca
 b
 us
d for draft mod

 tra



g.
    - **Draft mod

 tra



g support**: E2E tra



g support of s

g

 a
d mu
t
-
ay
r draft mod

s. Tra



g 
s support
d for both 
o
-MoE a
d MoE mod

s.
    - **Sta
dard
z
d, 
xt

s
b

 format**: Prov
d
s a Hugg

g Fac
-compat
b

 format for d
f



g sp
cu
at
v
 mod

s, 

th too
s to co
v
rt from 
xt
r
a
 r
s
arch r
pos
tor

s 

to a sta
dard sp
cu
ators format for 
asy adopt
o
.
    - **S
am

ss vLLM I
t
grat
o
**: Bu

t for d
r
ct d
p
oym

t 

to vLLM, 

ab


g 
o
-
at

cy, product
o
-grad
 

f
r

c
 

th m


ma
 ov
rh
ad.
## Why us
 Sp
cu
ators?
Larg
 
a
guag
 mod

s g


rat
 t
xt o

 tok

 at a t
m
, 
h
ch cr
at
s a fu
dam

ta
 bott



ck: 
ach tok

 r
qu
r
s a fu
 for
ard pass through th
 mod

, 

av

g GPU comput
 u
d
rut


z
d 
h


 
a
t

g for m
mory-bou
d op
rat
o
s.
Sp
cu
at
v
 d
cod

g addr
ss
s th
s by us

g a sma

r, fast
r "draft" mod

 (oft

 t
m
s, just a s

g

 tra
sform
r 
ay
r) to pr
d
ct mu
t
p

 tok

s ah
ad, a
d th

 v
r
fy

g tok

s 

 para


 

th th
 pr
mary mod

.
Sp
cu
at
v
 d
cod

g prov
d
s th
 fo
o


g b


f
ts:
    - **R
duc
d 
at

cy**: G


rat
s tok

s 2-3 t
m
s fast
r for 

t
ract
v
 app

cat
o
s such as chatbots a
d cod
 ass
sta
ts, 
h
r
 r
spo
s
 t
m
 d
r
ct
y 
mpacts us
r 
xp
r


c

    - **B
tt
r GPU ut


zat
o
**: Co
v
rts 
at

cy a
d m
mory-bou
d d
cod

g 

 th
 
arg
 mod

 

to comput
-bou
d para


 tok

 v
r
f
cat
o
, 
mprov

g hard
ar
 ut


zat
o
.
    - **No qua

ty 
oss**: Sp
cu
at
v
 d
cod

g do
s 
ot approx
mat
 th
 targ
t mod

. Acc
pt
d tok

s ar
 
xact
y thos
 th
 targ
t mod

 
ou
d hav
 produc
d u
d
r th
 sam
 samp


g co
f
gurat
o
; r
j
ct
d draft tok

s ar
 d
scard
d a
d r
g


rat
d by th
 targ
t mod

.
    - **Cost 
ff
c


cy**: S
rv
 mor
 r
qu
sts p
r GPU by r
duc

g th
 t
m
 
ach r
qu
st occup

s th
 hard
ar

Sp
cu
ators 
s part
cu
ar
y va
uab

 for 
at

cy-s

s
t
v
 app

cat
o
s 
h
r
 us
rs ar
 
a
t

g for r
spo
s
s 

 r
a
-t
m
, such as co
v
rsat
o
a
 AI, 

t
ract
v
 cod

g ass
sta
ts, a
d str
am

g t
xt g


rat
o
.
## R
sourc
s
    - [Sp
cu
ators 
xamp

s](https://g
thub.com/v
m-proj
ct/sp
cu
ators/tr
/ma

/
xamp

s)
    - [G
tHub R
pos
tory](https://g
thub.com/v
m-proj
ct/sp
cu
ators)
