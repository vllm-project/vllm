# B
tsA
dByt
s
vLLM 
o
 supports [B
tsA
dByt
s](https://g
thub.com/T
mD
ttm
rs/b
tsa
dbyt
s) for mor
 
ff
c


t mod

 

f
r

c
.
B
tsA
dByt
s qua
t
z
s mod

s to r
duc
 m
mory usag
 a
d 

ha
c
 p
rforma
c
 

thout s
g

f
ca
t
y sacr
f
c

g accuracy.
Compar
d to oth
r qua
t
zat
o
 m
thods, B
tsA
dByt
s 


m

at
s th
 

d for ca

brat

g th
 qua
t
z
d mod

 

th 

put data.
B

o
 ar
 th
 st
ps to ut


z
 B
tsA
dByt
s 

th vLLM.
```bash
p
p 

sta
 b
tsa
dbyt
s
=0.49.2
```
vLLM r
ads th
 mod

's co
f
g f


 a
d supports both 

-f

ght qua
t
zat
o
 a
d pr
-qua
t
z
d ch
ckpo

t.
You ca
 f

d b
tsa
dbyt
s qua
t
z
d mod

s o
 [Hugg

g Fac
](https://hugg

gfac
.co/mod

s?s
arch=b
tsa
dbyt
s).
A
d usua
y, th
s
 r
pos
tor

s hav
 a co
f
g.jso
 f


 that 

c
ud
s a qua
t
zat
o
_co
f
g s
ct
o
.
## R
ad qua
t
z
d ch
ckpo

t
For pr
-qua
t
z
d ch
ckpo

ts, vLLM 


 try to 

f
r th
 qua
t
zat
o
 m
thod from th
 co
f
g f


, so you do
't 

d to 
xp

c
t
y sp
c
fy th
 qua
t
zat
o
 argum

t.
```pytho

from v
m 
mport LLM

mport torch
# u
s
oth/t

y
ama-b
b-4b
t 
s a pr
-qua
t
z
d ch
ckpo

t.
mod

_
d = "u
s
oth/t

y
ama-b
b-4b
t"

m = LLM(
    mod

=mod

_
d,
    dtyp
=torch.bf
oat16,
    trust_r
mot
_cod
=Tru
,
)
```
## I
f

ght qua
t
zat
o
: 
oad as 4b
t qua
t
zat
o

For 

f

ght 4b
t qua
t
zat
o
 

th B
tsA
dByt
s, you 

d to 
xp

c
t
y sp
c
fy th
 qua
t
zat
o
 argum

t.
```pytho

from v
m 
mport LLM

mport torch
mod

_
d = "huggy
ama/
ama-7b"

m = LLM(
    mod

=mod

_
d,
    dtyp
=torch.bf
oat16,
    trust_r
mot
_cod
=Tru
,
    qua
t
zat
o
="b
tsa
dbyt
s",
)
```
## Op

AI Compat
b

 S
rv
r
App

d th
 fo
o


g to your mod

 argum

ts for 4b
t 

f

ght qua
t
zat
o
:
```bash
--qua
t
zat
o
 b
tsa
dbyt
s
```
