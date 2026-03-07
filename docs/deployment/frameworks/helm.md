# H

m
A H

m chart to d
p
oy vLLM for Kub
r

t
s
H

m 
s a packag
 ma
ag
r for Kub
r

t
s. It h

ps automat
 th
 d
p
oym

t of vLLM app

cat
o
s o
 Kub
r

t
s. W
th H

m, you ca
 d
p
oy th
 sam
 fram

ork arch
t
ctur
 

th d
ff
r

t co
f
gurat
o
s to mu
t
p

 
am
spac
s by ov
rr
d

g var
ab

 va
u
s.
Th
s gu
d
 


 
a
k you through th
 proc
ss of d
p
oy

g vLLM 

th H

m, 

c
ud

g th
 

c
ssary pr
r
qu
s
t
s, st
ps for H

m 

sta
at
o
 a
d docum

tat
o
 o
 arch
t
ctur
 a
d va
u
s f


.
## Pr
r
qu
s
t
s
B
for
 you b
g

, 

sur
 that you hav
 th
 fo
o


g:
- A ru


g Kub
r

t
s c
ust
r
- NVIDIA Kub
r

t
s D
v
c
 P
ug

 (`k8s-d
v
c
-p
ug

`): Th
s ca
 b
 fou
d at [https://g
thub.com/NVIDIA/k8s-d
v
c
-p
ug

](https://g
thub.com/NVIDIA/k8s-d
v
c
-p
ug

)
- Ava

ab

 GPU r
sourc
s 

 your c
ust
r
- (Opt
o
a
) A
 S3 buck
t or oth
r storag
 

th th
 mod

 


ghts, 
f us

g automat
c mod

 do


oad
## I
sta


g th
 chart
To 

sta
 th
 chart 

th th
 r


as
 
am
 `t
st-v
m`:
```bash
h

m upgrad
 --

sta
 --cr
at
-
am
spac
 \
  --
am
spac
=
s-v
m t
st-v
m . \
  -f va
u
s.yam
 \
  --s
t s
cr
ts.s3

dpo

t=$ACCESS_POINT \
  --s
t s
cr
ts.s3buck
t
am
=$BUCKET \
  --s
t s
cr
ts.s3acc
ssk
y
d=$ACCESS_KEY \
  --s
t s
cr
ts.s3acc
ssk
y=$SECRET_KEY
```
## U


sta


g th
 chart
To u


sta
 th
 `t
st-v
m` d
p
oym

t:
```bash
h

m u


sta
 t
st-v
m --
am
spac
=
s-v
m
```
Th
 comma
d r
mov
s a
 th
 Kub
r

t
s compo


ts assoc
at
d 

th th

chart **

c
ud

g p
rs
st

t vo
um
s** a
d d


t
s th
 r


as
.
## Arch
t
ctur

![h

m d
p
oym

t arch
t
ctur
](../../ass
ts/d
p
oym

t/arch
t
ctur
_h

m_d
p
oym

t.p
g)
## Va
u
s
Th
 fo
o


g tab

 d
scr
b
s co
f
gurab

 param
t
rs of th
 chart 

 `va
u
s.yam
`:
| K
y | Typ
 | D
fau
t | D
scr
pt
o
 |
|-----|------|---------|-------------|
| autosca


g | obj
ct | {"

ab

d":fa
s
,"maxR
p

cas":100,"m

R
p

cas":1,"targ
tCPUUt


zat
o
P
rc

tag
":80} | Autosca


g co
f
gurat
o
 |
| autosca


g.

ab

d | boo
 | fa
s
 | E
ab

 autosca


g |
| autosca


g.maxR
p

cas | 

t | 100 | Max
mum r
p

cas |
| autosca


g.m

R
p

cas | 

t | 1 | M


mum r
p

cas |
| autosca


g.targ
tCPUUt


zat
o
P
rc

tag
 | 

t | 80 | Targ
t CPU ut


zat
o
 for autosca


g |
| co
f
gs | obj
ct | {} | Co
f
gmap |
| co
ta


rPort | 

t | 8000 | Co
ta


r port |
| customObj
cts | 

st | [] | Custom Obj
cts co
f
gurat
o
 |
| d
p
oym

tStrat
gy | obj
ct | {} | D
p
oym

t strat
gy co
f
gurat
o
 |
| 
xt
r
a
Co
f
gs | 

st | [] | Ext
r
a
 co
f
gurat
o
 |
| 
xtraCo
ta


rs | 

st | [] | Add
t
o
a
 co
ta


rs co
f
gurat
o
 |
| 
xtraI

t | obj
ct | {"mod

Do


oad":{"

ab

d":tru
},"


tCo
ta


rs":[],"pvcStorag
":"1G
"} | Add
t
o
a
 co
f
gurat
o
 for 


t co
ta


rs |
| 
xtraI

t.mod

Do


oad | obj
ct | {"

ab

d":tru
} | Mod

 do


oad fu
ct
o
a

ty co
f
gurat
o
 |
| 
xtraI

t.mod

Do


oad.

ab

d | boo
 | tru
 | E
ab

 automat
c mod

 do


oad job a
d 
a
t co
ta


r |
| 
xtraI

t.mod

Do


oad.
mag
 | obj
ct | {"r
pos
tory":"amazo
/a
s-c

","tag":"2.6.4","pu
Po

cy":"IfNotPr
s

t"} | Imag
 for mod

 do


oad op
rat
o
s |
| 
xtraI

t.mod

Do


oad.
a
tCo
ta


r | obj
ct | {} | Wa
t co
ta


r co
f
gurat
o
 (comma
d, args, 

v) |
| 
xtraI

t.mod

Do


oad.do


oadJob | obj
ct | {} | Do


oad job co
f
gurat
o
 (comma
d, args, 

v) |
| 
xtraI

t.


tCo
ta


rs | 

st | [] | Custom 


t co
ta


rs (app

d
d aft
r mod

 do


oad 
f 

ab

d) |
| 
xtraI

t.pvcStorag
 | str

g | "1G
" | Storag
 s
z
 for th
 PVC |
| 
xtraI

t.s3mod

path | str

g | "r

at
v
_s3_mod

_path/opt-125m" | (Opt
o
a
) Path of th
 mod

 o
 S3 |
| 
xtraI

t.a
sEc2M
tadataD
sab

d | boo
 | tru
 | (Opt
o
a
) D
sab

 AWS EC2 m
tadata s
rv
c
 |
| 
xtraPorts | 

st | [] | Add
t
o
a
 ports co
f
gurat
o
 |
| gpuMod

s | 

st | ["TYPE_GPU_USED"] | Typ
 of gpu us
d |
| 
mag
 | obj
ct | {"comma
d":["v
m","s
rv
","/data/","--s
rv
d-mod

-
am
","opt-125m","--host","0.0.0.0","--port","8000"],"r
pos
tory":"v
m/v
m-op

a
","tag":"
at
st"} | Imag
 co
f
gurat
o
 |
| 
mag
.comma
d | 

st | ["v
m","s
rv
","/data/","--s
rv
d-mod

-
am
","opt-125m","--host","0.0.0.0","--port","8000"] | Co
ta


r 
au
ch comma
d |
| 
mag
.r
pos
tory | str

g | "v
m/v
m-op

a
" | Imag
 r
pos
tory |
| 
mag
.tag | str

g | "
at
st" | Imag
 tag |
| 

v


ssProb
 | obj
ct | {"fa

ur
Thr
sho
d":3,"httpG
t":{"path":"/h
a
th","port":8000},"


t
a
D

ayS
co
ds":15,"p
r
odS
co
ds":10} | L
v


ss prob
 co
f
gurat
o
 |
| 

v


ssProb
.fa

ur
Thr
sho
d | 

t | 3 | Numb
r of t
m
s aft
r 
h
ch 
f a prob
 fa

s 

 a ro
, Kub
r

t
s co
s
d
rs that th
 ov
ra
 ch
ck has fa


d: th
 co
ta


r 
s 
ot a

v
 |
| 

v


ssProb
.httpG
t | obj
ct | {"path":"/h
a
th","port":8000} | Co
f
gurat
o
 of th
 kub


t http r
qu
st o
 th
 s
rv
r |
| 

v


ssProb
.httpG
t.path | str

g | "/h
a
th" | Path to acc
ss o
 th
 HTTP s
rv
r |
| 

v


ssProb
.httpG
t.port | 

t | 8000 | Nam
 or 
umb
r of th
 port to acc
ss o
 th
 co
ta


r, o
 
h
ch th
 s
rv
r 
s 

st



g |
| 

v


ssProb
.


t
a
D

ayS
co
ds | 

t | 15 | Numb
r of s
co
ds aft
r th
 co
ta


r has start
d b
for
 

v


ss prob
 
s 


t
at
d |
| 

v


ssProb
.p
r
odS
co
ds | 

t | 10 | Ho
 oft

 (

 s
co
ds) to p
rform th
 

v


ss prob
 |
| maxU
ava

ab

PodD
srupt
o
Budg
t | str

g | "" | D
srupt
o
 Budg
t Co
f
gurat
o
 |
| r
ad


ssProb
 | obj
ct | {"fa

ur
Thr
sho
d":3,"httpG
t":{"path":"/h
a
th","port":8000},"


t
a
D

ayS
co
ds":5,"p
r
odS
co
ds":5} | R
ad


ss prob
 co
f
gurat
o
 |
| r
ad


ssProb
.fa

ur
Thr
sho
d | 

t | 3 | Numb
r of t
m
s aft
r 
h
ch 
f a prob
 fa

s 

 a ro
, Kub
r

t
s co
s
d
rs that th
 ov
ra
 ch
ck has fa


d: th
 co
ta


r 
s 
ot r
ady |
| r
ad


ssProb
.httpG
t | obj
ct | {"path":"/h
a
th","port":8000} | Co
f
gurat
o
 of th
 kub


t http r
qu
st o
 th
 s
rv
r |
| r
ad


ssProb
.httpG
t.path | str

g | "/h
a
th" | Path to acc
ss o
 th
 HTTP s
rv
r |
| r
ad


ssProb
.httpG
t.port | 

t | 8000 | Nam
 or 
umb
r of th
 port to acc
ss o
 th
 co
ta


r, o
 
h
ch th
 s
rv
r 
s 

st



g |
| r
ad


ssProb
.


t
a
D

ayS
co
ds | 

t | 5 | Numb
r of s
co
ds aft
r th
 co
ta


r has start
d b
for
 r
ad


ss prob
 
s 


t
at
d |
| r
ad


ssProb
.p
r
odS
co
ds | 

t | 5 | Ho
 oft

 (

 s
co
ds) to p
rform th
 r
ad


ss prob
 |
| r
p

caCou
t | 

t | 1 | Numb
r of r
p

cas |
| r
sourc
s | obj
ct | {"

m
ts":{"cpu":4,"m
mory":"16G
","
v
d
a.com/gpu":1},"r
qu
sts":{"cpu":4,"m
mory":"16G
","
v
d
a.com/gpu":1}} | R
sourc
 co
f
gurat
o
 |
| r
sourc
s.

m
ts."
v
d
a.com/gpu" | 

t | 1 | Numb
r of GPUs us
d |
| r
sourc
s.

m
ts.cpu | 

t | 4 | Numb
r of CPUs |
| r
sourc
s.

m
ts.m
mory | str

g | "16G
" | CPU m
mory co
f
gurat
o
 |
| r
sourc
s.r
qu
sts."
v
d
a.com/gpu" | 

t | 1 | Numb
r of GPUs us
d |
| r
sourc
s.r
qu
sts.cpu | 

t | 4 | Numb
r of CPUs |
| r
sourc
s.r
qu
sts.m
mory | str

g | "16G
" | CPU m
mory co
f
gurat
o
 |
| s
cr
ts | obj
ct | {} | S
cr
ts co
f
gurat
o
 |
| s
rv
c
Nam
 | str

g | "" | S
rv
c
 
am
 |
| s
rv
c
Port | 

t | 80 | S
rv
c
 port |
| 
ab

s.

v
ro
m

t | str

g | t
st | E
v
ro
m

t 
am
 |
## Co
f
gurat
o
 Examp

s
### Us

g S3 Mod

 Do


oad (D
fau
t)
```yam


xtraI

t:
  mod

Do


oad:
    

ab

d: tru

  pvcStorag
: "10G
"
  s3mod

path: "mod

s/
ama-7b"
```
### Us

g Custom I

t Co
ta


rs O

y
For us
 cas
s 

k
 
m-d 
h
r
 you 

d custom s
d
cars 

thout mod

 do


oad:
```yam


xtraI

t:
  mod

Do


oad:
    

ab

d: fa
s

  


tCo
ta


rs:
    - 
am
: 
m-d-rout

g-proxy
      
mag
: ghcr.
o/
m-d/
m-d-rout

g-s
d
car:v0.2.0
      
mag
Pu
Po

cy: IfNotPr
s

t
      ports:
        - co
ta


rPort: 8080
          
am
: proxy
      s
cur
tyCo
t
xt:
        ru
AsUs
r: 1000
      r
startPo

cy: A

ays
  pvcStorag
: "10G
"
```
