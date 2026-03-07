# --8
-- [start:

sta
at
o
]
vLLM 


t
a
y supports bas
c mod

 

f
r

c
 a
d s
rv

g o
 I
t

 GPU p
atform.
# --8
-- [

d:

sta
at
o
]
# --8
-- [start:r
qu
r
m

ts]
- Support
d Hard
ar
: I
t

 Data C

t
r GPU, I
t

 ARC GPU
- O

API r
qu
r
m

ts: o

API 2025.3
- D
p

d

cy: [v
m-xpu-k
r


s](https://g
thub.com/v
m-proj
ct/v
m-xpu-k
r


s): a packag
 prov
d
 a
 

c
ssary v
m custom k
r


 
h

 ru


g vLLM o
 I
t

 GPU p
atform, 
- Pytho
: 3.12
!!! 
ar


g
    Th
 prov
d
d v
m-xpu-k
r


s 
h
 
s Pytho
3.12 sp
c
f
c so th
s v
rs
o
 
s a MUST.
# --8
-- [

d:r
qu
r
m

ts]
# --8
-- [start:s
t-up-us

g-pytho
]
Th
r
 
s 
o 
xtra 

format
o
 o
 cr
at

g a 


 Pytho
 

v
ro
m

t for th
s d
v
c
.
# --8
-- [

d:s
t-up-us

g-pytho
]
# --8
-- [start:pr
-bu

t-
h

s]
Curr

t
y, th
r
 ar
 
o pr
-bu

t XPU 
h

s.
# --8
-- [

d:pr
-bu

t-
h

s]
# --8
-- [start:bu

d-
h

-from-sourc
]
- F
rst, 

sta
 r
qu
r
d [dr
v
r](https://dgpu-docs.

t

.com/dr
v
r/

sta
at
o
.htm
#

sta


g-gpu-dr
v
rs) a
d [I
t

 O

API](https://
.

t

.com/co
t

t/
/us/

/d
v

op
r/too
s/o

ap
/bas
-too
k
t.htm
) 2025.3 or 
at
r.
- S
co
d, 

sta
 Pytho
 packag
s for vLLM XPU back

d bu

d

g:
```bash
g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t
cd v
m
p
p 

sta
 --upgrad
 p
p
p
p 

sta
 -v -r r
qu
r
m

ts/xpu.txt
```
- Th

, bu

d a
d 

sta
 vLLM XPU back

d:
```bash
VLLM_TARGET_DEVICE=xpu p
p 

sta
 --
o-bu

d-
so
at
o
 -
 . -v
```
# --8
-- [

d:bu

d-
h

-from-sourc
]
# --8
-- [start:pr
-bu

t-
mag
s]
Curr

t
y, 

 r


as
 pr
bu

t XPU 
mag
s at dock
r [hub](https://hub.dock
r.com/r/

t

/v
m/tags) bas
d o
 vLLM r


as
d v
rs
o
. For mor
 

format
o
, p

as
 r
f
r r


as
 [
ot
](https://g
thub.com/

t

/a
-co
ta


rs/b
ob/ma

/v
m).
# --8
-- [

d:pr
-bu

t-
mag
s]
# --8
-- [start:bu

d-
mag
-from-sourc
]
```bash
dock
r bu

d -f dock
r/Dock
rf


.xpu -t v
m-xpu-

v --shm-s
z
=4g .
dock
r ru
 -
t \
             --rm \
             --

t
ork=host \
             --d
v
c
 /d
v/dr
:/d
v/dr
 \
             -v /d
v/dr
/by-path:/d
v/dr
/by-path \
             --
pc=host \
             --pr
v


g
d \
             v
m-xpu-

v
```
# --8
-- [

d:bu

d-
mag
-from-sourc
]
# --8
-- [start:support
d-f
atur
s]
XPU p
atform supports **t

sor para


** 

f
r

c
/s
rv

g a
d a
so supports **p
p




 para


** as a b
ta f
atur
 for o




 s
rv

g. For **p
p




 para


**, 

 support 
t o
 s

g

 
od
 

th mp as th
 back

d. For 
xamp

, a r
f
r

c
 
x
cut
o
 

k
 fo
o


g:
```bash
v
m s
rv
 fac
book/opt-13b \
     --dtyp
=bf
oat16 \
     --max_mod

_


=1024 \
     --d
str
but
d-
x
cutor-back

d=mp \
     --p
p




-para


-s
z
=2 \
     -tp=8
```
By d
fau
t, a ray 

sta
c
 


 b
 
au
ch
d automat
ca
y 
f 
o 
x
st

g o

 
s d
t
ct
d 

 th
 syst
m, 

th `
um-gpus` 
qua
s to `para


_co
f
g.
or
d_s
z
`. W
 r
comm

d prop
r
y start

g a ray c
ust
r b
for
 
x
cut
o
, r
f
rr

g to th
 [
xamp

s/o




_s
rv

g/ru
_c
ust
r.sh](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/
xamp

s/o




_s
rv

g/ru
_c
ust
r.sh) h

p
r scr
pt.
# --8
-- [

d:support
d-f
atur
s]
# --8
-- [start:d
str
but
d-back

d]
XPU p
atform us
s **torch-cc
** for torch
2.8 a
d **xcc
** for torch
=2.8 as d
str
but
d back

d, s

c
 torch 2.8 supports **xcc
** as bu

t-

 back

d for XPU.
# --8
-- [

d:d
str
but
d-back

d]
