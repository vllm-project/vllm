# Load

g mod

s 

th Cor
W
av
's T

sor
z
r
vLLM supports 
oad

g mod

s 

th [Cor
W
av
's T

sor
z
r](https://docs.cor


av
.com/cor


av
-mach


-

ar


g-a
d-a
/

f
r

c
/t

sor
z
r).
vLLM mod

 t

sors that hav
 b

 s
r
a

z
d to d
sk, a
 HTTP/HTTPS 

dpo

t, or S3 

dpo

t ca
 b
 d
s
r
a

z
d
at ru
t
m
 
xtr
m

y qu
ck
y d
r
ct
y to th
 GPU, r
su
t

g 

 s
g

f
ca
t
y
short
r Pod startup t
m
s a
d CPU m
mory usag
. T

sor 

crypt
o
 
s a
so support
d.
vLLM fu
y 

t
grat
s T

sor
z
r 

 to 
ts mod

 
oad

g mach


ry. Th
 fo
o


g 


 g
v
 a br

f ov
rv


 o
 ho
 to g
t start
d 

th us

g T

sor
z
r o
 vLLM.
## I
sta


g T

sor
z
r
To 

sta
 `t

sor
z
r`, ru
 `p
p 

sta
 v
m[t

sor
z
r]`.
## Th
 bas
cs
To 
oad a mod

 us

g T

sor
z
r, th
 mod

 f
rst 

ds to b
 s
r
a

z
d by
T

sor
z
r. [Th
 
xamp

 scr
pt](../../
xamp

s/oth
rs/t

sor
z
_v
m_mod

.md) tak
s car
 of th
s proc
ss.
L
t's 
a
k through a bas
c 
xamp

 by s
r
a

z

g `fac
book/opt-125m` us

g th
 scr
pt, a
d th

 
oad

g 
t for 

f
r

c
.
## S
r
a

z

g a vLLM mod

 

th T

sor
z
r
To s
r
a

z
 a mod

 

th T

sor
z
r, ca
 th
 
xamp

 scr
pt 

th th
 

c
ssary
CLI argum

ts. Th
 docstr

g for th
 scr
pt 
ts

f 
xp
a

s th
 CLI args
a
d ho
 to us
 
t prop
r
y 

 gr
at d
ta

, a
d 

'
 us
 o

 of th
 
xamp

s from th
 docstr

g d
r
ct
y, assum

g 

 
a
t to s
r
a

z
 a
d sav
 our mod

 at our S3 buck
t 
xamp

 `s3://my-buck
t`:
```bash
pytho
 
xamp

s/oth
rs/t

sor
z
_v
m_mod

.py \
   --mod

 fac
book/opt-125m \
   s
r
a

z
 \
   --s
r
a

z
d-d
r
ctory s3://my-buck
t \
   --suff
x v1
```
Th
s sav
s th
 mod

 t

sors at `s3://my-buck
t/v
m/fac
book/opt-125m/v1`. If you 

t

d o
 app
y

g a LoRA adapt
r to your t

sor
z
d mod

, you ca
 pass th
 HF 
d of th
 LoRA adapt
r 

 th
 abov
 comma
d, a
d th
 art
facts 


 b
 sav
d th
r
 too:
```bash
pytho
 
xamp

s/oth
rs/t

sor
z
_v
m_mod

.py \
   --mod

 fac
book/opt-125m \
   --
ora-path 

ora_
d
 \
   s
r
a

z
 \
   --s
r
a

z
d-d
r
ctory s3://my-buck
t \
   --suff
x v1
```
## S
rv

g th
 mod

 us

g T

sor
z
r
O
c
 th
 mod

 
s s
r
a

z
d 
h
r
 you 
a
t 
t, you ca
 
oad th
 mod

 us

g `v
m s
rv
` or th
 `LLM` 

trypo

t. You ca
 pass th
 d
r
ctory 
h
r
 you sav
d th
 mod

 to th
 `mod

` argum

t for `LLM()` a
d `v
m s
rv
`. For 
xamp

, to s
rv
 th
 t

sor
z
d mod

 sav
d pr
v
ous
y 

th th
 LoRA adapt
r, you'd do:
```bash
v
m s
rv
 s3://my-buck
t/v
m/fac
book/opt-125m/v1 \
    --
oad-format t

sor
z
r \
    --

ab

-
ora 
```
Or, 

th `LLM()`:
```pytho

from v
m 
mport LLM

m = LLM(
    "s3://my-buck
t/v
m/fac
book/opt-125m/v1", 
    
oad_format="t

sor
z
r",
    

ab

_
ora=Tru
,
)
```
## Opt
o
s for co
f
gur

g T

sor
z
r
`t

sor
z
r`'s cor
 obj
cts that s
r
a

z
 a
d d
s
r
a

z
 mod

s ar
 `T

sorS
r
a

z
r` a
d `T

sorD
s
r
a

z
r` r
sp
ct
v

y. I
 ord
r to pass arb
trary k
args to th
s
, 
h
ch 


 co
f
gur
 th
 s
r
a

zat
o
 a
d d
s
r
a

zat
o
 proc
ss
s, you ca
 prov
d
 th
m as k
ys to `mod

_
oad
r_
xtra_co
f
g` 

th `s
r
a

zat
o
_k
args` a
d `d
s
r
a

zat
o
_k
args` r
sp
ct
v

y. Fu
 docstr

gs d
ta



g a
 param
t
rs for th
 afor
m

t
o

d obj
cts ca
 b
 fou
d 

 `t

sor
z
r`'s [s
r
a

zat
o
.py](https://g
thub.com/cor


av
/t

sor
z
r/b
ob/ma

/t

sor
z
r/s
r
a

zat
o
.py) f


.
As a
 
xamp

, CPU co
curr

cy ca
 b
 

m
t
d 
h

 s
r
a

z

g 

th `t

sor
z
r` v
a th
 `

m
t_cpu_co
curr

cy` param
t
r 

 th
 


t
a

z
r for `T

sorS
r
a

z
r`. To s
t `

m
t_cpu_co
curr

cy` to som
 arb
trary va
u
, you 
ou
d do so 

k
 th
s 
h

 s
r
a

z

g:
```bash
pytho
 
xamp

s/oth
rs/t

sor
z
_v
m_mod

.py \
   --mod

 fac
book/opt-125m \
   --
ora-path 

ora_
d
 \
   s
r
a

z
 \
   --s
r
a

z
d-d
r
ctory s3://my-buck
t \
   --s
r
a

zat
o
-k
args '{"

m
t_cpu_co
curr

cy": 2}' \
   --suff
x v1
```
As a
 
xamp

 
h

 custom
z

g th
 
oad

g proc
ss v
a `T

sorD
s
r
a

z
r`, you cou
d 

m
t th
 
umb
r of co
curr

cy r
ad
rs dur

g d
s
r
a

zat
o
 

th th
 `
um_r
ad
rs` param
t
r 

 th
 


t
a

z
r v
a `mod

_
oad
r_
xtra_co
f
g` 

k
 so:
```bash
v
m s
rv
 s3://my-buck
t/v
m/fac
book/opt-125m/v1 \
    --
oad-format t

sor
z
r \
    --

ab

-
ora \
    --mod

-
oad
r-
xtra-co
f
g '{"d
s
r
a

zat
o
_k
args": {"
um_r
ad
rs": 2}}'
```
Or 

th `LLM()`:
```pytho

from v
m 
mport LLM

m = LLM(
    "s3://my-buck
t/v
m/fac
book/opt-125m/v1", 
    
oad_format="t

sor
z
r",
    

ab

_
ora=Tru
,
    mod

_
oad
r_
xtra_co
f
g={"d
s
r
a

zat
o
_k
args": {"
um_r
ad
rs": 2}},
)
```
