# I
t
grat
o
 

th Hugg

g Fac

Th
s docum

t d
scr
b
s ho
 vLLM 

t
grat
s 

th Hugg

g Fac
 

brar

s. W
 


 
xp
a

 st
p by st
p 
hat happ

s u
d
r th
 hood 
h

 

 ru
 `v
m s
rv
`.
L
t's say 

 
a
t to s
rv
 th
 popu
ar Q


 mod

 by ru


g `v
m s
rv
 Q


/Q


2-7B`.
1. Th
 `mod

` argum

t 
s `Q


/Q


2-7B`. vLLM d
t
rm


s 
h
th
r th
s mod

 
x
sts by ch
ck

g for th
 corr
spo
d

g co
f
g f


 `co
f
g.jso
`. S
 th
s [cod
 s

pp
t](https://g
thub.com/v
m-proj
ct/v
m/b
ob/10b67d865d92
376956345b
cafc249d4c3c0ab7/v
m/tra
sform
rs_ut

s/co
f
g.py#L162-L182) for th
 
mp

m

tat
o
. W
th

 th
s proc
ss:
    - If th
 `mod

` argum

t corr
spo
ds to a
 
x
st

g 
oca
 path, vLLM 


 
oad th
 co
f
g f


 d
r
ct
y from th
s path.
    - If th
 `mod

` argum

t 
s a Hugg

g Fac
 mod

 ID co
s
st

g of a us
r
am
 a
d mod

 
am
, vLLM 


 f
rst try to us
 th
 co
f
g f


 from th
 Hugg

g Fac
 
oca
 cach
, us

g th
 `mod

` argum

t as th
 mod

 
am
 a
d th
 `--r
v
s
o
` argum

t as th
 r
v
s
o
. S
 [th

r 

bs
t
](https://hugg

gfac
.co/docs/hugg

gfac
_hub/

/packag
_r
f
r

c
/

v
ro
m

t_var
ab

s#hfhom
) for mor
 

format
o
 o
 ho
 th
 Hugg

g Fac
 cach
 
orks.
    - If th
 `mod

` argum

t 
s a Hugg

g Fac
 mod

 ID but 
t 
s 
ot fou
d 

 th
 cach
, vLLM 


 do


oad th
 co
f
g f


 from th
 Hugg

g Fac
 mod

 hub. R
f
r to [th
s fu
ct
o
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/10b67d865d92
376956345b
cafc249d4c3c0ab7/v
m/tra
sform
rs_ut

s/co
f
g.py#L91) for th
 
mp

m

tat
o
. Th
 

put argum

ts 

c
ud
 th
 `mod

` argum

t as th
 mod

 
am
, th
 `--r
v
s
o
` argum

t as th
 r
v
s
o
, a
d th
 

v
ro
m

t var
ab

 `HF_TOKEN` as th
 tok

 to acc
ss th
 mod

 hub. I
 our cas
, vLLM 


 do


oad th
 [co
f
g.jso
](https://hugg

gfac
.co/Q


/Q


2-7B/b
ob/ma

/co
f
g.jso
) f


.
2. Aft
r co
f
rm

g th
 
x
st

c
 of th
 mod

, vLLM 
oads 
ts co
f
g f


 a
d co
v
rts 
t 

to a d
ct
o
ary. S
 th
s [cod
 s

pp
t](https://g
thub.com/v
m-proj
ct/v
m/b
ob/10b67d865d92
376956345b
cafc249d4c3c0ab7/v
m/tra
sform
rs_ut

s/co
f
g.py#L185-L186) for th
 
mp

m

tat
o
.
3. N
xt, vLLM [

sp
cts](https://g
thub.com/v
m-proj
ct/v
m/b
ob/10b67d865d92
376956345b
cafc249d4c3c0ab7/v
m/tra
sform
rs_ut

s/co
f
g.py#L189) th
 `mod

_typ
` f


d 

 th
 co
f
g d
ct
o
ary to [g


rat
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/10b67d865d92
376956345b
cafc249d4c3c0ab7/v
m/tra
sform
rs_ut

s/co
f
g.py#L190-L216) th
 co
f
g obj
ct to us
. Th
r
 ar
 som
 `mod

_typ
` va
u
s that vLLM d
r
ct
y supports; s
 [h
r
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/10b67d865d92
376956345b
cafc249d4c3c0ab7/v
m/tra
sform
rs_ut

s/co
f
g.py#L48) for th
 

st. If th
 `mod

_typ
` 
s 
ot 

 th
 

st, vLLM 


 us
 [AutoCo
f
g.from_pr
tra


d](https://hugg

gfac
.co/docs/tra
sform
rs/

/mod

_doc/auto#tra
sform
rs.AutoCo
f
g.from_pr
tra


d) to 
oad th
 co
f
g c
ass, 

th `mod

`, `--r
v
s
o
`, a
d `--trust_r
mot
_cod
` as th
 argum

ts. P

as
 
ot
 that:
    - Hugg

g Fac
 a
so has 
ts o

 
og
c to d
t
rm


 th
 co
f
g c
ass to us
. It 


 aga

 us
 th
 `mod

_typ
` f


d to s
arch for th
 c
ass 
am
 

 th
 tra
sform
rs 

brary; s
 [h
r
](https://g
thub.com/hugg

gfac
/tra
sform
rs/tr
/ma

/src/tra
sform
rs/mod

s) for th
 

st of support
d mod

s. If th
 `mod

_typ
` 
s 
ot fou
d, Hugg

g Fac
 


 us
 th
 `auto_map` f


d from th
 co
f
g JSON f


 to d
t
rm


 th
 c
ass 
am
. Sp
c
f
ca
y, 
t 
s th
 `AutoCo
f
g` f


d u
d
r `auto_map`. S
 [D
pS
k](https://hugg

gfac
.co/d
ps
k-a
/D
pS
k-V2.5/b
ob/ma

/co
f
g.jso
) for a
 
xamp

.
    - Th
 `AutoCo
f
g` f


d u
d
r `auto_map` po

ts to a modu

 path 

 th
 mod

's r
pos
tory. To cr
at
 th
 co
f
g c
ass, Hugg

g Fac
 


 
mport th
 modu

 a
d us
 th
 `from_pr
tra


d` m
thod to 
oad th
 co
f
g c
ass. Th
s ca
 g


ra
y caus
 arb
trary cod
 
x
cut
o
, so 
t 
s o

y 
x
cut
d 
h

 `--trust_r
mot
_cod
` 
s 

ab

d.
4. Subs
qu

t
y, vLLM app


s som
 h
stor
ca
 patch
s to th
 co
f
g obj
ct. Th
s
 ar
 most
y r

at
d to RoPE co
f
gurat
o
; s
 [h
r
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/127c07480
c
a15
4c2990820c457807ff78a057/v
m/tra
sform
rs_ut

s/co
f
g.py#L244) for th
 
mp

m

tat
o
.
5. F

a
y, vLLM ca
 r
ach th
 mod

 c
ass 

 
a
t to 


t
a

z
. vLLM us
s th
 `arch
t
ctur
s` f


d 

 th
 co
f
g obj
ct to d
t
rm


 th
 mod

 c
ass to 


t
a

z
, as 
t ma

ta

s th
 mapp

g from arch
t
ctur
 
am
 to mod

 c
ass 

 [
ts r
g
stry](https://g
thub.com/v
m-proj
ct/v
m/b
ob/127c07480
c
a15
4c2990820c457807ff78a057/v
m/mod

_
x
cutor/mod

s/r
g
stry.py#L80). If th
 arch
t
ctur
 
am
 
s 
ot fou
d 

 th
 r
g
stry, 
t m
a
s th
s mod

 arch
t
ctur
 
s 
ot support
d by vLLM. For `Q


/Q


2-7B`, th
 `arch
t
ctur
s` f


d 
s `["Q


2ForCausa
LM"]`, 
h
ch corr
spo
ds to th
 `Q


2ForCausa
LM` c
ass 

 [vLLM's cod
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/127c07480
c
a15
4c2990820c457807ff78a057/v
m/mod

_
x
cutor/mod

s/q


2.py#L364). Th
s c
ass 


 


t
a

z
 
ts

f d
p

d

g o
 var
ous co
f
gs.
B
yo
d that, th
r
 ar
 t
o mor
 th

gs vLLM d
p

ds o
 Hugg

g Fac
 for.
1. **Tok


z
r**: vLLM us
s th
 tok


z
r from Hugg

g Fac
 to tok


z
 th
 

put t
xt. Th
 tok


z
r 
s 
oad
d us

g [AutoTok


z
r.from_pr
tra


d](https://hugg

gfac
.co/docs/tra
sform
rs/

/mod

_doc/auto#tra
sform
rs.AutoTok


z
r.from_pr
tra


d) 

th th
 `mod

` argum

t as th
 mod

 
am
 a
d th
 `--r
v
s
o
` argum

t as th
 r
v
s
o
. It 
s a
so poss
b

 to us
 a tok


z
r from a
oth
r mod

 by sp
c
fy

g th
 `--tok


z
r` argum

t 

 th
 `v
m s
rv
` comma
d. Oth
r r


va
t argum

ts ar
 `--tok


z
r-r
v
s
o
` a
d `--tok


z
r-mod
`. P

as
 ch
ck Hugg

g Fac
's docum

tat
o
 for th
 m
a


g of th
s
 argum

ts. Th
s part of th
 
og
c ca
 b
 fou
d 

 th
 [g
t_tok


z
r](https://g
thub.com/v
m-proj
ct/v
m/b
ob/127c07480
c
a15
4c2990820c457807ff78a057/v
m/tra
sform
rs_ut

s/tok


z
r.py#L87) fu
ct
o
. Aft
r obta



g th
 tok


z
r, 
otab
y, vLLM 


 cach
 som
 
xp

s
v
 attr
but
s of th
 tok


z
r 

 [v
m.tok


z
rs.hf.g
t_cach
d_tok


z
r][].
2. **Mod

 


ght**: vLLM do


oads th
 mod

 


ght from th
 Hugg

g Fac
 mod

 hub us

g th
 `mod

` argum

t as th
 mod

 
am
 a
d th
 `--r
v
s
o
` argum

t as th
 r
v
s
o
. vLLM prov
d
s th
 argum

t `--
oad-format` to co
tro
 
hat f


s to do


oad from th
 mod

 hub. By d
fau
t, 
t 


 try to 
oad th
 


ghts 

 th
 saf
t

sors format a
d fa
 back to th
 PyTorch b

 format 
f th
 saf
t

sors format 
s 
ot ava

ab

. W
 ca
 a
so pass `--
oad-format dummy` to sk
p do


oad

g th
 


ghts.
    - It 
s r
comm

d
d to us
 th
 saf
t

sors format, as 
t 
s 
ff
c


t for 
oad

g 

 d
str
but
d 

f
r

c
 a
d a
so saf
 from arb
trary cod
 
x
cut
o
. S
 th
 [docum

tat
o
](https://hugg

gfac
.co/docs/saf
t

sors/

/

d
x) for mor
 

format
o
 o
 th
 saf
t

sors format. Th
s part of th
 
og
c ca
 b
 fou
d [h
r
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/10b67d865d92
376956345b
cafc249d4c3c0ab7/v
m/mod

_
x
cutor/mod

_
oad
r/
oad
r.py#L385). P

as
 
ot
 that:
Th
s comp

t
s th
 

t
grat
o
 b
t


 vLLM a
d Hugg

g Fac
.
I
 summary, vLLM r
ads th
 co
f
g f


 `co
f
g.jso
`, tok


z
r, a
d mod

 


ght from th
 Hugg

g Fac
 mod

 hub or a 
oca
 d
r
ctory. It us
s th
 co
f
g c
ass from 

th
r vLLM, Hugg

g Fac
 tra
sform
rs, or 
oads th
 co
f
g c
ass from th
 mod

's r
pos
tory.
