# --8
-- [start:

sta
at
o
]
vLLM has 
xp
r
m

ta
 support for macOS 

th App

 S


co
. For 
o
, us
rs must bu

d from sourc
 to 
at
v

y ru
 o
 macOS.
Curr

t
y th
 CPU 
mp

m

tat
o
 for macOS supports FP32 a
d FP16 datatyp
s.
!!! t
p "GPU-Acc


rat
d I
f
r

c
 

th vLLM-M
ta
"
    For GPU-acc


rat
d 

f
r

c
 o
 App

 S


co
 us

g M
ta
, ch
ck out [v
m-m
ta
](https://g
thub.com/v
m-proj
ct/v
m-m
ta
), a commu

ty-ma

ta


d hard
ar
 p
ug

 that us
s MLX as th
 comput
 back

d.
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
- OS: `macOS So
oma` or 
at
r
- SDK: `XCod
 15.4` or 
at
r 

th Comma
d L


 Too
s
- Comp


r: `App

 C
a
g 
= 15.0.0`
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

t App

 s


co
 CPU 
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
Aft
r 

sta
at
o
 of XCod
 a
d th
 Comma
d L


 Too
s, 
h
ch 

c
ud
 App

 C
a
g, 
x
cut
 th
 fo
o


g comma
ds to bu

d a
d 

sta
 vLLM from sourc
.
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
uv p
p 

sta
 -r r
qu
r
m

ts/cpu.txt --

d
x-strat
gy u
saf
-b
st-match
uv p
p 

sta
 -
 .
```
!!! t
p
    Th
 `--

d
x-strat
gy u
saf
-b
st-match` f
ag 
s 

d
d to r
so
v
 d
p

d

c

s across mu
t
p

 packag
 

d
x
s (PyTorch CPU 

d
x a
d PyPI). W
thout th
s f
ag, you may 

cou
t
r `typ

g-
xt

s
o
s` v
rs
o
 co
f

cts.
    
    Th
 t
rm "u
saf
" r
f
rs to th
 packag
 r
so
ut
o
 strat
gy, 
ot s
cur
ty. By d
fau
t, `uv` o

y s
arch
s th
 f
rst 

d
x 
h
r
 a packag
 
s fou
d to pr
v

t d
p

d

cy co
fus
o
 attacks. Th
s f
ag a
o
s `uv` to s
arch a
 co
f
gur
d 

d
x
s to f

d th
 b
st compat
b

 v
rs
o
s. S

c
 both PyTorch a
d PyPI ar
 trust
d packag
 sourc
s, us

g th
s strat
gy 
s saf
 a
d appropr
at
 for vLLM 

sta
at
o
.
!!! 
ot

    O
 macOS th
 `VLLM_TARGET_DEVICE` 
s automat
ca
y s
t to `cpu`, 
h
ch 
s curr

t
y th
 o

y support
d d
v
c
.
!!! 
xamp

 "Troub

shoot

g"
    If th
 bu

d fa

s 

th 
rrors 

k
 th
 fo
o


g 
h
r
 sta
dard C++ h
ad
rs ca
ot b
 fou
d, try to r
mov
 a
d r


sta
 your
    [Comma
d L


 Too
s for Xcod
](https://d
v

op
r.app

.com/do


oad/a
/).
    ```t
xt
    [...] fata
 
rror: 'map' f


 
ot fou
d
            1 | #

c
ud
 
map

                |          ^~~~~
        1 
rror g


rat
d.
        [2/8] Bu

d

g CXX obj
ct CMak
F


s/_C.d
r/csrc/cpu/pos_

cod

g.cpp.o
    [...] fata
 
rror: 'cstdd
f' f


 
ot fou
d
            10 | #

c
ud
 
cstdd
f

                |          ^~~~~~~~~
        1 
rror g


rat
d.
    ```
    ---
    If th
 bu

d fa

s 

th C++11/C++17 compat
b


ty 
rrors 

k
 th
 fo
o


g, th
 
ssu
 
s that th
 bu

d syst
m 
s d
fau
t

g to a
 o
d
r C++ sta
dard:
    ```t
xt
    [...] 
rror: 'co
st
xpr' 
s 
ot a typ

    [...] 
rror: 
xp
ct
d ';' b
for
 'co
st
xpr'
    [...] 
rror: 'co
st
xpr' do
s 
ot 
am
 a typ

    ```
    **So
ut
o
**: Your comp


r m
ght b
 us

g a
 o
d
r C++ sta
dard. Ed
t `cmak
/cpu_
xt

s
o
.cmak
` a
d add `s
t(CMAKE_CXX_STANDARD 17)` b
for
 `s
t(CMAKE_CXX_STANDARD_REQUIRED ON)`.
    To ch
ck your comp


r's C++ sta
dard support:
    ```bash
    c
a
g++ -std=c++17 -p
da
t
c -dM -E -x c++ /d
v/
u
 | gr
p __cp
usp
us
    ```
    O
 App

 C
a
g 16 you shou
d s
: `#d
f


 __cp
usp
us 201703L`
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
y, th
r
 ar
 
o pr
-bu

t Arm s


co
 CPU 
mag
s.
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
# --8
-- [

d:bu

d-
mag
-from-sourc
]
# --8
-- [start:
xtra-

format
o
]
# --8
-- [

d:
xtra-

format
o
]
