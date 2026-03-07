# Product
o
 M
tr
cs
vLLM 
xpos
s a 
umb
r of m
tr
cs that ca
 b
 us
d to mo

tor th
 h
a
th of th

syst
m. Th
s
 m
tr
cs ar
 
xpos
d v
a th
 `/m
tr
cs` 

dpo

t o
 th
 vLLM
Op

AI compat
b

 API s
rv
r.
You ca
 start th
 s
rv
r us

g Pytho
, or us

g [Dock
r](../d
p
oym

t/dock
r.md):
```bash
v
m s
rv
 u
s
oth/L
ama-3.2-1B-I
struct
```
Th

 qu
ry th
 

dpo

t to g
t th
 
at
st m
tr
cs from th
 s
rv
r:
??? co
so

 "Output"
    ```co
so


    $ cur
 http://0.0.0.0:8000/m
tr
cs
    # HELP v
m:
t
rat
o
_tok

s_tota
 H
stogram of 
umb
r of tok

s p
r 

g


_st
p.
    # TYPE v
m:
t
rat
o
_tok

s_tota
 h
stogram
    v
m:
t
rat
o
_tok

s_tota
_sum{mod

_
am
="u
s
oth/L
ama-3.2-1B-I
struct"} 0.0
    v
m:
t
rat
o
_tok

s_tota
_buck
t{

="1.0",mod

_
am
="u
s
oth/L
ama-3.2-1B-I
struct"} 3.0
    v
m:
t
rat
o
_tok

s_tota
_buck
t{

="8.0",mod

_
am
="u
s
oth/L
ama-3.2-1B-I
struct"} 3.0
    v
m:
t
rat
o
_tok

s_tota
_buck
t{

="16.0",mod

_
am
="u
s
oth/L
ama-3.2-1B-I
struct"} 3.0
    v
m:
t
rat
o
_tok

s_tota
_buck
t{

="32.0",mod

_
am
="u
s
oth/L
ama-3.2-1B-I
struct"} 3.0
    v
m:
t
rat
o
_tok

s_tota
_buck
t{

="64.0",mod

_
am
="u
s
oth/L
ama-3.2-1B-I
struct"} 3.0
    v
m:
t
rat
o
_tok

s_tota
_buck
t{

="128.0",mod

_
am
="u
s
oth/L
ama-3.2-1B-I
struct"} 3.0
    v
m:
t
rat
o
_tok

s_tota
_buck
t{

="256.0",mod

_
am
="u
s
oth/L
ama-3.2-1B-I
struct"} 3.0
    v
m:
t
rat
o
_tok

s_tota
_buck
t{

="512.0",mod

_
am
="u
s
oth/L
ama-3.2-1B-I
struct"} 3.0
    ...
```
Th
 fo
o


g m
tr
cs ar
 
xpos
d:
## G


ra
 M
tr
cs
--8
-- "docs/g


rat
d/m
tr
cs/g


ra
.

c.md"
## Sp
cu
at
v
 D
cod

g M
tr
cs
--8
-- "docs/g


rat
d/m
tr
cs/sp
c_d
cod
.

c.md"
## NIXL KV Co

ctor M
tr
cs
--8
-- "docs/g


rat
d/m
tr
cs/

x
_co

ctor.

c.md"
## Mod

 F
ops Ut


zat
o
 (MFU) P
rforma
c
 M
tr
cs
Th
s
 m
tr
cs ar
 ava

ab

 v
a `--

ab

-mfu-m
tr
cs`:
--8
-- "docs/g


rat
d/m
tr
cs/p
rf.

c.md"
## D
pr
cat
o
 Po

cy
Not
: 
h

 m
tr
cs ar
 d
pr
cat
d 

 v
rs
o
 `X.Y`, th
y ar
 h
dd

 

 v
rs
o
 `X.Y+1`
but ca
 b
 r
-

ab

d us

g th
 `--sho
-h
dd

-m
tr
cs-for-v
rs
o
=X.Y` 
scap
 hatch,
a
d ar
 th

 r
mov
d 

 v
rs
o
 `X.Y+2`.
