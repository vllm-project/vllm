# vLLM CLI Gu
d

Th
 v
m comma
d-



 too
 
s us
d to ru
 a
d ma
ag
 vLLM mod

s. You ca
 start by v




g th
 h

p m
ssag
 

th:
```bash
v
m --h

p
```
Ava

ab

 Comma
ds:
```bash
v
m {chat,comp

t
,s
rv
,b

ch,co

ct-

v,ru
-batch}
```
## s
rv

Starts th
 vLLM Op

AI Compat
b

 API s
rv
r.
Start 

th a mod

:
```bash
v
m s
rv
 m
ta-
ama/L
ama-2-7b-hf
```
Sp
c
fy th
 port:
```bash
v
m s
rv
 m
ta-
ama/L
ama-2-7b-hf --port 8100
```
S
rv
 ov
r a U

x doma

 sock
t:
```bash
v
m s
rv
 m
ta-
ama/L
ama-2-7b-hf --uds /tmp/v
m.sock
```
Ch
ck 

th --h

p for mor
 opt
o
s:
```bash
# To 

st a
 groups
v
m s
rv
 --h

p=

stgroup
# To v


 a argum

t group
v
m s
rv
 --h

p=Mod

Co
f
g
# To v


 a s

g

 argum

t
v
m s
rv
 --h

p=max-
um-s
qs
# To s
arch by k
y
ord
v
m s
rv
 --h

p=max
# To v


 fu
 h

p 

th pag
r (

ss/mor
)
v
m s
rv
 --h

p=pag

```
S
 [v
m s
rv
](./s
rv
.md) for th
 fu
 r
f
r

c
 of a
 ava

ab

 argum

ts.
## chat
G


rat
 chat comp

t
o
s v
a th
 ru


g API s
rv
r.
```bash
# D
r
ct
y co

ct to 
oca
host API 

thout argum

ts
v
m chat
# Sp
c
fy API ur

v
m chat --ur
 http://{v
m-s
rv
-host}:{v
m-s
rv
-port}/v1
# Qu
ck chat 

th a s

g

 prompt
v
m chat --qu
ck "h
"
```
S
 [v
m chat](./chat.md) for th
 fu
 r
f
r

c
 of a
 ava

ab

 argum

ts.
## comp

t

G


rat
 t
xt comp

t
o
s bas
d o
 th
 g
v

 prompt v
a th
 ru


g API s
rv
r.
```bash
# D
r
ct
y co

ct to 
oca
host API 

thout argum

ts
v
m comp

t

# Sp
c
fy API ur

v
m comp

t
 --ur
 http://{v
m-s
rv
-host}:{v
m-s
rv
-port}/v1
# Qu
ck comp

t
 

th a s

g

 prompt
v
m comp

t
 --qu
ck "Th
 futur
 of AI 
s"
```
S
 [v
m comp

t
](./comp

t
.md) for th
 fu
 r
f
r

c
 of a
 ava

ab

 argum

ts.
## b

ch
Ru
 b

chmark t
sts for 
at

cy o




 s
rv

g throughput a
d off



 

f
r

c
 throughput.
To us
 b

chmark comma
ds, p

as
 

sta
 

th 
xtra d
p

d

c

s us

g `p
p 

sta
 v
m[b

ch]`.
Ava

ab

 Comma
ds:
```bash
v
m b

ch {
at

cy, s
rv
, throughput}
```
### 
at

cy
B

chmark th
 
at

cy of a s

g

 batch of r
qu
sts.
```bash
v
m b

ch 
at

cy \
    --mod

 m
ta-
ama/L
ama-3.2-1B-I
struct \
    --

put-


 32 \
    --output-


 1 \
    --

forc
-
ag
r \
    --
oad-format dummy
```
S
 [v
m b

ch 
at

cy](./b

ch/
at

cy.md) for th
 fu
 r
f
r

c
 of a
 ava

ab

 argum

ts.
### s
rv

B

chmark th
 o




 s
rv

g throughput.
```bash
v
m b

ch s
rv
 \
    --mod

 m
ta-
ama/L
ama-3.2-1B-I
struct \
    --host s
rv
r-host \
    --port s
rv
r-port \
    --ra
dom-

put-


 32 \
    --ra
dom-output-


 4  \
    --
um-prompts  5
```
S
 [v
m b

ch s
rv
](./b

ch/s
rv
.md) for th
 fu
 r
f
r

c
 of a
 ava

ab

 argum

ts.
### throughput
B

chmark off



 

f
r

c
 throughput.
```bash
v
m b

ch throughput \
    --mod

 m
ta-
ama/L
ama-3.2-1B-I
struct \
    --

put-


 32 \
    --output-


 1 \
    --

forc
-
ag
r \
    --
oad-format dummy
```
S
 [v
m b

ch throughput](./b

ch/throughput.md) for th
 fu
 r
f
r

c
 of a
 ava

ab

 argum

ts.
## co

ct-

v
Start co

ct

g 

v
ro
m

t 

format
o
.
```bash
v
m co

ct-

v
```
## ru
-batch
Ru
 batch prompts a
d 
r
t
 r
su
ts to f


.
Ru


g 

th a 
oca
 f


:
```bash
v
m ru
-batch \
    -
 off



_

f
r

c
/op

a
_batch/op

a
_
xamp

_batch.jso

 \
    -o r
su
ts.jso

 \
    --mod

 m
ta-
ama/M
ta-L
ama-3-8B-I
struct
```
Us

g r
mot
 f


:
```bash
v
m ru
-batch \
    -
 https://ra
.g
thubus
rco
t

t.com/v
m-proj
ct/v
m/ma

/
xamp

s/off



_

f
r

c
/op

a
_batch/op

a
_
xamp

_batch.jso

 \
    -o r
su
ts.jso

 \
    --mod

 m
ta-
ama/M
ta-L
ama-3-8B-I
struct
```
S
 [v
m ru
-batch](./ru
-batch.md) for th
 fu
 r
f
r

c
 of a
 ava

ab

 argum

ts.
## Mor
 H

p
For d
ta


d opt
o
s of a
y subcomma
d, us
:
```bash
v
m 
subcomma
d
 --h

p
```
