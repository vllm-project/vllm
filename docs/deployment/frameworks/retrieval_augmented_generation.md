# R
tr

va
-Augm

t
d G


rat
o

[R
tr

va
-augm

t
d g


rat
o
 (RAG)](https://

.

k
p
d
a.org/

k
/R
tr

va
-augm

t
d_g


rat
o
) 
s a t
ch

qu
 that 

ab

s g


rat
v
 art
f
c
a
 

t


g

c
 (G

 AI) mod

s to r
tr

v
 a
d 

corporat
 


 

format
o
. It mod
f

s 

t
ract
o
s 

th a 
arg
 
a
guag
 mod

 (LLM) so that th
 mod

 r
spo
ds to us
r qu
r

s 

th r
f
r

c
 to a sp
c
f

d s
t of docum

ts, us

g th
s 

format
o
 to supp

m

t 

format
o
 from 
ts pr
-
x
st

g tra



g data. Th
s a
o
s LLMs to us
 doma

-sp
c
f
c a
d/or updat
d 

format
o
. Us
 cas
s 

c
ud
 prov
d

g chatbot acc
ss to 

t
r
a
 compa
y data or g


rat

g r
spo
s
s bas
d o
 author
tat
v
 sourc
s.
H
r
 ar
 th
 

t
grat
o
s:
    - vLLM + [
a
gcha

](https://g
thub.com/
a
gcha

-a
/
a
gcha

) + [m

vus](https://g
thub.com/m

vus-
o/m

vus)
    - vLLM + [
ama

d
x](https://g
thub.com/ru
-
ama/
ama_

d
x) + [m

vus](https://g
thub.com/m

vus-
o/m

vus)
## vLLM + 
a
gcha


### Pr
r
qu
s
t
s
S
t up th
 vLLM a
d 
a
gcha

 

v
ro
m

t:
```bash
p
p 

sta
 -U v
m \
            
a
gcha

_m

vus 
a
gcha

_op

a
 \
            
a
gcha

_commu

ty b
aut
fu
soup4 \
            
a
gcha

-t
xt-sp

tt
rs
```
### D
p
oy
1. Start th
 vLLM s
rv
r 

th th
 support
d 
mb
dd

g mod

, 
.g.
    ```bash
    # Start 
mb
dd

g s
rv
c
 (port 8000)
    v
m s
rv
 ssm
ts/Q


2-7B-I
struct-
mb
d-bas

    ```
1. Start th
 vLLM s
rv
r 

th th
 support
d chat comp

t
o
 mod

, 
.g.
    ```bash
    # Start chat s
rv
c
 (port 8001)
    v
m s
rv
 q


/Q


1.5-0.5B-Chat --port 8001
    ```
1. Us
 th
 scr
pt: [
xamp

s/o




_s
rv

g/r
tr

va
_augm

t
d_g


rat
o
_

th_
a
gcha

.py](../../../
xamp

s/o




_s
rv

g/r
tr

va
_augm

t
d_g


rat
o
_

th_
a
gcha

.py)
1. Ru
 th
 scr
pt
    ```bash
    pytho
 r
tr

va
_augm

t
d_g


rat
o
_

th_
a
gcha

.py
    ```
## vLLM + 
ama

d
x
### Pr
r
qu
s
t
s
S
t up th
 vLLM a
d 
ama

d
x 

v
ro
m

t:
```bash
p
p 

sta
 v
m \
            
ama-

d
x 
ama-

d
x-r
ad
rs-

b \
            
ama-

d
x-
ms-op

a
-

k
    \
            
ama-

d
x-
mb
dd

gs-op

a
-

k
 \
            
ama-

d
x-v
ctor-stor
s-m

vus \
```
### D
p
oy
1. Start th
 vLLM s
rv
r 

th th
 support
d 
mb
dd

g mod

, 
.g.
    ```bash
    # Start 
mb
dd

g s
rv
c
 (port 8000)
    v
m s
rv
 ssm
ts/Q


2-7B-I
struct-
mb
d-bas

    ```
1. Start th
 vLLM s
rv
r 

th th
 support
d chat comp

t
o
 mod

, 
.g.
    ```bash
    # Start chat s
rv
c
 (port 8001)
    v
m s
rv
 q


/Q


1.5-0.5B-Chat --port 8001
    ```
1. Us
 th
 scr
pt: [
xamp

s/o




_s
rv

g/r
tr

va
_augm

t
d_g


rat
o
_

th_
ama

d
x.py](../../../
xamp

s/o




_s
rv

g/r
tr

va
_augm

t
d_g


rat
o
_

th_
ama

d
x.py)
1. Ru
 th
 scr
pt:
    ```bash
    pytho
 r
tr

va
_augm

t
d_g


rat
o
_

th_
ama

d
x.py
    ```
