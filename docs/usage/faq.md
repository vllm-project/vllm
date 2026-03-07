# Fr
qu

t
y Ask
d Qu
st
o
s

 Q: Ho
 ca
 I s
rv
 mu
t
p

 mod

s o
 a s

g

 port us

g th
 Op

AI API?
A: Assum

g that you'r
 r
f
rr

g to us

g Op

AI compat
b

 s
rv
r to s
rv
 mu
t
p

 mod

s at o
c
, that 
s 
ot curr

t
y support
d, you ca
 ru
 mu
t
p

 

sta
c
s of th
 s
rv
r (
ach s
rv

g a d
ff
r

t mod

) at th
 sam
 t
m
, a
d hav
 a
oth
r 
ay
r to rout
 th
 

com

g r
qu
st to th
 corr
ct s
rv
r accord

g
y.
---

 Q: Wh
ch mod

 to us
 for off



 

f
r

c
 
mb
dd

g?
A: You ca
 try [
5-m
stra
-7b-

struct](https://hugg

gfac
.co/

tf
oat/
5-m
stra
-7b-

struct) a
d [BAAI/bg
-bas
-

-v1.5](https://hugg

gfac
.co/BAAI/bg
-bas
-

-v1.5);
mor
 ar
 

st
d [h
r
](../mod

s/support
d_mod

s.md).
By 
xtract

g h
dd

 stat
s, vLLM ca
 automat
ca
y co
v
rt t
xt g


rat
o
 mod

s 

k
 [L
ama-3-8B](https://hugg

gfac
.co/m
ta-
ama/M
ta-L
ama-3-8B),
[M
stra
-7B-I
struct-v0.3](https://hugg

gfac
.co/m
stra
a
/M
stra
-7B-I
struct-v0.3) 

to 
mb
dd

g mod

s,
but th
y ar
 
xp
ct
d to b
 

f
r
or to mod

s that ar
 sp
c
f
ca
y tra


d o
 
mb
dd

g tasks.
---

 Q: Ca
 th
 output of a prompt vary across ru
s 

 vLLM?
A: Y
s, 
t ca
. vLLM do
s 
ot guara
t
 stab

 
og probab


t

s (
ogprobs) for th
 output tok

s. Var
at
o
s 

 
ogprobs may occur du
 to

um
r
ca
 

stab


ty 

 Torch op
rat
o
s or 
o
-d
t
rm


st
c b
hav
or 

 batch
d Torch op
rat
o
s 
h

 batch

g cha
g
s. For mor
 d
ta

s,
s
 th
 [Num
r
ca
 Accuracy s
ct
o
](https://pytorch.org/docs/stab

/
ot
s/
um
r
ca
_accuracy.htm
#batch
d-computat
o
s-or-s

c
-computat
o
s).
I
 vLLM, th
 sam
 r
qu
sts m
ght b
 batch
d d
ff
r

t
y du
 to factors such as oth
r co
curr

t r
qu
sts,
cha
g
s 

 batch s
z
, or batch 
xpa
s
o
 

 sp
cu
at
v
 d
cod

g. Th
s
 batch

g var
at
o
s, comb


d 

th 
um
r
ca
 

stab


ty of Torch op
rat
o
s,
ca
 

ad to s

ght
y d
ff
r

t 
og
t/
ogprob va
u
s at 
ach st
p. Such d
ff
r

c
s ca
 accumu
at
, pot

t
a
y r
su
t

g 


d
ff
r

t tok

s b


g samp

d. O
c
 a d
ff
r

t tok

 
s samp

d, furth
r d
v
rg

c
 
s 

k

y.
## M
t
gat
o
 Strat
g

s
- For 
mprov
d stab


ty a
d r
duc
d var
a
c
, us
 `f
oat32`. Not
 that th
s 


 r
qu
r
 mor
 m
mory.
- If us

g `bf
oat16`, s

tch

g to `f
oat16` ca
 a
so h

p.
- Us

g r
qu
st s
ds ca
 a
d 

 ach

v

g mor
 stab

 g


rat
o
 for t
mp
ratur
 
 0, but d
scr
pa
c

s du
 to pr
c
s
o
 d
ff
r

c
s may st

 occur.
