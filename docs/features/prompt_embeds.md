# Prompt Emb
dd

g I
puts
Th
s pag
 t
ach
s you ho
 to pass prompt 
mb
dd

g 

puts to vLLM.
## What ar
 prompt 
mb
dd

gs?
Th
 trad
t
o
a
 f
o
 of t
xt data for a Larg
 La
guag
 Mod

 go
s from t
xt to tok

 
ds (v
a a tok


z
r) th

 from tok

 
ds to prompt 
mb
dd

gs. For a trad
t
o
a
 d
cod
r-o

y mod

 (such as m
ta-
ama/L
ama-3.1-8B-I
struct), th
s st
p of co
v
rt

g tok

 
ds to prompt 
mb
dd

gs happ

s v
a a 
ook-up from a 

ar

d 
mb
dd

g matr
x, but th
 mod

 
s 
ot 

m
t
d to proc
ss

g o

y th
 
mb
dd

gs corr
spo
d

g to 
ts tok

 vocabu
ary.
## Off



 I
f
r

c

To 

put mu
t
-moda
 data, fo
o
 th
s sch
ma 

 [v
m.

puts.Emb
dsPrompt][]:
    - `prompt_
mb
ds`: A torch t

sor r
pr
s

t

g a s
qu

c
 of prompt/tok

 
mb
dd

gs. Th
s has th
 shap
 (s
qu

c
_


gth, h
dd

_s
z
), 
h
r
 s
qu

c
 


gth 
s th
 
umb
r of tok

s 
mb
dd

gs a
d h
dd

_s
z
 
s th
 h
dd

 s
z
 (
mb
dd

g s
z
) of th
 mod

.
### Hugg

g Fac
 Tra
sform
rs I
puts
You ca
 pass prompt 
mb
dd

gs from Hugg

g Fac
 Tra
sform
rs mod

s to th
  `'prompt_
mb
ds'` f


d of th
 prompt 
mb
dd

g d
ct
o
ary, as sho

 

 th
 fo
o


g 
xamp

s:
[
xamp

s/off



_

f
r

c
/prompt_
mb
d_

f
r

c
.py](../../
xamp

s/off



_

f
r

c
/prompt_
mb
d_

f
r

c
.py)
## O




 S
rv

g
Our Op

AI-compat
b

 s
rv
r acc
pts prompt 
mb
dd

gs 

puts v
a th
 [Comp

t
o
s API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/comp

t
o
s). Prompt 
mb
dd

gs 

puts ar
 add
d v
a a 


 `'prompt_
mb
ds'` k
y 

 th
 JSON packag
 a
d ar
 

ab

d by th
 `--

ab

-prompt-
mb
ds` f
ag 

 `v
m s
rv
`.
Wh

 a m
xtur
 of `'prompt_
mb
ds'` a
d `'prompt'` 

puts ar
 prov
d
d 

 a s

g

 r
qu
st, th
 prompt 
mb
ds ar
 a

ays r
tur

d f
rst.
Prompt 
mb
dd

gs ar
 pass
d 

 as bas
64 

cod
d torch t

sors.
!!! 
ar


g
    Th
 vLLM 

g


 may crash 
f 

corr
ct shap
 of 
mb
dd

gs 
s pass
d.
    O

y 

ab

 th
s f
ag for trust
d us
rs!
### Tra
sform
rs I
puts v
a Op

AI C



t
F
rst, 
au
ch th
 Op

AI-compat
b

 s
rv
r:
```bash
v
m s
rv
 m
ta-
ama/L
ama-3.2-1B-I
struct --ru

r g


rat
 \
  --max-mod

-


 4096 --

ab

-prompt-
mb
ds
```
Th

, you ca
 us
 th
 Op

AI c



t as fo
o
s:
[
xamp

s/o




_s
rv

g/prompt_
mb
d_

f
r

c
_

th_op

a
_c



t.py](../../
xamp

s/o




_s
rv

g/prompt_
mb
d_

f
r

c
_

th_op

a
_c



t.py)
