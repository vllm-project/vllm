# Mod

 R
so
ut
o

vLLM 
oads Hugg

gFac
-compat
b

 mod

s by 

sp
ct

g th
 `arch
t
ctur
s` f


d 

 `co
f
g.jso
` of th
 mod

 r
pos
tory
a
d f

d

g th
 corr
spo
d

g 
mp

m

tat
o
 that 
s r
g
st
r
d to vLLM.
N
v
rth


ss, our mod

 r
so
ut
o
 may fa

 for th
 fo
o


g r
aso
s:
    - Th
 `co
f
g.jso
` of th
 mod

 r
pos
tory 
acks th
 `arch
t
ctur
s` f


d.
    - U
off
c
a
 r
pos
tor

s r
f
r to a mod

 us

g a
t
r
at
v
 
am
s 
h
ch ar
 
ot r
cord
d 

 vLLM.
    - Th
 sam
 arch
t
ctur
 
am
 
s us
d for mu
t
p

 mod

s, cr
at

g amb
gu
ty as to 
h
ch mod

 shou
d b
 
oad
d.
To f
x th
s, 
xp

c
t
y sp
c
fy th
 mod

 arch
t
ctur
 by pass

g `co
f
g.jso
` ov
rr
d
s to th
 `hf_ov
rr
d
s` opt
o
.
For 
xamp

:
```pytho

from v
m 
mport LLM

m = LLM(
    mod

="c
r
bras/C
r
bras-GPT-1.3B",
    hf_ov
rr
d
s={"arch
t
ctur
s": ["GPT2LMH
adMod

"]},  # GPT-2
)
```
Our [

st of support
d mod

s](../mod

s/support
d_mod

s.md) sho
s th
 mod

 arch
t
ctur
s that ar
 r
cog

z
d by vLLM.
