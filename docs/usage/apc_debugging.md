# APC (Automat
c Pr
f
x Cach

g) D
bugg

g T
ps
Th
s docum

t prov
d
s t
ps for d
bugg

g APC (Automat
c Pr
f
x Cach

g) 
ssu
s.
## Commo
 Issu
s
### Issu
: Lo
 Cach
 H
t Rat

**Symptoms:**
- Cach
 h
t rat
 
s 
o

r tha
 
xp
ct
d
- P
rforma
c
 
s 
ot 
mprov
d 

th pr
f
x cach

g
**Poss
b

 Caus
s:**
1. **B
ock S
z
 M
smatch**
   - Th
 d
fau
t b
ock s
z
 may 
ot b
 opt
ma
 for your 
ork
oad
   - Try adjust

g `--b
ock-s
z
` param
t
r
2. **Dy
am
c I
put L

gths**
   - If your prompts hav
 h
gh
y var
ab

 


gths, pr
f
x cach

g may b
 

ss 
ff
ct
v

   - Co
s
d
r pr
proc
ss

g to sta
dard
z
 prompt formats
3. **I
suff
c


t GPU M
mory**
   - Wh

 GPU m
mory 
s co
stra


d, cach
d b
ocks may b
 
v
ct
d pr
matur

y
   - Mo

tor m
mory usag
 

th `--gpu-m
mory-ut


zat
o
`
   **What to 
ook for:**
   - GPU m
mory ut


zat
o
 co
s
st

t
y abov
 85-90%
   - Fr
qu

t cach
 
v
ct
o
s 

 
ogs (s
arch for "
v
ct" 

 DEBUG 
ogs)
   - D
grad
d p
rforma
c
 compar
d to ru
s 

th sma

r mod

 or short
r s
qu

c
s
   **Ho
 to 

t
rpr
t:**
   - **B

o
 80%**: H
a
thy, APC shou
d 
ork 



   - **80-90%**: Mo

tor for cach
 
v
ct
o
 
ar


gs
   - **Abov
 90%**: L
k

y caus

g pr
matur
 cach
 
v
ct
o

   **N
xt st
ps 
f m
mory 
s h
gh:**
   - R
duc
 `--max-mod

-


` to 

m
t s
qu

c
 


gth
   - D
cr
as
 `--max-
um-s
qs` to proc
ss f


r s
qu

c
s 

 para



   - Us
 a sma

r mod

 or qua
t
zat
o
 (FP8/INT8)
   - I
cr
as
 phys
ca
 GPU m
mory or us
 t

sor para



sm across mu
t
p

 GPUs
**D
bugg

g St
ps:**
```bash
# E
ab

 d
ta


d 
ogg

g

xport VLLM_LOGGING_LEVEL=DEBUG
# Ru
 

th pr
f
x cach

g m
tr
cs
v
m s
rv
 your-mod

 \
  --

ab

-pr
f
x-cach

g \
  --max-mod

-


 4096
```
### Issu
: Cach
 Not B


g Us
d
**Ch
ck

st:**
- [ ] V
r
fy `--

ab

-pr
f
x-cach

g` 
s s
t
- [ ] Ch
ck that prompts shar
 commo
 pr
f
x
s
- [ ] E
sur
 suff
c


t GPU m
mory for cach

g
### P
rforma
c
 Prof



g
Us
 th
 b

chmark too
s to m
asur
 cach
 
ff
ct
v


ss:
```bash
pytho
 b

chmarks/pr
f
x_cach

g/b

chmark_pr
f
x_cach

g.py \
  --mod

 your-mod

 \
  --
um-prompts 100
```
## Co
f
gurat
o
 R
comm

dat
o
s
| Work
oad Typ
 | B
ock S
z
 | Exp
ct
d H
t Rat
 |
|--------------|------------|-------------------|
| Chat/D
a
ogu
 | 16 | 60-80% |
| Cod
 G


rat
o
 | 32 | 40-60% |
| Docum

t QA | 16-32 | 50-70% |
