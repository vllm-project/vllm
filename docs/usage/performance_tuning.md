# P
rforma
c
 Tu


g Gu
d

Th
s gu
d
 prov
d
s r
comm

dat
o
s for tu


g vLLM p
rforma
c
 for d
ff
r

t 
ork
oads a
d hard
ar
 co
f
gurat
o
s.
## G


ra
 R
comm

dat
o
s
### GPU M
mory Ut


zat
o

Th
 `--gpu-m
mory-ut


zat
o
` param
t
r co
tro
s ho
 much GPU m
mory vLLM 


 us
:
- **D
fau
t (0.9)**: Good ba
a
c
 for most 
ork
oads
- **H
gh
r (0.95+)**: B
tt
r throughput but h
gh
r r
sk of OOM
- **Lo

r (0.5-0.8)**: Saf
r for var
ab

 
ork
oads or 
h

 shar

g GPU
### Batch S
z

Larg
r batch s
z
s g


ra
y 
mprov
 throughput but 

cr
as
 
at

cy:
- **Throughput-opt
m
z
d**: Us
 
arg
r `--max-
um-s
qs` (256-512)
- **Lat

cy-opt
m
z
d**: Us
 sma

r `--max-
um-s
qs` (64-128)
- **D
fau
t (256)**: Good start

g po

t
## Work
oad-Sp
c
f
c Tu


g
### Chat/Co
v
rsat
o
a
 AI
For chat-bas
d app

cat
o
s:
```bash
v
m s
rv
 your-chat-mod

 \
  --max-mod

-


 8192 \
  --max-
um-s
qs 256 \
  --

ab

-pr
f
x-cach

g \
  --b
ock-s
z
 16
```
K
y param
t
rs:
- `--

ab

-pr
f
x-cach

g`: Ess

t
a
 for mu
t
-tur
 co
v
rsat
o
s
- `--b
ock-s
z
 16`: Good for short
r s
qu

c
s
- `--max-mod

-


 8192`: Typ
ca
 for chat co
t
xts
### Docum

t Proc
ss

g
For 
o
g docum

t proc
ss

g:
```bash
v
m s
rv
 your-mod

 \
  --max-mod

-


 32768 \
  --max-
um-s
qs 64 \
  --b
ock-s
z
 32 \
  --

ab

-pr
f
x-cach

g
```
K
y param
t
rs:
- `--max-mod

-


 32768+`: Support for 
o
g docum

ts
- `--max-
um-s
qs 64`: Sma

r batch
s for 
o
g s
qu

c
s
- `--b
ock-s
z
 32`: B
tt
r for 
o
g
r s
qu

c
s
### Cod
 G


rat
o

For cod
 comp

t
o
/ g


rat
o
:
```bash
v
m s
rv
 your-cod
-mod

 \
  --max-mod

-


 16384 \
  --max-
um-s
qs 128 \
  --b
ock-s
z
 32 \
  --

ab

-pr
f
x-cach

g
```
## Hard
ar
-Sp
c
f
c Tu


g
### S

g

 GPU
O
 a s

g

 GPU, focus o
 max
m
z

g throughput 

th

 m
mory co
stra

ts:
```bash
v
m s
rv
 your-mod

 \
  --gpu-m
mory-ut


zat
o
 0.95 \
  --max-
um-s
qs 256 \
  --

ab

-chu
k
d-pr
f


```
### Mu
t
-GPU (T

sor Para



sm)
For t

sor para



sm across mu
t
p

 GPUs:
```bash
v
m s
rv
 your-mod

 \
  --t

sor-para


-s
z
 2 \
  --p
p




-para


-s
z
 1 \
  --max-
um-s
qs 512
```
Not
: Us
 t

sor para



sm 
h

 a s

g

 GPU do
s
't hav
 

ough m
mory for th
 mod

.
### Mu
t
-Nod

For d
str
but
d s
rv

g across 
od
s:
```bash
# Nod
 0
v
m s
rv
 your-mod

 \
  --t

sor-para


-s
z
 2 \
  --p
p




-para


-s
z
 2 \
  --d
str
but
d-


t-m
thod 

v://
# Nod
 1 (s
t appropr
at
 

v
ro
m

t var
ab

s)
```
## Adva
c
d Opt
m
zat
o
s
### Qua
t
zat
o

Us
 qua
t
zat
o
 to r
duc
 m
mory usag
 a
d 
mprov
 p
rforma
c
:
```bash
# FP8 qua
t
zat
o
 (Hopp
r GPUs)
v
m s
rv
 your-mod

 --qua
t
zat
o
 fp8
# AWQ qua
t
zat
o

v
m s
rv
 your-mod

 --qua
t
zat
o
 a
q
# GPTQ qua
t
zat
o

v
m s
rv
 your-mod

 --qua
t
zat
o
 gptq
```
### Sp
cu
at
v
 D
cod

g
For 
mprov
d 
at

cy 

th compat
b

 mod

s:
```bash
v
m s
rv
 your-mod

 \
  --sp
cu
at
v
-mod

 sp
cu
at
v
-mod

-
am
 \
  --
um-sp
cu
at
v
-tok

s 5
```
### Pr
f
x Cach

g
E
ab

 for 
ork
oads 

th shar
d prompts:
```bash
v
m s
rv
 your-mod

 --

ab

-pr
f
x-cach

g
```
B


f
ts:
- Mu
t
-tur
 chat: 60-80% cach
 h
t rat

- Docum

t QA: 50-70% cach
 h
t rat

- Cod
 comp

t
o
: 40-60% cach
 h
t rat

## Mo

tor

g P
rforma
c

### K
y M
tr
cs
Mo

tor th
s
 m
tr
cs to 
va
uat
 p
rforma
c
:
- **Throughput**: tok

s/s
co
d
- **Lat

cy**: T
m
 to F
rst Tok

 (TTFT), T
m
 P
r Output Tok

 (TPOT)
- **Cach
 H
t Rat
**: Pr
f
x cach

g 
ff
ct
v


ss
- **GPU Ut


zat
o
**: `
v
d
a-sm
` or `--gpu-m
mory-ut


zat
o
`
### B

chmark

g
Us
 vLLM's bu

t-

 b

chmarks:
```bash
# Throughput b

chmark
v
m b

ch throughput \
  --mod

 your-mod

 \
  --datas
t b

chmarks/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

# Lat

cy b

chmark
v
m b

ch 
at

cy \
  --mod

 your-mod

 \
  --

put-


 512 \
  --output-


 128
```
## Troub

shoot

g P
rforma
c

### Lo
 Throughput
1. Ch
ck GPU ut


zat
o
 

th `
v
d
a-sm
`
2. I
cr
as
 `--max-
um-s
qs` 
f GPU 
s u
d
rut


z
d
3. E
ab

 chu
k
d pr
f

: `--

ab

-chu
k
d-pr
f

`
4. Co
s
d
r qua
t
zat
o
 for 
arg
r mod

s
### H
gh Lat

cy
1. R
duc
 `--max-
um-s
qs` for 
o

r qu
u

g d

ay
2. E
ab

 pr
f
x cach

g for r
p
at
d prompts
3. Us
 sp
cu
at
v
 d
cod

g 
f ava

ab


4. Co
s
d
r us

g a sma

r mod

 or qua
t
zat
o

### OOM Errors
1. R
duc
 `--max-mod

-


`
2. R
duc
 `--max-
um-s
qs`
3. Lo

r `--gpu-m
mory-ut


zat
o
`
4. E
ab

 qua
t
zat
o

5. Us
 t

sor para



sm across mu
t
p

 GPUs
## R
f
r

c
s
- [M
mory Ma
ag
m

t](../co
f
gurat
o
/co
s
rv

g_m
mory.md)
- [APC D
bugg

g](./apc_d
bugg

g.md)
- [B

chmark

g](../b

chmark

g/README.md)
