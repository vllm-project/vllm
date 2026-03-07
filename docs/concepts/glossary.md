# G
ossary
Th
s g
ossary prov
d
s d
f


t
o
s for commo
 t
rms a
d co
c
pts us
d throughout th
 vLLM docum

tat
o
.
## A
**Att

t
o
 Back

d**
: Th
 
mp

m

tat
o
 us
d for comput

g att

t
o
 m
cha

sms. vLLM supports mu
t
p

 back

ds 

c
ud

g FLASH_ATTN, FLASHINFER, a
d p
atform-sp
c
f
c opt
o
s 

k
 ROCM_ATTN for AMD GPUs.
**Automat
c Pr
f
x Cach

g (APC)**
: A f
atur
 that cach
s th
 

t
rm
d
at
 stat
s (KV cach
) of pr
v
ous
y comput
d tok

s. Wh

 a 


 r
qu
st shar
s a commo
 pr
f
x 

th cach
d co
t

t, vLLM ca
 r
us
 th
 cach
d stat
s 

st
ad of r
comput

g th
m, s
g

f
ca
t
y 
mprov

g throughput for 
ork
oads 

th shar
d prompts.
## B
**Batch I
f
r

c
**
: Proc
ss

g mu
t
p

 

put prompts tog
th
r 

 a s

g

 batch. vLLM's 

g


 
s opt
m
z
d for h
gh-throughput batch 

f
r

c
, mak

g 
ff
c


t us
 of GPU r
sourc
s.
**B
ock S
z
**
: Th
 s
z
 of m
mory b
ocks us
d for KV cach
 ma
ag
m

t 

 vLLM. Co
f
gurab

 v
a `--b
ock-s
z
`, commo
 va
u
s ar
 16 or 32. Sma

r b
ocks off
r f


r gra
u
ar
ty but mor
 ov
rh
ad; 
arg
r b
ocks ar
 mor
 
ff
c


t but may 
ast
 m
mory.
## C
**Chu
k
d Pr
f

**
: A f
atur
 that br
aks 
arg
 pr
f

 op
rat
o
s 

to sma

r chu
ks. Th
s pr
v

ts 
o
g prompts from b
ock

g oth
r r
qu
sts a
d 
mprov
s ov
ra
 syst
m throughput. E
ab

d 

th `--

ab

-chu
k
d-pr
f

`.
**CUDA Graph**
: A
 opt
m
zat
o
 that captur
s a
d r
p
ays GPU op
rat
o
s to r
duc
 CPU ov
rh
ad. vLLM us
s CUDA graphs to opt
m
z
 th
 

f
r

c
 
oop, but th
y ca
 b
 d
sab

d 

th `--

forc
-
ag
r` for d
bugg

g.
## D
**D
cod

g**
: Th
 proc
ss of g


rat

g output tok

s from a 
a
guag
 mod

. I
 vLLM, th
s 

c
ud
s both th
 


t
a
 prompt proc
ss

g (pr
f

) a
d tok

 g


rat
o
 phas
s.
**D
str
but
d I
f
r

c
**
: Ru


g vLLM across mu
t
p

 GPUs or 
od
s us

g t

sor para



sm or p
p




 para



sm to support 
arg
r mod

s or h
gh
r throughput.
## E
**Eag
r Mod
**
: Ex
cut
o
 mod
 
h
r
 op
rat
o
s ar
 p
rform
d 
mm
d
at

y 

thout CUDA graph opt
m
zat
o
. Us
fu
 for d
bugg

g but may hav
 
o

r p
rforma
c
. E
ab

d 

th `--

forc
-
ag
r`.
## F
**FP8 Qua
t
zat
o
**
: A
 8-b
t f
oat

g po

t qua
t
zat
o
 format support
d o
 NVIDIA Hopp
r GPUs. R
duc
s m
mory usag
 a
d ca
 
mprov
 p
rforma
c
. E
ab

d 

th `--qua
t
zat
o
 fp8`.
## G
**G


rat
o
 Co
f
g**
: A JSON co
f
gurat
o
 f


 that sp
c
f

s d
fau
t samp


g param
t
rs for a mod

. vLLM ca
 us
 th
 mod

's `g


rat
o
_co
f
g.jso
` or 
ts o

 d
fau
ts.
**GPU M
mory Ut


zat
o
**
: Th
 fract
o
 of GPU m
mory that vLLM 


 us
, co
f
gurab

 v
a `--gpu-m
mory-ut


zat
o
` (d
fau
t: 0.9). H
gh
r va
u
s a
o
 
arg
r batch
s but 

cr
as
 OOM r
sk.
## H
**Ha
f Pr
c
s
o
 (FP16/BF16)**
: 16-b
t f
oat

g po

t formats that r
duc
 m
mory usag
 compar
d to 32-b
t (FP32) 
h


 ma

ta



g acc
ptab

 accuracy for most 

f
r

c
 tasks.
## I
**I
-f

ght Batch

g**
: vLLM's ab


ty to dy
am
ca
y batch r
qu
sts at d
ff
r

t stag
s of proc
ss

g (pr
f

 vs. d
cod

g), max
m
z

g GPU ut


zat
o
.
## K
**KV Cach
**
: K
y-Va
u
 cach
 that stor
s 

t
rm
d
at
 att

t
o
 computat
o
s to avo
d r
du
da
t ca
cu
at
o
s dur

g autor
gr
ss
v
 g


rat
o
. vLLM's Pag
dAtt

t
o
 
ff
c


t
y ma
ag
s th
 KV cach
.
## L
**LLM E
g


**
: Th
 cor
 compo


t of vLLM that ma
ag
s mod

 
x
cut
o
, m
mory, a
d r
qu
st sch
du


g. Ca
 b
 us
d d
r
ct
y for off



 

f
r

c
 or v
a th
 API s
rv
r.
**Logprobs**
: Logar
thm
c probab


t

s ass
g

d to tok

s by th
 
a
guag
 mod

. Ca
 b
 r
tr

v
d v
a API us

g th
 `
ogprobs` param
t
r.
## M
**Max Mod

 L

gth**
: Th
 max
mum s
qu

c
 


gth support
d by th
 mod

. Co
f
gurab

 v
a `--max-mod

-


` but 

m
t
d by th
 mod

's arch
t
ctura
 co
stra

ts.
**Max Num S
qs**
: Th
 max
mum 
umb
r of s
qu

c
s to proc
ss co
curr

t
y. H
gh
r va
u
s 
mprov
 throughput but 

cr
as
 m
mory usag
. Co
f
gurab

 v
a `--max-
um-s
qs` (d
fau
t: 256).
**Mod

 Para



sm**
: D
str
but

g a mod

 across mu
t
p

 GPUs. vLLM supports both t

sor para



sm (sp

tt

g 
ay
rs across GPUs) a
d p
p




 para



sm (sp

tt

g mod

 stag
s).
## N
**Nuc

us Samp


g (Top-p)**
: A samp


g m
thod that s


cts from th
 sma

st s
t of tok

s 
hos
 cumu
at
v
 probab


ty 
xc
ds a thr
sho
d `p`. Co
f
gurab

 v
a `top_p` 

 Samp


gParams.
## O
**Off



 I
f
r

c
**
: Batch proc
ss

g of prompts 

thout s
tt

g up a s
rv
r. Us
s th
 `LLM` c
ass d
r
ct
y 

 Pytho
 scr
pts.
**O




 S
rv

g**
: D
p
oy

g vLLM as a s
rv
r that acc
pts r
qu
sts ov
r HTTP, 
mp

m

t

g th
 Op

AI API protoco
.
**OOM (Out of M
mory)**
: Error that occurs 
h

 vLLM att
mpts to a
ocat
 mor
 GPU m
mory tha
 ava

ab

. Ca
 b
 m
t
gat
d by adjust

g m
mory-r

at
d param
t
rs.
## P
**Pag
dAtt

t
o
**
: vLLM's cor
 a
gor
thm that 
ff
c


t
y ma
ag
s KV cach
 us

g v
rtua
 m
mory pag

g t
ch

qu
s, 

ab


g dy
am
c m
mory a
ocat
o
 a
d h
gh GPU ut


zat
o
.
**P
p




 Para



sm**
: D
str
but

g d
ff
r

t 
ay
rs or stag
s of a mod

 across mu
t
p

 GPUs or 
od
s. Co
f
gur
d 

th `--p
p




-para


-s
z
`.
**Pr
f
x Cach

g**
: S
 Automat
c Pr
f
x Cach

g (APC).
**Pr
f

**
: Th
 


t
a
 phas
 of 

f
r

c
 
h
r
 th
 mod

 proc
ss
s th
 

put prompt to g


rat
 th
 f
rst KV cach
 stat
s.
## Q
**Qua
t
zat
o
**
: T
ch

qu
s to r
duc
 mod

 pr
c
s
o
 (
.g., to 8-b
t or 4-b
t) to sav
 m
mory a
d 
mprov
 

f
r

c
 sp
d. vLLM supports AWQ, GPTQ, FP8, a
d oth
r qua
t
zat
o
 m
thods.
## S
**Samp


g Param
t
rs**
: Co
f
gurat
o
 opt
o
s that co
tro
 t
xt g


rat
o
, 

c
ud

g t
mp
ratur
, top_p, top_k, a
d pr
s

c
_p

a
ty. D
f


d us

g th
 `Samp


gParams` c
ass.
**Sch
du

r**
: Th
 compo


t that ma
ag
s r
qu
st qu
u


g, batch

g, a
d 
x
cut
o
 ord
r 

 vLLM.
**Sp
cu
at
v
 D
cod

g**
: A
 opt
m
zat
o
 that us
s a sma

r draft mod

 to pr
d
ct mu
t
p

 tok

s ah
ad, 
h
ch ar
 th

 v
r
f

d by th
 ma

 mod

. Ca
 s
g

f
ca
t
y r
duc
 
at

cy.
## T
**T
mp
ratur
**
: A samp


g param
t
r that co
tro
s ra
dom

ss 

 g


rat
o
. Lo

r va
u
s (
.g., 0.1) produc
 mor
 d
t
rm


st
c output; h
gh
r va
u
s (
.g., 0.9) produc
 mor
 ra
dom output.
**T

sor Para



sm**
: Sp

tt

g 

d
v
dua
 
ay
rs of a mod

 across mu
t
p

 GPUs. Co
f
gur
d 

th `--t

sor-para


-s
z
`.
**Throughput**
: Th
 rat
 at 
h
ch vLLM proc
ss
s tok

s, typ
ca
y m
asur
d 

 tok

s p
r s
co
d.
**Tok

**
: Th
 bas
c u

t of t
xt for 
a
guag
 mod

s. Tok

s ca
 b
 
ords, sub
ords, or charact
rs d
p

d

g o
 th
 tok


z
r.
**TTFT (T
m
 to F
rst Tok

)**
: Th
 
at

cy from r
c

v

g a r
qu
st to g


rat

g th
 f
rst output tok

.
## V
**vLLM V1**
: Th
 


 arch
t
ctur
 

troduc
d 

 vLLM v0.7.0+, f
atur

g 
mprov
d p
rforma
c
 a
d a r
d
s
g

d 

g


. Ca
 b
 

ab

d 

th `VLLM_USE_V1=1`.
**Vocab S
z
**
: Th
 
umb
r of u

qu
 tok

s 

 a mod

's vocabu
ary.
## S
 A
so
    - [Arch
t
ctur
 Ov
rv


](../d
s
g
/arch
t
ctur
.md)
    - [Support
d Mod

s](../mod

s/support
d_mod

s.md)
    - [Co
f
gurat
o
 Gu
d
](../co
f
gurat
o
/

g


_params.md)
