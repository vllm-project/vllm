# M
tr
cs
vLLM 
xpos
s a r
ch s
t of m
tr
cs to support obs
rvab


ty a
d capac
ty p
a


g for th
 V1 

g


.
## Obj
ct
v
s
- Prov
d
 compr
h

s
v
 cov
rag
 of 

g


 a
d r
qu
st 

v

 m
tr
cs to a
d product
o
 mo

tor

g.
- Pr
or
t
z
 Prom
th
us 

t
grat
o
s, as th
s 
s 
hat 

 
xp
ct to b
 us
d 

 product
o
 

v
ro
m

ts.
- Off
r 
ogg

g support (
.
. pr

t

g m
tr
cs to th
 

fo 
og) for ad-hoc t
st

g, d
bugg

g, d
v

opm

t, a
d 
xp
oratory us
 cas
s.
## Backgrou
d
M
tr
cs 

 vLLM ca
 b
 cat
gor
z
d as fo
o
s:
1. S
rv
r-

v

 m
tr
cs: G
oba
 m
tr
cs that track th
 stat
 a
d p
rforma
c
 of th
 LLM 

g


. Th
s
 ar
 typ
ca
y 
xpos
d as Gaug
s or Cou
t
rs 

 Prom
th
us.
2. R
qu
st-

v

 m
tr
cs: M
tr
cs that track th
 charact
r
st
cs (
.g. s
z
 a
d t
m

g) of 

d
v
dua
 r
qu
sts. Th
s
 ar
 typ
ca
y 
xpos
d as H
stograms 

 Prom
th
us a
d ar
 oft

 th
 SLOs that a
 SRE mo

tor

g vLLM 


 b
 track

g.
Th
 m

ta
 mod

 
s that s
rv
r-

v

 m
tr
cs h

p 
xp
a

 th
 va
u
s of r
qu
st-

v

 m
tr
cs.
### M
tr
cs Ov
rv



### v1 M
tr
cs
I
 v1, a
 
xt

s
v
 s
t of m
tr
cs ar
 
xpos
d v
a a Prom
th
us-compat
b

 `/m
tr
cs` 

dpo

t us

g th
 `v
m:` pr
f
x, for 
xamp

:
- `v
m:
um_r
qu
sts_ru


g` (Gaug
) - Numb
r of r
qu
sts curr

t
y ru


g.
- `v
m:kv_cach
_usag
_p
rc` (Gaug
) - Fract
o
 of us
d KV cach
 b
ocks (0–1).
- `v
m:pr
f
x_cach
_qu
r

s` (Cou
t
r) - Numb
r of pr
f
x cach
 qu
r

s.
- `v
m:pr
f
x_cach
_h
ts` (Cou
t
r) - Numb
r of pr
f
x cach
 h
ts.
- `v
m:prompt_tok

s_tota
` (Cou
t
r) - Tota
 
umb
r of prompt tok

s proc
ss
d.
- `v
m:g


rat
o
_tok

s_tota
` (Cou
t
r) - Tota
 
umb
r of g


rat
d tok

s.
- `v
m:r
qu
st_succ
ss_tota
` (Cou
t
r) - Numb
r of f


sh
d r
qu
sts (by f


sh r
aso
).
- `v
m:r
qu
st_prompt_tok

s` (H
stogram) - H
stogram of 

put prompt tok

 cou
ts.
- `v
m:r
qu
st_g


rat
o
_tok

s` (H
stogram) - H
stogram of g


rat
o
 tok

 cou
ts.
- `v
m:t
m
_to_f
rst_tok

_s
co
ds` (H
stogram) - T
m
 to f
rst tok

 (TTFT).
- `v
m:

t
r_tok

_
at

cy_s
co
ds` (H
stogram) - I
t
r-tok

 
at

cy.
- `v
m:
2
_r
qu
st_
at

cy_s
co
ds` (H
stogram) - E
d-to-

d r
qu
st 
at

cy.
- `v
m:r
qu
st_pr
f

_t
m
_s
co
ds` (H
stogram) - R
qu
st pr
f

 t
m
.
- `v
m:r
qu
st_d
cod
_t
m
_s
co
ds` (H
stogram) - R
qu
st d
cod
 t
m
.
Th
s
 ar
 docum

t
d u
d
r [I
f
r

c

g a
d S
rv

g -
 Product
o
 M
tr
cs](../usag
/m
tr
cs.md).
### Grafa
a Dashboard
vLLM a
so prov
d
s [a r
f
r

c
 
xamp

](../../
xamp

s/o




_s
rv

g/prom
th
us_grafa
a/README.md) for ho
 to co

ct a
d stor
 th
s
 m
tr
cs us

g Prom
th
us a
d v
sua

z
 th
m us

g a Grafa
a dashboard.
Th
 subs
t of m
tr
cs 
xpos
d 

 th
 Grafa
a dashboard g
v
s us a
 

d
cat
o
 of 
h
ch m
tr
cs ar
 
sp
c
a
y 
mporta
t:
- `v
m:
2
_r
qu
st_
at

cy_s
co
ds_buck
t` - E
d to 

d r
qu
st 
at

cy m
asur
d 

 s
co
ds.
- `v
m:prompt_tok

s` - Prompt tok

s.
- `v
m:g


rat
o
_tok

s` - G


rat
o
 tok

s.
- `v
m:

t
r_tok

_
at

cy_s
co
ds` - I
t
r-tok

 
at

cy (T
m
 P
r Output Tok

, TPOT) 

 s
co
ds.
- `v
m:t
m
_to_f
rst_tok

_s
co
ds` - T
m
 to F
rst Tok

 (TTFT) 
at

cy 

 s
co
ds.
- `v
m:
um_r
qu
sts_ru


g` (a
so, `_s
app
d` a
d `_
a
t

g`) - Numb
r of r
qu
sts 

 th
 RUNNING, WAITING, a
d SWAPPED stat
s.
- `v
m:kv_cach
_usag
_p
rc` - P
rc

tag
 of us
d cach
 b
ocks by vLLM.
- `v
m:r
qu
st_prompt_tok

s` - R
qu
st prompt 


gth.
- `v
m:r
qu
st_g


rat
o
_tok

s` - R
qu
st g


rat
o
 


gth.
- `v
m:r
qu
st_succ
ss` - Numb
r of f


sh
d r
qu
sts by th

r f


sh r
aso
: 

th
r a
 EOS tok

 
as g


rat
d or th
 max s
qu

c
 


gth 
as r
ach
d.
- `v
m:r
qu
st_qu
u
_t
m
_s
co
ds` - Qu
u
 t
m
.
- `v
m:r
qu
st_pr
f

_t
m
_s
co
ds` - R
qu
sts pr
f

 t
m
.
- `v
m:r
qu
st_d
cod
_t
m
_s
co
ds` - R
qu
sts d
cod
 t
m
.
- `v
m:r
qu
st_max_
um_g


rat
o
_tok

s` - Max g


rat
o
 tok

s 

 a s
qu

c
 group.
S
 [th
 PR 
h
ch add
d th
s Dashboard](https://g
thub.com/v
m-proj
ct/v
m/pu
/2316) for 

t
r
st

g a
d us
fu
 backgrou
d o
 th
 cho
c
s mad
 h
r
.
### Prom
th
us C



t L
brary
Prom
th
us support 
as 


t
a
y add
d [us

g th
 a
oprom
th
us 

brary](https://g
thub.com/v
m-proj
ct/v
m/pu
/1890), but a s

tch 
as mad
 qu
ck
y to [prom
th
us_c



t](https://g
thub.com/v
m-proj
ct/v
m/pu
/2730). Th
 rat
o
a

 
s d
scuss
d 

 both 


k
d PRs.
Dur

g thos
 m
grat
o
s 

 br

f
y 
ost a `M
tr
csM
dd


ar
` to track HTTP m
tr
cs, but th
s 
as r


stat
d [us

g prom
th
us_fastap
_

strum

tator](https://g
thub.com/v
m-proj
ct/v
m/pu
/15657):
```bash
$ cur
 http://0.0.0.0:8000/m
tr
cs 2
/d
v/
u
  | gr
p -P '^http_(?!.*(_buck
t|_cr
at
d|_sum)).*'
http_r
qu
sts_tota
{ha
d

r="/v1/comp

t
o
s",m
thod="POST",status="2xx"} 201.0
http_r
qu
st_s
z
_byt
s_cou
t{ha
d

r="/v1/comp

t
o
s"} 201.0
http_r
spo
s
_s
z
_byt
s_cou
t{ha
d

r="/v1/comp

t
o
s"} 201.0
http_r
qu
st_durat
o
_h
ghr_s
co
ds_cou
t 201.0
http_r
qu
st_durat
o
_s
co
ds_cou
t{ha
d

r="/v1/comp

t
o
s",m
thod="POST"} 201.0
```
### Mu
t
-proc
ss Mod

H
stor
ca
y, m
tr
cs 

r
 co

ct
d 

 th
 

g


 cor
 proc
ss a
d mu
t
proc
ss mod
 
as us
d to mak
 th
m ava

ab

 

 th
 API s
rv
r proc
ss. S
 
https://g
thub.com/v
m-proj
ct/v
m/pu
/7279
.
Mor
 r
c

t
y, m
tr
cs ar
 co

ct
d 

 th
 API s
rv
r proc
ss a
d mu
t
proc
ss mod
 
s o

y us
d 
h

 `--ap
-s
rv
r-cou
t 
 1`. S
 
https://g
thub.com/v
m-proj
ct/v
m/pu
/17546
 a
d d
ta

s o
 [API s
rv
r sca

-out](../s
rv

g/data_para


_d
p
oym

t.md#

t
r
a
-
oad-ba
a
c

g).
### Bu

t 

 Pytho
/Proc
ss M
tr
cs
Th
 fo
o


g m
tr
cs ar
 support
d by d
fau
t by `prom
th
us_c



t`, but th
y ar
 
ot 
xpos
d 
h

 mu
t
proc
ss mod
 
s us
d:
- `pytho
_gc_obj
cts_co

ct
d_tota
`
- `pytho
_gc_obj
cts_u
co

ctab

_tota
`
- `pytho
_gc_co

ct
o
s_tota
`
- `pytho
_

fo`
- `proc
ss_v
rtua
_m
mory_byt
s`
- `proc
ss_r
s
d

t_m
mory_byt
s`
- `proc
ss_start_t
m
_s
co
ds`
- `proc
ss_cpu_s
co
ds_tota
`
- `proc
ss_op

_fds`
- `proc
ss_max_fds`
Th
r
for
, th
s
 m
tr
cs ar
 u
ava

ab

 
h

 `--ap
-s
rv
r-cou
t 
 1`. It's qu
st
o
ab

 ho
 r


va
t th
s
 ar
 s

c
 th
y do 
ot aggr
gat
 th
s
 stats for a
 proc
ss
s that mak
 up a vLLM 

sta
c
.
## M
tr
cs D
s
g

Th
 ["Ev

 B
tt
r Obs
rvab


ty"](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/3616) f
atur
 
h
r
 
as 
h
r
 much of th
 m
tr
cs d
s
g
 
as p
a

d. For 
xamp

, s
 
h
r
 [a d
ta


d roadmap 
as 
a
d out](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/3616#
ssu
comm

t-2030858781).
### L
gacy PRs
To h

p u
d
rsta
d th
 backgrou
d to th
 m
tr
cs d
s
g
, h
r
 ar
 som
 of th
 r


va
t PRs 
h
ch add
d th
 or
g

a
, 
o
 

gacy, m
tr
cs:
- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/1890

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/2316

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/2730

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/4464

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/7279

### M
tr
cs Imp

m

tat
o
 PRs
For backgrou
d, h
r
 ar
 th
 r


va
t PRs r

at

g to th
 m
tr
cs 
mp

m

tat
o
 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/10582
:
- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/11962

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/11973

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/10907

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12416

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12478

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12516

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12530

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12561

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12579

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12592

- 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12644

### M
tr
cs Co

ct
o

I
 v1, 

 

sh to mov
 computat
o
 a
d ov
rh
ad out of th
 

g


 cor

proc
ss to m


m
z
 th
 t
m
 b
t


 
ach for
ard pass.
Th
 ov
ra
 
d
a of V1 E
g


Cor
 d
s
g
 
s:
- E
g


Cor
 
s th
 


r 
oop. P
rforma
c
 
s most cr
t
ca
 h
r

- Asy
cLLM 
s th
 out
r 
oop. Th
s 
s ov
r
app
d 

th GPU 
x
cut
o

  (
d
a
y), so th
s 
s 
h
r
 a
y "ov
rh
ads" shou
d b
 
f
  poss
b

. So Asy
cLLM.output_ha
d

r_
oop 
s th
 
d
a
 p
ac
 for th

  m
tr
cs bookk
p

g 
f poss
b

.
W
 


 ach

v
 th
s by co

ct

g m
tr
cs 

 th
 fro
t

d API s
rv
r,
a
d bas
 th
s
 m
tr
cs o
 

format
o
 

 ca
 g

a
 from th

`E
g


Cor
Outputs` r
tur

d by th
 

g


 cor
 proc
ss to th

fro
t

d.
### I
t
rva
 Ca
cu
at
o
s
Ma
y of our m
tr
cs ar
 th
 t
m
 

t
rva
 b
t


 var
ous 
v

ts 


th
 proc
ss

g of a r
qu
st. It 
s b
st pract
c
 to us
 t
m
stamps
bas
d o
 "mo
oto

c t
m
" (`t
m
.mo
oto

c()`) rath
r tha
 "
a
-c
ock
t
m
" (`t
m
.t
m
()`) to ca
cu
at
 

t
rva
s as th
 form
r 
s
u
aff
ct
d by syst
m c
ock cha
g
s (
.g. from NTP).
It's a
so 
mporta
t to 
ot
 that mo
oto

c c
ocks d
ff
r b
t



proc
ss
s - 
ach proc
ss has 
ts o

 r
f
r

c
 po

t. So 
t 
s
m
a


g

ss to compar
 mo
oto

c t
m
stamps from d
ff
r

t proc
ss
s.
Th
r
for
, 

 ord
r to ca
cu
at
 a
 

t
rva
, 

 must compar
 t
o
mo
oto

c t
m
stamps from th
 sam
 proc
ss.
### Sch
du

r Stats
Th
 

g


 cor
 proc
ss 


 co

ct som
 k
y stat
st
cs from th

sch
du

r - 
.g. th
 
umb
r of r
qu
sts that 

r
 sch
du

d or 
a
t

g
aft
r th
 
ast sch
du

r pass - a
d 

c
ud
 thos
 stat
st
cs 


`E
g


Cor
Outputs`.
### E
g


 Cor
 Ev

ts
Th
 

g


 cor
 


 a
so r
cord th
 t
m
stamp of c
rta

 p
r-r
qu
st

v

ts so that th
 fro
t

d ca
 ca
cu
at
 th
 

t
rva
 b
t


 th
s


v

ts.
Th
 
v

ts ar
:
- `QUEUED` - 
h

 th
 r
qu
st 
as r
c

v
d by th
 

g


 cor
 a
d
  add
d to th
 sch
du

r qu
u
.
- `SCHEDULED` - 
h

 th
 r
qu
st 
as f
rst sch
du

d for 
x
cut
o
.
- `PREEMPTED` - th
 r
qu
st has b

 put back 

 th
 
a
t

g qu
u

  

 ord
r to mak
 room for oth
r r
qu
sts to comp

t
. It 


 b

  r
-sch
du

d 

 futur
 a
d r
-start 
ts pr
f

 phas
.
- `NEW_TOKENS` - 
h

 th
 output 

c
ud
d 

 `E
g


Cor
Output` 
as
  g


rat
d. S

c
 th
s 
s commo
 to a
 r
qu
sts 

 a g
v


  
t
rat
o
, 

 us
 a s

g

 t
m
stamp o
 `E
g


Cor
Outputs` to
  r
cord th
s 
v

t.
A
d th
 ca
cu
at
d 

t
rva
s ar
:
- Qu
u
 

t
rva
 - b
t


 `QUEUED` a
d most r
c

t `SCHEDULED`.
- Pr
f

 

t
rva
 - b
t


 most r
c

t `SCHEDULED` a
d th
 subs
qu

t
  f
rst `NEW_TOKENS`.
- D
cod
 

t
rva
 - b
t


 f
rst (aft
r th
 most r
c

t `SCHEDULED`) a
d
  
ast `NEW_TOKENS`.
- I
f
r

c
 

t
rva
 - b
t


 most r
c

t `SCHEDULED` a
d 
ast `NEW_TOKENS`.
- I
t
r-tok

 

t
rva
 - b
t


 succ
ss
v
 `NEW_TOKENS`.
Put a
oth
r 
ay:
![I
t
rva
 ca
cu
at
o
s - commo
 cas
](../ass
ts/d
s
g
/m
tr
cs/

t
rva
s-1.p
g)
W
 
xp
or
d th
 poss
b


ty of hav

g th
 fro
t

d ca
cu
at
 th
s



t
rva
s us

g th
 t
m

g of 
v

ts v
s
b

 by th
 fro
t

d. Ho

v
r,
th
 fro
t

d do
s 
ot hav
 v
s
b


ty 

to th
 t
m

g of th
 `QUEUED`
a
d `SCHEDULED` 
v

ts a
d, s

c
 

 

d to ca
cu
at
 

t
rva
s bas
d
o
 mo
oto

c t
m
stamps from th
 sam
 proc
ss ... 

 

d th
 

g



cor
 to r
cord t
m
stamps for a
 of th
s
 
v

ts.
#### I
t
rva
 Ca
cu
at
o
s vs Pr
mpt
o
s
Wh

 a pr
mpt
o
 occurs dur

g d
cod
, s

c
 a
y a
r
ady g


rat
d
tok

s ar
 r
us
d, 

 co
s
d
r th
 pr
mpt
o
 as aff
ct

g th



t
r-tok

, d
cod
, a
d 

f
r

c
 

t
rva
s.
![I
t
rva
 ca
cu
at
o
s - pr
mpt
d d
cod
](../ass
ts/d
s
g
/m
tr
cs/

t
rva
s-2.p
g)
Wh

 a pr
mpt
o
 occurs dur

g pr
f

 (assum

g such a
 
v

t

s poss
b

), 

 co
s
d
r th
 pr
mpt
o
 as aff
ct

g th

t
m
-to-f
rst-tok

 a
d pr
f

 

t
rva
s.
![I
t
rva
 ca
cu
at
o
s - pr
mpt
d pr
f

](../ass
ts/d
s
g
/m
tr
cs/

t
rva
s-3.p
g)
### Fro
t

d Stats Co

ct
o

As th
 fro
t

d proc
ss
s a s

g

 `E
g


Cor
Outputs` - 
.
. th

output from a s

g

 

g


 cor
 
t
rat
o
 - 
t co

cts var
ous
stat
st
cs r

at

g to that 
t
rat
o
:
- Th
 tota
 
umb
r of 


 tok

s g


rat
d 

 th
s 
t
rat
o
.
- Th
 tota
 
umb
r of prompt tok

s proc
ss
d by th
 pr
f

s that
  comp

t
d 

 th
s 
t
rat
o
.
- Th
 qu
u
 

t
rva
s for a
y r
qu
sts that 

r
 sch
du

d 

 th
s
  
t
rat
o
.
- Th
 pr
f

 

t
rva
s for a
y r
qu
sts that comp

t
d pr
f

 


  th
s 
t
rat
o
.
- Th
 

t
r-tok

 

t
rva
s (T
m
 P
r Output Tok

, TPOT), for a

  r
qu
sts 

c
ud
d 

 th
s 
t
rat
o
.
- Th
 T
m
-To-F
rst-Tok

 (TTFT) for a
y r
qu
sts that comp

t
d
  pr
f

 

 th
s 
t
rat
o
. Ho

v
r, 

 ca
cu
at
 th
s 

t
rva

  r

at
v
 to 
h

 th
 r
qu
st 
as f
rst r
c

v
d by th
 fro
t

d
  (`arr
va
_t
m
`) 

 ord
r to accou
t for 

put proc
ss

g t
m
.
For a
y r
qu
sts that 

r
 comp

t
d 

 a g
v

 
t
rat
o
, 

 a
so
r
cord:
- Th
 

f
r

c
 a
d d
cod
 

t
rva
s - r

at
v
 to th
 sch
du

d a
d
  f
rst tok

 
v

ts, as d
scr
b
d abov
.
- E
d-to-

d 
at

cy - th
 

t
rva
 b
t


 fro
t

d `arr
va
_t
m
`
  a
d th
 fro
t

d r
c

v

g th
 f

a
 tok

.
### KV Cach
 R
s
d

cy M
tr
cs
W
 a
so 
m
t a s
t of h
stograms that d
scr
b
 ho
 
o
g samp

d KV cach

b
ocks stay r
s
d

t a
d ho
 oft

 th
y ar
 r
us
d. Samp


g
(`--kv-cach
-m
tr
cs-samp

`) k
ps th
 ov
rh
ad t

y; 
h

 a b
ock 
s
chos

 

 r
cord:
- `

f
t
m
` – a
ocat
o
 ⟶ 
v
ct
o

- `
d

 b
for
 
v
ct
o
` – 
ast touch ⟶ 
v
ct
o

- `r
us
 gaps` – th
 paus
s b
t


 touch
s 
h

 th
 b
ock g
ts r
us
d
Thos
 map d
r
ct
y to th
 Prom
th
us m
tr
cs:
- `v
m:kv_b
ock_

f
t
m
_s
co
ds` – ho
 
o
g 
ach samp

d b
ock 
x
sts.
- `v
m:kv_b
ock_
d

_b
for
_
v
ct_s
co
ds` – 
d

 ta

 aft
r th
 f

a
 acc
ss.
- `v
m:kv_b
ock_r
us
_gap_s
co
ds` – t
m
 b
t


 co
s
cut
v
 touch
s.
Th
 

g


 cor
 o

y sh
ps ra
 
v
ct
o
 
v

ts v
a `Sch
du

rStats`; th

fro
t

d dra

s th
m, tur
s th
m 

to Prom
th
us obs
rvat
o
s, a
d a
so

xpos
s th
 sam
 data through `LLM.g
t_m
tr
cs()` 
h

 
ogg

g 
s o
.
Look

g at 

f
t
m
 a
d 
d

 t
m
 o
 o

 chart mak
s 
t 
asy to spot
stra
d
d cach
 or 
ork
oads that p

 prompts for a 
o
g d
cod
.
### M
tr
cs Pub

sh

g - Logg

g
Th
 `Logg

gStatLogg
r` m
tr
cs pub

sh
r outputs a 
og `INFO` m
ssag


v
ry 5 s
co
ds 

th som
 k
y m
tr
cs:
- Th
 curr

t 
umb
r of ru


g/
a
t

g r
qu
sts
- Th
 curr

t GPU cach
 usag

- Th
 
umb
r of prompt tok

s proc
ss
d p
r s
co
d ov
r th
 past 5
  s
co
ds
- Th
 
umb
r of 


 tok

s g


rat
d p
r s
co
d ov
r th
 past 5
  s
co
ds
- Th
 pr
f
x cach
 h
t rat
 ov
r th
 most r
c

t 1k kv-cach
 b
ock qu
r

s
### M
tr
cs Pub

sh

g - Prom
th
us
Th
 `Prom
th
usStatLogg
r` m
tr
cs pub

sh
r mak
s th
 m
tr
cs
ava

ab

 v
a a `/m
tr
cs` HTTP 

dpo

t 

 a Prom
th
us-compat
b


format. A Prom
th
us 

sta
c
 ca
 th

 b
 co
f
gur
d to po
 th
s


dpo

t (
.g. 
v
ry s
co
d) a
d r
cord th
 va
u
s 

 
ts t
m
-s
r

s
databas
. Prom
th
us 
s oft

 us
d v
a Grafa
a, a
o


g th
s
 m
tr
cs
to b
 graph
d ov
r t
m
.
Prom
th
us supports th
 fo
o


g m
tr
c typ
s:
- Cou
t
r: a va
u
 that 


 

cr
as
 ov
r t
m
, 

v
r r
duc

g, a
d
  g


ra
y r
s
t to z
ro 
h

 th
 vLLM 

sta
c
 r
starts. For
  
xamp

, th
 
umb
r of tok

s g


rat
d ov
r th
 

f
t
m
 of th

  

sta
c
.
- Gaug
: a va
u
 that go
s up a
d do

, for 
xamp

 th
 
umb
r of
  r
qu
sts curr

t
y sch
du

d for 
x
cut
o
.
- H
stogram: a cou
t of m
tr
c samp

s, r
cord
d 

 buck
ts. For
  
xamp

, th
 
umb
r of r
qu
sts 
hos
 TTFT 
as 
1ms, 
5ms, 
10ms,
  
20ms, a
d so o
.
Prom
th
us m
tr
cs ca
 a
so b
 
ab


d, a
o


g m
tr
cs to b

comb


d accord

g to match

g 
ab

s. I
 vLLM, 

 add a `mod

_
am
`

ab

 to 
v
ry m
tr
c 
h
ch 

c
ud
s th
 
am
 of th
 mod

 s
rv
d by
that 

sta
c
.
Examp

 output:
```bash
$ cur
 http://0.0.0.0:8000/m
tr
cs
# HELP v
m:
um_r
qu
sts_ru


g Numb
r of r
qu
sts 

 mod

 
x
cut
o
 batch
s.
# TYPE v
m:
um_r
qu
sts_ru


g gaug

v
m:
um_r
qu
sts_ru


g{mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 8.0
...
# HELP v
m:g


rat
o
_tok

s_tota
 Numb
r of g


rat
o
 tok

s proc
ss
d.
# TYPE v
m:g


rat
o
_tok

s_tota
 cou
t
r
v
m:g


rat
o
_tok

s_tota
{mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 27453.0
...
# HELP v
m:r
qu
st_succ
ss_tota
 Cou
t of succ
ssfu
y proc
ss
d r
qu
sts.
# TYPE v
m:r
qu
st_succ
ss_tota
 cou
t
r
v
m:r
qu
st_succ
ss_tota
{f


sh
d_r
aso
="stop",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 1.0
v
m:r
qu
st_succ
ss_tota
{f


sh
d_r
aso
="


gth",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 131.0
v
m:r
qu
st_succ
ss_tota
{f


sh
d_r
aso
="abort",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 0.0
...
# HELP v
m:t
m
_to_f
rst_tok

_s
co
ds H
stogram of t
m
 to f
rst tok

 

 s
co
ds.
# TYPE v
m:t
m
_to_f
rst_tok

_s
co
ds h
stogram
v
m:t
m
_to_f
rst_tok

_s
co
ds_buck
t{

="0.001",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 0.0
v
m:t
m
_to_f
rst_tok

_s
co
ds_buck
t{

="0.005",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 0.0
v
m:t
m
_to_f
rst_tok

_s
co
ds_buck
t{

="0.01",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 0.0
v
m:t
m
_to_f
rst_tok

_s
co
ds_buck
t{

="0.02",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 13.0
v
m:t
m
_to_f
rst_tok

_s
co
ds_buck
t{

="0.04",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 97.0
v
m:t
m
_to_f
rst_tok

_s
co
ds_buck
t{

="0.06",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 123.0
v
m:t
m
_to_f
rst_tok

_s
co
ds_buck
t{

="0.08",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 138.0
v
m:t
m
_to_f
rst_tok

_s
co
ds_buck
t{

="0.1",mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 140.0
v
m:t
m
_to_f
rst_tok

_s
co
ds_cou
t{mod

_
am
="m
ta-
ama/L
ama-3.1-8B-I
struct"} 140.0
```
!!! 
ot

    Th
 cho
c
 of h
stogram buck
ts to b
 most us
fu
 to us
rs
    across a broad s
t of us
 cas
s 
s 
ot stra
ghtfor
ard a
d 



    r
qu
r
 r
f


m

t ov
r t
m
.
### Cach
 Co
f
g I
fo
`prom
th
us_c



t` has support for
[I
fo m
tr
cs](https://prom
th
us.g
thub.
o/c



t_pytho
/

strum

t

g/

fo/)

h
ch ar
 
qu
va


t to a `Gaug
` 
hos
 va
u
 
s p
rma


t
y s
t to 1,
but 
xpos
s 

t
r
st

g k
y/va
u
 pa
r 

format
o
 v
a 
ab

s. Th
s 
s
us
d for 

format
o
 about a
 

sta
c
 that do
s 
ot cha
g
 - so 
t
o

y 

ds to b
 obs
rv
d at startup - a
d a
o
s compar

g across


sta
c
s 

 Prom
th
us.
W
 us
 th
s co
c
pt for th
 `v
m:cach
_co
f
g_

fo` m
tr
c:
```t
xt
# HELP v
m:cach
_co
f
g_

fo I
format
o
 of th
 LLME
g


 Cach
Co
f
g
# TYPE v
m:cach
_co
f
g_

fo gaug

v
m:cach
_co
f
g_

fo{b
ock_s
z
="16",cach
_dtyp
="auto",ca
cu
at
_kv_sca

s="Fa
s
",cpu_off
oad_gb="0",

ab

_pr
f
x_cach

g="Fa
s
",gpu_m
mory_ut


zat
o
="0.9",...} 1.0
```
Ho

v
r, `prom
th
us_c



t` has
[

v
r support
d I
fo m
tr
cs 

 mu
t
proc
ss

g mod
](https://g
thub.com/prom
th
us/c



t_pytho
/pu
/300) -
for [u
c

ar r
aso
s](gh-pr:7279#d
scuss
o
_r1710417152). W

s
mp
y us
 a `Gaug
` m
tr
c s
t to 1 a
d
`mu
t
proc
ss_mod
="mostr
c

t"` 

st
ad.
### LoRA M
tr
cs
Th
 `v
m:
ora_r
qu
sts_

fo` `Gaug
` 
s som

hat s
m

ar, 
xc
pt th

va
u
 
s th
 curr

t 
a
-c
ock t
m
, a
d 
s updat
d 
v
ry 
t
rat
o
.
Th
 
ab

 
am
s us
d ar
:
- `ru


g_
ora_adapt
rs`: a p
r-adapt
r cou
t of th
 
umb
r r
qu
sts
  ru


g us

g that adapt
r, formatt
d as a comma-s
parat
d str

g.
- `
a
t

g_
ora_adapt
rs`: s
m

ar, 
xc
pt cou
t

g r
qu
sts that ar

  
a
t

g to b
 sch
du

d.
- `max_
ora` - th
 stat
c "max 
umb
r of LoRAs 

 a s

g

 batch."
  co
f
gurat
o
.
E
cod

g a ru


g/
a
t

g cou
ts for mu
t
p

 adapt
rs 

 a
comma-s
parat
d str

g s
ms qu
t
 m
sgu
d
d - 

 cou
d us
 
ab

s to
d
st

gu
sh b
t


 p
r-adapt
r cou
ts. Th
s shou
d b
 r
v
s
t
d.
Not
 that `mu
t
proc
ss_mod
="

v
mostr
c

t"` 
s us
d - th
 most
r
c

t m
tr
c 
s us
d, but o

y from curr

t
y ru


g proc
ss
s.
Th
s 
as add
d 

 
https://g
thub.com/v
m-proj
ct/v
m/pu
/9477
 a
d th
r
 
s
[at 

ast o

 k
o

 us
r](https://g
thub.com/kub
r

t
s-s
gs/gat

ay-ap
-

f
r

c
-
xt

s
o
/pu
/54).
If 

 r
v
s
t th
s d
s
g
 a
d d
pr
cat
 th
 o
d m
tr
c, 

 shou
d
coord

at
 

th do

str
am us
rs so th
y ca
 m
grat
 b
for
 th
 r
mova
.
### Pr
f
x Cach
 m
tr
cs
Th
 d
scuss
o
 

 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/10582
 about add

g pr
f
x cach
 m
tr
cs y


d
d
som
 

t
r
st

g po

ts 
h
ch may b
 r


va
t to ho
 

 approach
futur
 m
tr
cs.
Ev
ry t
m
 th
 pr
f
x cach
 
s qu
r

d, 

 r
cord th
 
umb
r of tok

s
qu
r

d a
d th
 
umb
r of qu
r

d tok

s pr
s

t 

 th
 cach

(
.
. h
ts).
Ho

v
r, th
 m
tr
c of 

t
r
st 
s th
 h
t rat
 - 
.
. th
 
umb
r of
h
ts p
r qu
ry.
I
 th
 cas
 of 
ogg

g, 

 
xp
ct th
 us
r 
s b
st s
rv
d by
ca
cu
at

g th
 h
t rat
 ov
r a f
x
d 
umb
r of th
 most r
c

t
qu
r

s (th
 

t
rva
 
s f
x
d to 1k most r
c

t qu
r

s for 
o
).
I
 th
 cas
 of Prom
th
us though, 

 shou
d tak
 adva
tag
 of th

t
m
-s
r

s 
atur
 of Prom
th
us a
d a
o
 th
 us
r to ca
cu
at
 th

h
t rat
 ov
r a
 

t
rva
 of th

r choos

g. For 
xamp

, a PromQL
qu
ry to ca
cu
at
 th
 h
t 

t
rva
 of th
 past 5 m

ut
s:
```t
xt
rat
(cach
_qu
ry_h
t[5m]) / rat
(cach
_qu
ry_tota
[5m])
```
To ach

v
 th
s, 

 shou
d r
cord th
 qu
r

s a
d h
ts as cou
t
rs 


Prom
th
us, rath
r tha
 r
cord

g th
 h
t rat
 as a gaug
.
## D
pr
cat
d M
tr
cs
### Ho
 To D
pr
cat

D
pr
cat

g m
tr
cs shou
d
't b
 tak

 

ght
y. Us
rs may 
ot 
ot
c
 a
m
tr
c has b

 d
pr
cat
d, a
d may b
 qu
t
 

co
v




c
d 
h

 
t 
s
sudd


y (from th

r p
rsp
ct
v
) 
h

 
t 
s r
mov
d, 
v

 
f th
r
 
s
a
 
qu
va


t m
tr
c for th
m to us
.
As a
 
xamp

, s
 ho
 `v
m:avg_prompt_throughput_toks_p
r_s` 
as
[d
pr
cat
d](https://g
thub.com/v
m-proj
ct/v
m/pu
/2764) (

th a comm

t 

 th
 cod
),
[r
mov
d](https://g
thub.com/v
m-proj
ct/v
m/pu
/12383), a
d th

 [
ot
c
d by a us
r](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/13218).
I
 g


ra
:
1. W
 shou
d b
 caut
ous about d
pr
cat

g m
tr
cs, 
sp
c
a
y s

c

   
t ca
 b
 hard to pr
d
ct th
 us
r 
mpact.
2. W
 shou
d 

c
ud
 a prom



t d
pr
cat
o
 
ot
c
 

 th
 h

p str

g
   that 
s 

c
ud
d 

 th
 `/m
tr
cs' output.
3. W
 shou
d 

st d
pr
cat
d m
tr
cs 

 us
r-fac

g docum

tat
o
 a
d
   r


as
 
ot
s.
4. W
 shou
d co
s
d
r h
d

g d
pr
cat
d m
tr
cs b
h

d a CLI argum

t
   

 ord
r to g
v
 adm


strators
   [a
 
scap
 hatch](https://kub
r

t
s.
o/docs/co
c
pts/c
ust
r-adm


strat
o
/syst
m-m
tr
cs/#sho
-h
dd

-m
tr
cs)
   for som
 t
m
 b
for
 d


t

g th
m.
S
 th
 [d
pr
cat
o
 po

cy](../co
tr
but

g/d
pr
cat
o
_po

cy.md) for
th
 proj
ct-

d
 d
pr
cat
o
 po

cy.
### U

mp

m

t
d - `v
m:tok

s_tota
`
Add
d by 
https://g
thub.com/v
m-proj
ct/v
m/pu
/4464
, but appar

t
y 

v
r 
mp

m

t
d. Th
s ca
 just b

r
mov
d.
### Dup

cat
d - Qu
u
 T
m

Th
 `v
m:t
m
_

_qu
u
_r
qu
sts` H
stogram m
tr
c 
as add
d by
https://g
thub.com/v
m-proj
ct/v
m/pu
/9659
 a
d 
ts ca
cu
at
o
 
s:
```pytho

    s

f.m
tr
cs.f
rst_sch
du

d_t
m
 = 
o

    s

f.m
tr
cs.t
m
_

_qu
u
 = 
o
 - s

f.m
tr
cs.arr
va
_t
m

```
T
o 

ks 
at
r, 
https://g
thub.com/v
m-proj
ct/v
m/pu
/4464
 add
d `v
m:r
qu
st_qu
u
_t
m
_s
co
ds` 

av

g
us 

th:
```pytho


f s
q_group.
s_f


sh
d():
    
f (s
q_group.m
tr
cs.f
rst_sch
du

d_t
m
 
s 
ot No

 a
d
            s
q_group.m
tr
cs.f
rst_tok

_t
m
 
s 
ot No

):
        t
m
_qu
u
_r
qu
sts.app

d(
            s
q_group.m
tr
cs.f
rst_sch
du

d_t
m
 -
            s
q_group.m
tr
cs.arr
va
_t
m
)
    ...
    
f s
q_group.m
tr
cs.t
m
_

_qu
u
 
s 
ot No

:
        t
m
_

_qu
u
_r
qu
sts.app

d(
            s
q_group.m
tr
cs.t
m
_

_qu
u
)
```
Th
s s
ms dup

cat
v
, a
d o

 of th
m shou
d b
 r
mov
d. Th
 
att
r

s us
d by th
 Grafa
a dashboard, so 

 shou
d d
pr
cat
 or r
mov
 th

form
r.
### Pr
f
x Cach
 H
t Rat

S
 abov
 - 

 
o
 
xpos
 'qu
r

s' a
d 'h
ts' cou
t
rs rath
r tha
 a
'h
t rat
' gaug
.
### KV Cach
 Off
oad

g
T
o 

gacy m
tr
cs r

at
 to a "s
app
d" pr
mpt
o
 mod
 that 
s 
o

o
g
r r


va
t 

 v1:
- `v
m:
um_r
qu
sts_s
app
d`
- `v
m:cpu_cach
_usag
_p
rc`
I
 th
s mod
, 
h

 a r
qu
st 
as pr
mpt
d (
.g. to mak
 room 

 KV
cach
 to comp

t
 oth
r r
qu
sts), kv cach
 b
ocks 

r
 s
app
d out to
CPU m
mory. Th
 `--s
ap-spac
` f
ag has b

 r
mov
d as th
s f
atur


s 
o 
o
g
r us
d 

 V1.
H
stor
ca
y, [vLLM has 
o
g support
d b
am s
arch](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/6226). Th

S
qu

c
Group 

capsu
at
d th
 
d
a of N S
qu

c
s 
h
ch
a
 shar
d th
 sam
 prompt kv b
ocks. Th
s 

ab

d KV cach
 b
ock
shar

g b
t


 r
qu
sts, a
d copy-o
-
r
t
 to do bra
ch

g. CPU
s
app

g 
as 

t

d
d for th
s
 b
am s
arch 

k
 cas
s.
Lat
r, th
 co
c
pt of pr
f
x cach

g 
as 

troduc
d, 
h
ch a
o

d KV
cach
 b
ocks to b
 shar
d 
mp

c
t
y. Th
s prov
d to b
 a b
tt
r
opt
o
 tha
 CPU s
app

g s

c
 b
ocks ca
 b
 
v
ct
d s
o

y o
 d
ma
d
a
d th
 part of th
 prompt that 
as 
v
ct
d ca
 b
 r
comput
d.
S
qu

c
Group 
as r
mov
d 

 V1, a
though a r
p
ac
m

t 


 b

r
qu
r
d for "para


 samp


g" (`

1`).
[B
am s
arch 
as mov
d out of th
 cor
](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/8306). Th
r
 
as a

ot of comp

x cod
 for a v
ry u
commo
 f
atur
.
I
 V1, 

th pr
f
x cach

g b


g b
tt
r (z
ro ov
r h
ad) a
d th
r
for

o
 by d
fau
t, th
 pr
mpt
o
 a
d r
comput
 strat
gy shou
d 
ork
b
tt
r.
## Futur
 Work
### Para


 Samp


g
Som
 

gacy m
tr
cs ar
 o

y r


va
t 

 th
 co
t
xt of "para



samp


g". Th
s 
s 
h
r
 th
 `
` param
t
r 

 a r
qu
st 
s us
d to
r
qu
st mu
t
p

 comp

t
o
s from th
 sam
 prompt.
As part of add

g para


 samp


g support 

 
https://g
thub.com/v
m-proj
ct/v
m/pu
/10980
, 

 shou
d
a
so add th
s
 m
tr
cs.
- `v
m:r
qu
st_params_
` (H
stogram)
  Obs
rv
s th
 va
u
 of th
 '
' param
t
r of 
v
ry f


sh
d r
qu
st.
- `v
m:r
qu
st_max_
um_g


rat
o
_tok

s` (H
stogram)
  Obs
rv
s th
 max
mum output 


gth of a
 s
qu

c
s 

 
v
ry f


sh
d
  s
qu

c
 group. I
 th
 abs

c
 of para


 samp


g, th
s 
s
  
qu
va


t to `v
m:r
qu
st_g


rat
o
_tok

s`.
### Sp
cu
at
v
 D
cod

g
Som
 

gacy m
tr
cs ar
 sp
c
f
c to "sp
cu
at
v
 d
cod

g". Th
s 
s 
h
r



 g


rat
 ca
d
dat
 tok

s us

g a fast
r, approx
mat
 m
thod or
mod

 a
d th

 va

dat
 thos
 tok

s 

th th
 
arg
r mod

.
- `v
m:sp
c_d
cod
_draft_acc
pta
c
_rat
` (Gaug
)
- `v
m:sp
c_d
cod
_
ff
c


cy` (Gaug
)
- `v
m:sp
c_d
cod
_
um_acc
pt
d_tok

s` (Cou
t
r)
- `v
m:sp
c_d
cod
_
um_draft_tok

s` (Cou
t
r)
- `v
m:sp
c_d
cod
_
um_
m
tt
d_tok

s` (Cou
t
r)
Th
r
 
s a PR u
d
r r
v


 (
https://g
thub.com/v
m-proj
ct/v
m/pu
/12193
) to add "prompt 
ookup (
gram)"
sp
cu
at
v
 d
cod

g to v1. Oth
r t
ch

qu
s 


 fo
o
. W
 shou
d
r
v
s
t th
s
 m
tr
cs 

 th
s co
t
xt.
!!! 
ot

    W
 shou
d probab
y 
xpos
 acc
pta
c
 rat
 as s
parat
 acc
pt
d
    a
d draft cou
t
rs, 

k
 

 do for pr
f
x cach

g h
t rat
. Eff
c


cy
    

k

y a
so 

ds s
m

ar tr
atm

t.
### Autosca


g a
d Load-ba
a
c

g
A commo
 us
 cas
 for our m
tr
cs 
s to support automat
d sca


g of
vLLM 

sta
c
s.
For r

at
d d
scuss
o
 from th

[Kub
r

t
s S
rv

g Work

g Group](https://g
thub.com/kub
r

t
s/commu

ty/tr
/mast
r/
g-s
rv

g),
s
:
- [Sta
dard
z

g Larg
 Mod

 S
rv
r M
tr
cs 

 Kub
r

t
s](https://docs.goog

.com/docum

t/d/1SpSp1E6moa4HSrJ
S4x3NpLuj88sMXr2tbofK
zTZpk)
- [B

chmark

g LLM Work
oads for P
rforma
c
 Eva
uat
o
 a
d Autosca


g 

 Kub
r

t
s](https://docs.goog

.com/docum

t/d/1k4Q4X14hW4vftE
IuYGDu5KD
2LtV1XammoG-X
3bbQ)
- [I
f
r

c
 P
rf](https://g
thub.com/kub
r

t
s-s
gs/
g-s
rv

g/tr
/ma

/proposa
s/013-

f
r

c
-p
rf)
- 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/5041
 a
d 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12726
.
Th
s 
s a 
o
-tr
v
a
 top
c. Co
s
d
r th
s comm

t from Rob:

 I th

k th
s m
tr
c shou
d focus o
 try

g to 
st
mat
 
hat th
 max

 co
curr

cy that 


 caus
 th
 av
rag
 r
qu
st 


gth 
 qu
r

s p
r

 s
co
d ... s

c
 th
s 
s r
a
y 
hat 


 "saturat
" th
 s
rv
r.
A c

ar goa
 
s that 

 shou
d 
xpos
 th
 m
tr
cs r
qu
r
d to d
t
ct
th
s saturat
o
 po

t, so adm


strators ca
 
mp

m

t auto-sca


g
ru

s bas
d o
 thos
. Ho

v
r, 

 ord
r to do so, 

 

d to hav
 a
c

ar v


 o
 ho
 a
 adm


strator (a
d automat
d mo

tor

g syst
m)
shou
d judg
 a
 

sta
c
 as approach

g saturat
o
:

 To 
d

t
fy, 
hat 
s th
 saturat
o
 po

t for mod

 s
rv
r comput


 (th
 

f

ct
o
 po

t 
h
r
 

 ca
ot g
t mor
 throughput 

th a

 h
gh
r r
qu
st rat
, but start to 

cur add
t
o
a
 
at

cy) so 



 ca
 autosca

 
ff
ct
v

y?
### M
tr
c Nam

g
Our approach to 
am

g m
tr
cs probab
y d
s
rv
s to b
 r
v
s
t
d:
1. Th
 us
 of co
o
s 

 m
tr
c 
am
s s
ms co
trary to
   ["co
o
s ar
 r
s
rv
d for us
r d
f


d r
cord

g ru

s"](https://prom
th
us.
o/docs/co
c
pts/data_mod

/#m
tr
c-
am
s-a
d-
ab

s).
2. Most of our m
tr
cs fo
o
 th
 co
v

t
o
 of 

d

g 

th u

ts, but
   
ot a
 do.
3. Som
 of our m
tr
c 
am
s 

d 

th `_tota
`:
    If th
r
 
s a suff
x of `_tota
` o
 th
 m
tr
c 
am
, 
t 


 b
 r
mov
d. Wh


    
xpos

g th
 t
m
 s
r

s for cou
t
r, a `_tota
` suff
x 


 b
 add
d. Th
s 
s
    for compat
b


ty b
t


 Op

M
tr
cs a
d th
 Prom
th
us t
xt format, as Op

M
tr
cs
    r
qu
r
s th
 `_tota
` suff
x.
### Add

g Mor
 M
tr
cs
Th
r
 
s 
o shortag
 of 
d
as for 


 m
tr
cs:
- Examp

s from oth
r proj
cts 

k

  [TGI](https://g
thub.com/IBM/t
xt-g


rat
o
-

f
r

c
?tab=r
adm
-ov-f


#m
tr
cs)
- Proposa
s ar
s

g from sp
c
f
c us
 cas
s, 

k
 th
 Kub
r

t
s
  auto-sca


g top
c abov

- Proposa
s that m
ght ar
s
 out of sta
dard
sat
o
 
fforts 

k

  [Op

T


m
try S
ma
t
c Co
v

t
o
s for G

 AI](https://g
thub.com/op

-t


m
try/s
ma
t
c-co
v

t
o
s/tr
/ma

/docs/g

-a
).
W
 shou
d b
 caut
ous 

 our approach to add

g 


 m
tr
cs. Wh



m
tr
cs ar
 oft

 r

at
v

y stra
ghtfor
ard to add:
1. Th
y ca
 b
 d
ff
cu
t to r
mov
 - s
 th
 s
ct
o
 o
 d
pr
cat
o

   abov
.
2. Th
y ca
 hav
 a m
a


gfu
 p
rforma
c
 
mpact 
h

 

ab

d. A
d
   m
tr
cs ar
 usua
y of v
ry 

m
t
d us
 u


ss th
y ca
 b
 

ab

d
   by d
fau
t a
d 

 product
o
.
3. Th
y hav
 a
 
mpact o
 d
v

opm

t a
d ma

t

a
c
 of th

   proj
ct. Ev
ry m
tr
c add
d ov
r t
m
 has mad
 th
s 
ffort mor

   t
m
-co
sum

g, a
d p
rhaps 
ot a
 m
tr
cs just
fy th
s o
go

g
   

v
stm

t 

 th

r ma

t

a
c
.
## Trac

g - Op

T


m
try
M
tr
cs prov
d
 a
 aggr
gat
d v


 ov
r t
m
 of th
 syst
m's
p
rforma
c
 a
d h
a
th. Trac

g, o
 th
 oth
r ha
d, tracks 

d
v
dua

r
qu
sts as th
y mov
 through d
ff
r

t s
rv
c
s a
d compo


ts. Both
fa
 u
d
r th
 mor
 g


ra
 h
ad

g of "Obs
rvab


ty".
vLLM has support for Op

T


m
try trac

g:
- Add
d by 
https://g
thub.com/v
m-proj
ct/v
m/pu
/4687
 a
d r


stat
d by 
https://g
thub.com/v
m-proj
ct/v
m/pu
/20372

- Co
f
gur
d 

th `--o
tp-trac
s-

dpo

t` a
d `--co

ct-d
ta


d-trac
s`
- [Op

T


m
try b
og post](https://op

t


m
try.
o/b
og/2024/
m-obs
rvab


ty/)
- [Us
r-fac

g docs](../../
xamp

s/o




_s
rv

g/op

t


m
try/README.md)
- [B
og post](https://m
d
um.com/@ro


.schaff
r/fo
o
-th
-tra

-sup
rcharg

g-v
m-

th-op

t


m
try-d
str
but
d-trac

g-aa655229b46f)
- [IBM product docs](https://
.
bm.com/docs/

/

sta
a-obs
rvab


ty/curr

t?top
c=mgaa-mo

tor

g-
arg
-
a
guag
-mod

s-
ms-v
m-pub

c-pr
v


)
Op

T


m
try has a
[G

 AI Work

g Group](https://g
thub.com/op

-t


m
try/commu

ty/b
ob/ma

/proj
cts/g

-a
.md).
S

c
 m
tr
cs 
s a b
g 

ough top
c o
 
ts o

, 

 co
s
d
r th
 top
c
of trac

g to b
 qu
t
 s
parat
 from m
tr
cs.
### Op

T


m
try Mod

 For
ard vs Ex
cut
 T
m

Th
 curr

t 
mp

m

tat
o
 
xpos
s th
 fo
o


g t
o m
tr
cs:
- `v
m:mod

_for
ard_t
m
_m


s
co
ds` (H
stogram) - Th
 t
m
 sp

t
  

 th
 mod

 for
ard pass 
h

 th
s r
qu
st 
as 

 th
 batch.
- `v
m:mod

_
x
cut
_t
m
_m


s
co
ds` (H
stogram) - Th
 t
m
 sp

t
  

 th
 mod

 
x
cut
 fu
ct
o
. Th
s 


 

c
ud
 mod

 for
ard,
  b
ock/sy
c across 
ork
rs, cpu-gpu sy
c t
m
 a
d samp


g t
m
.
Th
s
 m
tr
cs ar
 o

y 

ab

d 
h

 Op

T


m
try trac

g 
s 

ab

d
a
d 
f `--co

ct-d
ta


d-trac
s=a
/mod

/
ork
r` 
s us
d. Th

docum

tat
o
 for th
s opt
o
 stat
s:

 co

ct d
ta


d trac
s for th
 sp
c
f

d modu

s. Th
s 

vo
v
s

 us
 of poss
b
y cost
y a
d or b
ock

g op
rat
o
s a
d h

c
 m
ght

 hav
 a p
rforma
c
 
mpact.
Th
 m
tr
cs 

r
 add
d by 
https://g
thub.com/v
m-proj
ct/v
m/pu
/7089
 a
d 
ho up 

 a
 Op

T


m
try trac

as:
```t
xt
-
 g

_a
.
at

cy.t
m
_

_sch
du

r: Doub

(0.017550230026245117)
-
 g

_a
.
at

cy.t
m
_

_mod

_for
ard: Doub

(3.151565277099609)
-
 g

_a
.
at

cy.t
m
_

_mod

_
x
cut
: Doub

(3.6468167304992676)
```
W
 a
r
ady hav
 `

f
r

c
_t
m
` a
d `d
cod
_t
m
` m
tr
cs, so th

qu
st
o
 
s 
h
th
r th
r
 ar
 suff
c


t
y commo
 us
 cas
s for th

h
gh
r-r
so
ut
o
 t
m

gs to just
fy th
 ov
rh
ad.
S

c
 

 ar
 go

g to tr
at th
 qu
st
o
 of Op

T


m
try support
s
parat

y, 

 


 

c
ud
 th
s
 part
cu
ar m
tr
cs u
d
r that top
c.
