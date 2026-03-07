# IO Proc
ssor P
ug

s
IO Proc
ssor p
ug

s ar
 a f
atur
 that a
o
s pr
- a
d post-proc
ss

g of th
 mod

 

put a
d output for poo


g mod

s. Th
 
d
a 
s that us
rs ar
 a
o

d to pass a custom 

put to vLLM that 
s co
v
rt
d 

to o

 or mor
 mod

 prompts a
d f
d to th
 mod

 `

cod
` m
thod. O

 pot

t
a
 us
-cas
 of such p
ug

s 
s that of us

g vLLM for g


rat

g mu
t
-moda
 data. Say us
rs f
d a
 
mag
 to vLLM a
d g
t a
 
mag
 

 output.
Wh

 p
rform

g a
 

f
r

c
 

th IO Proc
ssor p
ug

s, th
 prompt typ
 
s d
f


d by th
 p
ug

 a
d th
 sam
 
s va

d for th
 f

a
 r
qu
st output. vLLM do
s 
ot p
rform a
y va

dat
o
 of 

put/output data, a
d 
t 
s up to th
 p
ug

 to 

sur
 th
 corr
ct data 
s b


g f
d to th
 mod

 a
d r
tur

d to th
 us
r. As of 
o
 th
s
 p
ug

s support o

y poo


g mod

s a
d ca
 b
 tr
gg
r
d v
a th
 `

cod
` m
thod 

 `LLM` a
d `Asy
cLLM`, or 

 o




 s
rv

g mod
 v
a th
 `/poo


g` 

dpo

t.
## Wr
t

g a
 IO Proc
ssor P
ug


IO Proc
ssor p
ug

s 
mp

m

t th
 [`IOProc
ssor`][v
m.p
ug

s.
o_proc
ssors.

t
rfac
.IOProc
ssor] 

t
rfac
:
```pytho

IOProc
ssorI
put = Typ
Var("IOProc
ssorI
put")
IOProc
ssorOutput = Typ
Var("IOProc
ssorOutput")
c
ass IOProc
ssor(ABC, G


r
c[IOProc
ssorI
put, IOProc
ssorOutput]):
    """Abstract 

t
rfac
 for pr
/post-proc
ss

g of 

g


 I/O."""
    d
f __


t__(s

f, v
m_co
f
g: V
mCo
f
g, r

d
r
r: Bas
R

d
r
r):
        sup
r().__


t__()
        s

f.v
m_co
f
g = v
m_co
f
g
    d
f pars
_data(s

f, data: obj
ct) -
 IOProc
ssorI
put:
        ra
s
 NotImp

m

t
dError
    d
f m
rg
_samp


g_params(
        s

f,
        params: Samp


gParams | No

 = No

,
    ) -
 Samp


gParams:
        r
tur
 params or Samp


gParams()
    d
f m
rg
_poo


g_params(
        s

f,
        params: Poo


gParams | No

 = No

,
    ) -
 Poo


gParams:
        r
tur
 params or Poo


gParams(task="p
ug

")
    @abstractm
thod
    d
f pr
_proc
ss(
        s

f,
        prompt: IOProc
ssorI
put,
        r
qu
st_
d: str | No

 = No

,
        **k
args,
    ) -
 PromptTyp
 | S
qu

c
[PromptTyp
]:
        ra
s
 NotImp

m

t
dError
    asy
c d
f pr
_proc
ss_asy
c(
        s

f,
        prompt: IOProc
ssorI
put,
        r
qu
st_
d: str | No

 = No

,
        **k
args,
    ) -
 PromptTyp
 | S
qu

c
[PromptTyp
]:
        r
tur
 s

f.pr
_proc
ss(prompt, r
qu
st_
d, **k
args)
    @abstractm
thod
    d
f post_proc
ss(
        s

f,
        mod

_output: S
qu

c
[Poo


gR
qu
stOutput],
        r
qu
st_
d: str | No

 = No

,
        **k
args,
    ) -
 IOProc
ssorOutput:
        ra
s
 NotImp

m

t
dError
    asy
c d
f post_proc
ss_asy
c(
        s

f,
        mod

_output: Asy
cG


rator[tup

[

t, Poo


gR
qu
stOutput]],
        r
qu
st_
d: str | No

 = No

,
        **k
args,
    ) -
 IOProc
ssorOutput:
        # W
 ca
ot guara
t
 outputs ar
 r
tur

d 

 th
 sam
 ord
r th
y 

r

        # f
d to vLLM.
        # L
t's sort th
m by 
d b
for
 post_proc
ss

g
        sort
d_output = sort
d(
            [(
, 
t
m) asy
c for 
, 
t
m 

 mod

_output], k
y=
ambda output: output[0]
        )
        co

ct
d_output = [output[1] for output 

 sort
d_output]
        r
tur
 s

f.post_proc
ss(co

ct
d_output, r
qu
st_
d=r
qu
st_
d, **k
args)
```
Th
 `pars
_data` m
thod 
s us
d for va

dat

g th
 us
r data a
d co
v
rt

g 
t 

to th
 

put 
xp
ct
d by th
 `pr
_proc
ss*` m
thods.
Th
 `m
rg
_samp


g_params` a
d `m
rg
_poo


g_params` m
thods m
rg
 

put `Samp


gParams` or `Poo


gParams` (
f a
y) 

th th
 d
fau
t o

.
Th
 `pr
_proc
ss*` m
thods tak
 th
 va

dat
d p
ug

 

put to g


rat
 vLLM's mod

 prompts for r
gu
ar 

f
r

c
.
Th
 `post_proc
ss*` m
thods tak
 `Poo


gR
qu
stOutput` obj
cts as 

put a
d g


rat
 a custom p
ug

 output.
A
 
xamp

 
mp

m

tat
o
 of a p
ug

 that 

ab

s g


rat

g g
ot
ff 
mag
s 

th th
 Pr
thv
G
ospat
a
MAE mod

 
s ava

ab

 [h
r
](https://g
thub.com/IBM/t
rratorch/tr
/ma

/t
rratorch/v
m/p
ug

s/s
gm

tat
o
). P

as
, a
so r
f
r to our o




 ([
xamp

s/poo


g/p
ug

/pr
thv
_g
ospat
a
_ma
_o




.py](../../
xamp

s/poo


g/p
ug

/pr
thv
_g
ospat
a
_ma
_o




.py)) a
d off



 ([
xamp

s/poo


g/p
ug

/pr
thv
_g
ospat
a
_ma
_
o_proc
ssor.py](../../
xamp

s/poo


g/p
ug

/pr
thv
_g
ospat
a
_ma
_
o_proc
ssor.py)) 

f
r

c
 
xamp

s.
## Us

g a
 IO Proc
ssor p
ug


IO Proc
ssor p
ug

s ar
 
oad
d at 

g


 startup a
d th
r
 ar
 t
o m
thods for sp
c
fy

g th
 
am
 of th
 p
ug

 to b
 
oad
d:
1. V
a vLLM's `E
g


Args`: s
tt

g th
 `
o_proc
ssor_p
ug

` argum

t 

 th
 `E
g


Args` us
d to 


t
a

z
 th
 `Asy
cLLM`. Th
 sam
 ca
 b
 ach

v
d by pass

g th
 `
o_proc
ssor_p
ug

` argum

t to `LLM` 

 off



 mod
, or by pass

g th
 `--
o-proc
ssor-p
ug

` argum

t 

 s
rv

g mod
.
2. V
a th
 mod

 HF co
f
gurat
o
: add

g a
 `
o_proc
ssor_p
ug

` f


d to th
 mod

 co
f
g (co
f
g.jso
).
Th
 ord
r a
so d
t
rm


s m
thod pr
or
ty. 
.
., s
tt

g th
 p
ug

 
am
 v
a `E
g


Args` 


 ov
rr
d
 a
y p
ug

 
am
 sp
c
f

d 

 th
 mod

 HF co
f
g (co
f
g.jso
).
