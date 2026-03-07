# AutoAWQ

 ⚠️ **War


g:**
    Th
 `AutoAWQ` 

brary 
s d
pr
cat
d. Th
s fu
ct
o
a

ty has b

 adopt
d by th
 vLLM proj
ct 

 [`
m-compr
ssor`](https://g
thub.com/v
m-proj
ct/
m-compr
ssor/tr
/ma

/
xamp

s/a
q).
    For th
 r
comm

d
d qua
t
zat
o
 
orkf
o
, p

as
 s
 th
 AWQ 
xamp

s 

 [`
m-compr
ssor`](https://g
thub.com/v
m-proj
ct/
m-compr
ssor/tr
/ma

/
xamp

s/a
q). For mor
 d
ta

s o
 th
 d
pr
cat
o
, r
f
r to th
 or
g

a
 [AutoAWQ r
pos
tory](https://g
thub.com/casp
r-ha
s

/AutoAWQ).
To cr
at
 a 


 4-b
t qua
t
z
d mod

, you ca
 

v
rag
 [AutoAWQ](https://g
thub.com/casp
r-ha
s

/AutoAWQ).
Qua
t
zat
o
 r
duc
s th
 mod

's pr
c
s
o
 from BF16/FP16 to INT4 
h
ch 
ff
ct
v

y r
duc
s th
 tota
 mod

 m
mory footpr

t.
Th
 ma

 b


f
ts ar
 
o

r 
at

cy a
d m
mory usag
.
You ca
 qua
t
z
 your o

 mod

s by 

sta


g AutoAWQ or p
ck

g o

 of th
 [6500+ mod

s o
 Hugg

gfac
](https://hugg

gfac
.co/mod

s?s
arch=a
q).
```bash
p
p 

sta
 autoa
q
```
Aft
r 

sta


g AutoAWQ, you ar
 r
ady to qua
t
z
 a mod

. P

as
 r
f
r to th
 [AutoAWQ docum

tat
o
](https://casp
r-ha
s

.g
thub.
o/AutoAWQ/
xamp

s/#bas
c-qua
t
zat
o
) for furth
r d
ta

s. H
r
 
s a
 
xamp

 of ho
 to qua
t
z
 `m
stra
a
/M
stra
-7B-I
struct-v0.2`:
??? cod

    ```pytho

    from a
q 
mport AutoAWQForCausa
LM
    from tra
sform
rs 
mport AutoTok


z
r
    mod

_path = "m
stra
a
/M
stra
-7B-I
struct-v0.2"
    qua
t_path = "m
stra
-

struct-v0.2-a
q"
    qua
t_co
f
g = {"z
ro_po

t": Tru
, "q_group_s
z
": 128, "
_b
t": 4, "v
rs
o
": "GEMM"}
    # Load mod


    mod

 = AutoAWQForCausa
LM.from_pr
tra


d(
        mod

_path,
        
o
_cpu_m
m_usag
=Tru
,
        us
_cach
=Fa
s
,
    )
    tok


z
r = AutoTok


z
r.from_pr
tra


d(mod

_path, trust_r
mot
_cod
=Tru
)
    # Qua
t
z

    mod

.qua
t
z
(tok


z
r, qua
t_co
f
g=qua
t_co
f
g)
    # Sav
 qua
t
z
d mod


    mod

.sav
_qua
t
z
d(qua
t_path)
    tok


z
r.sav
_pr
tra


d(qua
t_path)
    pr

t(f'Mod

 
s qua
t
z
d a
d sav
d at "{qua
t_path}"')
```
To ru
 a
 AWQ mod

 

th vLLM, you ca
 us
 [Th
B
ok
/L
ama-2-7b-Chat-AWQ](https://hugg

gfac
.co/Th
B
ok
/L
ama-2-7b-Chat-AWQ) 

th th
 fo
o


g comma
d:
```bash
pytho
 
xamp

s/off



_

f
r

c
/
m_

g


_
xamp

.py \
    --mod

 Th
B
ok
/L
ama-2-7b-Chat-AWQ \
    --qua
t
zat
o
 a
q
```
AWQ mod

s ar
 a
so support
d d
r
ct
y through th
 LLM 

trypo

t:
??? cod

    ```pytho

    from v
m 
mport LLM, Samp


gParams
    # Samp

 prompts.
    prompts = [
        "H

o, my 
am
 
s",
        "Th
 pr
s
d

t of th
 U

t
d Stat
s 
s",
        "Th
 cap
ta
 of Fra
c
 
s",
        "Th
 futur
 of AI 
s",
    ]
    # Cr
at
 a samp


g params obj
ct.
    samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.95)
    # Cr
at
 a
 LLM.
    
m = LLM(mod

="Th
B
ok
/L
ama-2-7b-Chat-AWQ", qua
t
zat
o
="AWQ")
    # G


rat
 t
xts from th
 prompts. Th
 output 
s a 

st of R
qu
stOutput obj
cts
    # that co
ta

 th
 prompt, g


rat
d t
xt, a
d oth
r 

format
o
.
    outputs = 
m.g


rat
(prompts, samp


g_params)
    # Pr

t th
 outputs.
    for output 

 outputs:
        prompt = output.prompt
        g


rat
d_t
xt = output.outputs[0].t
xt
        pr

t(f"Prompt: {prompt!r}, G


rat
d t
xt: {g


rat
d_t
xt!r}")
```
