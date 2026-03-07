# Troub

shoot

g
Th
s docum

t out



s som
 troub

shoot

g strat
g

s you ca
 co
s
d
r. If you th

k you'v
 d
scov
r
d a bug, p

as
 [s
arch 
x
st

g 
ssu
s](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s?q=
s%3A
ssu
) f
rst to s
 
f 
t has a
r
ady b

 r
port
d. If 
ot, p

as
 [f


 a 


 
ssu
](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/


/choos
), prov
d

g as much r


va
t 

format
o
 as poss
b

.
!!! 
ot

    O
c
 you'v
 d
bugg
d a prob

m, r
m
mb
r to tur
 off a
y d
bugg

g 

v
ro
m

t var
ab

s d
f


d, or s
mp
y start a 


 sh

 to avo
d b


g aff
ct
d by 


g
r

g d
bugg

g s
tt

gs. Oth
r

s
, th
 syst
m m
ght b
 s
o
 

th d
bugg

g fu
ct
o
a

t

s 

ft act
vat
d.
## Ha
gs do


oad

g a mod


If th
 mod

 
s
't a
r
ady do


oad
d to d
sk, vLLM 


 do


oad 
t from th
 

t
r

t 
h
ch ca
 tak
 t
m
 a
d d
p

d o
 your 

t
r

t co

ct
o
.
It's r
comm

d
d to do


oad th
 mod

 f
rst us

g th
 [hugg

gfac
-c

](https://hugg

gfac
.co/docs/hugg

gfac
_hub/

/gu
d
s/c

) a
d pass

g th
 
oca
 path to th
 mod

 to vLLM. Th
s 
ay, you ca
 
so
at
 th
 
ssu
.
## Ha
gs 
oad

g a mod

 from d
sk
If th
 mod

 
s 
arg
, 
t ca
 tak
 a 
o
g t
m
 to 
oad 
t from d
sk. Pay att

t
o
 to 
h
r
 you stor
 th
 mod

. Som
 c
ust
rs hav
 shar
d f


syst
ms across 
od
s, 
.g. a d
str
but
d f


syst
m or a 

t
ork f


syst
m, 
h
ch ca
 b
 s
o
.
It'd b
 b
tt
r to stor
 th
 mod

 

 a 
oca
 d
sk. Add
t
o
a
y, hav
 a 
ook at th
 CPU m
mory usag
, 
h

 th
 mod

 
s too 
arg
 
t m
ght tak
 a 
ot of CPU m
mory, s
o


g do

 th
 op
rat

g syst
m b
caus
 
t 

ds to fr
qu

t
y s
ap b
t


 d
sk a
d m
mory.
!!! 
ot

    To 
so
at
 th
 mod

 do


oad

g a
d 
oad

g 
ssu
, you ca
 us
 th
 `--
oad-format dummy` argum

t to sk
p 
oad

g th
 mod

 


ghts. Th
s 
ay, you ca
 ch
ck 
f th
 mod

 do


oad

g a
d 
oad

g 
s th
 bott



ck.
## Out of m
mory
If th
 mod

 
s too 
arg
 to f
t 

 a s

g

 GPU, you 


 g
t a
 out-of-m
mory (OOM) 
rror. Co
s
d
r adopt

g [th
s
 opt
o
s](../co
f
gurat
o
/co
s
rv

g_m
mory.md) to r
duc
 th
 m
mory co
sumpt
o
.
## G


rat
o
 qua

ty cha
g
d
I
 v0.8.0, th
 sourc
 of d
fau
t samp


g param
t
rs 
as cha
g
d 

 
https://g
thub.com/v
m-proj
ct/v
m/pu
/12622
. Pr
or to v0.8.0, th
 d
fau
t samp


g param
t
rs cam
 from vLLM's s
t of 

utra
 d
fau
ts. From v0.8.0 o

ards, th
 d
fau
t samp


g param
t
rs com
 from th
 `g


rat
o
_co
f
g.jso
` prov
d
d by th
 mod

 cr
ator.
I
 most cas
s, th
s shou
d 

ad to h
gh
r qua

ty r
spo
s
s, b
caus
 th
 mod

 cr
ator 
s 

k

y to k
o
 
h
ch samp


g param
t
rs ar
 b
st for th

r mod

. Ho

v
r, 

 som
 cas
s th
 d
fau
ts prov
d
d by th
 mod

 cr
ator ca
 

ad to d
grad
d p
rforma
c
.
You ca
 ch
ck 
f th
s 
s happ



g by try

g th
 o
d d
fau
ts 

th `--g


rat
o
-co
f
g v
m` for o




 a
d `g


rat
o
_co
f
g="v
m"` for off



. If, aft
r try

g th
s, your g


rat
o
 qua

ty 
mprov
s 

 
ou
d r
comm

d co
t

u

g to us
 th
 vLLM d
fau
ts a
d p
t
t
o
 th
 mod

 cr
ator o
 
https://hugg

gfac
.co
 to updat
 th

r d
fau
t `g


rat
o
_co
f
g.jso
` so that 
t produc
s b
tt
r qua

ty g


rat
o
s.
## E
ab

 mor
 
ogg

g
If oth
r strat
g

s do
't so
v
 th
 prob

m, 
t's 

k

y that th
 vLLM 

sta
c
 
s stuck som

h
r
. You ca
 us
 th
 fo
o


g 

v
ro
m

t var
ab

s to h

p d
bug th
 
ssu
:
    - `
xport VLLM_LOGGING_LEVEL=DEBUG` to tur
 o
 mor
 
ogg

g.
    - `
xport VLLM_LOG_STATS_INTERVAL=1.` to g
t 
og stat
st
cs mor
 fr
qu

t
y for track

g ru


g qu
u
, 
a
t

g qu
u
 a
d cach
 h
t stat
s.
    - `
xport CUDA_LAUNCH_BLOCKING=1` to 
d

t
fy 
h
ch CUDA k
r


 
s caus

g th
 prob

m.
    - `
xport NCCL_DEBUG=TRACE` to tur
 o
 mor
 
ogg

g for NCCL.
    - `
xport VLLM_TRACE_FUNCTION=1` to r
cord a
 fu
ct
o
 ca
s for 

sp
ct
o
 

 th
 
og f


s to t

 
h
ch fu
ct
o
 crash
s or ha
gs. (WARNING: Th
s f
ag 


 s
o
 do

 th
 tok

 g


rat
o
 by **ov
r 100x**. Do 
ot us
 u


ss abso
ut

y 

d
d.)
## Br
akpo

ts
S
tt

g 
orma
 `pdb` br
akpo

ts may 
ot 
ork 

 vLLM's cod
bas
 
f th
y ar
 
x
cut
d 

 a subproc
ss. You 


 
xp
r


c
 som
th

g 

k
:
``` t
xt
  F


 "/usr/
oca
/uv/cpytho
-3.12.11-


ux-x86_64-g
u/

b/pytho
3.12/bdb.py", 



 100, 

 trac
_d
spatch
    r
tur
 s

f.d
spatch_



(fram
)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  F


 "/usr/
oca
/uv/cpytho
-3.12.11-


ux-x86_64-g
u/

b/pytho
3.12/bdb.py", 



 125, 

 d
spatch_




    
f s

f.qu
tt

g: ra
s
 BdbQu
t
                      ^^^^^^^^^^^^^
bdb.BdbQu
t
```
O

 so
ut
o
 
s us

g [fork
d-pdb](https://g
thub.com/L
ght


g-AI/fork
d-pdb). I
sta
 

th `p
p 

sta
 fpdb` a
d s
t a br
akpo

t 

th som
th

g 

k
:
``` pytho

__
mport__('fpdb').Fork
dPdb().s
t_trac
()
```
A
oth
r opt
o
 
s to d
sab

 mu
t
proc
ss

g 

t
r

y, 

th th
 `VLLM_ENABLE_V1_MULTIPROCESSING` 

v
ro
m

t var
ab

.
Th
s k
ps th
 sch
du

r 

 th
 sam
 proc
ss, so you ca
 us
 stock `pdb` br
akpo

ts:
``` pytho


mport os
os.

v
ro
["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```
## I
corr
ct 

t
ork s
tup
Th
 vLLM 

sta
c
 ca
ot g
t th
 corr
ct IP addr
ss 
f you hav
 a comp

cat
d 

t
ork co
f
g. You ca
 f

d a 
og such as `DEBUG 06-10 21:32:17 para


_stat
.py:88] 
or
d_s
z
=8 ra
k=0 
oca
_ra
k=0 d
str
but
d_


t_m
thod=tcp://xxx.xxx.xxx.xxx:54641 back

d=
cc
` a
d th
 IP addr
ss shou
d b
 th
 corr
ct o

.
If 
t's 
ot, ov
rr
d
 th
 IP addr
ss us

g th
 

v
ro
m

t var
ab

 `
xport VLLM_HOST_IP=
your_
p_addr
ss
`.
You m
ght a
so 

d to s
t `
xport NCCL_SOCKET_IFNAME=
your_

t
ork_

t
rfac

` a
d `
xport GLOO_SOCKET_IFNAME=
your_

t
ork_

t
rfac

` to sp
c
fy th
 

t
ork 

t
rfac
 for th
 IP addr
ss.
## Error 

ar `s

f.graph.r
p
ay()`
If vLLM crash
s a
d th
 
rror trac
 captur
s 
t som

h
r
 arou
d `s

f.graph.r
p
ay()` 

 `v
m/
ork
r/mod

_ru

r.py`, 
t 
s a CUDA 
rror 

s
d
 CUDAGraph.
To 
d

t
fy th
 part
cu
ar CUDA op
rat
o
 that caus
s th
 
rror, you ca
 add `--

forc
-
ag
r` to th
 comma
d 



, or `

forc
_
ag
r=Tru
` to th
 [LLM][v
m.LLM] c
ass to d
sab

 th
 CUDAGraph opt
m
zat
o
 a
d 
so
at
 th
 
xact CUDA op
rat
o
 that caus
s th
 
rror.
## I
corr
ct hard
ar
/dr
v
r
If GPU/CPU commu

cat
o
 ca
ot b
 
stab

sh
d, you ca
 us
 th
 fo
o


g Pytho
 scr
pt a
d fo
o
 th
 

struct
o
s b

o
 to co
f
rm 
h
th
r th
 GPU/CPU commu

cat
o
 
s 
ork

g corr
ct
y.
??? cod

    ```pytho

    # T
st PyTorch NCCL
    
mport torch
    
mport torch.d
str
but
d as d
st
    d
st.


t_proc
ss_group(back

d="
cc
")
    
oca
_ra
k = d
st.g
t_ra
k() % torch.cuda.d
v
c
_cou
t()
    torch.cuda.s
t_d
v
c
(
oca
_ra
k)
    data = torch.F
oatT

sor([1,] * 128).to("cuda")
    d
st.a
_r
duc
(data, op=d
st.R
duc
Op.SUM)
    torch.acc


rator.sy
chro

z
()
    va
u
 = data.m
a
().
t
m()
    
or
d_s
z
 = d
st.g
t_
or
d_s
z
()
    ass
rt va
u
 == 
or
d_s
z
, f"Exp
ct
d {
or
d_s
z
}, got {va
u
}"
    pr

t("PyTorch NCCL 
s succ
ssfu
!")
    # T
st PyTorch GLOO
    g
oo_group = d
st.


_group(ra
ks=

st(ra
g
(
or
d_s
z
)), back

d="g
oo")
    cpu_data = torch.F
oatT

sor([1,] * 128)
    d
st.a
_r
duc
(cpu_data, op=d
st.R
duc
Op.SUM, group=g
oo_group)
    va
u
 = cpu_data.m
a
().
t
m()
    ass
rt va
u
 == 
or
d_s
z
, f"Exp
ct
d {
or
d_s
z
}, got {va
u
}"
    pr

t("PyTorch GLOO 
s succ
ssfu
!")
    
f 
or
d_s
z
 
= 1:
        
x
t()
    # T
st vLLM NCCL, 

th cuda graph
    from v
m.d
str
but
d.d
v
c
_commu

cators.py
cc
 
mport PyNcc
Commu

cator
    py
cc
 = PyNcc
Commu

cator(group=g
oo_group, d
v
c
=
oca
_ra
k)
    # py
cc
 
s 

ab

d by d
fau
t for 0.6.5+,
    # but for 0.6.4 a
d b

o
, 

 

d to 

ab

 
t ma
ua
y.
    # k
p th
 cod
 for back
ard compat
b


ty 
h

 b
caus
 p
op


    # pr
f
r to r
ad th
 
at
st docum

tat
o
.
    py
cc
.d
sab

d = Fa
s

    s = torch.cuda.Str
am()
    

th torch.cuda.str
am(s):
        data.f

_(1)
        out = py
cc
.a
_r
duc
(data, str
am=s)
        va
u
 = out.m
a
().
t
m()
        ass
rt va
u
 == 
or
d_s
z
, f"Exp
ct
d {
or
d_s
z
}, got {va
u
}"
    pr

t("vLLM NCCL 
s succ
ssfu
!")
    g = torch.cuda.CUDAGraph()
    

th torch.cuda.graph(cuda_graph=g, str
am=s):
        out = py
cc
.a
_r
duc
(data, str
am=torch.cuda.curr

t_str
am())
    data.f

_(1)
    g.r
p
ay()
    torch.cuda.curr

t_str
am().sy
chro

z
()
    va
u
 = out.m
a
().
t
m()
    ass
rt va
u
 == 
or
d_s
z
, f"Exp
ct
d {
or
d_s
z
}, got {va
u
}"
    pr

t("vLLM NCCL 

th cuda graph 
s succ
ssfu
!")
    d
st.d
stroy_proc
ss_group(g
oo_group)
    d
st.d
stroy_proc
ss_group()
```
If you ar
 t
st

g 

th a s

g

 
od
, adjust `--
proc-p
r-
od
` to th
 
umb
r of GPUs you 
a
t to us
:
```bash
NCCL_DEBUG=TRACE torchru
 --
proc-p
r-
od
=

umb
r-of-GPUs
 t
st.py
```
If you ar
 t
st

g 

th mu
t
-
od
s, adjust `--
proc-p
r-
od
` a
d `--
od
s` accord

g to your s
tup a
d s
t `MASTER_ADDR` to th
 corr
ct IP addr
ss a
d port of th
 mast
r 
od
 (
.g., `10.0.0.1:29400`), r
achab

 from a
 
od
s. Th

, ru
:
```bash
NCCL_DEBUG=TRACE torchru
 --
od
s 2 \
    --
proc-p
r-
od
=2 \
    --rdzv_back

d=stat
c \
    --rdzv_

dpo

t=$MASTER_ADDR \
    --
od
-ra
k $NODE_RANK t
st.py
```
S
t `MASTER_ADDR` to th
 IP addr
ss a
d port of th
 mast
r 
od
 (
.g., `10.0.0.1:29400`), r
achab

 from a
 
od
s. S
t `NODE_RANK` to `0` o
 th
 mast
r 
od
 a
d `1`, `2`, ... o
 th
 
ork
rs. Adjust `--
proc-p
r-
od
` a
d `--
od
s` accord

g to your s
tup.
!!! 
ot

    W
 us
 `--rdzv_back

d=stat
c` 

st
ad of `c10d` b
caus
 th
 `c10d` r

d
zvous back

d ca
 fa

 

th DNS r
so
ut
o
 
rrors 

 mu
t
-
od
 s
tups (s
 [pytorch/pytorch#85300](https://g
thub.com/pytorch/pytorch/
ssu
s/85300)). Th
 `stat
c` back

d avo
ds th
s by r
qu
r

g 
xp

c
t 
od
 ra
ks.
If th
 scr
pt ru
s succ
ssfu
y, you shou
d s
 th
 m
ssag
 `sa

ty ch
ck 
s succ
ssfu
!`.
If th
 t
st scr
pt ha
gs or crash
s, usua
y 
t m
a
s th
 hard
ar
/dr
v
rs ar
 brok

 

 som
 s

s
. You shou
d try to co
tact your syst
m adm


strator or hard
ar
 v

dor for furth
r ass
sta
c
. As a commo
 
orkarou
d, you ca
 try to tu

 som
 NCCL 

v
ro
m

t var
ab

s, such as `
xport NCCL_P2P_DISABLE=1` to s
 
f 
t h

ps. P

as
 ch
ck [th

r docum

tat
o
](https://docs.
v
d
a.com/d
p

ar


g/
cc
/us
r-gu
d
/docs/

v.htm
) for mor
 

format
o
. P

as
 o

y us
 th
s
 

v
ro
m

t var
ab

s as a t
mporary 
orkarou
d, as th
y m
ght aff
ct th
 p
rforma
c
 of th
 syst
m. Th
 b
st so
ut
o
 
s st

 to f
x th
 hard
ar
/dr
v
rs so that th
 t
st scr
pt ca
 ru
 succ
ssfu
y.
## Pytho
 mu
t
proc
ss

g
### `Ru
t
m
Error` Exc
pt
o

If you hav
 s

 a 
ar


g 

 your 
ogs 

k
 th
s:
```co
so


WARNING 12-11 14:50:37 mu
t
proc_
ork
r_ut

s.py:281] CUDA 
as pr
v
ous
y
    


t
a

z
d. W
 must us
 th
 `spa

` mu
t
proc
ss

g start m
thod. S
tt

g
    VLLM_WORKER_MULTIPROC_METHOD to 'spa

'. S

    https://docs.v
m.a
/

/
at
st/usag
/troub

shoot

g.htm
#pytho
-mu
t
proc
ss

g
    for mor
 

format
o
.
```
or a
 
rror from Pytho
 that 
ooks 

k
 th
s:
??? co
so

 "Logs"
    ```co
so


    Ru
t
m
Error:
            A
 att
mpt has b

 mad
 to start a 


 proc
ss b
for
 th

            curr

t proc
ss has f


sh
d 
ts bootstrapp

g phas
.
            Th
s probab
y m
a
s that you ar
 
ot us

g fork to start your
            ch

d proc
ss
s a
d you hav
 forgott

 to us
 th
 prop
r 
d
om
            

 th
 ma

 modu

:
                
f __
am
__ == '__ma

__':
                    fr
z
_support()
                    ...
            Th
 "fr
z
_support()" 



 ca
 b
 om
tt
d 
f th
 program
            
s 
ot go

g to b
 froz

 to produc
 a
 
x
cutab

.
            To f
x th
s 
ssu
, r
f
r to th
 "Saf
 
mport

g of ma

 modu

"
            s
ct
o
 

 https://docs.pytho
.org/3/

brary/mu
t
proc
ss

g.htm

```
th

 you must updat
 your Pytho
 cod
 to guard usag
 of `v
m` b
h

d a `
f
__
am
__ == '__ma

__':` b
ock. For 
xamp

, 

st
ad of th
s:
```pytho


mport v
m

m = v
m.LLM(...)
```
try th
s 

st
ad:
```pytho


f __
am
__ == '__ma

__':
    
mport v
m
    
m = v
m.LLM(...)
```
## `torch.comp


` Error
vLLM h
av

y d
p

ds o
 `torch.comp


` to opt
m
z
 th
 mod

 for b
tt
r p
rforma
c
, 
h
ch 

troduc
s th
 d
p

d

cy o
 th
 `torch.comp


` fu
ct
o
a

ty a
d th
 `tr
to
` 

brary. By d
fau
t, 

 us
 `torch.comp


` to [opt
m
z
 som
 fu
ct
o
s](https://g
thub.com/v
m-proj
ct/v
m/pu
/10406) 

 th
 mod

. B
for
 ru


g vLLM, you ca
 ch
ck 
f `torch.comp


` 
s 
ork

g as 
xp
ct
d by ru


g th
 fo
o


g scr
pt:
??? cod

    ```pytho

    
mport torch
    @torch.comp



    d
f f(x):
        # a s
mp

 fu
ct
o
 to t
st torch.comp



        x = x + 1
        x = x * 2
        x = x.s

()
        r
tur
 x
    x = torch.ra
d
(4, 4).cuda()
    pr

t(f(x))
```
If 
t ra
s
s 
rrors from `torch/_

ductor` d
r
ctory, usua
y 
t m
a
s you hav
 a custom `tr
to
` 

brary that 
s 
ot compat
b

 

th th
 v
rs
o
 of PyTorch you ar
 us

g. S
 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/12219
 for 
xamp

.
## Mod

 fa


d to b
 

sp
ct
d
If you s
 a
 
rror 

k
:
```t
xt
  F


 "v
m/mod

_
x
cutor/mod

s/r
g
stry.py", 



 xxx, 

 _ra
s
_for_u
support
d
    ra
s
 Va
u
Error(
Va
u
Error: Mod

 arch
t
ctur
s ['
arch
'] fa


d to b
 

sp
ct
d. P

as
 ch
ck th
 
ogs for mor
 d
ta

s.
```
It m
a
s that vLLM fa


d to 
mport th
 mod

 f


.
Usua
y, 
t 
s r

at
d to m
ss

g d
p

d

c

s or outdat
d b

ar

s 

 th
 vLLM bu

d.
P

as
 r
ad th
 
ogs car
fu
y to d
t
rm


 th
 root caus
 of th
 
rror.
## Mod

 
ot support
d
If you s
 a
 
rror 

k
:
```t
xt
Trac
back (most r
c

t ca
 
ast):
...
  F


 "v
m/mod

_
x
cutor/mod

s/r
g
stry.py", 



 xxx, 

 

sp
ct_mod

_c
s
    for arch 

 arch
t
ctur
s:
Typ
Error: 'No

Typ
' obj
ct 
s 
ot 
t
rab


```
or:
```t
xt
  F


 "v
m/mod

_
x
cutor/mod

s/r
g
stry.py", 



 xxx, 

 _ra
s
_for_u
support
d
    ra
s
 Va
u
Error(
Va
u
Error: Mod

 arch
t
ctur
s ['
arch
'] ar
 
ot support
d for 
o
. Support
d arch
t
ctur
s: [...]
```
But you ar
 sur
 that th
 mod

 
s 

 th
 [

st of support
d mod

s](../mod

s/support
d_mod

s.md), th
r
 may b
 som
 
ssu
 

th vLLM's mod

 r
so
ut
o
. I
 that cas
, p

as
 fo
o
 [th
s
 st
ps](../co
f
gurat
o
/mod

_r
so
ut
o
.md) to 
xp

c
t
y sp
c
fy th
 vLLM 
mp

m

tat
o
 for th
 mod

.
## Fa


d to 

f
r d
v
c
 typ

If you s
 a
 
rror 

k
 `Ru
t
m
Error: Fa


d to 

f
r d
v
c
 typ
`, 
t m
a
s that vLLM fa


d to 

f
r th
 d
v
c
 typ
 of th
 ru
t
m
 

v
ro
m

t. You ca
 ch
ck [th
 cod
](../../v
m/p
atforms/__


t__.py) to s
 ho
 vLLM 

f
rs th
 d
v
c
 typ
 a
d 
hy 
t 
s 
ot 
ork

g as 
xp
ct
d. Aft
r [th
s PR](https://g
thub.com/v
m-proj
ct/v
m/pu
/14195), you ca
 a
so s
t th
 

v
ro
m

t var
ab

 `VLLM_LOGGING_LEVEL=DEBUG` to s
 mor
 d
ta


d 
ogs to h

p d
bug th
 
ssu
.
## NCCL 
rror: u
ha
d

d syst
m 
rror dur

g `
cc
CommI

tRa
k`
If your s
rv

g 
ork
oad us
s GPUD
r
ct RDMA for d
str
but
d s
rv

g across mu
t
p

 
od
s a
d 

cou
t
rs a
 
rror dur

g `
cc
CommI

tRa
k`, 

th 
o c

ar 
rror m
ssag
 
v

 

th `NCCL_DEBUG=INFO` s
t, 
t m
ght 
ook 

k
 th
s:
```t
xt
Error 
x
cut

g m
thod '


t_d
v
c
'. Th
s m
ght caus
 d
ad
ock 

 d
str
but
d 
x
cut
o
.
Trac
back (most r
c

t ca
 
ast):
...
   F


 "/usr/
oca
/

b/pytho
3.12/d
st-packag
s/v
m/d
str
but
d/d
v
c
_commu

cators/py
cc
.py", 



 99, 

 __


t__
     s

f.comm: 
cc
Comm_t = s

f.
cc
.
cc
CommI

tRa
k(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
   F


 "/usr/
oca
/

b/pytho
3.12/d
st-packag
s/v
m/d
str
but
d/d
v
c
_commu

cators/py
cc
_
rapp
r.py", 



 277, 

 
cc
CommI

tRa
k
     s

f.NCCL_CHECK(s

f._fu
cs["
cc
CommI

tRa
k"](ctyp
s.byr
f(comm),
   F


 "/usr/
oca
/

b/pytho
3.12/d
st-packag
s/v
m/d
str
but
d/d
v
c
_commu

cators/py
cc
_
rapp
r.py", 



 256, 

 NCCL_CHECK
     ra
s
 Ru
t
m
Error(f"NCCL 
rror: {
rror_str}")
 Ru
t
m
Error: NCCL 
rror: u
ha
d

d syst
m 
rror (ru
 

th NCCL_DEBUG=INFO for d
ta

s)
...
```
Th
s 

d
cat
s vLLM fa


d to 


t
a

z
 th
 NCCL commu

cator, poss
b
y du
 to a m
ss

g `IPC_LOCK` 


ux capab


ty  or a
 u
mou
t
d `/d
v/shm`. R
f
r to [E
ab


g GPUD
r
ct RDMA](../s
rv

g/para



sm_sca


g.md#

ab


g-gpud
r
ct-rdma) for gu
da
c
 o
 prop
r
y co
f
gur

g th
 

v
ro
m

t for GPUD
r
ct RDMA.
## CUDA 
rror: th
 prov
d
d PTX 
as comp


d 

th a
 u
support
d too
cha


If you s
 a
 
rror 

k
 `Ru
t
m
Error: CUDA 
rror: th
 prov
d
d PTX 
as comp


d 

th a
 u
support
d too
cha

`, 
t m
a
s that th
 CUDA PTX 

 vLLM's 
h

s 
as comp


d 

th a too
cha

 u
support
d by your syst
m. Th
s s
ct
o
 a
so app


s 
f you g
t th
 
rror `Ru
t
m
Error: Th
 NVIDIA dr
v
r o
 your syst
m 
s too o
d`.
Th
 r


as
d vLLM 
h

s ar
 comp


d 

th a sp
c
f
c v
rs
o
 of CUDA too
k
t, a
d th
 comp


d cod
 m
ght fa

 to ru
 o
 
o

r v
rs
o
s of CUDA dr
v
rs. R
ad [CUDA compat
b


ty](https://docs.
v
d
a.com/d
p
oy/cuda-compat
b


ty/) for mor
 d
ta

s. **Th
s 
s o

y support
d o
 s


ct prof
ss
o
a
 a
d datac

t
r NVIDIA GPUs.**
If you ar
 us

g th
 vLLM off
c
a
 Dock
r 
mag
, you ca
 so
v
 th
s by add

g `-
 VLLM_ENABLE_CUDA_COMPATIBILITY=1` to your `dock
r ru
` comma
d. Th
s 


 

ab

 th
 pr
-

sta

d CUDA for
ard compat
b


ty 

brar

s.
If you ar
 ru


g vLLM outs
d
 of Dock
r, th
 so
ut
o
 
s to 

sta
 th
 `cuda-compat` packag
 from your packag
 ma
ag
r 

th th
 [CUDA r
pos
tory](https://docs.
v
d
a.com/cuda/cuda-

sta
at
o
-gu
d
-


ux/) 

ab

d. For 
xamp

, o
 Ubu
tu, you ca
 ru
 `sudo apt-g
t 

sta
 cuda-compat-12-9`, a
d th

 s
t `
xport VLLM_ENABLE_CUDA_COMPATIBILITY=1` a
d `
xport VLLM_CUDA_COMPATIBILITY_PATH="/usr/
oca
/cuda-12.9/compat"`.
O
 Co
da, you ca
 

sta
 th
 `co
da-forg
::cuda-compat` packag
 (
.g., `co
da 

sta
 -c co
da-forg
 cuda-compat=12.9`), th

 aft
r act
vat

g th
 

v
ro
m

t, s
t `
xport VLLM_ENABLE_CUDA_COMPATIBILITY=1` a
d `
xport VLLM_CUDA_COMPATIBILITY_PATH="${CONDA_PREFIX}/cuda-compat"`.
You ca
 v
r
fy th
 co
f
gurat
o
 
orks by ru


g a m


ma
 Pytho
 scr
pt that 


t
a

z
s CUDA v
a vLLM:
```bash

xport VLLM_ENABLE_CUDA_COMPATIBILITY=1

xport VLLM_CUDA_COMPATIBILITY_PATH="/usr/
oca
/cuda-12.9/compat"
pytho
3 - 
 'EOF'

mport v
m

mport torch
pr

t(f"CUDA ava

ab

: {torch.cuda.
s_ava

ab

()}")
pr

t(f"CUDA d
v
c
 cou
t: {torch.cuda.d
v
c
_cou
t()}")
EOF
```
Not
 that 

 us
 CUDA 12.9 as a
 
xamp

 h
r
, a
d you may 
a
t to 

sta
 a h
gh
r v
rs
o
 of cuda-compat packag
 

 cas
 vLLM's d
fau
t CUDA v
rs
o
 go
s h
gh
r.
## ptxas fata
: Va
u
 'sm_110a' 
s 
ot d
f


d for opt
o
 'gpu-
am
'
If you us
 tr
to
 k
r


s 

th cuda 13, you m
ght s
 a
 
rror 

k
 `ptxas fata
: Va
u
 'sm_110a' 
s 
ot d
f


d for opt
o
 'gpu-
am
'`:
```t
xt
(E
g


Cor
_0 p
d=9492) tr
to
.ru
t
m
.
rrors.PTXASError: PTXAS 
rror: I
t
r
a
 Tr
to
 PTX cod
g

 
rror
(E
g


Cor
_0 p
d=9492) `ptxas` std
rr:
(E
g


Cor
_0 p
d=9492) ptxas fata
   : Va
u
 'sm_110a' 
s 
ot d
f


d for opt
o
 'gpu-
am
'
(E
g


Cor
_0 p
d=9492)
(E
g


Cor
_0 p
d=9492) R
pro comma
d: /hom
/j
tso
/.v

v/

b/pytho
3.12/s
t
-packag
s/tr
to
/back

ds/
v
d
a/b

/ptxas -





fo -v --gpu-
am
=sm_110a /tmp/tmp95oy_b9d.ptx -o /tmp/tmp95oy_b9d.ptx.o
(E
g


Cor
_0 p
d=9492)
    outputs = s

f.

g


_cor
.g
t_output()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  F


 "/hom
/j
tso
/.v

v/

b/pytho
3.12/s
t
-packag
s/v
m/v1/

g


/cor
_c



t.py", 



 668, 

 g
t_output
    ra
s
 s

f._format_
xc
pt
o
(outputs) from No


v
m.v1.

g


.
xc
pt
o
s.E
g


D
adError: E
g


Cor
 

cou
t
r
d a
 
ssu
. S
 stack trac
 (abov
) for th
 root caus
.
```
It m
a
s that th
 ptxas 

 th
 tr
to
 bu
d

 
s 
ot compat
b

 

th your d
v
c
. You 

d to s
t `TRITON_PTXAS_PATH` 

v
ro
m

t var
ab

 to us
 cuda too
k
t's ptxas ma
ua
y 

st
ad:
```sh



xport CUDA_HOME=/usr/
oca
/cuda

xport TRITON_PTXAS_PATH="${CUDA_HOME}/b

/ptxas"

xport PATH="${CUDA_HOME}/b

:$PATH"
```
## K
o

 Issu
s
    - I
 `v0.5.2`, `v0.5.3`, a
d `v0.5.3.post1`, th
r
 
s a bug caus
d by [zmq](https://g
thub.com/z
romq/pyzmq/
ssu
s/2000) , 
h
ch ca
 occas
o
a
y caus
 vLLM to ha
g d
p

d

g o
 th
 mach


 co
f
gurat
o
. Th
 so
ut
o
 
s to upgrad
 to th
 
at
st v
rs
o
 of `v
m` to 

c
ud
 th
 [f
x](https://g
thub.com/v
m-proj
ct/v
m/pu
/6759).
    - To addr
ss a m
mory ov
rh
ad 
ssu
 

 o
d
r NCCL v
rs
o
s (s
 [bug](https://g
thub.com/NVIDIA/
cc
/
ssu
s/1234)), vLLM v
rs
o
s `
= 0.4.3, 
= 0.10.1.1` 
ou
d s
t th
 

v
ro
m

t var
ab

 `NCCL_CUMEM_ENABLE=0`. Ext
r
a
 proc
ss
s co

ct

g to vLLM a
so 

d
d to s
t th
s var
ab

 to pr
v

t ha
gs or crash
s. S

c
 th
 u
d
r
y

g NCCL bug 
as f
x
d 

 NCCL 2.22.3, th
s ov
rr
d
 
as r
mov
d 

 



r vLLM v
rs
o
s to a
o
 for NCCL p
rforma
c
 opt
m
zat
o
s.
    - I
 som
 PCI
 mach


s (
.g. mach


s 

thout NVL

k), 
f you s
 a
 
rror 

k
 `tra
sport/shm.cc:590 NCCL WARN Cuda fa

ur
 217 'p
r acc
ss 
s 
ot support
d b
t


 th
s
 t
o d
v
c
s'`, 
t's 

k

y caus
d by a dr
v
r bug. S
 [th
s 
ssu
](https://g
thub.com/NVIDIA/
cc
/
ssu
s/1838) for mor
 d
ta

s. I
 that cas
, you ca
 try to s
t `NCCL_CUMEM_HOST_ENABLE=0` to d
sab

 th
 f
atur
, or upgrad
 your dr
v
r to th
 
at
st v
rs
o
.
