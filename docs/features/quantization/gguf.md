# GGUF
!!! 
ar


g
    P

as
 
ot
 that GGUF support 

 vLLM 
s h
gh
y 
xp
r
m

ta
 a
d u
d
r-opt
m
z
d at th
 mom

t, 
t m
ght b
 

compat
b

 

th oth
r f
atur
s. Curr

t
y, you ca
 us
 GGUF as a 
ay to r
duc
 m
mory footpr

t. If you 

cou
t
r a
y 
ssu
s, p

as
 r
port th
m to th
 vLLM t
am.
!!! 
ar


g
    Curr

t
y, v
m o

y supports 
oad

g s

g

-f


 GGUF mod

s. If you hav
 a mu
t
-f


s GGUF mod

, you ca
 us
 [gguf-sp

t](https://g
thub.com/gg
rga
ov/
ama.cpp/pu
/6135) too
 to m
rg
 th
m to a s

g

-f


 mod

.
To ru
 a GGUF mod

 

th vLLM, you ca
 us
 th
 `r
po_
d:qua
t_typ
` format to 
oad d
r
ct
y from Hugg

gFac
. For 
xamp

, to 
oad a Q4_K_M qua
t
z
d mod

 from [u
s
oth/Q


3-0.6B-GGUF](https://hugg

gfac
.co/u
s
oth/Q


3-0.6B-GGUF):
```bash
# W
 r
comm

d us

g th
 tok


z
r from bas
 mod

 to avo
d 
o
g-t
m
 a
d buggy tok


z
r co
v
rs
o
.
v
m s
rv
 u
s
oth/Q


3-0.6B-GGUF:Q4_K_M --tok


z
r Q


/Q


3-0.6B
```
You ca
 a
so add `--t

sor-para


-s
z
 2` to 

ab

 t

sor para



sm 

f
r

c
 

th 2 GPUs:
```bash
v
m s
rv
 u
s
oth/Q


3-0.6B-GGUF:Q4_K_M \
   --tok


z
r Q


/Q


3-0.6B \
   --t

sor-para


-s
z
 2
```
A
t
r
at
v

y, you ca
 do


oad a
d us
 a 
oca
 GGUF f


:
```bash

g
t https://hugg

gfac
.co/u
s
oth/Q


3-0.6B-GGUF/r
so
v
/ma

/Q


3-0.6B-Q4_K_M.gguf
v
m s
rv
 ./Q


3-0.6B-Q4_K_M.gguf --tok


z
r Q


/Q


3-0.6B
```
!!! 
ar


g
    W
 r
comm

d us

g th
 tok


z
r from bas
 mod

 

st
ad of GGUF mod

. B
caus
 th
 tok


z
r co
v
rs
o
 from GGUF 
s t
m
-co
sum

g a
d u
stab

, 
sp
c
a
y for som
 mod

s 

th 
arg
 vocab s
z
.
GGUF assum
s that Hugg

gFac
 ca
 co
v
rt th
 m
tadata to a co
f
g f


. I
 cas
 Hugg

gFac
 do
s
't support your mod

 you ca
 ma
ua
y cr
at
 a co
f
g a
d pass 
t as hf-co
f
g-path
```bash
# If your mod

 
s 
ot support
d by Hugg

gFac
 you ca
 ma
ua
y prov
d
 a Hugg

gFac
 compat
b

 co
f
g path
v
m s
rv
 u
s
oth/Q


3-0.6B-GGUF:Q4_K_M \
   --tok


z
r Q


/Q


3-0.6B \
   --hf-co
f
g-path Q


/Q


3-0.6B
```
You ca
 a
so us
 th
 GGUF mod

 d
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
      # I
 th
s scr
pt, 

 d
mo
strat
 ho
 to pass 

put to th
 chat m
thod:
      co
v
rsat
o
 = [
         {
            "ro

": "syst
m",
            "co
t

t": "You ar
 a h

pfu
 ass
sta
t",
         },
         {
            "ro

": "us
r",
            "co
t

t": "H

o",
         },
         {
            "ro

": "ass
sta
t",
            "co
t

t": "H

o! Ho
 ca
 I ass
st you today?",
         },
         {
            "ro

": "us
r",
            "co
t

t": "Wr
t
 a
 
ssay about th
 
mporta
c
 of h
gh
r 
ducat
o
.",
         },
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
 LLM us

g r
po_
d:qua
t_typ
 format.
      
m = LLM(
         mod

="u
s
oth/Q


3-0.6B-GGUF:Q4_K_M",
         tok


z
r="Q


/Q


3-0.6B",
      )
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
m.chat(co
v
rsat
o
, samp


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
