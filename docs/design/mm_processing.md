# Mu
t
-Moda
 Data Proc
ss

g
To 

ab

 var
ous opt
m
zat
o
s 

 vLLM such as [chu
k
d pr
f

](../co
f
gurat
o
/opt
m
zat
o
.md#chu
k
d-pr
f

) a
d [pr
f
x cach

g](../f
atur
s/automat
c_pr
f
x_cach

g.md), 

 us
 [Bas
Mu
t
Moda
Proc
ssor][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor] to prov
d
 th
 corr
spo
d

c
 b
t


 p
ac
ho
d
r f
atur
 tok

s (
.g. `

mag

`) a
d mu
t
-moda
 

puts (
.g. th
 ra
 

put 
mag
) bas
d o
 th
 outputs of HF proc
ssor.
H
r
 ar
 th
 ma

 f
atur
s of [Bas
Mu
t
Moda
Proc
ssor][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor]:
## Prompt Updat
 D
t
ct
o

O

 of th
 ma

 r
spo
s
b


t

s of HF proc
ssor 
s to updat
 th
 prompt 

th p
ac
ho
d
r tok

s. For 
xamp

:
- I
s
rt f
atur
 p
ac
ho
d
r tok

s (
.g. `

mag



mag

...

mag

`, th
 
umb
r of 
h
ch 
qua
s to th
 f
atur
 s
z
) at th
 start of th
 str

g.
- R
p
ac
 
x
st

g 

put p
ac
ho
d
r tok

s (
.g. `

mag

` for a s

g

 
mag
) 

th f
atur
 p
ac
ho
d
r tok

s (
.g. `

mag



mag

...

mag

`, th
 
umb
r of 
h
ch 
qua
s to th
 f
atur
 s
z
).
Th
 

format
o
 about 
h
ch tok

s hav
 b

 updat
d 
s k
y to f

d

g th
 corr
spo
d

c
 b
t


 p
ac
ho
d
r f
atur
 tok

s a
d mu
t
-moda
 

puts.
I
 vLLM, th
s 

format
o
 
s sp
c
f

d us

g [PromptUpdat
][v
m.mu
t
moda
.proc
ss

g.PromptUpdat
] 

 [_g
t_prompt_updat
s][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._g
t_prompt_updat
s]. W
 ca
 automat
ca
y d
t
ct 
h
th
r HF has updat
d th
 prompt by ch
ck

g th
 
x
st

c
 of th
 updat
d tok

s.
## Tok


z
d Prompt I
puts
To 

ab

 tok


zat
o
 

 a s
parat
 proc
ss, 

 support pass

g 

put tok

 IDs a
o
gs
d
 mu
t
-moda
 data.
### Th
 prob

m
Co
s
d
r that HF proc
ssors fo
o
 th
s
 ma

 st
ps:
1. Tok


z
 th
 t
xt
2. Proc
ss mu
t
-moda
 

puts
3. P
rform prompt updat
s
A
d 

 r
qu
r
 that:
- For t
xt + mu
t
-moda
 

puts, app
y a
 st
ps 1--3.
- For tok


z
d + mu
t
-moda
 

puts, app
y o

y st
ps 2--3.
Ho
 ca
 

 ach

v
 th
s 

thout r

r
t

g HF proc
ssors? W
 ca
 try to ca
 th
 HF proc
ssor s
v
ra
 t
m
s o
 d
ff
r

t 

puts:
- For t
xt + mu
t
-moda
 

puts, s
mp
y ca
 th
 HF proc
ssor d
r
ct
y.
- For tok


z
d + mu
t
-moda
 

puts, ca
 th
 proc
ssor o

y o
 th
 mu
t
-moda
 

puts.
Wh


 HF proc
ssors support t
xt + mu
t
-moda
 

puts 
at
v

y, th
s 
s 
ot so for tok


z
d + mu
t
-moda
 

puts: a
 
rror 
s thro

 
f th
 
umb
r of 

put p
ac
ho
d
r tok

s do 
ot corr
spo
d to th
 
umb
r of mu
t
-moda
 

puts.
Mor
ov
r, s

c
 th
 tok


z
d t
xt has 
ot pass
d through th
 HF proc
ssor, 

 hav
 to app
y St
p 3 by ours

v
s to k
p th
 output tok

s a
d mu
t
-moda
 data co
s
st

t 

th 
ach oth
r.
### Dummy t
xt
W
 
ork arou
d th
 f
rst 
ssu
 by r
qu
r

g 
ach mod

 to d
f


 ho
 to g


rat
 dummy t
xt bas
d o
 th
 
umb
r of mu
t
-moda
 

puts, v
a [g
t_dummy_t
xt][v
m.mu
t
moda
.proc
ss

g.Bas
DummyI
putsBu

d
r.g
t_dummy_t
xt]. Th
s 

ts us g


rat
 dummy t
xt corr
spo
d

g to th
 mu
t
-moda
 

puts a
d 

put th
m tog
th
r to obta

 th
 proc
ss
d mu
t
-moda
 data.
### Automat
c prompt updat

g
W
 addr
ss th
 s
co
d 
ssu
 by 
mp

m

t

g mod

-ag
ost
c cod
 


[_app
y_prompt_updat
s][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._app
y_prompt_updat
s] to automat
ca
y updat
 th
 prompt 

th f
atur
 p
ac
ho
d
r tok

s bas
d o
 th
 sp
c
f
cat
o
 outputt
d by [_g
t_prompt_updat
s][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._g
t_prompt_updat
s].
### Summary
W
th th
 h

p of dummy t
xt a
d automat
c prompt updat

g, our mu
t
-moda
 proc
ssor ca
 f

a
y acc
pt both t
xt a
d tok

 prompts 

th mu
t
-moda
 data. Th
 d
ta


d 
og
c 
s sho

 

 [_app
y_hf_proc
ssor_ma

][v
m.mu
t
moda
.proc
ss

g.Bas
Mu
t
Moda
Proc
ssor._app
y_hf_proc
ssor_ma

].
## Proc
ssor Output Cach

g
Som
 HF proc
ssors, such as th
 o

 for Q


2-VL, ar
 [v
ry s
o
](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/9238). To a

v
at
 th
s prob

m, 

 cach
 th
 mu
t
-moda
 outputs of HF proc
ssor to avo
d proc
ss

g th
 sam
 mu
t
-moda
 

put (
.g. 
mag
) aga

.
Wh

 


 data 
s pass
d 

, 

 f
rst ch
ck 
h
ch 
t
ms ar
 

 th
 cach
, a
d 
h
ch o

s ar
 m
ss

g. Th
 m
ss

g 
t
ms ar
 pass
d 

to th
 HF proc
ssor 

 a s

g

 batch a
d cach
d, b
for
 b


g m
rg
d 

th th
 
x
st

g 
t
ms 

 th
 cach
.
S

c
 

 o

y proc
ss th
 m
ss

g mu
t
-moda
 data 
t
ms, th
 
umb
r of 

put p
ac
ho
d
r tok

s 
o 
o
g
r corr
spo
ds to th
 
umb
r of th
 mu
t
-moda
 

puts, so th
y ca
't b
 pass
d a
o
gs
d
 th
 t
xt prompt to HF proc
ssor. Th
r
for
, 

 proc
ss th
 t
xt a
d mu
t
-moda
 

puts s
parat

y, us

g [dummy t
xt](#dummy-t
xt) to avo
d HF 
rrors. S

c
 th
s sk
ps HF's prompt updat

g cod
, 

 app
y [automat
c prompt updat

g](#automat
c-prompt-updat

g) aft
r
ards to k
p th
 output tok

s a
d mu
t
-moda
 data co
s
st

t 

th 
ach oth
r.
