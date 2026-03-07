# v
m b

ch mm-proc
ssor
## Ov
rv



`v
m b

ch mm-proc
ssor` prof


s th
 mu
t
moda
 

put proc
ssor p
p




 of
v
s
o
-
a
guag
 mod

s. It m
asur
s p
r-stag
 
at

cy from th
 Hugg

gFac

proc
ssor through to th
 

cod
r for
ard pass, h

p

g you 
d

t
fy
pr
proc
ss

g bott



cks a
d u
d
rsta
d ho
 d
ff
r

t 
mag
 r
so
ut
o
s or

t
m cou
ts aff
ct 

d-to-

d r
qu
st t
m
.
Th
 b

chmark supports t
o data sourc
s: sy
th
t
c ra
dom mu
t
moda
 

puts
(`ra
dom-mm`) a
d Hugg

gFac
 datas
ts (`hf`). Warmup r
qu
sts ar
 ru
 b
for

m
asur
m

t to 

sur
 stab

 r
su
ts.
## Qu
ck Start
```bash
v
m b

ch mm-proc
ssor \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --datas
t-
am
 ra
dom-mm \
  --
um-prompts 50 \
  --ra
dom-

put-


 300 \
  --ra
dom-output-


 40 \
  --ra
dom-mm-bas
-
t
ms-p
r-r
qu
st 2 \
  --ra
dom-mm-

m
t-mm-p
r-prompt '{"
mag
": 3, "v
d
o": 0}' \
  --ra
dom-mm-buck
t-co
f
g '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```
## M
asur
d Stag
s
| Stag
 | D
scr
pt
o
 |
|-------|-------------|
| `g
t_mm_hash
s_s
cs` | T
m
 sp

t hash

g mu
t
moda
 

puts |
| `g
t_cach
_m
ss

g_
t
ms_s
cs` | T
m
 sp

t 
ook

g up th
 proc
ssor cach
 |
| `app
y_hf_proc
ssor_s
cs` | T
m
 sp

t 

 th
 Hugg

gFac
 proc
ssor |
| `m
rg
_mm_k
args_s
cs` | T
m
 sp

t m
rg

g mu
t
moda
 k
args |
| `app
y_prompt_updat
s_s
cs` | T
m
 sp

t updat

g prompt tok

s |
| `pr
proc
ssor_tota
_s
cs` | Tota
 pr
proc
ss

g t
m
 |
| `

cod
r_for
ard_s
cs` | T
m
 sp

t 

 th
 

cod
r mod

 for
ard pass |
| `
um_

cod
r_ca
s` | Numb
r of 

cod
r 

vocat
o
s p
r r
qu
st |
Th
 b

chmark a
so r
ports 

d-to-

d 
at

cy (TTFT + d
cod
 t
m
) p
r
r
qu
st. Us
 `--m
tr
c-p
rc

t


s` to s


ct 
h
ch p
rc

t


s to r
port
(d
fau
t: p99) a
d `--output-jso
` to sav
 r
su
ts.
For mor
 
xamp

s (HF datas
ts, 
armup, JSON output), s

[B

chmark

g CLI — Mu
t
moda
 Proc
ssor B

chmark](../../b

chmark

g/c

.md#mu
t
moda
-proc
ssor-b

chmark).
## JSON CLI Argum

ts
--8
-- "docs/c

/jso
_t
p.

c.md"
## Argum

ts
--8
-- "docs/g


rat
d/argpars
/b

ch_mm_proc
ssor.

c.md"
