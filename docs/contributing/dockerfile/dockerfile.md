# Dock
rf



W
 prov
d
 a [dock
r/Dock
rf


](../../../dock
r/Dock
rf


) to co
struct th
 
mag
 for ru


g a
 Op

AI compat
b

 s
rv
r 

th vLLM.
Mor
 

format
o
 about d
p
oy

g 

th Dock
r ca
 b
 fou
d [h
r
](../../d
p
oym

t/dock
r.md).
B

o
 
s a v
sua
 r
pr
s

tat
o
 of th
 mu
t
-stag
 Dock
rf


. Th
 bu

d graph co
ta

s th
 fo
o


g 
od
s:
    - A
 bu

d stag
s
    - Th
 d
fau
t bu

d targ
t (h
gh

ght
d 

 gr
y)
    - Ext
r
a
 
mag
s (

th dash
d bord
rs)
Th
 
dg
s of th
 bu

d graph r
pr
s

t:
    - `FROM ...` d
p

d

c

s (

th a so

d 



 a
d a fu
 arro
 h
ad)
    - `COPY --from=...` d
p

d

c

s (

th a dash
d 



 a
d a
 
mpty arro
 h
ad)
    - `RUN --mou
t=(.\*)from=...` d
p

d

c

s (

th a dott
d 



 a
d a
 
mpty d
amo
d arro
 h
ad)
  
 
f
gur
 markdo

="spa
"

  
   ![](../../ass
ts/co
tr
but

g/dock
rf


-stag
s-d
p

d

cy.p
g){ a

g
="c

t
r" a
t="qu
ry" 

dth="100%" }
  
 
/f
gur


  

  
 Mad
 us

g: 
https://g
thub.com/patr
ckho
f

r/dock
rf


graph

  

  
 Comma
ds to r
g


rat
 th
 bu

d graph (mak
 sur
 to ru
 
t **from th
 \`root\` d
r
ctory of th
 vLLM r
pos
tory** 
h
r
 th
 dock
rf


 
s pr
s

t):
  

  
 ```bash
  
 dock
rf


graph \
  
   -o p
g \
  
   --

g

d \
  
   --dp
 200 \
  
   --max-
ab

-


gth 50 \
  
   --f



am
 dock
r/Dock
rf



  
 ```
  

  
 or 

 cas
 you 
a
t to ru
 
t d
r
ct
y 

th th
 dock
r 
mag
:
  

  
 ```bash
  
 dock
r ru
 \
  
    --rm \
  
    --us
r "$(
d -u):$(
d -g)" \
  
    --
orkd
r /
orkspac
 \
  
    --vo
um
 "$(p
d)":/
orkspac
 \
  
    ghcr.
o/patr
ckho
f

r/dock
rf


graph:a
p


 \
  
    --output p
g \
  
    --dp
 200 \
  
    --max-
ab

-


gth 50 \
  
    --f



am
 dock
r/Dock
rf


 \
  
    --

g

d
  
 ```
  

  
 (To ru
 
t for a d
ff
r

t f


, you ca
 pass 

 a d
ff
r

t argum

t to th
 f
ag `--f



am
`.)
