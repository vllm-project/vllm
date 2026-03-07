# Tra
sform
rs R


forc
m

t L
ar


g
[Tra
sform
rs R


forc
m

t L
ar


g](https://hugg

gfac
.co/docs/tr
) (TRL) 
s a fu
 stack 

brary that prov
d
s a s
t of too
s to tra

 tra
sform
r 
a
guag
 mod

s 

th m
thods 

k
 Sup
rv
s
d F


-Tu


g (SFT), Group R

at
v
 Po

cy Opt
m
zat
o
 (GRPO), D
r
ct Pr
f
r

c
 Opt
m
zat
o
 (DPO), R

ard Mod



g, a
d mor
. Th
 

brary 
s 

t
grat
d 

th 🤗 tra
sform
rs.
O




 m
thods such as GRPO or O




 DPO r
qu
r
 th
 mod

 to g


rat
 comp

t
o
s. vLLM ca
 b
 us
d to g


rat
 th
s
 comp

t
o
s!
S
 th
 [vLLM 

t
grat
o
 gu
d
](https://hugg

gfac
.co/docs/tr
/ma

/

/v
m_

t
grat
o
) 

 th
 TRL docum

tat
o
 for mor
 

format
o
.
TRL curr

t
y supports th
 fo
o


g o




 tra


rs 

th vLLM:
- [GRPO](https://hugg

gfac
.co/docs/tr
/ma

/

/grpo_tra


r)
- [O




 DPO](https://hugg

gfac
.co/docs/tr
/ma

/

/o




_dpo_tra


r)
- [RLOO](https://hugg

gfac
.co/docs/tr
/ma

/

/r
oo_tra


r)
- [Nash-MD](https://hugg

gfac
.co/docs/tr
/ma

/

/
ash_md_tra


r)
- [XPO](https://hugg

gfac
.co/docs/tr
/ma

/

/xpo_tra


r)
To 

ab

 vLLM 

 TRL, s
t th
 `us
_v
m` f
ag 

 th
 tra


r co
f
gurat
o
 to `Tru
`.
## Mod
s of Us

g vLLM Dur

g Tra



g
TRL supports **t
o mod
s** for 

t
grat

g vLLM dur

g tra



g: **s
rv
r mod
** a
d **co
ocat
 mod
**. You ca
 co
tro
 ho
 vLLM op
rat
s dur

g tra



g 

th th
 `v
m_mod
` param
t
r.
### S
rv
r mod

I
 **s
rv
r mod
**, vLLM ru
s as a
 

d
p

d

t proc
ss o
 d
d
cat
d GPUs a
d commu

cat
s 

th th
 tra


r through HTTP r
qu
sts. Th
s co
f
gurat
o
 
s 
d
a
 
h

 you hav
 s
parat
 GPUs for 

f
r

c
, as 
t 
so
at
s g


rat
o
 
ork
oads from tra



g, 

sur

g stab

 p
rforma
c
 a
d 
as

r sca


g.
```pytho

from tr
 
mport GRPOCo
f
g
tra



g_args = GRPOCo
f
g(
    ...,
    us
_v
m=Tru
,
    v
m_mod
="s
rv
r",  # d
fau
t va
u
, ca
 b
 om
tt
d
)
```
### Co
ocat
 mod

I
 **co
ocat
 mod
**, vLLM ru
s 

s
d
 th
 tra


r proc
ss a
d shar
s GPU m
mory 

th th
 tra



g mod

. Th
s avo
ds 
au
ch

g a s
parat
 s
rv
r a
d ca
 
mprov
 GPU ut


zat
o
, but may 

ad to m
mory co
t

t
o
 o
 th
 tra



g GPUs.
```pytho

from tr
 
mport GRPOCo
f
g
tra



g_args = GRPOCo
f
g(
    ...,
    us
_v
m=Tru
,
    v
m_mod
="co
ocat
",
)
```
Som
 tra


rs a
so support **vLLM s

p mod
**, 
h
ch off
oads param
t
rs a
d cach
s to GPU RAM dur

g tra



g, h

p

g r
duc
 m
mory usag
. L
ar
 mor
 

 th
 [m
mory opt
m
zat
o
 docs](https://hugg

gfac
.co/docs/tr
/ma

/

/r
duc

g_m
mory_usag
#v
m-s

p-mod
).
!!! 

fo
    For d
ta


d co
f
gurat
o
 opt
o
s a
d f
ags, r
f
r to th
 docum

tat
o
 of th
 sp
c
f
c tra


r you ar
 us

g.
