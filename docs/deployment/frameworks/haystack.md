# Haystack
[Haystack](https://g
thub.com/d
ps
t-a
/haystack) 
s a
 

d-to-

d LLM fram

ork that a
o
s you to bu

d app

cat
o
s po

r
d by LLMs, Tra
sform
r mod

s, v
ctor s
arch a
d mor
. Wh
th
r you 
a
t to p
rform r
tr

va
-augm

t
d g


rat
o
 (RAG), docum

t s
arch, qu
st
o
 a
s

r

g or a
s

r g


rat
o
, Haystack ca
 orch
strat
 stat
-of-th
-art 
mb
dd

g mod

s a
d LLMs 

to p
p




s to bu

d 

d-to-

d NLP app

cat
o
s a
d so
v
 your us
 cas
.
It a
o
s you to d
p
oy a 
arg
 
a
guag
 mod

 (LLM) s
rv
r 

th vLLM as th
 back

d, 
h
ch 
xpos
s Op

AI-compat
b

 

dpo

ts.
## Pr
r
qu
s
t
s
S
t up th
 vLLM a
d Haystack 

v
ro
m

t:
```bash
p
p 

sta
 v
m haystack-a

```
## D
p
oy
1. Start th
 vLLM s
rv
r 

th th
 support
d chat comp

t
o
 mod

, 
.g.
    ```bash
    v
m s
rv
 m
stra
a
/M
stra
-7B-I
struct-v0.1
    ```
1. Us
 th
 `Op

AIG


rator` a
d `Op

AIChatG


rator` compo


ts 

 Haystack to qu
ry th
 vLLM s
rv
r.
??? cod

    ```pytho

    from haystack.compo


ts.g


rators.chat 
mport Op

AIChatG


rator
    from haystack.datac
ass
s 
mport ChatM
ssag

    from haystack.ut

s 
mport S
cr
t
    g


rator = Op

AIChatG


rator(
        # for compat
b


ty 

th th
 Op

AI API, a p
ac
ho
d
r ap
_k
y 
s 

d
d
        ap
_k
y=S
cr
t.from_tok

("VLLM-PLACEHOLDER-API-KEY"),
        mod

="m
stra
a
/M
stra
-7B-I
struct-v0.1",
        ap
_bas
_ur
="http://{your-vLLM-host-
p}:{your-vLLM-host-port}/v1",
        g


rat
o
_k
args={"max_tok

s": 512},
    )
    r
spo
s
 = g


rator.ru
(
      m
ssag
s=[ChatM
ssag
.from_us
r("H
. Ca
 you h

p m
 p
a
 my 

xt tr
p to Ita
y?")]
    )
    pr

t("-"*30)
    pr

t(r
spo
s
)
    pr

t("-"*30)
    ```
```co
so


------------------------------
{'r
p


s': [ChatM
ssag
(_ro

=
ChatRo

.ASSISTANT: 'ass
sta
t'
, _co
t

t=[T
xtCo
t

t(t
xt=' Of cours
! Wh
r
 

 Ita
y 
ou
d you 

k
 to go a
d 
hat typ
 of tr
p ar
 you 
ook

g to p
a
?')], _
am
=No

, _m
ta={'mod

': 'm
stra
a
/M
stra
-7B-I
struct-v0.1', '

d
x': 0, 'f


sh_r
aso
': 'stop', 'usag
': {'comp

t
o
_tok

s': 23, 'prompt_tok

s': 21, 'tota
_tok

s': 44, 'comp

t
o
_tok

s_d
ta

s': No

, 'prompt_tok

s_d
ta

s': No

}})]}
------------------------------
```
For d
ta

s, s
 th
 tutor
a
 [Us

g vLLM 

 Haystack](https://g
thub.com/d
ps
t-a
/haystack-

t
grat
o
s/b
ob/ma

/

t
grat
o
s/v
m.md).
