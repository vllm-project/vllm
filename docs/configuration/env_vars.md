# E
v
ro
m

t Var
ab

s
vLLM us
s th
 fo
o


g 

v
ro
m

t var
ab

s to co
f
gur
 th
 syst
m:
!!! 
ar


g
    P

as
 
ot
 that `VLLM_PORT` a
d `VLLM_HOST_IP` s
t th
 port a
d 
p for vLLM's **

t
r
a
 usag
**. It 
s 
ot th
 port a
d 
p for th
 API s
rv
r. If you us
 `--host $VLLM_HOST_IP` a
d `--port $VLLM_PORT` to start th
 API s
rv
r, 
t 


 
ot 
ork.
    A
 

v
ro
m

t var
ab

s us
d by vLLM ar
 pr
f
x
d 

th `VLLM_`. **Sp
c
a
 car
 shou
d b
 tak

 for Kub
r

t
s us
rs**: p

as
 do 
ot 
am
 th
 s
rv
c
 as `v
m`, oth
r

s
 

v
ro
m

t var
ab

s s
t by Kub
r

t
s m
ght co
f

ct 

th vLLM's 

v
ro
m

t var
ab

s, b
caus
 [Kub
r

t
s s
ts 

v
ro
m

t var
ab

s for 
ach s
rv
c
 

th th
 cap
ta

z
d s
rv
c
 
am
 as th
 pr
f
x](https://kub
r

t
s.
o/docs/co
c
pts/s
rv
c
s-

t
ork

g/s
rv
c
/#

v
ro
m

t-var
ab

s).
```pytho

--8
-- "v
m/

vs.py:

v-vars-d
f


t
o
"
```
