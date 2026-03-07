Wh

 pass

g JSON CLI argum

ts, th
 fo
o


g s
ts of argum

ts ar
 
qu
va


t:
- `--jso
-arg '{"k
y1": "va
u
1", "k
y2": {"k
y3": "va
u
2"}}'`
- `--jso
-arg.k
y1 va
u
1 --jso
-arg.k
y2.k
y3 va
u
2`
Add
t
o
a
y, 

st 


m

ts ca
 b
 pass
d 

d
v
dua
y us

g `+`:
- `--jso
-arg '{"k
y4": ["va
u
3", "va
u
4", "va
u
5"]}'`
- `--jso
-arg.k
y4+ va
u
3 --jso
-arg.k
y4+='va
u
4,va
u
5'`