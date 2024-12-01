import sympy as sp
import math as m

d, H = sp.symbols("d H")

ro = 0.3
pi = m.pi
B = 30
P = 33e3
t = 0.1
E = 3e7
escoamento = 1e5

x1 = d
x2 = H

f = 2 * ro * pi * d * t * sp.sqrt(H**2 + B**2)

df_dx1 = sp.diff(f, x1)
df_dx2 = sp.diff(f, x2)

print(f"Gradiente: {df_dx1}, {df_dx2}")

hessiana = sp.hessian(f, (x1, x2))
print(f"Hessiana: {hessiana}")
