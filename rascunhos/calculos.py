import sympy as sp
import math

x1, x2 = sp.symbols("x1 x2")

d = x1
H = x2
# ro = sp.symbols("ro")
# B = sp.symbols("B")
# P = sp.symbols("P")
# t = sp.symbols("t")
# E = sp.symbols("E")
# sigma_y = sp.symbols("sigma_y")
# pi = sp.symbols("pi")

pi = math.pi
P = 33e3
E = 3e7
sigma_y = 1e5
ro = 0.3
B = 30.0
t = 0.1


f = P * (H**2 + B**2) ** (1 / 2) / (pi * d * t * H) - pi**2 * E * (d**2 + t**2) / (
    8 * (H**2 + B**2)
)

df_dx1 = sp.diff(f, x1)
df_dx2 = sp.diff(f, x2)

print(f"Gradiente: {df_dx1}, {df_dx2}")

hessiana = sp.hessian(f, (x1, x2))
print(f"Hessiana: {hessiana}")
