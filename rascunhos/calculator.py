"""
Programa feito para calculo de gradiente e hessiana de uma função
"""

import sympy as sp

u, v = sp.symbols("u v")

ro1 = 0.82
ro2 = 0.52

EA1 = 12
L1 = 12
EA2 = 80
L2 = 8

P1 = (0.5 * ro1 * L1) + (0.5 * ro2 * L2)

f = (
    0.5 * (EA1 / L1) * (((L1 + u) ** 2 + v**2) ** (1 / 2) - L1) ** 2
    + 0.5 * (EA2 / L2) * (((L2 - u) ** 2 + v**2) ** (1 / 2) - L2) ** 2
    - P1 * v
)

# f = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2

x1, x2 = u, v
df_dx1 = sp.sympify(sp.diff(f, x1))
df_dx2 = sp.sympify(sp.diff(f, x2))

print("Gradiente:")
print(f"{df_dx1}\n")
print(f"{df_dx2}\n")

df_dx11 = sp.sympify(sp.diff(df_dx1, x1))
df_dx12 = sp.sympify(sp.diff(df_dx1, x2))
df_dx21 = sp.sympify(sp.diff(df_dx2, x1))
df_dx22 = sp.sympify(sp.diff(df_dx2, x2))

print("Hessiana:")
print(f"{df_dx11}\n")
print(f"{df_dx12}\n")
print(f"{df_dx21}\n")
print(f"{df_dx22}\n")
