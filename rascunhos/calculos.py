import sympy as sp

x1, x2 = sp.symbols("x1 x2")

f = (x1 - 1) ** 2 + (x2 - 2) ** 2

df_dx1 = sp.diff(f, x1)
df_dx2 = sp.diff(f, x2)

print(f"Gradiente: {df_dx1}, {df_dx2}")
