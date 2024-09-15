# %%
import sympy as sp
import numpy as np

# %%
x1, x2 = sp.symbols("x1 x2")

f = (11 - x1 - x2) ** 2 + (1 + x1 + 10 * x2 - x1 * x2) ** 2

df_dx1 = sp.diff(f, x1)
df_dx2 = sp.diff(f, x2)

df_dxx = sp.diff(df_dx1, x1)
df_dyy = sp.diff(df_dx2, x2)
df_dxy = sp.diff(df_dx1, x2)
df_dyx = sp.diff(df_dx2, x1)

# %%
# Exemplo de matriz
matriz = np.array([[2, -20], [-20, 2]])

# Verifica se a matriz é simétrica
if np.array_equal(matriz, matriz.T):
    # Calcula os autovalores
    autovalores = np.linalg.eigvals(matriz)
    print(autovalores)
    # Verifica se todos os autovalores são positivos
    if np.all(autovalores > 0):
        print("A matriz é positiva definida.")
    else:
        print("A matriz não é positiva definida.")
else:
    print("A matriz não é simétrica e, portanto, não pode ser positiva definida.")
