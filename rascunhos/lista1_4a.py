"""
Programa para plotar o plano de 1 função na direção de p1 -> p2
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Definindo função simbólica e variáveis
x1, x2 = sp.symbols("x1 x2")


f = (
    0.5 * (((12 + x1) ** 2 + x2**2) ** 0.5 - 12) ** 2
    + 5 * (((8 - x1) ** 2 + x2**2) ** 0.5 - 8) ** 2
    - 7 * x2
)

# Definindo pontos e verificando comprimento total
p1 = 2, 13
p2 = 13, -5

dist = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
print(f"Distância de p1 a p2: {dist}")

# Discretizando plano 3D
num_points = 20
x1_points = np.linspace(p1[0], p2[0], num_points)
x2_points = np.linspace(p1[1], p2[1], num_points)

points = np.column_stack((x1_points, x2_points))


# Calculando os valores de f para os pontos
f_lambdified = sp.lambdify((x1, x2), f, "numpy")
f_values = f_lambdified(points[:, 0], points[:, 1])

# Armazenando valores de x no eixo local
alfa_values = np.linspace(0, dist, num_points)

# Plotando os valores de f para os pontos
plt.figure(figsize=(10, 6))
plt.plot(alfa_values, f_values, marker="o")
plt.title("Valores de f para os pontos dados")
plt.xlabel("alfa")
plt.ylabel("f(x, alfa, d)")
plt.grid(True)
plt.show()

# Gerando um domínio para plotar a superfície de f
dominio = 10
x_domain = np.linspace(-dominio, dominio, 100)
y_domain = np.linspace(-dominio, dominio, 100)
X, Y = np.meshgrid(x_domain, y_domain)
Z = f_lambdified(X, Y)

# Plotando a superfície de f
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")

# Configurações adicionais do gráfico
ax.set_title("Superfície da função f(x1, x2)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1, x2)")
plt.show()
