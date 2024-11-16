import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from optimization.linear import make_step, passo_constante


def f_(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return x1**2 - 3 * x1 * x2 + 4 * x2**2 + x1 - x2


# Definindo pontos de análise
p1 = 2, 2
d = np.array([-2.71428571, -2.14285714])
p2 = tuple(passo_constante(np.array(p1), 0.5, d, f_))

# Definindo função simbólica e variáveis
x1, x2 = sp.symbols("x1 x2")
f = x1**2 - 3 * x1 * x2 + 4 * x2**2 + x1 - x2

# Discretizando plano 3D
num_points = 20
x1_points = np.linspace(p1[0], p2[0], num_points)
x2_points = np.linspace(p1[1], p2[1], num_points)
points = np.column_stack((x1_points, x2_points))

# Calculando os valores de f para os pontos
f_lambdified = sp.lambdify((x1, x2), f, "numpy")
f_values = f_lambdified(points[:, 0], points[:, 1])


# Armazenando valores de x no eixo local
dist = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
print(f"Distância de p1 a p2: {dist}")
alfa_values = np.linspace(0, dist, num_points)


# Gerando um domínio para plotar a superfície de f
dominio = 10
discretization = 100
x_domain = np.linspace(-dominio, dominio, discretization)
y_domain = np.linspace(-dominio, dominio, discretization)
X, Y = np.meshgrid(x_domain, y_domain)
Z = f_lambdified(X, Y)
minimo = np.array(p2)
print(f_lambdified(minimo[0], minimo[1]))


# Plotando os valores de f para os pontos
plt.figure(figsize=(10, 6))
plt.plot(alfa_values, f_values, marker="o")
plt.title("Valores de f para os pontos dados")
plt.xlabel("alfa")
plt.ylabel("f(x, alfa, d)")
plt.grid(True)
plt.show()


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
