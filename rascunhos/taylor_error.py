"""
Programa desenvolvido para plotar curvas de erro e gráfico de algum número de
graus da série de Taylor
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Definindo a variável simbólica
x = sp.symbols("x")

# Função original
func = sp.exp(x)

# Ponto de expansão
x0 = 2

# Graus da série de Taylor
graus = [0, 1, 2, 3]

# Geração das séries de Taylor
series = [sp.series(func, x, x0, n=grau + 1).removeO() for grau in graus]

# Conversão para funções lambda (usáveis numericamente)
series_funcs = [sp.lambdify(x, s, "numpy") for s in series]

# Intervalo de valores para x
x_vals = np.linspace(0, 3, 100)

# Função original e^x
func_original = np.exp(x_vals)

# Plotando a função original
plt.figure(figsize=(10, 6))
plt.plot(x_vals, func_original, label="e^x", color="black", linewidth=2)

# Lista para armazenar os erros relativos
errors = []

# Plotando os polinômios de Taylor e calculando os erros relativos
for i, serie_func in enumerate(series_funcs):
    y_vals = serie_func(x_vals)
    # Garantindo que y_vals tenha o mesmo tamanho que x_vals
    if np.isscalar(y_vals):  # Se y_vals for escalar (grau 0)
        y_vals = np.full_like(x_vals, y_vals)

    plt.plot(x_vals, y_vals, label=f"Grau {graus[i]}")

    # Calculando o erro relativo
    error = np.abs(func_original - y_vals)
    errors.append(error)

# Configurações do gráfico
plt.title("Expansão da Série de Taylor de e^x em torno de x=2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Plotando os erros relativos
plt.figure(figsize=(10, 6))
for i, error in enumerate(errors):
    plt.plot(x_vals, error, label=f"Erro Relativo - Grau {graus[i]}")

# Configurações do gráfico de erros
plt.title("Erro Relativo das Expansões da Série de Taylor de e^x")
plt.xlabel("x")
plt.ylabel("Erro Relativo")
plt.yscale("log")  # Usando escala logarítmica para o erro
plt.legend()
plt.grid(True)
plt.show()
