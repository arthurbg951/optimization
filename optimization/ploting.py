import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# Settings
__fig_size = (8, 6)


def get_surface_points(
    func, p1, p2, discretization=100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_domain = np.linspace(p1[0], p2[0], discretization)
    y_domain = np.linspace(p1[1], p2[1], discretization)
    X, Y = np.meshgrid(x_domain, y_domain)
    # func = np.vectorize(func)
    Z = func(X, Y)
    return X, Y, Z


def plot_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    func: Callable,
    point: np.ndarray = None,
):
    # Plotando a superfície de f
    fig = plt.figure(figsize=__fig_size)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")

    # Adicionar o ponto vermelho se fornecido
    if point is not None:
        x_p, y_p = point
        z_p = func(x_p, y_p)
        ax.scatter(
            x_p, y_p, z_p, color="red", s=100, label=f"Mínimo ({x_p}, {y_p}, {z_p:.2f})"
        )
        ax.legend()

    # Configurações adicionais do gráfico
    ax.set_title("Superfície da função f(x, y) com ponto destacado")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    plt.show()


def plot_local(alfa: np.ndarray, z: np.ndarray):
    # Plotando os valores de f para os pontos
    plt.figure(figsize=__fig_size)
    plt.plot(alfa, z, marker="o")
    plt.title("Descida do gradiente no eixo local")
    plt.xlabel("Índice do ponto")
    plt.ylabel("f(x, y)")
    plt.grid(True)
    plt.show()


def plot_curves(
    points: list[np.ndarray],
    func: Callable[[float, float], float],
    n_points=100,
    countour_levels=50,
    title="Curvas de Nível de f(x, y)",
):
    pi = points[0]
    pf = points[-1]

    # Calcular o min e max
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    margin = 0.5

    # Definir os limites do gráfico e a discretização
    x = np.linspace(min_x - margin, max_x + margin, n_points)
    y = np.linspace(min_y - margin, max_y + margin, n_points)

    # Criar a malha de pontos (grid) para x e y
    X, Y = np.meshgrid(x, y)

    # Calcular Z = f(X, Y)
    f_vec = np.vectorize(lambda x, y: func(np.array([x, y])))
    Z = f_vec(X, Y)

    # Criar o gráfico de curvas de nível
    plt.figure(figsize=__fig_size)
    contour = plt.contour(X, Y, Z, levels=countour_levels, cmap="viridis")

    # Plotar os pontos inicial e final
    plt.scatter(pi[0], pi[1], color="red", label="Ponto Inicial")
    plt.scatter(pf[0], pf[1], color="blue", label="Ponto Final")

    # Adicionar legendas aos pontos
    plt.text(
        pi[0],
        pi[1],
        f"pi [{pi[0]:.2f}, {pi[1]:.2f}]",
        color="red",
        fontsize=12,
        ha="right",
    )
    plt.text(
        pf[0],
        pf[1],
        f"pf [{pf[0]:.2f}, {pf[1]:.2f}]",
        color="blue",
        fontsize=12,
        ha="left",
    )

    # Traçar uma linha ligando os dois pontos
    for i in range(1, len(points)):
        p0 = points[i - 1]
        p1 = points[i]
        plt.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color="green",
            linestyle="--",
        )

    # Adicionar rótulos às curvas de nível
    plt.clabel(contour, inline=True, fontsize=8)

    # Configurações adicionais do gráfico
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    # Exibir o gráfico com a barra de cores
    # plt.colorbar(contour)

    # Exibir o gráfico
    plt.show()


def show_surface(func, p1, p2, discretization=100, min_point: np.ndarray = None):
    X, Y, Z = get_surface_points(func, p1, p2, discretization)
    plot_surface(X, Y, Z, min_point)
