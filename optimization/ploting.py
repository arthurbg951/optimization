import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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
    show_fig=False,
):
    # TODO: alterar nome da função para plot_curve ou semelhante
    pi = points[0]
    pf = points[-1]

    # Calcular o min e max
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    margin = 1

    # Definir os limites do gráfico e a discretização
    x = np.linspace(min_x - margin, max_x + margin, n_points)
    y = np.linspace(min_y - margin, max_y + margin, n_points)

    # Criar a malha de pontos (grid) para x e y
    X, Y = np.meshgrid(x, y)

    # Calcular Z = f(X, Y)
    f_vec = np.vectorize(lambda x, y: func(np.array([x, y])))
    Z = f_vec(X, Y)

    # Criar o gráfico de curvas de nível
    fig = plt.figure(figsize=__fig_size)
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
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")

    # Exibir o gráfico com a barra de cores
    # plt.colorbar(contour)

    # Exibir o gráfico
    if show_fig:
        plt.show()

    return fig


# def plot_restriction_curves(
#     points: list[np.ndarray],
#     f: Callable[[np.ndarray], np.ndarray],
#     h_list: list[Callable[[np.ndarray], np.ndarray]],
#     c_list: list[Callable[[np.ndarray], np.ndarray]],
#     n_points=100,
#     countour_levels=50,
#     title="Gráfico das funções f(X), h(X) e c(X)",
#     show_fig=False,
# ):
#     pi = points[0]
#     pf = points[-1]

#     margin = 0.1 * np.linalg.norm(pf - pi)

#     # Calcular o min e max
#     min_x = min(p[0] for p in points)
#     min_y = min(p[1] for p in points)
#     max_x = max(p[0] for p in points)
#     max_y = max(p[1] for p in points)

#     margin = 1

#     # Definir os limites do gráfico e a discretização
#     x = np.linspace(min_x - margin, max_x + margin, n_points)
#     y = np.linspace(min_y - margin, max_y + margin, n_points)

#     # Criar a malha de pontos (grid) para x e y
#     X, Y = np.meshgrid(x, y)

#     # Plotagem
#     fig = plt.figure(figsize=__fig_size)

#     # Plotar os pontos inicial e final
#     plt.scatter(pi[0], pi[1], color="red", label="Ponto Inicial")
#     plt.scatter(pf[0], pf[1], color="blue", label="Ponto Final")

#     # Adicionar legendas aos pontos
#     plt.text(
#         pi[0],
#         pi[1],
#         f"pi [{pi[0]:.2f}, {pi[1]:.2f}]",
#         color="red",
#         fontsize=12,
#         ha="right",
#     )
#     plt.text(
#         pf[0],
#         pf[1],
#         f"pf [{pf[0]:.2f}, {pf[1]:.2f}]",
#         color="blue",
#         fontsize=12,
#         ha="left",
#     )

#     # Traçar uma linha ligando os dois pontos
#     for i in range(1, len(points)):
#         p0 = points[i - 1]
#         p1 = points[i]
#         plt.plot(
#             [p0[0], p1[0]],
#             [p0[1], p1[1]],
#             color="green",
#             linestyle="--",
#         )

#     # Avaliar f(X)
#     ZF = np.vectorize(lambda x1, x2: f(np.array([x1, x2])))(X, Y)
#     contour = plt.contour(X, Y, ZF, levels=countour_levels, cmap="viridis")
#     plt.clabel(contour, inline=True, fontsize=8)

#     # Plotar cada função em h_list
#     for i, h in enumerate(h_list, start=1):
#         ZH = np.vectorize(lambda x1, x2: h(np.array([x1, x2])))(X, Y)
#         plt.contour(
#             X,
#             Y,
#             ZH,
#             levels=[0],
#             colors="yellow",
#             linestyles="--",
#             linewidths=1.2,
#         )

#     # Plotar cada função em c_list
#     for i, c in enumerate(c_list, start=1):
#         ZC = np.vectorize(lambda x1, x2: c(np.array([x1, x2])))(X, Y)
#         plt.contour(
#             X,
#             Y,
#             ZC,
#             levels=[0],
#             colors="red",
#             linestyles="-.",
#             linewidths=1.2,
#         )

#     # Configuração do gráfico
#     # GERANDO ERRO NO PLOT DEVIDO AOS LIMITES DO GRÁFICO
#     # plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # Linha do eixo X
#     # plt.axvline(0, color="black", linewidth=0.5, linestyle="--")  # Linha do eixo Y
#     plt.title(title)
#     plt.xlabel("x1")
#     plt.ylabel("x2")
#     plt.grid(True)
#     plt.tight_layout()

#     # Mostrar o gráfico
#     if show_fig:
#         plt.show()


#     return fig
def plot_restriction_curves(
    points: list[np.ndarray],
    f: Callable[[np.ndarray], np.ndarray],
    h_list: list[Callable[[np.ndarray], np.ndarray]],
    c_list: list[Callable[[np.ndarray], np.ndarray]],
    n_points=100,
    countour_levels=50,
    title="Gráfico das funções f(X), h(X) e c(X)",
    show_fig=False,
):
    pi = points[0]
    pf = points[-1]

    margin = 0.1 * np.linalg.norm(pf - pi)

    # Calcular o min e max
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    # margin = 1

    # Definir os limites do gráfico e a discretização
    # x = np.linspace(min_x - margin, max_x + margin, n_points)
    # y = np.linspace(min_y - margin, max_y + margin, n_points)
    x = np.linspace(0, 10, n_points)
    y = np.linspace(0, 100, n_points)

    # Criar a malha de pontos (grid) para x e y
    X, Y = np.meshgrid(x, y)

    # Plotagem
    fig = plt.figure(figsize=(10, 8))

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

    # Avaliar f(X)
    ZF = np.vectorize(lambda x1, x2: f(np.array([x1, x2])))(X, Y)
    contour = plt.contour(X, Y, ZF, levels=countour_levels, cmap="viridis")
    plt.clabel(contour, inline=True, fontsize=8)

    # Plotar cada função em h_list
    for i, h in enumerate(h_list, start=1):
        ZH = np.vectorize(lambda x1, x2: h(np.array([x1, x2])))(X, Y)
        plt.contour(
            X,
            Y,
            ZH,
            levels=[0],
            colors="yellow",
            linestyles="--",
            linewidths=1.2,
        )

    # Lista de cores para as curvas
    colors = [
        "blue",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "green",
    ]

    # Criar a região viável para c_list
    viable_region = np.ones_like(X, dtype=bool)  # Inicialmente, toda a região é viável

    for i, c in enumerate(c_list):
        ZC = np.vectorize(lambda x1, x2: c(np.array([x1, x2])))(X, Y)
        viable_region &= ZC <= 0

        # Usar uma cor específica para cada curva
        plt.contour(
            X,
            Y,
            ZC,
            levels=[0],
            colors=colors[i % len(colors)],
            linestyles="-",
            linewidths=2,
        )

    # Preencher a região viável
    # plt.contourf(X, Y, viable_region, levels=[0.5, 1], colors=["#ff9999"], alpha=0.4)
    # plt.contour(
    #     X, Y, viable_region, levels=[0.5], colors="red", linestyles="-", linewidths=1
    # )

    # Configuração do gráfico
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Mostrar o gráfico
    if show_fig:
        plt.show()

    return fig


def plot_images(figuras: list[np.ndarray]):
    # Criação da figura principal com 2 linhas e 3 colunas de subplots
    fig, axs = plt.subplots(2, 3, figsize=__fig_size)

    # Loop para desenhar cada subfigura na grade
    for i, figura in enumerate(figuras):
        ax = axs[i // 3, i % 3]  # Determina a posição do subplot na grade 2x3
        for (
            child
        ) in figura.get_children():  # Copia o conteúdo de cada figura para o subplot
            ax.add_artist(child)
        ax.axis("on")  # Exibe os eixos

    # Ajuste do layout e exibição
    plt.tight_layout()
    plt.show()

    return fig


def show_surface(func, p1, p2, discretization=100, min_point: np.ndarray = None):
    X, Y, Z = get_surface_points(func, p1, p2, discretization)
    plot_surface(X, Y, Z, min_point)


def figure_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    return np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        canvas.get_width_height()[::-1] + (3,)
    )


def plot_figs(fig1, fig2, show_fig=False):
    fig1 = figure_to_image(fig1)
    fig2 = figure_to_image(fig2)
    # Criando um subplot com 1 linha e 2 colunas
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 linha, 2 colunas

    # Adicionando a primeira imagem
    axs[0].imshow(fig1, cmap="jet")
    # axs[0].set_title("Figura 1")  # Título para o primeiro gráfico
    axs[0].axis("off")  # Remove os eixos

    # Adicionando a segunda imagem
    axs[1].imshow(fig2, cmap="jet")
    # axs[1].set_title("Figura 2")  # Título para o segundo gráfico
    axs[1].axis("off")  # Remove os eixos

    # Ajustando o layout e exibindo
    plt.tight_layout()
    if show_fig:
        plt.show()
    return fig
