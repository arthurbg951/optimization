"""
Código destinado para resoler a segunda questão da lista 2 de otimização
"""

import numpy as np
from typing import Callable
from optimization.linear import passo_constante, bissecao, secao_aurea
from optimization.functions import a, mc_cormick, himmelblau
from optimization.matrices import rotate_z, expand, local
from optimization.ploting import plot_local, plot_curves
from optimization.colors import yellow as y, green as g


alfa = 0.01
tol = 1e-5


def analise(
    p0: np.ndarray,
    n: np.ndarray,
    f: Callable[[float, float], float],
    show_curves=False,
    show_local=False,
):
    # TODO: CORRIGIR FUNÇÕES DE PLOT DOS GRÁFICOS
    road_monitor_cte = []
    cte_min = passo_constante(
        p0,
        alfa,
        n,
        f,
        verbose=False,
        monitor=road_monitor_cte,
    )
    print(
        y("Passo Constante")
        + f" aU = {road_monitor_cte[-2]} aL = {road_monitor_cte[-1]}"
    )
    print(
        f"Mínimo encontrado: {cte_min} com valor {f(cte_min)} com steps = {len(road_monitor_cte)}"
    )
    if show_local:
        road_points = expand(np.array(road_monitor_cte), f)
        road_local = rotate_z(local(road_points))[:, 1:3]
        plot_local(road_local[:, 0], road_local[:, 1])
    if show_curves:
        plot_curves(p0, cte_min, f)

    road_monitor_bi = []
    bi_min = bissecao(
        p0,
        n,
        road_monitor_cte[0],
        road_monitor_cte[-1],
        f,
        tol,
        verbose=False,
        monitor=road_monitor_bi,
    )
    print(y("Bisseção") + f" aM = {(road_monitor_bi[-2] + road_monitor_bi[-1])/2}")
    print(
        f"Mínimo encontrado: {bi_min} com valor {f(bi_min)} com steps = {len(road_monitor_bi)}"
    )
    if show_local:
        road_points = expand(np.array(road_monitor_bi), f)
        road_local = rotate_z(local(road_points))[:, 1:3]
        plot_local(road_local[:, 0], road_local[:, 1])
    if show_curves:
        plot_curves(p0, bi_min, f)

    # road_monitor_au = []
    # au_min = secao_aurea(
    #     p0,
    #     road_monitor_cte[-1],
    #     # n,
    #     f,
    #     tol,
    #     verbose=True,
    #     monitor=road_monitor_au,
    # )
    # road_points = expand(np.array(road_monitor_au), f)
    # road_local = rotate_z(local(road_points))[:, 1:3]
    # print(y("Seção Áurea"))
    # # print(road_local[0], road_local[-1])
    # print(
    #     f"Mínimo encontrado: {au_min} com valor {f(au_min[0], au_min[1])} com steps = {len(road_monitor_au)}"
    # )
    # if show_local:
    #     plot_local(road_local[:, 0], road_local[:, 1])
    # if show_curves:
    #     plot_curves(p0, au_min, f)


def letra_a():
    print(g(" -> Letra A"))
    p0 = np.array([1, 2])
    n = np.array([-1, -2])
    analise(p0, n, a, show_curves=False, show_local=False)


def letra_b():
    print(g(" -> Letra B"))
    p0 = np.array([-2, 3])
    n = np.array([1.453, -4.547])
    analise(p0, n, mc_cormick, show_curves=False, show_local=False)


def letra_c():
    print(g(" -> Letra C"))
    p0 = np.array([0, 5])
    n = np.array([3, 1.5])
    analise(p0, n, himmelblau, show_curves=False, show_local=False)


if __name__ == "__main__":
    letra_a()
    letra_b()
    letra_c()
