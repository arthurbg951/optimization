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
tol = 10e-5


def analise(
    p0: np.ndarray, n: np.ndarray, f: Callable[[float, float], float], show_curves=False
):
    grad_monitor_cte = []
    cte_min = passo_constante(
        p0,
        alfa,
        n,
        f,
        verbose=False,
        monitor=grad_monitor_cte,
    )
    grad_points = expand(np.array(grad_monitor_cte), a)
    grad_local = rotate_z(local(grad_points))[:, 1:3]
    print(y("Passo Constante"))
    # print(grad_local[0], grad_local[-1])
    print(
        f"Mínimo encontrado: {cte_min} com valor {a(cte_min[0], cte_min[1])} com steps = {len(grad_monitor_cte)-1}"
    )
    # plot_local(grad_local[:, 0], grad_local[:, 1])
    if show_curves:
        plot_curves(p0, cte_min, f)

    grad_monitor_bi = []
    bi_min = bissecao(
        p0,
        grad_monitor_cte[-1],
        n,
        f,
        tol,
        verbose=False,
        monitor=grad_monitor_bi,
    )
    grad_points = expand(np.array(grad_monitor_bi), a)
    grad_local = rotate_z(local(grad_points))[:, 1:3]
    print(y("Bisseção"))
    # print(grad_local[0], grad_local[-1])
    print(
        f"Mínimo encontrado: {bi_min} com valor {a(bi_min[0], bi_min[1])} com steps = {len(grad_monitor_bi)-1}"
    )
    # plot_local(grad_local[:, 0], grad_local[:, 1])
    if show_curves:
        plot_curves(p0, bi_min, f)

    grad_monitor_au = []
    au_min = secao_aurea(
        p0,
        grad_monitor_cte[-1],
        n,
        f,
        tol,
        verbose=False,
        monitor=grad_monitor_au,
    )
    grad_points = expand(np.array(grad_monitor_au), a)
    grad_local = rotate_z(local(grad_points))[:, 1:3]
    print(y("Seção Áurea"))
    # print(grad_local[0], grad_local[-1])
    print(
        f"Mínimo encontrado: {au_min} com valor {a(au_min[0], au_min[1])} com steps = {len(grad_monitor_au)-1}"
    )
    # plot_local(grad_local[:, 0], grad_local[:, 1])
    if show_curves:
        plot_curves(p0, au_min, f)


def letra_a():
    print(g(" -> Letra A"))
    p0 = np.array([1, 2])
    n = np.array([-1, -2])
    analise(p0, n, a, show_curves=False)


def letra_b():
    print(g(" -> Letra B"))
    p0 = np.array([-2, 3])
    n = np.array([1.453, -4.547])
    analise(p0, n, mc_cormick, show_curves=False)


def letra_c():
    print(g(" -> Letra C"))
    p0 = np.array([0, 5])
    n = np.array([3, 1.5])
    analise(p0, n, himmelblau, show_curves=False)


if __name__ == "__main__":
    letra_a()
    letra_b()
    letra_c()
