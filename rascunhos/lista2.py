"""
Código destinado para resoler a segunda questão da lista 2 de otimização
"""

import numpy as np
from typing import Callable
from optimization.linear import passo_constante, bissecao, secao_aurea, make_step
from optimization.functions import a, mc_cormick, himmelblau
from optimization.matrices import rotate_z, expand, local
from optimization.ploting import plot_local, plot_curves
from optimization.colors import yellow as y, green as g


alfa = 0.01
tol = 1e-6


def main():
    letra_a(show_curves=True)
    letra_b(show_curves=True)
    letra_c(show_curves=True)


def run_analise(
    p0: np.ndarray,
    n: np.ndarray,
    f: Callable[[np.ndarray], float],
    show_curves: bool = False,
):
    aU, aL, min_pc = run_passo_constante(p0, alfa, n, f, verbose=False)
    min_bi = run_bissecao(p0, n, aU, aL, f, tol, verbose=False)
    min_au = run_secao_aurea(p0, n, aU, aL, f, tol, verbose=False)
    if show_curves:
        plot_curves(p0, min_pc, f)
        plot_curves(p0, min_bi, f)
        plot_curves(p0, min_au, f)


def run_passo_constante(
    p0: np.ndarray,
    alfa: float,
    n: np.ndarray,
    f: Callable[[np.ndarray], float],
    verbose=False,
):
    monitor_percurso = []
    aU, aL = passo_constante(
        p0,
        alfa,
        n,
        f,
        verbose=verbose,
        monitor=monitor_percurso,
    )

    print(y("Passo Constante") + f" aU = {aU} aL = {aL}")
    aM = (aU + aL) / 2
    p_min = make_step(p0, aM, n)
    print(
        f"Mínimo encontrado: {p_min} com valor {f(p_min)} com steps = {len(monitor_percurso)}"
    )
    return aU, aL, p_min


def run_bissecao(
    p0: np.ndarray,
    n: np.ndarray,
    aL: float,
    aU: float,
    f: Callable[[np.ndarray], float],
    tol: float,
    verbose=False,
):
    aM, n_steps = bissecao(
        p0,
        n,
        aL,
        aU,
        f,
        tol,
        verbose=verbose,
        out_n_steps=True,
    )
    p_min = make_step(p0, aM, n)
    print(y("Bisseção"))
    print(f"Mínimo encontrado: {p_min} com valor {f(p_min)} com steps = {n_steps}")
    return p_min


def run_secao_aurea(
    p0: np.ndarray,
    n: np.ndarray,
    aL: float,
    aU: float,
    f: Callable[[np.ndarray], float],
    tol: float,
    verbose=False,
):
    aM, n_steps = secao_aurea(
        p0,
        n,
        aL,
        aU,
        f,
        tol,
        verbose=verbose,
        out_n_steps=True,
    )
    p_min = make_step(p0, aM, n)
    print(y("Seção Áurea"))
    print(f"Mínimo encontrado: {p_min} com valor {f(p_min)} com steps = {n_steps}")
    return p_min


def letra_a(show_curves=False):
    print(g(" -> Letra A"))
    p0 = np.array([1, 2])
    n = np.array([-1, -2])
    run_analise(p0, n, a, show_curves=show_curves)


def letra_b(show_curves=False):
    print(g(" -> Letra B"))
    p0 = np.array([-2, 3])
    n = np.array([1.453, -4.547])
    run_analise(p0, n, mc_cormick, show_curves=show_curves)


def letra_c(show_curves=False):
    print(g(" -> Letra C"))
    p0 = np.array([0, 5])
    n = np.array([3, 1.5])
    run_analise(p0, n, himmelblau, show_curves=show_curves)


if __name__ == "__main__":
    main()
