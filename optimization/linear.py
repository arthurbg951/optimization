import numpy as np
from typing import Callable
from .colors import red as r, green as g, yellow as y


def make_step(p0: np.ndarray, alfa: float, n: np.ndarray):
    return p0 + alfa * n


def passo_constante(
    p0: np.ndarray,
    alfa: float,
    n: np.ndarray,
    func: Callable[[float, float], float],
    n_max_step: int = 1000,
    verbose: bool = False,
    monitor: list = None,
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: Remover float, float da função callable, e adicionar apenas np.ndarray
    if verbose:
        print(y("Inicializando Método do Passo Constante"))
    if monitor is not None:
        monitor.append(p0)
    for i in range(n_max_step):
        p1 = make_step(p0, alfa, n)
        f0 = func(p0[0], p0[1])
        f1 = func(p1[0], p1[1])
        if verbose:
            print(f"passo: {g(i+1)}, p0: {p0}, p1: {p1}")
        if monitor is not None:
            monitor.append(p1)
        if f1 > f0:
            return p0
        p0 = p1
    if verbose:
        print(r("Número máximo de passos atingido."))
    return p1


def bissecao(
    pi: np.ndarray,
    pf: np.ndarray,
    n: np.ndarray,
    func: Callable[[float, float], float],
    tol: float,
    ε: float = 1e-8,
    n_max_step: int = 1000,
    verbose: bool = False,
    monitor: list = None,
) -> np.ndarray:
    if verbose:
        print(y("Inicializando Método da Bicessão"))
    if monitor is not None:
        monitor.append(pi)

    # norm_dir = n / np.linalg.norm(n)
    norm_dir = n

    for i in range(n_max_step):
        mid_point = (pi + pf) / 2

        p_f1 = make_step(mid_point, -ε, norm_dir)
        p_f2 = make_step(mid_point, ε, norm_dir)

        f1 = func(p_f1[0], p_f1[1])
        f2 = func(p_f2[0], p_f2[1])

        if verbose:
            print(
                f"passo: {g(i+1)}, f({mid_point[0]}, {mid_point[1]})={func(mid_point[0], mid_point[1])}"
            )
        if monitor is not None:
            monitor.append(mid_point)
        if np.linalg.norm(pi - pf) <= tol:
            if verbose:
                print(g(f"Tolerância de {tol} atingida."))
            return (p_f1 + p_f2) / 2
        if f1 > f2:
            pi = mid_point
            continue
        if f2 > f1:
            pf = mid_point
            continue
        raise Exception("Erro inesperado.")
    if verbose:
        print(r("Número máximo de passos atingido."))
    return (p_f1 + p_f2) / 2


def secao_aurea(
    pi: np.ndarray,
    pf: np.ndarray,
    n: np.ndarray,
    func: Callable[[float, float], float],
    tol: float,
    n_max_step: int = 1000,
    verbose: bool = False,
    monitor: list = None,
):
    if verbose:
        print(y("Inicializando Método da Seção Áurea"))
    if monitor is not None:
        monitor.append(pi)

    norm_dir = n / np.linalg.norm(n)
    ra = (np.sqrt(5) - 1) / 2
    for i in range(n_max_step):
        beta = np.linalg.norm(pf - pi)
        beta_dir = (1 / beta) * norm_dir

        p_f1 = make_step(pi, (1 - ra) * (1 / beta), beta_dir)
        p_f2 = make_step(pi, ra * (1 / beta), beta_dir)

        mid_point = (p_f1 + p_f2) / 2
        f1 = func(p_f1[0], p_f1[1])
        f2 = func(p_f2[0], p_f2[1])

        if verbose:
            print(
                f"passo: {g(i+1)}, f({p_f1[0]}, {mid_point[1]})={func(mid_point[0], mid_point[1])}"
            )
        if monitor is not None:
            monitor.append(mid_point)
        if np.linalg.norm(beta_dir) <= tol:
            if verbose:
                print(g(f"Tolerância de {tol} atingida."))
            return (p_f1 + p_f2) / 2
        if f1 > f2:
            pi = mid_point
            continue
        if f2 > f1:
            pf = mid_point
            continue
        raise Exception("Erro inesperado.")
    if verbose:
        print(r("Número máximo de passos atingido."))
    return (p_f1 + p_f2) / 2
