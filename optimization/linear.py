import numpy as np
from typing import Callable
from .colors import red as r, green as g, yellow as y


def make_step(p0: np.ndarray, alfa: float, n: np.ndarray):
    return p0 + alfa * n


def passo_constante(
    p0: np.ndarray,
    alfa: float,
    n: np.ndarray,
    func: Callable[[np.ndarray], float],
    n_max_step: int = 1000,
    verbose: bool = False,
    monitor: list[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if verbose:
        print(y("Inicializando Método do Passo Constante"))

    # Realiza 1 passo para verificar se é necessário inverter a direção
    p_check = make_step(p0, 1e-6, n)
    f_check = func(p_check)
    if f_check > func(p0):
        alfa = -alfa

    p_buff = p0
    for i in range(n_max_step):
        new_alfa = alfa * (i + 1)
        p1 = make_step(p0, new_alfa, n)
        f0 = func(p_buff)
        f1 = func(p1)
        if verbose:
            print(f"passo: {g(i+1)}, p_buff: {p_buff}, p1: {p1}")
        if monitor is not None:
            monitor.append(new_alfa)
        if f1 > f0:
            return p_buff
        p_buff = p1

    if verbose:
        print(r("Número máximo de passos atingido."))
    return p_buff


def bissecao(
    p0: np.ndarray,
    n: np.ndarray,
    aL: float,
    aU: float,
    func: Callable[[np.ndarray], float],
    tol: float,
    ε: float = 1e-5,
    n_max_step: int = 1000,
    verbose: bool = False,
    monitor: list[float] = None,
) -> np.ndarray:
    if verbose:
        print(y("Inicializando Método da Bicessão"))

    for i in range(n_max_step):
        aM = (aL + aU) / 2
        mid_point = make_step(p0, aM, n)

        p_f1 = make_step(mid_point, -ε, n)
        p_f2 = make_step(mid_point, ε, n)

        f1 = func(p_f1)
        f2 = func(p_f2)

        if verbose:
            print(
                f"passo: {g(i+1)}, f({mid_point[0]}, {mid_point[1]})={func(mid_point)}"
            )
        if monitor is not None:
            monitor.append(aM)
        if abs(aU - aL) <= tol:
            if verbose:
                print(g(f"Tolerância de {tol} atingida."))
            return (p_f1 + p_f2) / 2
        if f1 > f2:
            aL = aM
            continue
        if f2 > f1:
            aU = aM
            continue
        raise Exception("Erro inesperado.")

    if verbose:
        print(r("Número máximo de passos atingido."))
    return (p_f1 + p_f2) / 2


def secao_aurea(
    pi: np.ndarray,
    pf: np.ndarray,
    # n: np.ndarray,
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

    n = pf - pi
    norm_dir = n / np.linalg.norm(n)
    ra = (np.sqrt(5) - 1) / 2

    # Realiza 1 passo para verificar se é necessário inverter a direção
    # beta = np.linalg.norm(pf - pi)
    # beta_dir = (1 / beta) * norm_dir

    # p_f1 = make_step(pi, (1 - ra) * (1 / beta), beta_dir)
    # p_f2 = make_step(pi, ra * (1 / beta), beta_dir)

    # mid_point = (p_f1 + p_f2) / 2
    # f1 = func(p_f1[0], p_f1[1])
    # f2 = func(p_f2[0], p_f2[1])
    # f_check = func(mid_point[0], mid_point[1])
    # if f_check > func(pi[0], pi[1]):
    #     norm_dir = -n

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
