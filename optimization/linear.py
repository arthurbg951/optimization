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
    monitor: list[np.ndarray] = None,
) -> np.ndarray:
    # TODO: Remover n_max_step da função
    if verbose:
        print(y("Inicializando Método do Passo Constante"))

    # Realiza 1 passo para verificar se é necessário inverter a direção
    p_check = make_step(p0, 1e-6, n)
    f_check = func(p_check)
    if f_check > func(p0):
        alfa = -alfa

    p_buff = p0
    i = 0
    while True:
        new_alfa = alfa * (i + 1)
        p1 = make_step(p0, new_alfa, n)
        f0 = func(p_buff)
        f1 = func(p1)
        if verbose:
            print(f"passo: {g(i+1)}, p_buff: {p_buff}, p1: {p1}")
        if monitor is not None:
            monitor.append(p_buff)
        if f1 > f0:
            return alfa * (i + 1), alfa * i
        p_buff = p1
        i += 1

    print(r("Número máximo de passos atingido passo constante."))
    return alfa * (i + 1), alfa * i


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
    out_n_steps: bool = False,
) -> np.ndarray:
    if verbose:
        print(y("Inicializando Método da Bicessão"))

    for i in range(n_max_step):
        aM = (aL + aU) / 2

        f1 = func(make_step(p0, (aM - ε), n))
        f2 = func(make_step(p0, (aM + ε), n))

        if verbose:
            mid_point = make_step(p0, aM, n)
            print(
                f"passo: {g(i+1)}, f({mid_point[0]}, {mid_point[1]})={func(mid_point)}"
            )
        if abs(aU - aL) <= tol:
            if verbose:
                print(g(f"Tolerância de {tol} atingida."))
            if out_n_steps:
                return aM, i + 1
            else:
                return aM

        if f1 > f2:
            aL = aM
            continue
        if f2 >= f1:
            aU = aM
            continue
        raise Exception("Erro inesperado.")

    print(r("Número máximo de passos atingido bissecao."))
    if out_n_steps:
        return aM, i + 1
    else:
        return aM


def secao_aurea(
    p0: np.ndarray,
    n: np.ndarray,
    aL: float,
    aU: float,
    func: Callable[[np.ndarray], float],
    tol: float,
    n_max_step: int = 1000,
    verbose: bool = False,
    out_n_steps: bool = False,
) -> tuple[float, float] | float:
    if verbose:
        print(y("Inicializando Método da Seção Áurea"))

    n_steps = 0

    ra = (np.sqrt(5) - 1) / 2
    comp_ra = 1 - ra

    beta = aU - aL
    aE = aL + comp_ra * beta
    aD = aL + ra * beta

    p_aE = make_step(p0, aE, n)
    p_aD = make_step(p0, aD, n)

    fE = func(p_aE)
    fD = func(p_aD)

    if verbose:
        mid_point = (p_aE + p_aD) / 2
        print(f"passo: {g(1)}, f({mid_point[0]}, {mid_point[1]})={func(mid_point)}")
    i = 0
    while True:
        n_steps += 1
        if verbose:
            mid_point = (p_aE + p_aD) / 2
            print(
                f"passo: {g(i+1)}, f({mid_point[0]}, {mid_point[1]})={func(mid_point)}"
            )
        if abs(beta) < tol:
            if out_n_steps:
                return (aE + aD) / 2, n_steps
            else:
                return (aE + aD) / 2

        beta = abs(aU - aL)

        if fE > fD:
            aL = aE
            aE = aD
            fE = fD
            aD = aL + ra * (aU - aL)
            p_aD = make_step(p0, aD, n)
            fD = func(p_aD)
        if fD >= fE:
            aU = aD
            aD = aE
            fD = fE
            aE = aL + comp_ra * (aU - aL)
            p_aE = make_step(p0, aE, n)
            fE = func(p_aE)
        i += 1

    print(r("Número máximo de passos atingido golden."))
    if out_n_steps:
        return (aE + aD) / 2, n_steps
    else:
        return (aE + aD) / 2
