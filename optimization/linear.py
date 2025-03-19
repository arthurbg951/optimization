import numpy as np
from typing import Callable
from .colors import red as r, green as g, yellow as y


def make_step(p0: np.ndarray, alfa: float, n: np.ndarray):
    return p0 + alfa * n


def passo_constante(
    x0: np.ndarray,
    delta_alpha: float,
    d: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray, float], float],
    eps: float = 1e-6,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Método do Passo Constante adaptado do código MATLAB.

    Parâmetros:
    - x0: Ponto inicial (np.ndarray).
    - d: Direção de busca (np.ndarray).
    - delta_alpha: Tamanho do passo constante.
    - func: Função objetivo, que deve aceitar (x0, direção, alpha) como parâmetros.
    - eps: Tolerância para verificação inicial.
    - verbose: Exibe informações adicionais durante o cálculo.

    Retorno:
    - aL: Limite inferior do intervalo alfa.
    - aU: Limite superior do intervalo alfa.
    """
    alpha = 0
    f0 = func(x0)

    f1 = func(make_step(x0, eps, d))
    flag_dir = 1  

    if f1 > f0:
        d = -d  
        flag_dir = -1

    while True:
        alpha += delta_alpha
        f = func(make_step(x0, alpha, d))
        f_prev = func(make_step(x0, alpha - eps, d))

        if f_prev < f:
            aL = alpha - delta_alpha
            aU = alpha
            break

    if flag_dir == -1:
        aL, aU = -aU, -aL

    if verbose:
        print(f"aL: {aL}, aU: {aU}")

    return aL, aU


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
                f"passo: {g(i + 1)}, f({mid_point[0]}, {mid_point[1]})={func(mid_point)}"
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
) -> float:
    RA = (np.sqrt(5.0) - 1.0) / 2.0

    beta = abs(aU - aL)

    aE = aL + (1.0 - RA) * beta
    aD = aL + RA * beta
    fE = func(p0 + aE * n)
    fD = func(p0 + aD * n)

    # while beta >= 1e-10:
    while beta >= tol:
        if fE > fD:
            aL = aE
            aE = aD
            fE = fD
            beta = abs(aU - aL)
            aD = aL + RA * beta
            fD = func(p0 + aD * n)
        else:
            aU = aD
            aD = aE
            fD = fE
            beta = abs(aU - aL)
            aE = aL + (1.0 - RA) * beta
            fE = func(p0 + aE * n)

    return (aL + aU) / 2  # TODO: Considerar cancelamento catastrófico
