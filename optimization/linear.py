import numpy as np
from typing import Callable
from .colors import red as r, green as g, yellow as y


def make_step(p0: np.ndarray, alfa: float, n: np.ndarray):
    return p0 + alfa * n


# def passo_constante(
#     p0: np.ndarray,
#     alfa: float,
#     d: np.ndarray,
#     f: Callable,
#     verbose: bool = False,
#     monitor: list[np.ndarray] = None,
# ) -> tuple[float, float]:
#     """
#     Performs a constant step search to minimize the function `f` along a given direction.

#     Parameters:
#     - p0: initial point as a numpy array.
#     - alfa: step size.
#     - f: function to minimize, takes two inputs.
#     - d: direction for the step, numpy array.

#     Returns:
#     - aL: lower bound of the interval containing the minimum.
#     - aU: upper bound of the interval containing the minimum.
#     """

#     def f_alpha(alpha, d):
#         return p0 + alpha * d

#     alpha = 0
#     eps = 1e-6
#     f0 = f(p0)
#     x1 = f_alpha(eps, d)
#     f1 = f(p0)

#     # Check if direction needs to be flipped
#     flag_dir = 1
#     if f1 > f0:
#         d = -d
#         flag_dir = -1

#     steps = 0
#     while True:
#         steps += 1
#         alpha += alfa
#         x = f_alpha(alpha, d)
#         f_val = f(x)
#         x1 = f_alpha(alpha - eps, d)
#         f1 = f(x1)

#         if f1 < f_val:
#             minimum = f_val
#             aL = alpha - alfa
#             aU = alpha
#             break

#     # Adjust the bounds if direction was reversed
#     if flag_dir == -1:
#         aL, aU = -aU, -aL

#     # return minimum, x, aL, aU, steps, direction
#     return aL, aU


# def passo_constante(
#     p0: np.ndarray,
#     alfa: float,
#     n: np.ndarray,
#     func: Callable[[np.ndarray], float],
#     n_max_step: int = 1000,
#     verbose: bool = False,
#     monitor: list[np.ndarray] = None,
# ) -> np.ndarray:
#     # TODO: Remover n_max_step da função
#     if verbose:
#         print(y("Inicializando Método do Passo Constante"))

#     # Realiza 1 passo para verificar se é necessário inverter a direção
#     p_check = make_step(p0, 1e-6, n)
#     f_check = func(p_check)
#     dir_changed = False

#     if f_check > func(p0):
#         dir_changed = True
#         n = -n

#     p_buff = p0
#     i = 0
#     while True:
#         new_alfa = alfa * (i + 1)
#         # print(f"alfa: {new_alfa}, p: {p_buff}")
#         p1 = make_step(p0, new_alfa, n)

#         f0 = func(p_buff)
#         f1 = func(p1)

#         if verbose:
#             print(f"passo: {g(i+1)}, p_buff: {p_buff}, p1: {p1}")

#         if monitor is not None:
#             monitor.append(p_buff)

#         if f1 > f0:
#             result = alfa * (i - 1), alfa * (i + 1)
#             alfa_min = min(result)
#             alfa_max = max(result)
#             if dir_changed:
#                 alfa_min, alfa_max = -alfa_max, -alfa_min
#             return alfa_min, alfa_max

#         p_buff = p1
#         i += 1

#     print(r("Número máximo de passos atingido passo constante."))
#     return alfa * (i + 1), alfa * i


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
    # Calcula o valor da função no ponto inicial
    alpha = 0
    f0 = func(x0)

    # Verifica o sentido positivo da busca
    f1 = func(make_step(x0, eps, d))
    flag_dir = 1  # Inicialmente direção é positiva

    if f1 > f0:
        d = -d  # Inverte a direção
        flag_dir = -1

    # Itera para encontrar os limites do intervalo
    while True:
        alpha += delta_alpha
        f = func(make_step(x0, alpha, d))
        f_prev = func(make_step(x0, alpha - eps, d))

        if f_prev < f:
            aL = alpha - delta_alpha
            aU = alpha
            break

    # Ajusta os limites se a direção foi invertida
    if flag_dir == -1:
        aL, aU = -aU, -aL

    if verbose:
        print(f"aL: {aL}, aU: {aU}")

    return aL, aU


# def passo_constante(
#     p0: np.ndarray,
#     alfa: float,
#     n: np.ndarray,
#     func: Callable[[np.ndarray], float],
#     n_max_step: int = 1000,
#     verbose: bool = False,
#     monitor: list[np.ndarray] = None,
# ) -> tuple[float, float]:
#     if verbose:
#         print("Inicializando Método do Passo Constante")

#     # Verifica a direção inicial do passo
#     p_check = make_step(p0, 1e-10, n)
#     f_check = func(p_check)

#     if f_check > func(p0):
#         alfa = -alfa  # Inverte a direção se a função aumenta

#     p_buff = p0
#     i = 0
#     while i < n_max_step:
#         new_alfa = alfa * (i + 1)
#         p1 = make_step(p0, new_alfa, n)

#         f0 = func(p_buff)
#         f1 = func(p1)

#         if verbose:
#             print(f"Passo: {i+1}, p_buff: {p_buff}, p1: {p1}, f0: {f0}, f1: {f1}")

#         if monitor is not None:
#             monitor.append(p_buff)

#         # Verifica a condição de parada
#         if f1 > f0:
#             # Retorna os valores de alfa para o intervalo de mínima e máxima aproximação
#             alfa_min = min(alfa * i, alfa * (i + 1))
#             alfa_max = max(alfa * i, alfa * (i + 1))
#             return alfa_min, alfa_max

#         # Avança o ponto atual para o próximo
#         p_buff = p1
#         i += 1

#     print("Número máximo de passos atingido no método do passo constante.")
#     return alfa * (i - 1), alfa * i


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


# def secao_aurea(
#     p0: np.ndarray,
#     n: np.ndarray,
#     aL: float,
#     aU: float,
#     func: Callable[[np.ndarray], float],
#     tol: float,
#     n_max_step: int = 1000,
#     verbose: bool = False,
#     out_n_steps: bool = False,
# ) -> tuple[float, float] | float:
#     if verbose:
#         print(y("Inicializando Método da Seção Áurea"))

#     n_steps = 0

#     ra = (np.sqrt(5.0) - 1.0) / 2.0
#     comp_ra = 1 - ra

#     beta = aU - aL
#     aE = aL + comp_ra * beta
#     aD = aL + ra * beta

#     p_aE = make_step(p0, aE, n)
#     p_aD = make_step(p0, aD, n)

#     fE = func(p_aE)
#     fD = func(p_aD)

#     if verbose:
#         mid_point = (p_aE + p_aD) / 2.0
#         print(f"passo: {g(1)}, f({mid_point[0]}, {mid_point[1]})={func(mid_point)}")
#     i = 0
#     while True:
#         n_steps += 1
#         if verbose:
#             mid_point = (p_aE + p_aD) / 2.0
#             print(
#                 f"passo: {g(i+1)}, f({mid_point[0]}, {mid_point[1]})={func(mid_point)}"
#             )
#         if abs(beta) < tol:
#             if out_n_steps:
#                 return (aE + aD) / 2.0, n_steps
#             else:
#                 return (aE + aD) / 2.0

#         beta = aU - aL

#         if fE > fD:
#             aL = aE
#             aE = aD
#             fE = fD
#             aD = aL + ra * (aU - aL)
#             p_aD = make_step(p0, aD, n)
#             fD = func(p_aD)
#         if fD >= fE:
#             aU = aD
#             aD = aE
#             fD = fE
#             aE = aL + comp_ra * (aU - aL)
#             p_aE = make_step(p0, aE, n)
#             fE = func(p_aE)
#         i += 1

#     print(r("Número máximo de passos atingido golden."))
#     if out_n_steps:
#         return (aE + aD) / 2, n_steps
#     else:
#         return (aE + aD) / 2


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
    # Razão Áurea
    RA = (np.sqrt(5.0) - 1.0) / 2.0

    # Tamanho inicial do intervalo
    beta = abs(aU - aL)

    # Pontos intermediários do intervalo
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

    # Retorna o ponto médio como melhor aproximação do mínimo
    return (aL + aU) / 2
