"""
Considerações de notação:

f -> Função objetivo
h -> Restrições de igualdade
c -> Restrições de desigualdade

x -> Ponto no espaço de busca

OSR -> Otimização sem restrições
"""

from typing import Callable
from datetime import datetime

import numpy as np

from optimization.minimize import (
    univariante,
    powell,
    steepest_descent,
    fletcher_reeves,
    bfgs,
    newton_raphson,
)
from optimization.colors import red as r, green as g, yellow as y, blue as b
from optimization.ploting import plot_curves, plot_images

# CODE DESCRIPTION FOR THE IMPLEMENTED OCR METHODS
PENALIDADE = "Penalidade"
BARREIRA = "Barreira"

# MINIMIZATION METHODS
UNIVARIANTE = "Univariante"
POWELL = "Powell"
STEEPEST_DESCENT = "Steepest Descent"
FLETCHER_REEVES = "Fletcher Reeves"
BFGS = "BFGS"
NEWTON_RAPHSON = "Newton Raphson"


## PROBLEMA 1
def problema1(
    method: str = PENALIDADE or BARREIRA,
) -> (
    tuple[
        float,
        float,
        np.ndarray,
        Callable[[np.ndarray], float],
        Callable[[np.ndarray], np.ndarray],
        Callable[[np.ndarray], np.ndarray],
        list[Callable[[np.ndarray], float]],
        list[Callable[[np.ndarray], np.ndarray]],
        list[Callable[[np.ndarray], np.ndarray]],
        list[Callable[[np.ndarray], float]],
        list[Callable[[np.ndarray], np.ndarray]],
        list[Callable[[np.ndarray], np.ndarray]],
    ]
    | NotImplementedError
):
    """
    Função para retornar os parâmetros necessários para solucionar o problema 1
    do trabalho 2.

    Parâmetros:
    -----------
    method: str
        Método a ser utilizado para resolver o problema.
        Opções: PENALIDADE, BARREIRA

    Retorno:
    --------
    rp: float
        Parâmetro de penalidade

    beta: float
        Parâmetro de penalidade

    x0: np.ndarray
        Ponto inicial

    f: Callable
        Função objetivo

    grad_f: Callable
        Gradiente da função objetivo

    hess_f: Callable
        Hessiana da função objetivo

    hs: list[Callable]
        Restrições de igualdade

    grad_hs: list[Callable]
        Gradientes das restrições de igualdade

    hess_hs: list[Callable]
        Hessiana das restrições de igualdade

    cs: list[Callable]
        Restrições de desigualdade

    grad_cs: list[Callable]
        Gradientes das restrições de desigualdade

    hess_cs: list[Callable]
        Hessiana das restrições de desigualdade
    """

    def f(x: np.ndarray) -> float:
        """Função objetivo"""
        x1 = x[0]
        x2 = x[1]
        return (x1 - 2.0) ** 2 + (x2 - 2.0) ** 2

    def grad_f(x: np.ndarray) -> np.ndarray:
        """Gradiente da função objetivo"""
        x1 = x[0]
        x2 = x[1]
        return np.array([2 * (x1 - 2), 2 * (x2 - 2)], dtype=np.float64)

    def hess_f(x: np.ndarray) -> np.ndarray:
        """Hessiana da função objetivo"""
        return np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)

    def c(x: np.ndarray) -> float:
        """Restrição de desigualdade"""
        x1 = x[0]
        x2 = x[1]
        return x1 + x2 + 3.0

    def grad_c(x: np.ndarray) -> np.ndarray:
        """Gradiente da restrição de desigualdade"""
        return np.array([1.0, 1.0], dtype=np.float64)

    def hess_c(x: np.ndarray) -> np.ndarray:
        """Hessiana da restrição de desigualdade"""
        return np.array([[0, 0], [0, 0]], dtype=np.float64)

    if method == PENALIDADE:
        rp = 0.1
        beta = 10
        x0 = np.array([-5.0, -2.0], dtype=np.float64)
        return rp, beta, x0, f, grad_f, hess_f, [], [], [], [c], [grad_c], [hess_c]

    if method == BARREIRA:
        rp = 0.1
        beta = 10
        x0 = np.array([-5.0, -2.0], dtype=np.float64)
        return rp, beta, x0, f, grad_f, hess_f, [], [], [], [c], [grad_c], [hess_c]

    raise NotImplementedError


## PROBLEMA 2
def problema2() -> tuple[Callable, list[Callable], list[Callable]]:
    """
    Retorna a função objetivo e as restrições do problema 2.
    """

    def f(x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (x1 - 2) ** 4 + (x1 - 2 * x2) ** 2

    def c(x: np.ndarray) -> float:
        """
        Menor igual a 0
        """
        x1 = x[0]
        x2 = x[1]
        return x1**2 - x2 + 3

    return f, [], [c]


## PROBLEMA 3
def problema3() -> tuple[Callable, list[Callable], list[Callable]]:
    """
    Retorna a função objetivo e as restrições do problema 2.
    """

    def f(x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return NotImplementedError

    def h(x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return NotImplementedError

    def c(x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return NotImplementedError

    return f, [h], [c]


def main():
    method = PENALIDADE
    tol = 1e-6
    show_fig = True
    min_verbose = True
    rp, beta, x0, f, grad_f, hess_f, hs, grad_hs, hess_hs, cs, grad_cs, hess_cs = (
        problema1(method)
    )

    # uni_min = OCR(
    #     x0,
    #     rp,
    #     beta,
    #     f,
    #     grad_f,
    #     hess_f,
    #     hs,
    #     grad_hs,
    #     hess_hs,
    #     cs,
    #     grad_cs,
    #     hess_cs,
    #     UNIVARIANTE,
    #     minimizer,
    #     ocr_method=method,
    #     alfa=0.001,
    #     tol=tol,
    #     show_fig=show_fig,
    #     min_verbose=min_verbose,
    # )
    # pow_min = OCR(
    #     x0,
    #     rp,
    #     beta,
    #     f,
    #     grad_f,
    #     hess_f,
    #     hs,
    #     grad_hs,
    #     hess_hs,
    #     cs,
    #     grad_cs,
    #     hess_cs,
    #     POWELL,
    #     minimizer,
    #     ocr_method=method,
    #     alfa=0.1,
    #     tol=tol,
    #     show_fig=show_fig,
    #     min_verbose=min_verbose,
    # )
    # ste_min = OCR(
    #     x0,
    #     rp,
    #     beta,
    #     f,
    #     grad_f,
    #     hess_f,
    #     hs,
    #     grad_hs,
    #     hess_hs,
    #     cs,
    #     grad_cs,
    #     hess_cs,
    #     STEEPEST_DESCENT,
    #     minimizer,
    #     ocr_method=method,
    #     alfa=0.5,
    #     tol=tol,
    #     show_fig=show_fig,
    #     min_verbose=min_verbose,
    # )
    fle_min = OCR(
        x0,
        rp,
        beta,
        f,
        grad_f,
        hess_f,
        hs,
        grad_hs,
        hess_hs,
        cs,
        grad_cs,
        hess_cs,
        FLETCHER_REEVES,
        minimizer,
        ocr_method=method,
        alfa=0.001,
        tol=tol,
        show_fig=show_fig,
        min_verbose=min_verbose,
    )
    # new_min = OCR(
    #     x0,
    #     rp,
    #     beta,
    #     f,
    #     grad_f,
    #     hess_f,
    #     hs,
    #     grad_hs,
    #     hess_hs,
    #     cs,
    #     grad_cs,
    #     hess_cs,
    #     NEWTON_RAPHSON,
    #     minimizer,
    #     ocr_method=method,
    #     alfa=0.001,
    #     tol=tol,
    #     show_fig=show_fig,
    #     min_verbose=min_verbose,
    # )
    # bfgs_min = OCR(
    #     x0,
    #     rp,
    #     beta,
    #     f,
    #     grad_f,
    #     hess_f,
    #     hs,
    #     grad_hs,
    #     hess_hs,
    #     cs,
    #     grad_cs,
    #     hess_cs,
    #     BFGS,
    #     minimizer,
    #     ocr_method=method,
    #     alfa=0.001,
    #     tol=tol,
    #     show_fig=show_fig,
    #     min_verbose=min_verbose,
    # )

    print(f"Resultados para OCR com método: {method}")
    # print(f"{method} {UNIVARIANTE}:      {uni_min}")
    # print(f"{method} {POWELL}:           {pow_min}")
    # print(f"{method} {STEEPEST_DESCENT}: {ste_min}")
    print(f"{method} {FLETCHER_REEVES}:  {fle_min}")
    # print(f"{method} {NEWTON_RAPHSON}:   {new_min}")
    # print(f"{method} {BFGS}:             {bfgs_min}")


# IMPLEMENTAR ESSA ABORDAGEM NESSE CÓDIGO
def methods(
    p0: np.ndarray,
    func: Callable,
    func_grad: Callable,
    func_hess: Callable,
    show_curves=False,
):
    start_time = datetime.now()
    min_uni, points_uni = univariante(p0, func, func_grad, verbose=False, monitor=True)
    time_uni = datetime.now() - start_time

    start_time = datetime.now()
    min_pow, points_pow = powell(p0, func, func_grad, verbose=False, monitor=True)
    time_pow = datetime.now() - start_time

    start_time = datetime.now()
    min_ste, points_ste = steepest_descent(
        p0, func, func_grad, verbose=False, monitor=True
    )
    time_ste = datetime.now() - start_time

    start_time = datetime.now()
    min_fle, points_fle = fletcher_reeves(
        p0, func, func_grad, verbose=False, monitor=True
    )
    time_fle = datetime.now() - start_time

    start_time = datetime.now()
    min_bfgs, points_bfgs = bfgs(p0, func, func_grad, verbose=False, monitor=True)
    time_bfgs = datetime.now() - start_time

    start_time = datetime.now()
    min_rap, points_rap = newton_raphson(
        p0, func, func_grad, func_hess, verbose=False, monitor=True
    )
    time_rap = datetime.now() - start_time

    print(
        f"Univariante:      [{min_uni[0]:<22}, {min_uni[1]:<22}] f={func(min_uni):<23} passos={len(points_uni)-1:>4} tempo={g(time_uni):>14}"
    )
    print(
        f"Powell:           [{min_pow[0]:<22}, {min_pow[1]:<22}] f={func(min_pow):<23} passos={len(points_pow)-1:>4} tempo={g(time_pow):>14}"
    )
    print(
        f"Steepest Descent: [{min_ste[0]:<22}, {min_ste[1]:<22}] f={func(min_ste):<23} passos={len(points_ste)-1:>4} tempo={g(time_ste):>14}"
    )
    print(
        f"Fletcher Reeves:  [{min_fle[0]:<22}, {min_fle[1]:<22}] f={func(min_fle):<23} passos={len(points_fle):>4} tempo={g(time_fle):>14}"
    )
    print(
        f"BFGS:             [{min_bfgs[0]:<22}, {min_bfgs[1]:<22}] f={func(min_bfgs):<23} passos={len(points_bfgs):>4} tempo={g(time_bfgs):>14}"
    )
    print(
        f"Newton Raphson:   [{min_rap[0]:<22}, {min_rap[1]:<22}] f={func(min_rap):<23} passos={len(points_rap):>4} tempo={g(time_rap):>14}"
    )

    if show_curves:
        uni_fig = plot_curves(points_uni, func, title="Univariante")
        pow_fig = plot_curves(points_pow, func, title="Powell")
        ste_fig = plot_curves(points_ste, func, title="Steepest Descent")
        fle_fig = plot_curves(points_fle, func, title="Fletcher Reeves")
        bfgs_fig = plot_curves(points_bfgs, func, title="BFGS")
        rap_fig = plot_curves(points_rap, func, title="Newton Raphson")
        plot_images([uni_fig, pow_fig, ste_fig, fle_fig, bfgs_fig, rap_fig])


def minimizer(
    method: str,
    x: np.ndarray,
    f: Callable,
    func_grad: Callable,
    func_hess: Callable,
    alfa: float,
    tol: float,
    verbose: bool = False,
) -> np.ndarray | NotImplementedError:
    """
    Parameters:
    -----------
    f: Callable
        Função objetivo a ser minimizada

    x: np.ndarray
        Ponto inicial

    method: str
        Método de otimização a ser utilizado.
        Opções: UNIVARIANTE, POWELL, STEEPEST_DESCENT, FLETCHER_REEVES, BFGS, NEWTON_RAPHSON
    """
    func = f
    p0 = x

    if method == UNIVARIANTE:
        return univariante(
            p0,
            func,
            func_grad,
            alfa,
            tol,
            n_max_steps=1000,
            verbose=verbose,
            monitor=True,
        )

    if method == POWELL:
        return powell(
            p0,
            func,
            func_grad,
            alfa,
            tol,
            n_max_steps=1000,
            verbose=verbose,
            monitor=True,
        )

    if method == STEEPEST_DESCENT:
        return steepest_descent(
            p0,
            func,
            func_grad,
            alfa,
            tol,
            n_max_steps=1000,
            verbose=verbose,
            monitor=True,
        )

    if method == FLETCHER_REEVES:
        return fletcher_reeves(
            p0,
            func,
            func_grad,
            alfa,
            tol,
            n_max_steps=1000,
            verbose=verbose,
            monitor=True,
        )

    if method == BFGS:
        return bfgs(
            p0,
            func,
            func_grad,
            alfa,
            tol,
            n_max_steps=1000,
            verbose=verbose,
            monitor=True,
        )

    if method == NEWTON_RAPHSON:
        return newton_raphson(
            p0,
            func,
            func_grad,
            func_hess,
            alfa,
            tol,
            n_max_steps=1000,
            verbose=verbose,
            monitor=True,
        )

    raise NotImplementedError


def OCR(
    x0: np.ndarray,
    rp: float,
    beta: float,
    f: Callable,
    func_grad: Callable,
    func_hess: Callable,
    hs: list[Callable],
    grad_hs: list[Callable],
    hess_hs: list[Callable],
    cs: list[Callable],
    grad_cs: list[Callable],
    hess_cs: list[Callable],
    min_method: str,
    minimizer: Callable,
    ocr_method: str,
    alfa: float,
    tol: float,
    show_fig=False,
    min_verbose=False,
):
    if beta <= 1:
        raise Exception(r("beta deve ser maior que 1"))

    print(y(f" -> Inicializando OCR com {min_method}"))

    # TODO: utilização ou não da restrição de desigualdade por vetores
    def p(x: np.ndarray) -> float:
        """Função penalidade"""
        return sum(h(x) ** 2 for h in hs) + sum(max(0.0, c(x)) ** 2 for c in cs)

    def grad_p(x: np.ndarray) -> np.ndarray:
        """Gradiente da função penalidade"""
        igualdade = 0.0
        for h, grad_h in zip(hs, grad_hs):
            igualdade += 2.0 * h(x) * grad_h(x)

        desigualdade = 0.0
        for c, grad_c in zip(cs, grad_cs):
            desigualdade += 2.0 * max(0.0, c(x)) * grad_c(x)

        return igualdade + desigualdade

    def hess_p(x: np.ndarray) -> np.ndarray:
        """Hessiana da função penalidade"""
        igualdade = 0.0
        for h, grad_h, hess_h in zip(hs, grad_hs, hess_hs):
            igualdade += 2.0 * h(x) * hess_h(x) + 2.0 * np.outer(grad_h(x), grad_h(x))

        desigualdade = 0.0
        for c, grad_c, hess_c in zip(cs, grad_cs, hess_cs):
            desigualdade += 2.0 * max(0.0, c(x)) * hess_c(x) + 2.0 * np.outer(
                grad_c(x), grad_c(x)
            )

        return igualdade + desigualdade

    # print(f"p({x0}) = {p(x0)}")
    # print(f"grad_p({x0}) = {grad_p(x0)}")
    # print(f"hess_p({x0}) = {hess_p(x0)}")

    x_buff = x0
    rp_buff = rp
    __n_max_iter = 20

    i = 0
    while True:
        # Definir pseudo função objetivo
        def fi(x: np.ndarray):
            return f(x) + (1.0 / 2.0) * rp_buff * p(x)

        def grad_fi(x: np.ndarray):
            return func_grad(x) + (1.0 / 2.0) * rp_buff * grad_p(x)

        def hess_fi(x: np.ndarray):
            return func_hess(x) + (1.0 / 2.0) * rp_buff * hess_p(x)

        # Minimizar a pseudo função objetivo utilizando OSR
        x_min, points = minimizer(
            min_method, x_buff, fi, grad_fi, hess_fi, alfa, tol, min_verbose
        )

        # Verificar convergência
        conv_value = (1.0 / 2.0) * rp_buff * p(x_min)
        print(
            f"Iteração {g(f'{i+1:>3}')}: "
            f"Critério de convergência: {conv_value:<22} "
            f"mínimo: {x_min[0]:<22}, {x_min[1]:<22} "
            f"n_passos {min_method}: {y(len(points)-1):>5}"
        )

        if show_fig:
            plot_curves(
                points,
                fi,
                title=f"Pseudo função objetivo {i+1} - metodo: {min_method} rp: {rp_buff}",
                show_fig=True,
            )
        # print(f"Mínimo encontrado: {x_min}")

        if abs(conv_value) < tol:
            print(
                f"Convergência para {y(min_method)} atingida com niter={y(__n_max_iter)}"
            )
            return x_min

        # Atualizar rp
        rp_buff = beta * rp_buff
        x_buff = x_min
        i += 1

        # Evitando loop infinito
        if i == __n_max_iter:
            raise Exception(r("Número máximo de iterações atingido"))


if __name__ == "__main__":
    main()
