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
from optimization.colors import red as r, green as g
from optimization.ploting import plot_curves, plot_images


## PROBLEMA 1
def problema1() -> (
    tuple[float, float, np.ndarray, Callable, list[Callable], list[Callable]]
):
    """
    Função para retornar os parâmetros necessários para solucionar o problema 1
    do trabalho 2.

    f: Função objetivo

    h: Restrições de igualdade

    c: Restrições de desigualdade
    """
    rp = 0.1
    beta = 10
    x0 = np.array([-5, -2])

    def f(x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (x1 - 2) ** 2 + (x2 - 2) ** 2

    def grad_f(x: np.ndarray) -> np.ndarray:
        x1 = x[0]
        x2 = x[1]
        return np.array([2 * (x1 - 2), 2 * (x2 - 2)])

    def hess_f(x: np.ndarray) -> np.ndarray:
        return np.array([[2, 0], [0, 2]])

    def c(x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return x1 + x2 + 3

    def grad_c(x: np.ndarray) -> np.ndarray:
        x1 = x[0]
        x2 = x[1]
        return np.array([1, 1])

    def hess_c(x: np.ndarray) -> np.ndarray:
        return np.zeros((2, 2))

    return rp, beta, x0, f, grad_f, hess_f, [], [], [], [c], [grad_c], [hess_c]


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


def main(method: str = "penalidade" or "barreira"):
    alfa = 1e-4
    tol = 1e-6
    show_fig = False
    rp, beta, x0, f, grad_f, hess_f, hs, grad_hs, hess_hs, cs, grad_cs, hess_cs = (
        problema1()
    )

    if method == "penalidade":
        uni_min = penalidade(
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
            "univariante",
            minimizer,
            alfa=alfa,
            tol=tol,
            show_fig=show_fig,
        )
        pow_min = penalidade(
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
            "powell",
            minimizer,
            alfa=alfa,
            tol=tol,
            show_fig=show_fig,
        )
        ste_min = penalidade(
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
            "steepest_descent",
            minimizer,
            alfa=alfa,
            tol=tol,
            show_fig=show_fig,
        )
        fle_min = penalidade(
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
            "fletcher_reeves",
            minimizer,
            alfa=alfa,
            tol=tol,
            show_fig=show_fig,
        )
        bfgs_min = penalidade(
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
            "bfgs",
            minimizer,
            alfa=alfa,
            tol=tol,
            show_fig=show_fig,
        )
        new_min = penalidade(
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
            "newton_raphson",
            minimizer,
            alfa=alfa,
            tol=tol,
            show_fig=show_fig,
        )

        print(f"{method} Univariante:      {uni_min}")
        print(f"{method} Powell:           {pow_min}")
        print(f"{method} Steepest Descent: {ste_min}")
        print(f"{method} Fletcher Reeves:  {fle_min}")
        print(f"{method} BFGS:             {bfgs_min}")
        print(f"{method} Newton Raphson:   {new_min}")
    elif method == "barreira":
        min = barreira()
    else:
        raise NotImplementedError

    # print(f"Minimo utilizando método da {method}: {min}")


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
        Opções: "univariante", "powell", "steepest_descent", "fletcher_reeves", "bfgs", "newton_raphson"
    """
    func = f
    p0 = x
    verbose = False

    if method == "univariante":
        return univariante(
            p0, func, func_grad, alfa, tol, verbose=verbose, monitor=True
        )
    elif method == "powell":
        return powell(p0, func, func_grad, alfa, tol, verbose=verbose, monitor=True)
    elif method == "steepest_descent":
        return steepest_descent(
            p0, func, func_grad, alfa, tol, verbose=verbose, monitor=True
        )
    elif method == "fletcher_reeves":
        return fletcher_reeves(
            p0, func, func_grad, alfa, tol, verbose=verbose, monitor=True
        )
    elif method == "bfgs":
        return bfgs(p0, func, func_grad, alfa, tol, verbose=verbose, monitor=True)
    elif method == "newton_raphson":
        return newton_raphson(
            p0, func, func_grad, func_hess, alfa, tol, verbose=verbose, monitor=True
        )
    else:
        raise NotImplementedError


def penalidade(
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
    method: str,
    minimizer: Callable,
    alfa: float,
    tol: float,
    show_fig=False,
):
    if beta <= 1:
        raise Exception(r("beta deve ser maior que 1"))

    def p(x: np.ndarray) -> float:
        """Função penalidade"""
        return sum(h(x) ** 2 for h in hs) + sum(max(0, c(x)) for c in cs)

    def grad_p(x: np.ndarray) -> np.ndarray:
        """Gradiente da função penalidade"""
        igualdade = 0
        for h, grad_h in zip(hs, grad_hs):
            igualdade += 2 * h(x) * grad_h(x)

        desigualdade = 0
        for c, grad_c in zip(cs, grad_cs):
            desigualdade += np.maximum(0, 2 * c(x) * grad_c(x))

        return igualdade + desigualdade

    def hess_p(x: np.ndarray) -> np.ndarray:
        """Hessiana da função penalidade"""
        igualdade = 0
        for h, grad_h, hess_h in zip(hs, grad_hs, hess_hs):
            igualdade += 2 * h(x) * hess_h(x) + 2 * np.outer(grad_h(x), grad_h(x))

        desigualdade = 0
        for c, grad_c, hess_c in zip(cs, grad_cs, hess_cs):
            desigualdade += np.maximum(
                0, 2 * c(x) * hess_c(x) + 2 * np.outer(grad_c(x), grad_c(x))
            )

        return igualdade + desigualdade

    x_buff = x0
    rp_buff = rp
    __n_max_iter = 20
    while True:
        # Definir pseudo função objetivo
        def fi(x: np.ndarray):
            return f(x) + 1 / 2 * rp_buff * p(x)

        def grad_fi(x: np.ndarray):
            return func_grad(x) + 1 / 2 * rp_buff * grad_p(x)

        def hess_fi(x: np.ndarray):
            return func_hess(x) + 1 / 2 * rp_buff * hess_p(x)

        # Minimizar a pseudo função objetivo utilizando OSR
        x_min, points = minimizer(method, x_buff, fi, grad_fi, hess_fi, alfa, tol)
        if show_fig:
            plot_curves(
                points,
                fi,
                title=f"Pseudo função objetivo - rp={rp_buff}",
                show_fig=True,
            )
        # print(f"Mínimo encontrado: {x_min}") 

        # Verificar convergência
        if 1 / 2 * rp_buff * p(x_min) < tol:
            print(f"Convergência atingida com niter={__n_max_iter}")
            return x_min

        # Atualizar rp
        rp_buff = beta * rp_buff

        # Evitando loop infinito
        if __n_max_iter == 0:
            raise Exception(r("Número máximo de iterações atingido"))
        __n_max_iter -= 1


def barreira():
    pass


if __name__ == "__main__":
    main()
