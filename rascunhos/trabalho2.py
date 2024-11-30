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
from optimization.ploting import (
    plot_curves,
    plot_images,
    plot_restriction_curves,
    plot_figs,
)

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
        beta = 10.0
        x0 = np.array([-5.0, -2.0], dtype=np.float64)
        return rp, beta, x0, f, grad_f, hess_f, [], [], [], [c], [grad_c], [hess_c]

    if method == BARREIRA:
        rb = 10.0
        beta = 0.1
        x0 = np.array([-5.0, -2.0], dtype=np.float64)
        return rb, beta, x0, f, grad_f, hess_f, [], [], [], [c], [grad_c], [hess_c]

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
    method = PENALIDADE  # PENALIDADE ou BARREIRA
    show_fig = True
    min_verbose = True
    (
        r_method,
        beta,
        x0,
        f,
        grad_f,
        hess_f,
        hs,
        grad_hs,
        hess_hs,
        cs,
        grad_cs,
        hess_cs,
    ) = problema1(method)

    start_time = datetime.now()
    uni_min = OCR(
        x0,
        r_method,
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
        UNIVARIANTE,
        minimizer,
        ocr_method=method,
        alfa=0.1,
        tol_ocr=1e-6,
        tol_grad=1e-3,
        tol_line=1e-9,
        max_steps_min=1000,
        show_fig=show_fig,
        min_verbose=min_verbose,
    )
    time_uni = datetime.now() - start_time

    start_time = datetime.now()
    pow_min = OCR(
        x0,
        r_method,
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
        POWELL,
        minimizer,
        ocr_method=method,
        alfa=0.1,
        tol_ocr=1e-6,
        tol_grad=1e-4,
        tol_line=1e-6,
        max_steps_min=250,
        show_fig=show_fig,
        min_verbose=min_verbose,
    )
    time_pow = datetime.now() - start_time

    start_time = datetime.now()
    ste_min = OCR(
        x0,
        r_method,
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
        STEEPEST_DESCENT,
        minimizer,
        ocr_method=method,
        alfa=0.1,
        tol_ocr=1e-4,
        tol_grad=1e-4,
        tol_line=1e-6,
        max_steps_min=20,
        show_fig=show_fig,
        min_verbose=min_verbose,
    )
    time_ste = datetime.now() - start_time

    start_time = datetime.now()
    fle_min = OCR(
        x0,
        r_method,
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
        alfa=0.1,
        tol_ocr=1e-3,
        tol_grad=1e-4,
        tol_line=1e-6,
        max_steps_min=20,
        show_fig=show_fig,
        min_verbose=min_verbose,
    )
    time_fle = datetime.now() - start_time

    start_time = datetime.now()
    new_min = OCR(
        x0,
        r_method,
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
        NEWTON_RAPHSON,
        minimizer,
        ocr_method=method,
        alfa=0.1,
        tol_ocr=1e-5,
        tol_grad=1e-3,
        tol_line=1e-5,
        max_steps_min=10,
        show_fig=show_fig,
        min_verbose=min_verbose,
    )
    time_bfgs = datetime.now() - start_time

    start_time = datetime.now()
    bfgs_min = OCR(
        x0,
        r_method,
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
        BFGS,
        minimizer,
        ocr_method=method,
        alfa=0.1,
        tol_ocr=1e-6,
        tol_grad=1e-5,
        tol_line=1e-6,
        max_steps_min=300,
        show_fig=show_fig,
        min_verbose=min_verbose,
    )
    time_rap = datetime.now() - start_time

    # TODO: Adicionar o número de passos na análise
    print(f"Resultados para OCR com método: {y(method)}")
    print(f"{method} {UNIVARIANTE}:      {uni_min}, tempo: {g(time_uni)}")
    print(f"{method} {POWELL}:           {pow_min}, tempo: {g(time_pow)}")
    print(f"{method} {STEEPEST_DESCENT}: {ste_min}, tempo: {g(time_ste)}")
    print(f"{method} {FLETCHER_REEVES}:  {fle_min}, tempo: {g(time_fle)}")
    print(f"{method} {NEWTON_RAPHSON}:   {new_min}, tempo: {g(time_bfgs)}")
    print(f"{method} {BFGS}:             {bfgs_min}, tempo: {g(time_rap)}")


def minimizer(
    method: str,
    x: np.ndarray,
    f: Callable,
    func_grad: Callable,
    func_hess: Callable,
    alfa: float,
    tol_grad: float,
    tol_line: float,
    n_max_steps: int = 1000,
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
            tol_grad,
            tol_line,
            n_max_steps=n_max_steps,
            verbose=verbose,
            monitor=True,
        )

    if method == POWELL:
        return powell(
            p0,
            func,
            func_grad,
            alfa,
            tol_grad,
            tol_line,
            n_max_steps=n_max_steps,
            verbose=verbose,
            monitor=True,
        )

    if method == STEEPEST_DESCENT:
        return steepest_descent(
            p0,
            func,
            func_grad,
            alfa,
            tol_grad,
            tol_line,
            n_max_steps=n_max_steps,
            verbose=verbose,
            monitor=True,
        )

    if method == FLETCHER_REEVES:
        return fletcher_reeves(
            p0,
            func,
            func_grad,
            alfa,
            tol_grad,
            tol_line,
            n_max_steps=n_max_steps,
            verbose=verbose,
            monitor=True,
        )

    if method == BFGS:
        return bfgs(
            p0,
            func,
            func_grad,
            alfa,
            tol_grad,
            tol_line,
            n_max_steps=n_max_steps,
            verbose=verbose,
            monitor=True,
        )

    if method == NEWTON_RAPHSON:
        return newton_raphson(
            p0=p0,
            func=func,
            f_grad=func_grad,
            f_hess=func_hess,
            alfa=alfa,
            tol_grad=tol_grad,
            tol_line=tol_line,
            n_max_steps=n_max_steps,
            verbose=verbose,
            monitor=True,
        )

    raise NotImplementedError


def OCR(
    x0: np.ndarray,
    r_method: float,
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
    tol_ocr: float,
    tol_grad: float,
    tol_line: float,
    max_steps_ocr: int = 20,
    max_steps_min: int = 1000,
    show_fig=False,
    min_verbose=False,
):
    if ocr_method == PENALIDADE and beta <= 1:
        raise Exception(r("beta deve ser maior que 1"))
    if ocr_method == BARREIRA and beta <= 0:
        raise Exception(r("beta deve ser maior que 0"))

    print(y(f" -> Inicializando OCR com {min_method}"))

    x_buff = x0
    r_buff = r_method
    alfa_buff = alfa

    i = 0
    while True:
        # hs_indexes = np.where(np.array([h(x_buff) for h in hs]) != 0)[0]
        cs_indexes = np.argwhere(
            np.array([max(0, c(x_buff)) for c in cs]) > 0
        ).flatten()

        if len(cs_indexes) == 0:
            print(b(f"Ponto {x_buff} interno da restrição de desigualdade"))

        def p(x: np.ndarray) -> float:
            """Função penalidade"""
            igualdade = sum(h(x) ** 2 for h in hs)

            if ocr_method == PENALIDADE:
                desigualdade = sum(cs[idx](x) ** 2 for idx in cs_indexes)

            if ocr_method == BARREIRA:
                desigualdade = sum(-(cs[idx](x) ** -1) for idx in cs_indexes)

            return igualdade + desigualdade

        def grad_p(x: np.ndarray) -> np.ndarray:
            """Gradiente da função penalidade"""
            igualdade = np.zeros(x.shape, dtype=np.float64)
            for h, grad_h in zip(hs, grad_hs):
                igualdade += h(x) * grad_h(x)

            desigualdade = np.zeros(x.shape, dtype=np.float64)
            if ocr_method == PENALIDADE:
                for idx in cs_indexes:
                    desigualdade += 2 * cs[idx](x) * grad_cs[idx](x)

            if ocr_method == BARREIRA:
                for idx in cs_indexes:
                    desigualdade += cs[idx](x) ** -2

            return igualdade + desigualdade

        def hess_p(x: np.ndarray) -> np.ndarray:
            """Hessiana da função penalidade"""
            igualdade = np.zeros((2, 2), dtype=np.float64)
            for h, grad_h, hess_h in zip(hs, grad_hs, hess_hs):
                igualdade += h(x) * hess_h(x) + np.outer(grad_h(x), grad_h(x))

            desigualdade = np.zeros((2, 2), dtype=np.float64)
            if ocr_method == PENALIDADE:
                for idx in cs_indexes:
                    desigualdade += cs[idx](x) * hess_cs[idx](x) + np.outer(
                        grad_cs[idx](x), grad_cs[idx](x)
                    )

            if ocr_method == BARREIRA:
                for idx in cs_indexes:
                    desigualdade += -2 * (cs[idx](x) ** -3)

            return igualdade + desigualdade

        # Definir pseudo função objetivo
        def fi(x: np.ndarray):
            return f(x) + (1 / 2) * r_buff * p(x)

        def grad_fi(x: np.ndarray):
            return func_grad(x) + (1 / 2) * r_buff * grad_p(x)

        def hess_fi(x: np.ndarray):
            return func_hess(x) + r_buff * hess_p(x)

        # Minimizar a pseudo função objetivo utilizando OSR
        x_min, points = minimizer(
            method=min_method,
            x=x_buff,
            f=fi,
            func_grad=grad_fi,
            func_hess=hess_fi,
            alfa=alfa_buff,
            tol_grad=tol_grad,
            tol_line=tol_line,
            n_max_steps=max_steps_min,
            verbose=min_verbose,
        )
        # Implementando alfa adaptativo para BARREIRA
        if ocr_method == BARREIRA:
            k = np.argwhere(np.array([max(0, c(x_min)) for c in cs]) > 0).flatten()
            """
            caso k seja vazio, o ponto está dentro da região da restrição
            caso contrário, o ponto está fora da região da restrição
            se estiver fora da região de restrição, siginifica que o alfa é muito grande
            e deve ser reduzido
            """
            if not len(k) == 0:  # saiu da região
                alfa_buff = alfa_buff / 10
                print(f"Alfa adaptativo: {alfa_buff} " f"x_min: {x_min}")
                continue
            else:
                alfa_buff = alfa

        # Contabilizando iterações
        i += 1

        # Analisando convergência
        conv_value = r_buff * p(x_min)
        print(
            f"Iteração {g(f'{i:>3}')}: "
            f"Critério de convergência: {conv_value:<22} "
            f"mínimo: [{x_min[0]:>22}, {x_min[1]:>22}] "
            f"n_passos: {min_method}: {y(len(points)-1):>5}"
        )
        if show_fig:
            fig_fi = plot_curves(
                points,
                fi,
                title=f"Pseudo função objetivo {i} - metodo: {min_method} r: {r_buff}",
                show_fig=False,
            )
            fig_ocr = plot_restriction_curves(
                points,
                fi,
                hs,
                cs,
                show_fig=False,
                title=f"Gráfico de f(x), h(x) e c(x) passo {i} OCR - metodo: {min_method} r: {r_buff}",
            )
            fig_ocr.savefig(f"ocr_{ocr_method}_{min_method}_{i}.png")
            # fig_result = plot_figs(fig_fi, fig_ocr, show_fig=False)
            # fig_result.savefig(f"ocr_{ocr_method}_{min_method}_{i}.png")

        # Atualizar rp
        r_buff = beta * r_buff
        x_buff = x_min

        if abs(conv_value) < tol_ocr:
            if len(cs_indexes) == 0:
                continue
            print(
                f"Convergência para OCR com {y(min_method)} atingida com niter={y(i)}"
            )
            return x_min

        # Evitando loop infinito
        if i == max_steps_ocr:
            raise Exception(r("Número máximo de iterações atingido"))


if __name__ == "__main__":
    main()
