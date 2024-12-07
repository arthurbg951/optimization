"""
Considerações de notação:

f -> Função objetivo
h -> Restrições de igualdade
c -> Restrições de desigualdade

x -> Ponto no espaço de busca

OSR -> Otimização sem restrições
"""

import os
import math
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

# SETTINGS
SAVE_MIN_FIG = True


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
def problema2(
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
    def f(x: np.ndarray) -> float:
        """Função objetivo"""
        x1 = x[0]
        x2 = x[1]
        return (x1 - 2) ** 4 + (x1 - 2 * x2) ** 2

    def grad_f(x: np.ndarray) -> np.ndarray:
        """Gradiente da função objetivo"""
        x1 = x[0]
        x2 = x[1]
        return np.array(
            [2 * x1 - 4 * x2 + 4 * (x1 - 2) ** 3, -4 * x1 + 8 * x2], dtype=np.float64
        )

    def hess_f(x: np.ndarray) -> np.ndarray:
        """Hessiana da função objetivo"""
        x1 = x[0]
        return np.array([[12 * (x1 - 2) ** 2 + 2, -4], [-4, 8]], dtype=np.float64)

    def c(x: np.ndarray) -> float:
        """Restrição de desigualdade"""
        x1 = x[0]
        x2 = x[1]
        return x1**2 - x2

    def grad_c(x: np.ndarray) -> np.ndarray:
        """Gradiente da restrição de desigualdade"""
        x1 = x[0]
        return np.array([2 * x1, -1], dtype=np.float64)

    def hess_c(x: np.ndarray) -> np.ndarray:
        """Hessiana da restrição de desigualdade"""
        return np.array([[2, 0], [0, 0]], dtype=np.float64)

    if method == PENALIDADE:
        rp = 1
        beta = 10.0
        x0 = np.array([3.0, 2.0], dtype=np.float64)
        return rp, beta, x0, f, grad_f, hess_f, [], [], [], [c], [grad_c], [hess_c]

    if method == BARREIRA:
        rb = 10.0
        beta = 0.1
        x0 = np.array([0.0, 1.0], dtype=np.float64)
        return rb, beta, x0, f, grad_f, hess_f, [], [], [], [c], [grad_c], [hess_c]

    raise NotImplementedError


## PROBLEMA 3
def problema3(
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
    pi = math.pi
    P = 33e3
    E = 3e7
    sigma_y = 1e5
    ro = 0.3
    B = 30.0
    t = 0.1

    def f(x: np.ndarray) -> float:
        """Função objetivo"""
        x1 = x[0]
        x2 = x[1]
        d = x1
        H = x2
        return 2 * ro * pi * d * t * (H**2 + B**2) ** (1 / 2)

    def grad_f(x: np.ndarray) -> np.ndarray:
        """Gradiente da função objetivo"""
        x1 = x[0]
        x2 = x[1]

        # COM VÁRIÁVEIS
        # return np.array(
        #     [
        #         2 * pi * ro * t * (B**2 + x2**2) ** 0.5,
        #         2.0 * pi * ro * t * x1 * x2 / (B**2 + x2**2) ** 0.5,
        #     ],
        #     dtype=np.float64,
        # )
        # SEM VARIÁVEIS
        return np.array(
            [
                5.65486677646163 * (0.00111111111111111 * x2**2 + 1) ** 0.5,
                0.00628318530717959
                * x1
                * x2
                / (0.00111111111111111 * x2**2 + 1) ** 0.5,
            ],
            dtype=np.float64,
        )

    def hess_f(x: np.ndarray) -> np.ndarray:
        """Hessiana da função objetivo"""
        x1 = x[0]
        x2 = x[1]
        # return np.array(
        #     [
        #         [0, 2.0 * pi * ro * t * x2 / (B**2 + x2**2) ** 0.5],
        #         [
        #             2.0 * pi * ro * t * x2 / (B**2 + x2**2) ** 0.5,
        #             -2.0 * pi * ro * t * x1 * x2**2 / (B**2 + x2**2) ** 1.5
        #             + 2.0 * pi * ro * t * x1 / (B**2 + x2**2) ** 0.5,
        #         ],
        #     ],
        #     dtype=np.float64,
        # )
        return np.array(
            [
                [
                    0,
                    0.00628318530717959 * x2 / (0.00111111111111111 * x2**2 + 1) ** 0.5,
                ],
                [
                    0.00628318530717959 * x2 / (0.00111111111111111 * x2**2 + 1) ** 0.5,
                    -6.98131700797732e-6
                    * x1
                    * x2**2
                    / (0.00111111111111111 * x2**2 + 1) ** 1.5
                    + 0.00628318530717959
                    * x1
                    / (0.00111111111111111 * x2**2 + 1) ** 0.5,
                ],
            ],
            np.float64,
        )

    def c1(x: np.ndarray) -> float:
        """Restrição de desigualdade"""
        x1 = x[0]
        x2 = x[1]
        d = x1
        H = x2
        return P * (H**2 + B**2) ** (1 / 2) / (pi * d * t * H) - sigma_y

    def grad_c1(x: np.ndarray) -> np.ndarray:
        """Gradiente da restrição de desigualdade"""
        x1 = x[0]
        x2 = x[1]
        return np.array(
            [
                -3151267.87321953
                * (0.00111111111111111 * x2**2 + 1) ** 0.5
                / (x1**2 * x2),
                3501.4087480217 / (x1 * (0.00111111111111111 * x2**2 + 1) ** 0.5)
                - 3151267.87321953
                * (0.00111111111111111 * x2**2 + 1) ** 0.5
                / (x1 * x2**2),
            ],
            dtype=np.float64,
        )

    def hess_c1(x: np.ndarray) -> np.ndarray:
        """Hessiana da restrição de desigualdade"""
        x1 = x[0]
        x2 = x[1]
        return np.array(
            [
                [
                    6302535.74643906
                    * (0.00111111111111111 * x2**2 + 1) ** 0.5
                    / (x1**3 * x2),
                    -3501.4087480217
                    / (x1**2 * (0.00111111111111111 * x2**2 + 1) ** 0.5)
                    + 3151267.87321953
                    * (0.00111111111111111 * x2**2 + 1) ** 0.5
                    / (x1**2 * x2**2),
                ],
                [
                    -3501.4087480217
                    / (x1**2 * (0.00111111111111111 * x2**2 + 1) ** 0.5)
                    + 3151267.87321953
                    * (0.00111111111111111 * x2**2 + 1) ** 0.5
                    / (x1**2 * x2**2),
                    -3.89045416446855
                    * x2
                    / (x1 * (0.00111111111111111 * x2**2 + 1) ** 1.5)
                    - 3501.4087480217
                    / (x1 * x2 * (0.00111111111111111 * x2**2 + 1) ** 0.5)
                    + 6302535.74643906
                    * (0.00111111111111111 * x2**2 + 1) ** 0.5
                    / (x1 * x2**3),
                ],
            ],
            dtype=np.float64,
        )

    def c2(x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        d = x1
        H = x2
        return P * (H**2 + B**2) ** (1 / 2) / (pi * d * t * H) - pi**2 * E * (
            d**2 + t**2
        ) / (8 * (H**2 + B**2))

    def grad_c2(x: np.ndarray) -> np.ndarray:
        x1 = x[0]
        x2 = x[1]
        d = x1
        H = x2
        return np.array(
            [
                -592176264.065361 * x1 / (8 * x2**2 + 7200.0)
                - 3151267.87321953
                * (0.00111111111111111 * x2**2 + 1) ** 0.5
                / (x1**2 * x2),
                3.08641975308642e-7
                * x2
                * (296088132.032681 * x1**2 + 2960881.32032681)
                / (0.00111111111111111 * x2**2 + 1) ** 2
                + 3501.4087480217 / (x1 * (0.00111111111111111 * x2**2 + 1) ** 0.5)
                - 3151267.87321953
                * (0.00111111111111111 * x2**2 + 1) ** 0.5
                / (x1 * x2**2),
            ],
            dtype=np.float64,
        )

    def hess_c2(x: np.ndarray) -> np.ndarray:
        x1 = x[0]
        x2 = x[1]
        d = x1
        H = x2
        return np.array(
            [
                [
                    -592176264.065361 / (8 * x2**2 + 7200.0)
                    + 6302535.74643906
                    * (0.00111111111111111 * x2**2 + 1) ** 0.5
                    / (x1**3 * x2),
                    182.770451872025 * x1 * x2 / (0.00111111111111111 * x2**2 + 1) ** 2
                    - 3501.4087480217
                    / (x1**2 * (0.00111111111111111 * x2**2 + 1) ** 0.5)
                    + 3151267.87321953
                    * (0.00111111111111111 * x2**2 + 1) ** 0.5
                    / (x1**2 * x2**2),
                ],
                [
                    182.770451872025 * x1 * x2 / (0.00111111111111111 * x2**2 + 1) ** 2
                    - 3501.4087480217
                    / (x1**2 * (0.00111111111111111 * x2**2 + 1) ** 0.5)
                    + 3151267.87321953
                    * (0.00111111111111111 * x2**2 + 1) ** 0.5
                    / (x1**2 * x2**2),
                    -1.37174211248285e-9
                    * x2**2
                    * (296088132.032681 * x1**2 + 2960881.32032681)
                    / (0.00111111111111111 * x2**2 + 1) ** 3
                    + 3.08641975308642e-7
                    * (296088132.032681 * x1**2 + 2960881.32032681)
                    / (0.00111111111111111 * x2**2 + 1) ** 2
                    - 3.89045416446855
                    * x2
                    / (x1 * (0.00111111111111111 * x2**2 + 1) ** 1.5)
                    - 3501.4087480217
                    / (x1 * x2 * (0.00111111111111111 * x2**2 + 1) ** 0.5)
                    + 6302535.74643906
                    * (0.00111111111111111 * x2**2 + 1) ** 0.5
                    / (x1 * x2**3),
                ],
            ],
            dtype=np.float64,
        )

    if method == PENALIDADE:
        rp = 1e-7
        beta = 10.0
        x0 = np.array([1.0, 15.0], dtype=np.float64)
        return (
            rp,
            beta,
            x0,
            f,
            grad_f,
            hess_f,
            [],
            [],
            [],
            [c1, c2],
            [grad_c1, grad_c2],
            [hess_c1, hess_c2],
        )

    if method == BARREIRA:
        rb = 1e7
        beta = 0.1
        x0 = np.array([4.0, 25.0], dtype=np.float64)
        return (
            rb,
            beta,
            x0,
            f,
            grad_f,
            hess_f,
            [],
            [],
            [],
            [c1, c2],
            [grad_c1, grad_c2],
            [hess_c1, hess_c2],
        )

    raise NotImplementedError


def __create_folder(folder_name: str = "imgs") -> str:
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def main():
    ocr_method = BARREIRA
    problem = problema1
    min_verbose = False
    show_min_fig = False
    show_ocr_fig = False
    save_ocr_fig = True

    start_time = datetime.now()
    uni_min, uni_points = ocr_minimizer(
        ocr_method=ocr_method,
        variables=problem,
        min_method=UNIVARIANTE,
        alfa=0.01,
        tol_ocr=1e-4,
        tol_grad=1e-5,
        tol_line=1e-6,
        max_steps_min=1000,
        show_fig=show_min_fig,
        min_verbose=min_verbose,
    )
    time_uni = datetime.now() - start_time

    start_time = datetime.now()
    pow_min, pow_points = ocr_minimizer(
        ocr_method=ocr_method,
        variables=problem,
        min_method=POWELL,
        alfa=0.001,
        tol_ocr=1e-4,
        tol_grad=1e-3,
        tol_line=1e-6,
        max_steps_min=20,
        show_fig=show_min_fig,
        min_verbose=min_verbose,
    )
    time_pow = datetime.now() - start_time

    start_time = datetime.now()
    ste_min, ste_points = ocr_minimizer(
        ocr_method=ocr_method,
        variables=problem,
        min_method=STEEPEST_DESCENT,
        alfa=0.01,
        tol_ocr=1e-3,
        tol_grad=1e-4,
        tol_line=1e-5,
        max_steps_min=50,
        show_fig=show_min_fig,
        min_verbose=min_verbose,
    )
    time_ste = datetime.now() - start_time

    start_time = datetime.now()
    fle_min, fle_points = ocr_minimizer(
        ocr_method=ocr_method,
        variables=problem,
        min_method=FLETCHER_REEVES,
        alfa=0.01,
        tol_ocr=1e-3,
        tol_grad=1e-5,
        tol_line=1e-6,
        max_steps_min=20,
        show_fig=show_min_fig,
        min_verbose=min_verbose,
    )
    time_fle = datetime.now() - start_time

    start_time = datetime.now()
    new_min, new_points = ocr_minimizer(
        ocr_method=ocr_method,
        variables=problem,
        min_method=NEWTON_RAPHSON,
        alfa=0.1,
        tol_ocr=1e-5,
        tol_grad=1e-4,
        tol_line=1e-7,
        max_steps_min=20,
        show_fig=show_min_fig,
        min_verbose=min_verbose,
    )
    time_bfgs = datetime.now() - start_time

    start_time = datetime.now()
    bfgs_min, bfgs_points = ocr_minimizer(
        ocr_method=ocr_method,
        variables=problem,
        min_method=BFGS,
        alfa=0.01,
        tol_ocr=1e-3,
        tol_grad=1e-5,
        tol_line=1e-6,
        max_steps_min=20,
        show_fig=show_min_fig,
        min_verbose=min_verbose,
    )
    time_rap = datetime.now() - start_time

    # TODO: Adicionar o número de passos na análise
    print(f"Resultados para OCR com método: {y(ocr_method)}")
    print(f"{ocr_method} {UNIVARIANTE}:      {uni_min}, tempo: {g(time_uni)}")
    print(f"{ocr_method} {POWELL}:           {pow_min}, tempo: {g(time_pow)}")
    print(f"{ocr_method} {STEEPEST_DESCENT}: {ste_min}, tempo: {g(time_ste)}")
    print(f"{ocr_method} {FLETCHER_REEVES}:  {fle_min}, tempo: {g(time_fle)}")
    print(f"{ocr_method} {NEWTON_RAPHSON}:   {new_min}, tempo: {g(time_bfgs)}")
    print(f"{ocr_method} {BFGS}:             {bfgs_min}, tempo: {g(time_rap)}")

    if show_ocr_fig or save_ocr_fig:
        # TODO: Repensar uma forma alternativa para requisitar os dados
        # talvez uma abordagem OOP resolva o problema elegantemente
        # utilizar um dict também pode ser uma solução

        variables = problem(ocr_method)
        f = lambda x: variables[3](x)  # noqa
        hs = variables[6]
        cs = variables[9]

        uni_fig = plot_restriction_curves(
            uni_points, f, hs, cs, title=UNIVARIANTE, show_fig=show_ocr_fig
        )
        pow_fig = plot_restriction_curves(
            pow_points, f, hs, cs, title=POWELL, show_fig=show_ocr_fig
        )
        ste_fig = plot_restriction_curves(
            ste_points, f, hs, cs, title=STEEPEST_DESCENT, show_fig=show_ocr_fig
        )
        fle_fig = plot_restriction_curves(
            fle_points, f, hs, cs, title=FLETCHER_REEVES, show_fig=show_ocr_fig
        )
        new_fig = plot_restriction_curves(
            new_points, f, hs, cs, title=NEWTON_RAPHSON, show_fig=show_ocr_fig
        )
        bfgs_fig = plot_restriction_curves(
            bfgs_points, f, hs, cs, title=BFGS, show_fig=show_ocr_fig
        )

        if save_ocr_fig:
            results_folder = __create_folder()

            # uni_folder = __create_folder(os.path.join(results_folder, UNIVARIANTE))
            # pow_folder = __create_folder(os.path.join(results_folder, POWELL))
            # ste_folder = __create_folder(os.path.join(results_folder, STEEPEST_DESCENT))
            # fle_folder = __create_folder(os.path.join(results_folder, FLETCHER_REEVES))
            # new_folder = __create_folder(os.path.join(results_folder, NEWTON_RAPHSON))
            # bfgs_folder = __create_folder(os.path.join(results_folder, BFGS))

            uni_fig.savefig(
                os.path.join(results_folder, f"{UNIVARIANTE}_{ocr_method}.png")
            )
            pow_fig.savefig(os.path.join(results_folder, f"{POWELL}_{ocr_method}.png"))
            ste_fig.savefig(
                os.path.join(results_folder, f"{STEEPEST_DESCENT}_{ocr_method}.png")
            )
            fle_fig.savefig(
                os.path.join(results_folder, f"{FLETCHER_REEVES}_{ocr_method}.png")
            )
            new_fig.savefig(
                os.path.join(results_folder, f"{NEWTON_RAPHSON}_{ocr_method}.png")
            )
            bfgs_fig.savefig(os.path.join(results_folder, f"{BFGS}_{ocr_method}.png"))


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


def ocr_minimizer(
    ocr_method=BARREIRA,
    variables=problema2,
    min_method=NEWTON_RAPHSON,
    alfa=0.1,
    tol_ocr=1e-5,
    tol_grad=1e-4,
    tol_line=1e-6,
    max_steps_min=20,
    min_verbose=False,
    show_fig=False,
):
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
    ) = variables(ocr_method)

    min, points = OCR(
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
        min_method=min_method,
        ocr_method=ocr_method,
        alfa=alfa,
        tol_ocr=tol_ocr,
        tol_grad=tol_grad,
        tol_line=tol_line,
        max_steps_min=max_steps_min,
        show_fig=show_fig,
        min_verbose=min_verbose,
    )

    if show_fig:
        plot_restriction_curves(
            points, f, hs, cs, title=f"{ocr_method} com {min_method}", show_fig=show_fig
        )

    return min, points


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
    ocr_method: str,
    alfa: float,
    tol_ocr: float,
    tol_grad: float,
    tol_line: float,
    max_steps_ocr: int = 20,
    max_steps_min: int = 1000,
    show_fig=False,
    save_min_fig=SAVE_MIN_FIG,
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

    hist_points = [x0]

    i = 0
    while True:
        # hs_indexes = np.where(np.array([h(x_buff) for h in hs]) != 0)[0]
        if ocr_method == PENALIDADE:
            cs_indexes = np.argwhere(
                np.array([max(0, c(x_buff)) for c in cs]) > 0
            ).flatten()

        if ocr_method == BARREIRA:
            cs_indexes = np.argwhere(
                np.array([-(c(x_buff) ** -1) for c in cs]) > 0
            ).flatten()

        if len(cs_indexes) == 0:
            print(b(f"Ponto {x_buff} interno da restrição de desigualdade"))

        def p(x: np.ndarray) -> float:
            """Função penalidade"""
            igualdade = sum(h(x) ** 2 for h in hs)

            if ocr_method == PENALIDADE:
                desigualdade = sum(cs[idx](x) ** 2 for idx in cs_indexes)

            if ocr_method == BARREIRA:
                desigualdade = 2 * -sum(cs[idx](x) ** -1 for idx in cs_indexes)

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
                    desigualdade += 2 * cs[idx](x) ** -2 * grad_cs[idx](x)

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
                    primeira = -2 * (cs[idx](x) ** -3) * grad_cs[idx](x)
                    segunda = cs[idx](x) ** -2 * hess_cs[idx](x)
                    desigualdade += 2 * (primeira + segunda)

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
            # k = np.argwhere(np.array([-(c(x_min) ** -1) for c in cs]) > 0).flatten()
            k = float(p(x_min))
            if k < 0.0:
                alfa_buff = alfa_buff / 2
                if min_verbose:
                    print(r(f"Quebrando Passo: {alfa_buff} ") + f"x_buff: {x_buff}")
                continue
            # else:
            #     alfa_buff = alfa
            #     print(g(f"Passo Normal: {alfa_buff} " f"x_min: {x_min}"))

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

        if show_fig or save_min_fig:
            fig_fi = plot_curves(
                points,
                fi,
                title=f"Pseudo função objetivo {i} - metodo: {min_method} r: {r_buff}",
                show_fig=show_fig,
            )
            # JÁ ESTÁ PRESENTE NA FUNÇÃO MAIN
            # fig_ocr = plot_restriction_curves(
            #     hist_points,
            #     f,
            #     hs,
            #     cs,
            #     show_fig=show_fig,
            #     title=f"Gráfico de f(x), h(x) e c(x) passo {i} OCR - metodo: {min_method} r: {r_buff}",
            # )

            if save_min_fig:
                # TODO: remover esse import e alterar o controle do plot das figuras
                # no próprio módulo de plotagem.
                # Utilizar uma figura global privativa e alterar todas as funções dentro
                # do módulo para consumirem a mesma figura. será necessário alterar a
                # adicionar um método para limpar a figura
                import matplotlib.pyplot as plt

                output_folder = __create_folder("ocr_imgs")
                ocr_folder = __create_folder(os.path.join(output_folder, ocr_method))
                min_folder = __create_folder(os.path.join(ocr_folder, min_method))

                fig_fi.savefig(
                    os.path.join(min_folder, f"fi_{ocr_method}_{min_method}_{i}.png")
                )
                plt.close(fig_fi)

                # fig_ocr.savefig(os.path.join(ocr_folder, f"ocr_{ocr_method}_{i}.png"))
                # plt.close(fig_ocr)

            # fig_result = plot_figs(fig_fi, fig_ocr, show_fig=False)
            # fig_result.savefig(f"ocr_{ocr_method}_{min_method}_{i}.png")

        # Atualizar rp
        r_buff = beta * r_buff
        x_buff = x_min

        # Salvando histórico
        hist_points.append(x_min)

        if abs(conv_value) < tol_ocr:
            if len(cs_indexes) == 0:
                continue
            print(
                f"Convergência para OCR com {y(min_method)} atingida com niter={y(i)}"
            )
            print(f"Resultado final: {x_min} f(x): {f(x_min)}")
            return x_min, hist_points

        # Evitando loop infinito
        if i == max_steps_ocr:
            raise Exception(r("Número máximo de iterações atingido"))


if __name__ == "__main__":
    main()
