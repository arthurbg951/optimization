import numpy as np
import math
from datetime import datetime
from typing import Callable

from optimization.linear import make_step, secao_aurea, passo_constante
from optimization.colors import yellow as y, green as g, red as r, blue as b
from optimization.ploting import plot_curves

alfa = 0.01
tol = 1e-5


def f_letra_a(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return x1**2 - 3 * x1 * x2 + 4 * x2**2 + x1 - x2


def grad_letra_a(p: np.ndarray) -> np.ndarray:
    x1, x2 = p[0], p[1]
    return np.array([2 * x1 - 3 * x2 + 1, -3 * x1 + 8 * x2 - 1], dtype=np.float64)


def hessiana_letra_a(p: np.ndarray) -> np.ndarray:
    return np.array([[2, -3], [-3, 8]], dtype=np.float64)


def f_letra_b(p: np.ndarray, a: float = 10, b: float = 1) -> float:
    x1, x2 = p[0], p[1]
    return (1 + a - b * x1 - b * x2) ** 2 + (b + x1 + a * x2 - b * x1 * x2) ** 2


def grad_letra_b(p: np.ndarray, a: float = 10, b: float = 1) -> np.ndarray:
    x1, x2 = p[0], p[1]
    return np.array(
        [
            -2 * b * (a - b * x1 - b * x2 + 1)
            + (-2 * b * x2 + 2) * (a * x2 - b * x1 * x2 + b + x1),
            -2 * b * (a - b * x1 - b * x2 + 1)
            + (2 * a - 2 * b * x1) * (a * x2 - b * x1 * x2 + b + x1),
        ],
        dtype=np.float64,
    )


def hessiana_letra_b(p: np.ndarray, a: float = 10, b: float = 1) -> np.ndarray:
    x1, x2 = p[0], p[1]
    return np.array(
        [
            [
                2 * b**2 + (-2 * b * x2 + 2) * (-b * x2 + 1),
                2 * b**2
                - 2 * b * (a * x2 - b * x1 * x2 + b + x1)
                + (a - b * x1) * (-2 * b * x2 + 2),
            ],
            [
                2 * b**2
                - 2 * b * (a * x2 - b * x1 * x2 + b + x1)
                + (2 * a - 2 * b * x1) * (-b * x2 + 1),
                2 * b**2 + (a - b * x1) * (2 * a - 2 * b * x1),
            ],
        ],
        dtype=np.float64,
    )


def f_letra_c(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return x1**4 + x2**4 - 4 * x1**3 - 3 * x2**3 + 2 * x1**2 + 2 * x1 * x2


def grad_letra_c(p: np.ndarray) -> np.ndarray:
    x1, x2 = p[0], p[1]
    return np.array(
        [4 * x1**3 - 12 * x1**2 + 4 * x1 + 2 * x2, 2 * x1 + 4 * x2**3 - 9 * x2**2],
        dtype=np.float64,
    )


def hessiana_letra_c(p: np.ndarray) -> np.ndarray:
    x1, x2 = p[0], p[1]
    return np.array(
        [[12 * x1**2 - 24 * x1 + 4, 2], [2, 12 * x2**2 - 18 * x2]], dtype=np.float64
    )


def f_letra_d(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2


def grad_letra_d(p: np.ndarray) -> np.ndarray:
    x1, x2 = p[0], p[1]
    return np.array(
        [-400 * x1 * (-(x1**2) + x2) + 2 * x1 - 2, -200 * x1**2 + 200 * x2],
        dtype=np.float64,
    )


def hessiana_letra_d(p: np.ndarray) -> np.ndarray:
    x1, x2 = p[0], p[1]
    return np.array(
        [[1200 * x1**2 - 400 * x2 + 2, -400 * x1], [-400 * x1, 200]], dtype=np.float64
    )


def f_segunda(p: np.ndarray) -> float:
    u, v = p[0], p[1]
    ro1 = 0.82
    ro2 = 0.52

    EA1 = 12
    L1 = 12
    EA2 = 80
    L2 = 8

    P1 = (0.5 * ro1 * L1) + (0.5 * ro2 * L2)
    return (
        0.5 * (EA1 / L1) * (((L1 + u) ** 2 + v**2) ** (1 / 2) - L1) ** 2
        + 0.5 * (EA2 / L2) * (((L2 - u) ** 2 + v**2) ** (1 / 2) - L2) ** 2
        - P1 * v
    )


def grad_segunda(p: np.ndarray) -> np.ndarray:
    u, v = p[0], p[1]
    ro1 = 0.82
    ro2 = 0.52

    EA1 = 12
    L1 = 12
    EA2 = 80
    L2 = 8

    P1 = (0.5 * ro1 * L1) + (0.5 * ro2 * L2)
    return np.array(
        [
            10.0
            * (1.0 * u - 8.0)
            * ((v**2 + (8 - u) ** 2) ** 0.5 - 8)
            / (v**2 + (8 - u) ** 2) ** 0.5
            + 1.0
            * (1.0 * u + 12.0)
            * ((v**2 + (u + 12) ** 2) ** 0.5 - 12)
            / (v**2 + (u + 12) ** 2) ** 0.5,
            10.0 * v * ((v**2 + (8 - u) ** 2) ** 0.5 - 8) / (v**2 + (8 - u) ** 2) ** 0.5
            + 1.0
            * v
            * ((v**2 + (u + 12) ** 2) ** 0.5 - 12)
            / (v**2 + (u + 12) ** 2) ** 0.5
            - 7.0,
        ]
    )


def hessiana_segunda(p: np.ndarray) -> np.ndarray:
    u, v = p[0], p[1]
    ro1 = 0.82
    ro2 = 0.52

    EA1 = 12
    L1 = 12
    EA2 = 80
    L2 = 8

    P1 = (0.5 * ro1 * L1) + (0.5 * ro2 * L2)
    return np.array(
        [
            [
                10.0
                * (8.0 - 1.0 * u)
                * (1.0 * u - 8.0)
                * ((v**2 + (8 - u) ** 2) ** 0.5 - 8)
                / (v**2 + (8 - u) ** 2) ** 1.5
                + 1.0
                * (-1.0 * u - 12.0)
                * (1.0 * u + 12.0)
                * ((v**2 + (u + 12) ** 2) ** 0.5 - 12)
                / (v**2 + (u + 12) ** 2) ** 1.5
                + 1.0
                * (144.0 * (0.0833333333333333 * u + 1) ** 2)
                / (v**2 + (u + 12) ** 2) ** 1.0
                + 10.0 * (64.0 * (0.125 * u - 1) ** 2) / (v**2 + (8 - u) ** 2) ** 1.0
                + 10.0
                * ((v**2 + (8 - u) ** 2) ** 0.5 - 8)
                / (v**2 + (8 - u) ** 2) ** 0.5
                + 1.0
                * ((v**2 + (u + 12) ** 2) ** 0.5 - 12)
                / (v**2 + (u + 12) ** 2) ** 0.5,
                -10.0
                * v
                * (1.0 * u - 8.0)
                * ((v**2 + (8 - u) ** 2) ** 0.5 - 8)
                / (v**2 + (8 - u) ** 2) ** 1.5
                + 10.0 * v * (1.0 * u - 8.0) / (v**2 + (8 - u) ** 2) ** 1.0
                - 1.0
                * v
                * (1.0 * u + 12.0)
                * ((v**2 + (u + 12) ** 2) ** 0.5 - 12)
                / (v**2 + (u + 12) ** 2) ** 1.5
                + 1.0 * v * (1.0 * u + 12.0) / (v**2 + (u + 12) ** 2) ** 1.0,
            ],
            [
                10.0
                * v
                * (8.0 - 1.0 * u)
                * ((v**2 + (8 - u) ** 2) ** 0.5 - 8)
                / (v**2 + (8 - u) ** 2) ** 1.5
                + 1.0
                * v
                * (-1.0 * u - 12.0)
                * ((v**2 + (u + 12) ** 2) ** 0.5 - 12)
                / (v**2 + (u + 12) ** 2) ** 1.5
                + 10.0 * v * (1.0 * u - 8.0) / (v**2 + (8 - u) ** 2) ** 1.0
                + 1.0 * v * (1.0 * u + 12.0) / (v**2 + (u + 12) ** 2) ** 1.0,
                -10.0
                * v**2
                * ((v**2 + (8 - u) ** 2) ** 0.5 - 8)
                / (v**2 + (8 - u) ** 2) ** 1.5
                + 10.0 * v**2 / (v**2 + (8 - u) ** 2) ** 1.0
                - 1.0
                * v**2
                * ((v**2 + (u + 12) ** 2) ** 0.5 - 12)
                / (v**2 + (u + 12) ** 2) ** 1.5
                + 1.0 * v**2 / (v**2 + (u + 12) ** 2) ** 1.0
                + 10.0
                * ((v**2 + (8 - u) ** 2) ** 0.5 - 8)
                / (v**2 + (8 - u) ** 2) ** 0.5
                + 1.0
                * ((v**2 + (u + 12) ** 2) ** 0.5 - 12)
                / (v**2 + (u + 12) ** 2) ** 0.5,
            ],
        ]
    )


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
        plot_curves(points_uni, func, title="Univariante")
        plot_curves(points_pow, func, title="Powell")
        plot_curves(points_ste, func, title="Steepest Descent")
        plot_curves(points_fle, func, title="Fletcher Reeves")
        plot_curves(points_bfgs, func, title="BFGS")
        plot_curves(points_rap, func, title="Newton Raphson")


def letra_a():
    print(b(" -> Letra A"))
    p0_1 = np.array([2, 2], dtype=np.float64)
    p0_2 = np.array([-1, -3], dtype=np.float64)

    print(f"p0_1: {p0_1} tol={tol} alfa={alfa}")
    methods(p0_1, f_letra_a, grad_letra_a, hessiana_letra_a)
    print(f"p0_2: {p0_2} tol={tol} alfa={alfa}")
    methods(p0_2, f_letra_a, grad_letra_a, hessiana_letra_a)


def letra_b():
    print(b(" -> Letra B"))
    p0_1 = np.array([10, 2], dtype=np.float64)
    p0_2 = np.array([-2, -3], dtype=np.float64)

    print(f"p0_1: {p0_1} tol={tol} alfa={alfa}")
    methods(p0_1, f_letra_b, grad_letra_b, hessiana_letra_b)
    print(f"p0_2: {p0_2} tol={tol} alfa={alfa}")
    methods(p0_2, f_letra_b, grad_letra_b, hessiana_letra_b)


def letra_c():
    print(b(" -> Letra C"))
    p0_1 = np.array([-1, -1], dtype=np.float64)
    p0_2 = np.array([2.8, -1.5], dtype=np.float64)

    print(f"p0_1: {p0_1} tol={tol} alfa={alfa}")
    methods(p0_1, f_letra_c, grad_letra_c, hessiana_letra_c)
    print(f"p0_2: {p0_2} tol={tol} alfa={alfa}")
    methods(p0_2, f_letra_c, grad_letra_c, hessiana_letra_c)


def letra_d():
    print(b(" -> Letra D"))
    p0_1 = np.array([-3, 1], dtype=np.float64)
    p0_2 = np.array([3, -1], dtype=np.float64)

    print(f"p0_1: {p0_1} tol={tol} alfa={alfa}")
    methods(p0_1, f_letra_d, grad_letra_d, hessiana_letra_d)
    print(f"p0_2: {p0_2} tol={tol} alfa={alfa}")
    methods(p0_2, f_letra_d, grad_letra_d, hessiana_letra_d)


def primeira_questao():
    letra_a()
    letra_b()
    letra_c()
    letra_d()


def segunda_questao():
    print(b(" -> Segunda Questão"))
    p0_1 = np.array([15, 12], dtype=np.float64)
    p0_2 = np.array([9, -2], dtype=np.float64)

    print(f"p0_1: {p0_1} tol={tol} alfa={alfa}")
    methods(p0_1, f_segunda, grad_segunda, hessiana_segunda, show_curves=True)
    print(f"p0_2: {p0_2} tol={tol} alfa={alfa}")
    methods(p0_2, f_segunda, grad_segunda, hessiana_segunda, show_curves=True)


def main():
    primeira_questao()
    # segunda_questao()


def univariante(
    p0: np.ndarray,
    func: Callable,
    f_grad: Callable,
    n_max_steps=200,
    verbose=False,
    monitor=False,
) -> np.ndarray:
    if verbose:
        print(y("Inicializando Método Univariante"))

    points: list[np.ndarray] = [p0]  # Percurso da minimização

    d1 = np.array([1.0, 0.0], dtype=np.float64)
    d2 = np.array([0.0, 1.0], dtype=np.float64)

    actual_d = d1
    actual_p = p0

    for i in range(n_max_steps):
        aL, aU = passo_constante(actual_p, alfa, actual_d, func, n_max_step=n_max_steps)
        a_min = secao_aurea(
            actual_p, actual_d, aL, aU, func, tol / 10, n_max_step=n_max_steps
        )
        next_p = make_step(actual_p, a_min, actual_d)
        points.append(next_p)
        if verbose:
            print(
                f"Passo {g(i+1)}: f({next_p[0]}, {next_p[1]})={func(next_p)} norm gradient={np.linalg.norm(f_grad(next_p))}"
            )

        # Verificar se convergiu
        if np.linalg.norm(f_grad(next_p)) < tol:
            if verbose:
                print(g(f"Convergiu em {i+1} passos"))
            if monitor:
                return next_p, points
            return next_p

        # Trocar direção
        if np.any(actual_d == d1):
            actual_d = d2
        else:
            actual_d = d1

        # Trocar ponto
        actual_p = next_p

    print(r("Número máximo de iterações atingido univariante."))
    if monitor:
        return actual_p, points
    return actual_p


def powell(
    p0: np.ndarray,
    func: Callable,
    f_grad: Callable,
    n_max_steps=200,
    verbose=False,
    monitor=False,
) -> np.ndarray:
    if verbose:
        print(y("Inicializando Método de Powell"))
    e1 = np.array([1.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 1.0], dtype=np.float64)
    directions: list[np.ndarray] = [e1, e2, None]

    n_max_ciclos = math.ceil(n_max_steps / 3)

    points: list[np.ndarray] = [p0]  # Percurso da minimização
    actual_p = p0
    ciclo_count = 0
    steps_count = 0

    for ciclo in range(n_max_ciclos):
        # A cada 3 ciclos, resetar direções
        if ciclo_count == 3:
            directions: list[np.ndarray] = [e1, e2, None]
            ciclo_count = 0

        for d in directions:
            # Verifica se atingiu o número máximo de passos
            if steps_count >= n_max_steps:
                break

            # Verificar se convergiu
            if np.linalg.norm(f_grad(actual_p)) < tol:
                if verbose:
                    print(g(f"Convergiu em {steps_count+1} passos"))
                if monitor:
                    return actual_p, points
                return actual_p

            if d is None:
                d = points[-1] - points[-2]
                directions[2] = d
                # d_base = d

            # Realiza passo
            aL, aU = passo_constante(actual_p, alfa, d, func)
            a_min = secao_aurea(
                actual_p,
                d,
                aL,
                aU,
                func,
                tol / 10,
                n_max_step=n_max_steps,
            )
            next_p = make_step(actual_p, a_min, d)  # Próximo ponto
            points.append(next_p)

            if verbose:
                # print(f"Vetor de direções : {directions} d: {d}")
                print(
                    f"Passo {g(steps_count+1)}: f({next_p[0]}, {next_p[1]}) = {func(next_p)}"
                )

            # Atualizar posição atual
            actual_p = next_p

            steps_count += 1

        _, d2, d3 = directions
        directions = [d2, d3, points[-1] - points[-3]]
        ciclo_count += 1
    print(r("Número máximo de iterações atingido Powell."))
    if monitor:
        return actual_p, points
    return actual_p


def steepest_descent(
    p0: np.ndarray,
    func: Callable,
    f_grad: Callable,
    n_max_steps=200,
    verbose=False,
    monitor=False,
) -> np.ndarray:
    if verbose:
        print(y("Inicializando Método Steepest Descent"))

    points: list[np.ndarray] = [p0]  # Percurso da minimização

    p_atual = p0
    for i in range(n_max_steps):
        d = -f_grad(p_atual)
        if np.linalg.norm(d) < tol:
            if verbose:
                print(g(f"Convergiu em {i} passos"))  # i, pois ainda não realizou passo
            if monitor:
                return p_atual, points
            return p_atual

        aU, aL = passo_constante(p_atual, alfa, d, func)
        a_min = secao_aurea(p_atual, d, aU, aL, func, tol / 10)
        next_p = make_step(p_atual, a_min, d)
        points.append(next_p)
        p_atual = next_p
        if verbose:
            print(f"Passo {g(i+1)}: f({next_p[0]}, {next_p[1]})={func(next_p)}")

    print(r("Número de iterações máximas atingido Steepest Descent."))
    if monitor:
        return p_atual, points
    return p_atual


# def fletcher_reeves(
#     p0: np.ndarray,
#     func: Callable,
#     f_grad: Callable,
#     f_hess: Callable,
#     n_max_steps=200,
#     verbose=False,
#     monitor=False,
# ) -> np.ndarray:
#     if verbose:
#         print(y("Inicializando Método Fletcher Reeves"))

#     points: list[np.ndarray] = [p0]  # Percurso da minimização

#     p_atual = p0
#     d = -f_grad(p0)

#     for i in range(n_max_steps):
#         if np.linalg.norm(f_grad(p_atual)) < tol:
#             if verbose:
#                 print(g(f"Convergiu em {i} passos"))  # i, pois ainda não realizou passo
#             if monitor:
#                 return p_atual, points
#             return p_atual

#         alfa = -f_grad(p_atual) @ d / (d @ f_hess(p_atual) @ d)
#         next_p = make_step(p_atual, alfa, d)
#         points.append(next_p)

#         if verbose:
#             print(
#                 f"Passo {g(i+1)}: f({next_p[0]}, {next_p[1]})={func(next_p)} grad norm={np.linalg.norm(f_grad(next_p))}"
#             )

#         beta = f_grad(next_p) @ f_hess(next_p) @ d / (d @ f_hess(p_atual) @ d)
#         d = -f_grad(next_p) + beta * d

#         p_atual = next_p

#     print(r("Número máximo de iterações atingido Fletcher Reeves"))
#     if monitor:
#         return p_atual, points
#     return p_atual


def fletcher_reeves(
    p0: np.ndarray,
    func: Callable,
    f_grad: Callable,
    n_max_steps=200,
    verbose=False,
    monitor=False,
) -> np.ndarray:
    if verbose:
        print(y("Inicializando Método Fletcher Reeves"))

    points: list[np.ndarray] = [p0]  # Percurso da minimização

    p_atual = p0
    grad_atual = f_grad(p_atual)
    d = -grad_atual  # Primeira direção de descida

    for i in range(n_max_steps):
        # Critério de parada baseado na norma do gradiente
        if np.linalg.norm(grad_atual) < tol:
            if verbose:
                print(g(f"Convergiu em {i} passos"))
            if monitor:
                return p_atual, points
            return p_atual

        # Busca de linha (secao_aurea ou método equivalente)
        aL, aU = passo_constante(p_atual, alfa, d, func)
        a_min = secao_aurea(p_atual, d, aL, aU, func, tol / 10)

        # Atualiza o ponto
        next_p = make_step(p_atual, a_min, d)
        points.append(next_p)

        if verbose:
            print(
                f"Passo {g(i+1)}: f({next_p[0]}, {next_p[1]}) = {func(next_p)}, "
                f"grad norm = {np.linalg.norm(f_grad(next_p))}"
            )

        # Atualiza o gradiente e calcula beta (Fletcher-Reeves)
        grad_prox = f_grad(next_p)
        beta = (grad_prox.T @ grad_prox) / (
            grad_atual.T @ grad_atual
        )  # Correto cálculo de beta

        # Atualiza a direção de descida
        d = -grad_prox + beta * d

        # Atualiza o gradiente e o ponto atual
        grad_atual = grad_prox
        p_atual = next_p

    print(r("Número máximo de iterações atingido Fletcher Reeves."))
    if monitor:
        return p_atual, points
    return p_atual


def newton_raphson(
    p0: np.ndarray,
    func: Callable,
    f_grad: Callable,
    f_hess: Callable,
    n_max_steps=200,
    verbose=False,
    monitor=False,
) -> np.ndarray:
    if verbose:
        print(y("Inicializando Método Newton Raphson"))

    points: list[np.ndarray] = [p0]  # Percurso da minimização

    p_atual = p0

    for i in range(n_max_steps):
        if np.linalg.norm(f_grad(p_atual)) < tol:
            if verbose:
                print(g(f"Convergiu em {i} passos"))  # i, pois ainda não realizou passo
            if monitor:
                return p_atual, points
            return p_atual

        d = -np.linalg.inv(f_hess(p_atual)) @ f_grad(p_atual)
        aL, aU = passo_constante(p_atual, alfa, d, func)
        a_min = secao_aurea(p_atual, d, aL, aU, func, tol / 10)
        next_p = make_step(p_atual, a_min, d)
        points.append(next_p)
        if verbose:
            print(
                f"Passo {g(i+1)}: f({next_p[0]}, {next_p[1]})={func(next_p)} grad norm={np.linalg.norm(f_grad(next_p))}"
            )
        p_atual = next_p

    print(r("Número máximo de iterações atingido Newton Raphson"))
    if monitor:
        return p_atual, points
    return p_atual


def bfgs(
    p0: np.ndarray,
    func: Callable,
    f_grad: Callable,
    n_max_steps=200,
    verbose=False,
    monitor=False,
) -> np.ndarray:
    if verbose:
        print(y("Inicializando Método BFGS"))

    points: list[np.ndarray] = [p0]  # Percurso da minimização

    p_atual = p0
    S = np.eye(len(p0))  # Matriz identidade para o tamanho de p0

    for i in range(n_max_steps):
        grad_atual = f_grad(p_atual)

        if np.linalg.norm(grad_atual) < tol:
            if verbose:
                print(g(f"Convergiu em {i} passos"))
            if monitor:
                return p_atual, points
            return p_atual

        # Direção de descida
        d = -S @ grad_atual

        # Busca de linha usando secao_aurea (ou sua função equivalente)
        aL, aU = passo_constante(p_atual, alfa, d, func)
        a_min = secao_aurea(p_atual, d, aL, aU, func, tol / 10)

        # Realiza o próximo passo
        next_p = make_step(p_atual, a_min, d)
        points.append(next_p)

        # Atualiza o gradiente
        grad_prox = f_grad(next_p)

        # Atualização da matriz BFGS
        delta_p = (next_p - p_atual).reshape(-1, 1)
        delta_grad = (grad_prox - grad_atual).reshape(-1, 1)

        denom = delta_p.T @ delta_grad
        if denom != 0:  # Evitar divisão por zero
            expr1 = (
                (delta_p @ delta_p.T)
                * (delta_p.T @ delta_grad + delta_grad.T @ S @ delta_grad)
                / denom**2
            )
            expr2 = (S @ delta_grad @ delta_p.T + delta_p @ delta_grad.T @ S) / denom
            S = S + expr1 - expr2

        if verbose:
            print(
                f"Passo {g(i+1)}: f({next_p[0]}, {next_p[1]}) = {func(next_p)}, "
                f"grad norm = {np.linalg.norm(grad_prox)}"
            )

        # Atualiza o ponto atual
        p_atual = next_p

    print(r("Número máximo de iterações atingido no BFGS."))
    if monitor:
        return p_atual, points
    return p_atual


if __name__ == "__main__":
    main()
