import math
from typing import Callable

import numpy as np

from .linear import make_step, passo_constante, secao_aurea
from .colors import red as r, green as g, yellow as y

n_max_steps = 500


def univariante(
    p0: np.ndarray,
    func: Callable,
    f_grad: Callable,
    alfa: float,
    tol: float,
    n_max_steps=n_max_steps,
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
    alfa: float,
    tol: float,
    n_max_steps=n_max_steps,
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
    alfa: float,
    tol: float,
    n_max_steps=n_max_steps,
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


def fletcher_reeves(
    p0: np.ndarray,
    func: Callable,
    f_grad: Callable,
    alfa: float,
    tol: float,
    n_max_steps=n_max_steps,
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
    alfa: float,
    tol: float,
    n_max_steps=n_max_steps,
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
    alfa: float,
    tol: float,
    n_max_steps=n_max_steps,
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