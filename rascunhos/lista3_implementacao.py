import numpy as np

from optimization.linear import make_step, secao_aurea, passo_constante
from optimization.functions import a
from optimization.colors import yellow as y, green as g

alfa = 0.01
tol = 1e-6


def main():
    p0 = np.array([2, 2])
    min = univariante(p0)
    # min = powell(p0)
    # min = steepest_descent(p0)


def univariante(p0: np.ndarray, n_max_steps=1000) -> np.ndarray:
    print(y("Inicializando Método Univariante"))
    d1 = np.array([1, 0])
    d2 = np.array([0, 1])

    actual_d = d1
    actual_p = p0

    for i in range(n_max_steps):
        aU, aL = passo_constante(actual_p, alfa, actual_d, a, n_max_step=n_max_steps)
        a_min, n_steps = secao_aurea(
            p0, actual_d, aU, aL, a, tol, n_max_step=n_max_steps, out_n_steps=True
        )
        next_p = make_step(actual_p, a_min, actual_d)
        print(
            f"Passo {g(i + 1)}: Alfa {a_min} f({next_p[0]}, {next_p[1]})={a(next_p)} com {n_steps} passos"
        )

        # Verificar se convergiu
        if a(actual_p) - a(next_p) < tol:
            print(g(f"Convergiu em {i + 1} passos"))
            return next_p

        # Trocar direção
        if np.any(actual_d == d1):
            actual_d = d2
        else:
            actual_d = d1

        # Trocar ponto
        actual_p = next_p


def powell(p0: np.ndarray, n_max_steps=1000):
    print(y("Inicializando Método de Powell"))
    d1 = np.array([1, 0])
    d2 = np.array([0, 1])

    def get_direction(p: np.ndarray):
        # e1
        aU, aL = passo_constante(p, alfa, d1, a, n_max_step=n_max_steps)
        a_min, n_steps = secao_aurea(
            p0, d1, aU, aL, a, tol, n_max_step=n_max_steps, out_n_steps=True
        )
        p_e1 = make_step(p, a_min, d1)

        # e2
        aU, aL = passo_constante(p, alfa, d2, a, n_max_step=n_max_steps)
        a_min, n_steps = secao_aurea(
            p0, d2, aU, aL, a, tol, n_max_step=n_max_steps, out_n_steps=True
        )
        p_e2 = make_step(p, a_min, d2)
        return p_e2 - p_e1

    actual_p = p0

    for i in range(n_max_steps):
        d = get_direction(actual_p)
        aU, aL = passo_constante(actual_p, alfa, d, a, n_max_step=n_max_steps)
        a_min, n_steps = secao_aurea(
            p0, d, aU, aL, a, tol, n_max_step=n_max_steps, out_n_steps=True
        )
        next_p = make_step(actual_p, a_min, d)

        print(
            f"Passo {g(i + 1)}. Alfa {a_min} Minimo encontrado: {a(next_p)} com {n_steps} passos"
        )

        # Verificar se convergiu
        if a(actual_p) - a(next_p) < tol:
            print(g(f"Convergiu em {i + 1} passos"))
            return next_p

        # Trocar ponto
        actual_p = next_p


def steepest_descent(p0: np.ndarray, n_max_steps=1000): ...


if __name__ == "__main__":
    main()
