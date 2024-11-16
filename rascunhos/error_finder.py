import numpy as np

from optimization.minimize import (
    univariante,
    powell,
    steepest_descent,
    fletcher_reeves,
    newton_raphson,
    bfgs,
)


def f(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return x1**2 - 3 * x1 * x2 + 4 * x2**2 + x1 - x2


def gradf(p: np.ndarray) -> np.ndarray:
    x1, x2 = p[0], p[1]
    return np.array([2 * x1 - 3 * x2 + 1, -3 * x1 + 8 * x2 - 1], dtype=float)


def hessf(p: np.ndarray) -> np.ndarray:
    return np.array([[2, -3], [-3, 8]], dtype=float)


def main():
    p0 = np.array([2, 2], dtype=float)
    tol = 1e-5
    alfa = 1.0 / 2.0

    test_methods(p0, f, gradf, hessf, alfa, tol, 3)


def test_methods(p0, f, gradf, hessf, alfa, tol, n_max_steps):
    univariante(
        p0=p0,
        func=f,
        f_grad=gradf,
        alfa=alfa,
        tol=tol,
        verbose=True,
        monitor=False,
        n_max_steps=n_max_steps,
    )

    powell(
        p0, f, gradf, alfa, tol, verbose=True, monitor=False, n_max_steps=n_max_steps
    )

    steepest_descent(
        p0, f, gradf, alfa, tol, verbose=True, monitor=False, n_max_steps=n_max_steps
    )

    fletcher_reeves(
        p0,
        f,
        gradf,
        alfa,
        tol,
        verbose=True,
        monitor=False,
        n_max_steps=n_max_steps,
    )

    newton_raphson(
        p0=p0,
        func=f,
        f_grad=gradf,
        f_hess=hessf,
        alfa=alfa,
        tol=tol,
        verbose=True,
        monitor=False,
        n_max_steps=n_max_steps,
    )

    bfgs(p0, f, gradf, alfa, tol, verbose=True, monitor=False, n_max_steps=n_max_steps)


if __name__ == "__main__":
    main()
