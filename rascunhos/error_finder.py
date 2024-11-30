import numpy as np

from optimization.minimize import (
    univariante,
    powell,
    steepest_descent,
    fletcher_reeves,
    newton_raphson,
    bfgs,
)
from optimization.ploting import plot_curves


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
    tol = 1e-4
    alfa = 1.0 / 2.0

    test_methods(p0, f, gradf, hessf, alfa, tol, 3)


def test_methods(p0, f, gradf, hessf, alfa, tol, n_max_steps):
    min, points = univariante(
        p0=p0,
        func=f,
        f_grad=gradf,
        alfa=alfa,
        tol_grad=tol,
        tol_line=tol / 10,
        verbose=True,
        monitor=True,
        n_max_steps=n_max_steps,
    )
    plot_curves(points, f, title="Univariante", show_fig=True)

    min, points = powell(
        p0,
        f,
        gradf,
        alfa,
        tol_grad=tol,
        tol_line=tol / 10,
        verbose=True,
        monitor=True,
        n_max_steps=n_max_steps,
    )
    plot_curves(points, f, title="Powell", show_fig=True)

    min, points = steepest_descent(
        p0,
        f,
        gradf,
        alfa,
        tol_grad=tol,
        tol_line=tol / 10,
        verbose=True,
        monitor=True,
        n_max_steps=n_max_steps,
    )
    plot_curves(points, f, title="Steepest Descent", show_fig=True)

    min, points = fletcher_reeves(
        p0,
        f,
        gradf,
        alfa,
        tol_grad=tol,
        tol_line=tol / 10,
        verbose=True,
        monitor=True,
        n_max_steps=n_max_steps,
    )
    plot_curves(points, f, title="Fletcher Reeves", show_fig=True)

    min, points = newton_raphson(
        p0=p0,
        func=f,
        f_grad=gradf,
        f_hess=hessf,
        alfa=alfa,
        tol_grad=tol,
        tol_line=tol / 10,
        verbose=True,
        monitor=True,
        n_max_steps=n_max_steps,
    )
    plot_curves(points, f, title="Newton Raphson", show_fig=True)

    min, points = bfgs(
        p0,
        f,
        gradf,
        alfa,
        tol_grad=tol,
        tol_line=tol / 10,
        verbose=True,
        monitor=True,
        n_max_steps=n_max_steps,
    )
    plot_curves(points, f, title="BFGS", show_fig=True)


if __name__ == "__main__":
    main()
