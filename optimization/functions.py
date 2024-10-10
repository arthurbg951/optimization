import numpy as np
import sympy as sp
from typing import Callable


def lista1_4(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return (
        0.5 * (((12 + x1) ** 2 + x2**2) ** 0.5 - 12) ** 2
        + 5 * (((8 - x1) ** 2 + x2**2) ** 0.5 - 8) ** 2
        - 7 * x2
    )


def a(p: np.ndarray = None) -> float:
    x1, x2 = p[0], p[1]
    return x1**2 - 3 * x1 * x2 + 4 * x2**2 + x1 - x2


def mc_cormick(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2


def himmelblau(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


def get_jacobian(f: Callable):
    x1, x2 = sp.symbols("x1 x2")
    df_dx1 = sp.diff(f, x1)
    df_dx2 = sp.diff(f, x2)

    def grad(p: np.ndarray) -> np.ndarray:
        return np.array(
            [df_dx1.subs({x1: p[0], x2: p[1]}), df_dx2.subs({x1: p[0], x2: p[1]})]
        )

    return np.array([df_dx1, df_dx2])


# def get_hessian(f: Callable, x1, x2):
#     df_dx1, df_dx2 = get_jacobian(f, x1, x2)
#     df_dxx = sp.diff(df_dx1, x1)
#     df_dyy = sp.diff(df_dx2, x2)
#     df_dxy = sp.diff(df_dx1, x2)
#     df_dyx = sp.diff(df_dx2, x1)
#     return np.array([[df_dxx, df_dxy], [df_dyx, df_dyy]])


if __name__ == "__main__":
    # TODO: mover para teste unitário
    print(lista1_4(np.array([0, 2])))
    print(a(np.array([0, 2])))
    print(mc_cormick(np.array([0, 2])))
    print(himmelblau(np.array([0, 2])))
