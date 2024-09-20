import numpy as np


def lista1_4(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return (
        0.5 * (((12 + x1) ** 2 + x2**2) ** 0.5 - 12) ** 2
        + 5 * (((8 - x1) ** 2 + x2**2) ** 0.5 - 8) ** 2
        - 7 * x2
    )


def a(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return x1**2 - 3 * x1 * x2 + 4 * x2**2 + x1 - x2


def mc_cormick(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2


def himmelblau(p: np.ndarray) -> float:
    x1, x2 = p[0], p[1]
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


if __name__ == "__main__":
    # TODO: mover para teste unitário
    print(lista1_4(np.array([0, 2])))
    print(a(np.array([0, 2])))
    print(mc_cormick(np.array([0, 2])))
    print(himmelblau(np.array([0, 2])))
