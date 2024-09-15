import numpy as np


def lista1_4(x1, x2):
    return (
        0.5 * (((12 + x1) ** 2 + x2**2) ** 0.5 - 12) ** 2
        + 5 * (((8 - x1) ** 2 + x2**2) ** 0.5 - 8) ** 2
        - 7 * x2
    )


def a(x1, x2):
    return x1**2 - 3 * x1 * x2 + 4 * x2**2 + x1 - x2


def mc_cormick(x1, x2):
    return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2


def himmelblau(x1, x2):
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2
