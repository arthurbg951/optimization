import numpy as np
import sympy as sp
from typing import Callable


def is_symetric(matriz: np.ndarray) -> bool:
    # Verifica se a matriz é simétrica
    if np.array_equal(matriz, matriz.T):
        # Calcula os autovalores
        autovalores = np.linalg.eigvals(matriz)
        # Verifica se todos os autovalores são positivos
        if np.all(autovalores > 0):
            return True
        else:
            return False
    else:
        return False


def get_jacobian(f: Callable, x1, x2):
    df_dx1 = sp.diff(f, x1)
    df_dx2 = sp.diff(f, x2)
    return np.array([df_dx1, df_dx2])


def get_hessian(f: Callable, x1, x2):
    df_dx1, df_dx2 = get_jacobian(f, x1, x2)
    df_dxx = sp.diff(df_dx1, x1)
    df_dyy = sp.diff(df_dx2, x2)
    df_dxy = sp.diff(df_dx1, x2)
    df_dyx = sp.diff(df_dx2, x1)
    return np.array([[df_dxx, df_dxy], [df_dyx, df_dyy]])


def local(points: np.ndarray):
    return points - points[0]


def expand(
    points: np.ndarray, func: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    z = func(points[:, 0], points[:, 1]).reshape(-1, 1)
    local_coords = np.hstack((points, z))
    return local_coords


def rotate_z(points: np.ndarray):
    cos = points[-1][1] / np.linalg.norm(points[-1])
    sin = points[-1][0] / np.linalg.norm(points[-1])
    Q = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    return np.dot(Q, points.T).T
