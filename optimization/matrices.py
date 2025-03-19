import numpy as np
from typing import Callable


def is_symetric(matriz: np.ndarray) -> bool:
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
