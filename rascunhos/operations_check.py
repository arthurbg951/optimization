import numpy as np


def h(x: np.ndarray) -> float:
    x1 = x[0]
    x2 = x[1]
    return (x1 - 1) ** 2 + (x2 - 2) ** 2


def grad_h(x: np.ndarray) -> np.ndarray:
    x1 = x[0]
    x2 = x[1]
    return np.array([2 * (x1 - 1), 2 * (x2 - 2)])


def hess_h(x: np.ndarray) -> np.ndarray:
    return np.array([[2, 0], [0, 2]])


def main():
    p = np.array([0, 0])
    rp = 1

    gradiente = 2 * rp * h(p) * grad_h(p)
    # print(gradiente.shape)
    hessiana = 2 * rp * (h(p) * hess_h(p) + np.outer(grad_h(p), grad_h(p)))
    # print(hessiana.shape)
    valor = grad_h(p).reshape(-1, 1) @ grad_h(p).reshape(-1, 1).T
    print(grad_h(p))
    print(np.outer(grad_h(p), grad_h(p)))


if __name__ == "__main__":
    main()
