import numpy as np
import sympy as sp
from optimization.linear import make_step
from optimization.colors import blue as b, yellow as y, green as g, red as r


def func(p: np.ndarray):
    x1, x2 = p[0], p[1]
    return x1**2 - 3 * x1 * x2 + 4 * x2**2 + x1 - x2


def grad(p: np.ndarray):
    x1, x2 = p[0], p[1]
    grad_f = np.array([2 * x1 - 3 * x2 + 1, -3 * x1 + 8 * x2 - 1])
    print(g(f"∇f({p[0]}, {p[1]}) = {grad_f}"))
    return grad_f


def Q(p: np.ndarray):
    x1, x2 = p[0], p[1]
    return np.array([[2, -3], [-3, 8]])


def alfa(p: np.ndarray, d: np.ndarray):
    # Metodo 1
    # cima = -np.dot(grad(p), d)
    # baixo = np.dot(np.dot(d, Q(p)), d)
    # calculo = cima / baixo
    # print(b(f"alfa = {cima} / {baixo} = {calculo}"))
    # Metodo 2
    cima = -grad(p) @ d
    baixo = d @ Q(p) @ d
    calculo = cima / baixo
    print(b(f"alfa = {cima} / {baixo} = {calculo}"))
    return calculo


def beta(p_init: np.ndarray, p_next: np.ndarray):
    # Metodo 1
    # cima = np.dot(grad(p), np.dot(Q(p), d))
    # baixo = np.dot(np.dot(d, Q(p)), d)
    # calculo = cima / baixo
    # print(b(f"beta = {cima} / {baixo} = {calculo}"))
    # Metodo 2
    # cima = grad(p) @ Q(p) @ d
    # baixo = d @ Q(p) @ d
    # calculo = cima / baixo
    # print(b(f"beta = {cima} / {baixo} = {calculo}"))
    # Syllas
    grad_next = grad(p_next)
    grad_init = grad(p_init)
    cima = grad_next @ grad_next
    baixo = grad_init @ grad_init
    calculo = cima / baixo
    print(b(f"beta = {cima} / {baixo} = {calculo}"))
    return calculo


def univariante(x0: np.ndarray):
    print(r("Inicializando Método Univariante"))

    print(y("Passo 01"))
    d0 = np.array([1, 0])
    a0 = alfa(x0, d0)
    x1 = make_step(x0, a0, d0)

    print(f"d1: {d0} alfa: {a0} x1: {x1}")

    print(y("Passo 02"))
    d1 = np.array([0, 1])
    a1 = alfa(x1, d1)
    x2 = make_step(x1, a1, d1)

    print(f"d2: {d1} alfa: {a1} x2: {x2}")

    print(y("Passo 03"))
    d2 = d0
    a2 = alfa(x2, d2)
    x3 = make_step(x2, a2, d2)

    print(f"d3: {d2} alfa: {a2} x3: {x3}")


def powell(x0: np.ndarray):
    print(r("Inicializando Método de Powell"))

    print(y("Passo 01"))
    d0 = np.array([1, 0])
    a0 = alfa(x0, d0)
    x1 = make_step(x0, a0, d0)

    print(f"d1: {d0} alfa: {a0} x1: {x1}")

    print(y("Passo 02"))
    d1 = np.array([0, 1])
    a1 = alfa(x1, d1)
    x2 = make_step(x1, a1, d1)

    print(f"d2: {d1} alfa: {a1} x2: {x2}")

    print(y("Passo 03"))
    d2 = x2 - x0
    a2 = alfa(x2, d2)
    x3 = make_step(x2, a2, d2)

    print(f"d3: {d2} alfa: {a2} x3: {x3}")


def steepest_descent(x0: np.ndarray):
    print(r("Inicializando Método Steepest Descent"))

    print(y("Passo 01"))
    d0 = -grad(x0)
    a0 = alfa(x0, d0)
    x1 = make_step(x0, a0, d0)

    print(f"d1: {d0} alfa: {a0} x1: {x1}")

    print(y("Passo 02"))
    d1 = -grad(x1)
    a1 = alfa(x1, d1)
    x2 = make_step(x1, a1, d1)

    print(f"d2: {d1} alfa: {a1} x2: {x2}")

    print(y("Passo 03"))
    d2 = -grad(x2)
    a2 = alfa(x2, d2)
    x3 = make_step(x2, a2, d2)

    print(f"d3: {d2} alfa: {a2} x3: {x3}")


def fletcher_reeves(x0: np.ndarray):
    print(r("Inicializando Método Fletcher Reeves"))

    print(y("Passo 01"))
    grad0 = grad(x0)
    d0 = -grad0
    alfa0 = alfa(x0, d0)
    x1 = make_step(x0, alfa0, d0)
    f1 = func(x1)

    print(f"grad0: {grad0} d0: {d0} alfa0: {alfa0} x1: {x1} f1: {f1}")

    print(y("Passo 02"))
    grad1 = grad(x1)
    beta0 = beta(x0, x1)
    d1 = -grad1 + beta0 * d0
    alfa1 = alfa(x1, d1)
    x2 = make_step(x1, alfa1, d1)
    f2 = func(x2)

    print(f"grad1: {grad1} beta0: {beta0} d1: {d1} alfa1: {alfa1} x2: {x2} f2: {f2}")

    print(y("Passo 03"))
    grad2 = grad(x2)

    print(f"grad2: {grad2}")


def main():
    x0 = np.array([2, 2])

    # univariante(x0)
    # powell(x0)
    # steepest_descent(x0)
    fletcher_reeves(x0)


if __name__ == "__main__":
    main()
