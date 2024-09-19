function f = FuncaoObjetivo( x0, d, alpha )

x = x0 + alpha * d;

f = x(1)^2 - 3 * x(1) * x(2) + 4 * x(2)^2 + x(1) - x(2);
