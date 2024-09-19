function [aM] = Bissecao( x0, d, aL, aU, TOL )

% N�mero suficientemente pequeno
eps = 1e-5;

erro = abs ( aU - aL );

while ( erro >= TOL )
    aM = ( aL + aU ) / 2;
    f1 = FuncaoObjetivo ( x0, d, (aM-eps) );
    f2 = FuncaoObjetivo ( x0, d, (aM+eps) );
    if f1 > f2
        aL = aM;
    else
        aU = aM;
    end
    erro = abs ( aU - aL );
end
aM = ( aL + aU ) / 2;

end