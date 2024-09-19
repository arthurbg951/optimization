function [a] = SecaoAurea( x0, d, aL, aU, TOL )

% Razão Áurea
RA = ( sqrt(5) - 1) / 2;

% "Tamanho" do intervalo
beta = abs ( aU - aL );

% Pontos "intermediários" do intervalo
aE = aL + ( 1 - RA ) * beta;   
aD = aL +  RA * beta;   
fE = FuncaoObjetivo ( x0, d, aE );
fD = FuncaoObjetivo ( x0, d, aD );

while ( beta >= TOL )
	if fE > fD
        aL = aE;
        aE = aD;
        fE = fD;
        beta = abs ( aU - aL );
        aD = aL +  RA * beta;
        fD = FuncaoObjetivo ( x0, d, aD );
    else
        aU = aD;
        aD = aE;
        fD = fE; 
        beta = abs ( aU - aL );
        aE = aL +  ( 1 - RA ) * beta;
        fE = FuncaoObjetivo ( x0, d, aE );
    end
end
a = ( aL + aU ) / 2;

end