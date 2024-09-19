
function [aL,aU] = PassoConstante( x0, d, DeltaAlpha )

% Calcula o valor da Função no Ponto Inicial
alpha = 0;
f0 = FuncaoObjetivo ( x0, d, alpha );

% Verifica o sentido positivo da busca
eps = 1e-6;

f1 = FuncaoObjetivo ( x0, d, eps );

dir = d;
flagdir = 1;
if f1 > f0
   dir = -d;
   flagdir = -1;
end

while 1
   alpha = alpha + DeltaAlpha;
   f = FuncaoObjetivo ( x0, dir, alpha );
   f1 = FuncaoObjetivo ( x0, dir, (alpha-eps) );
   if  f1 < f
      aL = alpha - DeltaAlpha;
	  aU = alpha;
	  break;
   end
end

if flagdir == -1
   temp = aL;
   aL = -aU;
   aU = -temp;
end

end