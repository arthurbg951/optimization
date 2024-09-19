%% ENG1786 & MEC2403: Curso de Otimização 
%
%  CURVAS DE NÍVEL ==> Usando a Função "contour"
%
%  27.Ago.2024 - Ivan Menezes
% ----------------------------------------------

%% Inicializa a memória
clear
clc
close all

%% Define a Função "f"   (EXEMPLO 1)  
f = @(x1,x2) x1.^2 - 3.*x1.*x2 + 4.*x2.^2 + x1 - x2;
% Define os Limites da "Janela" de Visualização
Lx = -10.0;
Ux =  10.0;
Ly = -10.0;
Uy =  10.0;

%% Define a Função "f"   (EXEMPLO 2)
% % % f = @(x1,x2) x1.^4 - 2.*x1.^2.*x2 + x1.^2 + x2.^2 - 2.*x1 + 5;
% % % % Define os Limites da "Janela" de Visualização
% % % Lx = -1.0;
% % % Ux =  2.0;
% % % Ly = -2.0;
% % % Uy =  4.0;

%% Define a Função "f"   (EXEMPLO 3)
% % % f = @(x1, x2)  0.5.*(((12 + x1).^2 + x2.^2).^0.5 - 12).^2 + 5.*(((8 - x1).^2 + x2.^2).^0.5 - 8).^2 - 7.*x2;
% % % % Define os Limites da "Janela" de Visualização
% % % Lx = -6.0;
% % % Ux =  18.0;
% % % Ly = -12.0;
% % % Uy =  15.0;


%% Cria uma Figura 
figure(1);
hold on

%% Define um Grid de Pontos: 
% Nesse exemplo, o Grid é um Quadrado, onde: X=[Lx,Ux] e Y=[Ly,Uy]
% A resolução da Curva de Nível é dada pelo parâmetro "R" 
R = 0.1;
[X,Y] = meshgrid(Lx:R:Ux,Ly:R:Uy);

%% Avalia a Função "f" nos Pontos do Grid (e armazena em "Z")
Z= f(X,Y);

%% Plota as Curvas de Nível usando "contour"
% A quantidade de Curvas de Nível é dada pelo parâmetro "N" 
N = 50;
contour(X,Y,Z,N);
            
hold off

%% Plota a Função "f" (superfície em 3D) na Figura 2
figure(2);
surf(X,Y,Z);
