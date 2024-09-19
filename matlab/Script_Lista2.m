%% ENG1786 & MEC2403:  Curso de Otimiza��o
%
%  C�digo MATLAB: Lista 2
%  Data: 10/SET/2024

clear all
close all
format long

%% Dados Iniciais

% Fun��o Objetivo:  Definida no arquivo "FuncaoObjetivo.m"

% Toler�ncia Num�rica
TOL = 1e-6;

% Ponto Inicial
x0 = [ 2; 2 ];

% Dire��o
d = [ 1; 0 ];

% Delta Alpha  (para o m�todo do Passo Constante)
DeltaAlpha = 0.01;


%% Define o Intervalo de Busca
[aL,aU] = PassoConstante( x0, d, DeltaAlpha );

%% Determina o ponto de m�nimo pelo M�todo da Bisse��o
[alpha] = Bissecao( x0, d, aL, aU, TOL );

display('Valor de ALPHA para o M�todo da Bisse��o: ');
alpha 

%% Determina o ponto de m�nimo pelo M�todo da Se��o �urea
[alpha] = SecaoAurea( x0, d, aL, aU, TOL );

display('Valor de ALPHA para o M�todo da Se��o �urea: ');
alpha 
