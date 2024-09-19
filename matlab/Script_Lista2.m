%% ENG1786 & MEC2403:  Curso de Otimização
%
%  Código MATLAB: Lista 2
%  Data: 10/SET/2024

clear all
close all
format long

%% Dados Iniciais

% Função Objetivo:  Definida no arquivo "FuncaoObjetivo.m"

% Tolerância Numérica
TOL = 1e-6;

% Ponto Inicial
x0 = [ 2; 2 ];

% Direção
d = [ 1; 0 ];

% Delta Alpha  (para o método do Passo Constante)
DeltaAlpha = 0.01;


%% Define o Intervalo de Busca
[aL,aU] = PassoConstante( x0, d, DeltaAlpha );

%% Determina o ponto de mínimo pelo Método da Bisseção
[alpha] = Bissecao( x0, d, aL, aU, TOL );

display('Valor de ALPHA para o Método da Bisseção: ');
alpha 

%% Determina o ponto de mínimo pelo Método da Seção Áurea
[alpha] = SecaoAurea( x0, d, aL, aU, TOL );

display('Valor de ALPHA para o Método da Seção Áurea: ');
alpha 
