% Definindo o intervalo de x
x_vals = linspace(2, 3, 5);

% Função original e^x
func_original = exp(x_vals);

% Ponto de expansão
x0 = 2;

% Graus da série de Taylor
graus = 0:3;

% Inicializando a matriz de erros
errors = zeros(length(graus), length(x_vals));

% Plotando a função original
figure;
plot(x_vals, func_original, 'k', 'LineWidth', 2);
hold on;

% Calculando e plotando as expansões da série de Taylor
for i = 1:length(graus)
    % Calcula a série de Taylor de grau i
    taylor_series = 0;
    for n = 0:graus(i)
        taylor_series = taylor_series + (exp(x0) * (x_vals - x0).^n) / factorial(n);
    end
    
    % Plotando a série de Taylor
    plot(x_vals, taylor_series, 'DisplayName', sprintf('Grau %d', graus(i)));
    
    % Calculando o erro absoluto
    errors(i, :) = abs(func_original - taylor_series);
end

% Configurações do gráfico
title('Expansão da Série de Taylor de e^x em torno de x=2');
xlabel('x');
ylabel('y');
legend('show');
grid on;
set(gca, 'XTick', x_vals); % Configura o eixo x para mostrar os valores de x_vals
xlim([min(x_vals) max(x_vals)]); % Define os limites do eixo x
hold off;

% Plotando os erros absolutos
figure;
for i = 1:length(graus)
    semilogy(x_vals, errors(i, :), 'DisplayName', sprintf('Erro Absoluto - Grau %d', graus(i)));
    hold on;
end

% Configurações do gráfico de erros
title('Erro Absoluto das Expansões da Série de Taylor de e^x');
xlabel('x');
ylabel('Erro Absoluto');
legend('show');
grid on;
set(gca, 'XTick', x_vals); % Configura o eixo x para mostrar os valores de x_vals
xlim([min(x_vals) max(x_vals)]); % Define os limites do eixo x
hold off;
