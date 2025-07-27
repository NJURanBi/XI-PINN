clc;clear
%% 总NTK
NTK_xi = readmatrix('NTK_XI.txt');
NTK_van = readmatrix('NTK_VAN.txt');
[~, D_xi] = eig(NTK_xi);
[~, D_van] = eig(NTK_van);
D_xi = sort(abs(diag(D_xi)), 'descend');
D_van = sort(abs(diag(D_van)), 'descend');
f1 = figure(1);
set(f1, 'Position', [100, 100, 400, 400]);
loglog(1:length(D_xi), D_xi, 'LineWidth', 2, 'Color', 'k'); hold on;
loglog(1:length(D_van), D_van, 'LineWidth', 2, 'Color', 'k', 'LineStyle', '--'); hold on;
xlim([1 2000]);
legend('XI-PINN', 'Vanilla-PINN');
title('Eigenvalue of $\mathcal{K}$', 'Interpreter', 'latex', 'FontSize', 25);
xlabel('Index');
ylabel('$\lambda$', 'Interpreter', 'latex');
set(gca,'LooseInset',get(gca,'TightInset'))
%% NTK_o
NTK_xi_o = readmatrix('NTK_XI_O.txt');
NTK_van_o = readmatrix('NTK_VAN_O.txt');
[~, D_xi_o] = eig(NTK_xi_o);
[~, D_van_o] = eig(NTK_van_o);
D_xi_o = sort(abs(diag(D_xi_o)), 'descend');
D_van_o = sort(abs(diag(D_van_o)), 'descend');
f2 = figure(2);
set(f2, 'Position', [100, 100, 400, 400]);
loglog(1:length(D_xi_o), D_xi_o, 'LineWidth', 2, 'Color', 'k'); hold on;
loglog(1:length(D_van_o), D_van_o, 'LineWidth', 2, 'Color', 'k', 'LineStyle', '--'); hold on;
xlim([1 1000]);
legend('XI-PINN', 'Vanilla-PINN');
title('Eigenvalue of $\mathcal{K}_{\Omega}$', 'Interpreter', 'latex', 'FontSize', 25);
xlabel('Index');
ylabel('$\lambda$', 'Interpreter', 'latex');
set(gca,'LooseInset',get(gca,'TightInset'))
%% NTK_b
NTK_xi_b = readmatrix('NTK_XI_B.txt');
NTK_van_b = readmatrix('NTK_VAN_B.txt');
[~, D_xi_b] = eig(NTK_xi_b);
[~, D_van_b] = eig(NTK_van_b);
D_xi_b = sort(abs(diag(D_xi_b)), 'descend');
D_van_b = sort(abs(diag(D_van_b)), 'descend');
f4 = figure(4);
set(f4, 'Position', [100, 100, 400, 400]);
loglog(1:length(D_xi_b), D_xi_b, 'LineWidth', 2, 'Color', 'k'); hold on;
loglog(1:length(D_van_b), D_van_b, 'LineWidth', 2, 'Color', 'k', 'LineStyle', '--'); hold on;
xlim([1 400]);
legend('XI-PINN', 'Vanilla-PINN');
title('Eigenvalue of $\mathcal{K}_b$', 'Interpreter', 'latex', 'FontSize', 25);
xlabel('Index');
ylabel('$\lambda$', 'Interpreter', 'latex');
set(gca,'LooseInset',get(gca,'TightInset'))
%% NTK_ini
NTK_xi_ini = readmatrix('NTK_XI_INI.txt');
NTK_van_ini = readmatrix('NTK_VAN_INI.txt');
[~, D_xi_ini] = eig(NTK_xi_ini);
[~, D_van_ini] = eig(NTK_van_ini);
D_xi_ini = sort(abs(diag(D_xi_ini)), 'descend');
D_van_ini = sort(abs(diag(D_van_ini)), 'descend');
f5 = figure(5);
set(f5, 'Position', [100, 100, 400, 400]);
loglog(1:length(D_xi_ini), D_xi_ini, 'LineWidth', 2, 'Color', 'k'); hold on;
loglog(1:length(D_van_ini), D_van_ini, 'LineWidth', 2, 'Color', 'k', 'LineStyle', '--'); hold on;
xlim([1 200]);
legend('XI-PINN', 'Vanilla-PINN');
title('Eigenvalue of $\mathcal{K}_0$', 'Interpreter', 'latex', 'FontSize', 25);
xlabel('Index');
ylabel('$\lambda$', 'Interpreter', 'latex');
set(gca,'LooseInset',get(gca,'TightInset'))
%%
NTK_xi_if = readmatrix('NTK_XI_IF.txt');
NTK_van_if = readmatrix('NTK_VAN_IF.txt');
[~, D_xi_if] = eig(NTK_xi_if);
[~, D_van_if] = eig(NTK_van_if);
D_xi_if = sort(abs(diag(D_xi_if)), 'descend');
D_van_if = sort(abs(diag(D_van_if)), 'descend');
f6 = figure(6);
set(f6, 'Position', [100, 100, 400, 400]);
loglog(1:length(D_xi_if), D_xi_if, 'LineWidth', 2, 'Color', 'k'); hold on;
loglog(1:length(D_van_if), D_van_if, 'LineWidth', 2, 'Color', 'k', 'LineStyle', '--'); hold on;
xlim([1 400]);
legend('XI-PINN', 'Vanilla-PINN');
title('Eigenvalue of $\mathcal{K}_\Gamma$', 'Interpreter', 'latex', 'FontSize', 25);
xlabel('Index');
ylabel('$\lambda$', 'Interpreter', 'latex');
set(gca,'LooseInset',get(gca,'TightInset'))
%% covergence rate
c_xi_tol = sum(D_xi) / 2000;
c_van_tol = sum(D_van) / 2000;

c_xi_par = sum(D_xi_o) / 1000 + sum(D_xi_b) / 400 + sum(D_xi_if) / 400 + sum(D_xi_ini) / 200;
c_van_par = sum(D_van_o) / 1000 + sum(D_van_b) / 400 + sum(D_van_if) / 400 + sum(D_van_ini) / 200;