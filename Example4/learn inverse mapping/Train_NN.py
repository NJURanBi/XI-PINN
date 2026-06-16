# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 下午3:29
# @Author  : NJU_RanBi
import numpy as np
import torch
from Generate_data import generate_train_data
from Network import Vanilla_Net
from Optimize_parameters import optimize_parameters_SOAP, optimize_parameters_Adam, optimize_parameters_LM
import matplotlib.pyplot as plt


def train_neural_network(n_input, n_hidden, n_output, n_layers, num_if, num_b, t_ini, t_end, tau, delta, optimizer, device):
    model = Vanilla_Net(n_input, n_hidden, n_output, n_layers).double().to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    T = t_ini + tau
    detJ = 1
    data_tr = []
    data_tr_empty = True
    while (T < t_end+1e-6) and (detJ >= delta):
        print(f"\n===== Time: {T:.3f}s =====")
        data_tr = generate_train_data(num_if, num_b, T, t_ini, data_tr, data_tr_empty, device)
        data_tr_empty = False
        # Optimizer parametes
        if optimizer == 'Adam':
            total_epochs = 50000
            lr = 1e-3
            model = optimize_parameters_SOAP(data_tr, t_ini, model, total_epochs, lr)
        elif optimizer == 'SOAP':
            total_epochs = 2000
            lr = 1e-3
            model = optimize_parameters_SOAP(data_tr, t_ini, model, total_epochs, lr)
        elif optimizer == 'LM':
            total_epochs = 500
            mu = 1e8
            mu_div = 3.
            mu_mul = 2.
            model = optimize_parameters_LM(data_tr, t_ini, model, total_epochs, mu, mu_div, mu_mul, device)
        else:
            ValueError('error!')
        detJ = compute_det_Jacobian(model, T, t_ini, device)
        print('detJ:', detJ)
        T = T + tau

    torch.save(model.state_dict(), 'inverse_mapping.mdl')
    print(f"\nFinal time: {(T - tau):.3f}s")
    return model

def compute_det_Jacobian(model, T, t_ini, device):
    # X(x,t;t_0) = (t-t_0)*F(x,t) + x
    grid_resolution = 101
    x = torch.linspace(-1, 1, grid_resolution, device=device)
    y = torch.linspace(-1, 1, grid_resolution, device=device)
    X_grid, Y_grid = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X_grid.reshape(-1), Y_grid.reshape(-1)], dim=1)
    t_input = torch.full((points.shape[0], 1), T, device=device)
    inputs = torch.cat([points, t_input], dim=1)
    inputs.requires_grad_(True)

    F = model(inputs)
    grad_F1 = torch.autograd.grad(F[:, 0].sum(), inputs, retain_graph=True, create_graph=False)[0]
    grad_F2 = torch.autograd.grad(F[:, 1].sum(), inputs, retain_graph=False, create_graph=False)[0]

    dF1_dx = grad_F1[:, 0]
    dF1_dy = grad_F1[:, 1]
    dF2_dx = grad_F2[:, 0]
    dF2_dy = grad_F2[:, 1]

    dt = T - t_ini
    J = 1 + dt * (dF1_dx + dF2_dy) + (dt ** 2) * (dF1_dx * dF2_dy - dF1_dy * dF2_dx)
    min_J = J.min()
    return min_J