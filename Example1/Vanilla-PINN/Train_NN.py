# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 下午3:29
# @Author  : NJU_RanBi
from Generate_data import generate_train_data
from Network import Vanilla_Net
from Optimize_parameters import optimize_parameters_Adam, optimize_parameters_LM, optimize_parameters_SOAP
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import random
import torch

def train_neural_network(n_input, n_hidden, n_output, n_layers, num_o, num_b, num_ini, num_if, beta_p, beta_n, optimizer, device, n_runs):
    master_seed = 2026
    run_seeds = generate_run_seeds(master_seed, n_runs)
    all_lossval = []
    all_relative_errors = []
    for run in range(n_runs):
        print(f"\n===== Run {run + 1}/{n_runs} =====")
        seed = run_seeds[run]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        # allocate loss
        lossval = []
        lossval_dbg = []
        relative_error = []
        lossval.append(1.)
        lossval_dbg.append([1., 1., 1.])
        relative_error.append(1.)

        model = Vanilla_Net(n_input, n_hidden, n_output, n_layers).to(device)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr = generate_train_data(num_o, num_ini, num_b, num_if, beta_p, beta_n, device)
        # Optimizer parametes
        if optimizer == 'Adam':
            total_epochs = 50000
            lr = 1e-3
            model, lossval, lossval_dbg, relative_error = optimize_parameters_Adam(data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr, beta_p, beta_n, model, total_epochs, lr, lossval, lossval_dbg, relative_error, device)
        elif optimizer == 'SOAP':
            total_epochs = 10000
            lr = 1e-3
            model, lossval, lossval_dbg, relative_error = optimize_parameters_SOAP(data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr, beta_p, beta_n, model, total_epochs, lr, lossval, lossval_dbg, relative_error, device)
        elif optimizer == 'LM':
            total_epochs = 2000
            mu = 1e8
            mu_div = 3.
            mu_mul = 2.
            model, lossval, lossval_dbg, relative_error = optimize_parameters_LM(data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr, beta_p, beta_n, model, total_epochs, mu, lossval, lossval_dbg, relative_error, mu_div, mu_mul, device)
        else:
            ValueError('error!')

        all_lossval.append(np.array(lossval)[1:])
        all_relative_errors.append(np.array(relative_error)[1:])

    errors_matrix = np.array(all_relative_errors)
    loss_matrix = np.array(all_lossval)
    epochs = np.arange(errors_matrix.shape[1])

    np.savez('relative_errors_data.npz',
             errors_matrix=errors_matrix,
             epochs=epochs,
             n_runs=n_runs,
             optimizer=optimizer)

    np.savez('loss_data.npz',
             errors_matrix=loss_matrix,
             epochs=epochs,
             n_runs=n_runs,
             optimizer=optimizer)

    mean_error = np.mean(errors_matrix, axis=0)
    mean_loss = np.mean(loss_matrix, axis=0)
    std_error = np.std(errors_matrix, axis=0, ddof=1)
    std_loss = np.std(loss_matrix, axis=0, ddof=1)
    start_idx = 10
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[start_idx:], mean_error[start_idx:], color='b', label='Mean Relative Error')
    plt.fill_between(epochs[start_idx:], (mean_error - std_error)[start_idx:], (mean_error + std_error)[start_idx:], alpha=0.3)
    plt.plot(epochs[start_idx:], mean_loss[start_idx:], color='r', label='Mean Loss')
    plt.fill_between(epochs[start_idx:], (mean_loss - std_loss)[start_idx:], (mean_loss + std_loss)[start_idx:], alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Relative Error')
    plt.title(f'Relative Error over {n_runs} runs ({optimizer})')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return

def generate_run_seeds(master_seed, n_runs):
    rng = random.Random(master_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n_runs)]