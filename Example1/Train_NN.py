# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 下午3:29
# @Author  : NJU_RanBi
import numpy as np
import torch

from Generate_data import generate_train_data
from Network import Vanilla_Net
from Optimize_parameters import optimize_parameters_adam, optimize_parameters_LM
import matplotlib.pyplot as plt

def train_neural_network(n_input, n_hidden, n_output, n_layers, num_o, num_b, num_ini, num_if, beta_p, beta_n, Adam_lr, Adam_epochs, LM_epochs, device):
    # 初始的mu值，将会放到初始参数设置
    mu = 1e5
    mu_div = 3.
    mu_mul = 2.
    # allocate loss
    lossval = []
    lossval_dbg = []
    relative_error = []
    lossval.append(1.)
    lossval_dbg.append([1., 1., 1.])
    relative_error.append(1.)
    model = Vanilla_Net(n_input, n_hidden, n_output, n_layers).double().to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr = generate_train_data(num_o, num_ini, num_b, num_if, beta_p, beta_n, device)
    # adam
    #model, func_params = optimize_parameters_adam(data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr, beta_p, beta_n, model, device, Adam_epochs, Adam_lr)
    # LM
    LM_epochs = 2000
    model, func_params, lossval, lossval_dbg, relative_error = optimize_parameters_LM(data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr, beta_p, beta_n, model, LM_epochs, mu, lossval, lossval_dbg, relative_error, mu_div, mu_mul, device)

    print(lossval_dbg[-1])
    torch.save(model.state_dict(), 'best_model.mdl')
    return model, func_params