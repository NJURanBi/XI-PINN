# -*- coding: utf-8 -*-
# @Time    : 2025/2/27 上午10:46
# @Author  : NJU_RanBi
import numpy as np
import torch
from Optimize_parameters import optimize_parameters_soap, optimize_parameters_LM
from Generate_data import generate_train_data
from Network import Vanilla_Net
from check import check_loss

import matplotlib.pyplot as plt

def train_neural_network(n_input, n_hidden, n_output, n_layers, num_op, num_on, num_b, num_ini, num_if, num_t, v_p, v_n, SOAP_lr, SOAP_epochs, LM_epochs, device):
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
    model.load_state_dict(torch.load('best_model.mdl'))
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    data_op_tr, data_on_tr, data_b_tr, data_if_phi_tr, data_if_psi_tr, data_ini_tr = generate_train_data(num_op, num_on, num_b, num_if, num_ini, num_t, v_p, v_n, device)
    check_loss(data_op_tr, data_on_tr, data_b_tr, data_if_phi_tr, data_if_psi_tr, data_ini_tr, v_p, v_n)
    # SOAP
    #model, func_params = optimize_parameters_soap(data_op_tr, data_on_tr, data_b_tr, data_if_phi_tr, data_if_psi_tr, data_ini_tr, v_p, v_n, model, device, SOAP_epochs, SOAP_lr)
    # LM
    #model, func_params, lossval, lossval_dbg, relative_error = optimize_parameters_LM(data_op_tr, data_on_tr, data_b_tr, data_if_phi_tr, data_if_psi_tr, data_ini_tr, v_p, v_n, model, LM_epochs, mu, lossval, lossval_dbg, relative_error, mu_div, mu_mul, device)
    print(lossval_dbg[-1])
    torch.save(model.state_dict(), 'best_model.mdl')
    return model