import torch
import torch.nn as nn
import numpy as np
import argparse
from Network import Vanilla_Net
from Compute_NTK import compute_NTK_XI_PINN, compute_NTK_Vanilla_PINN
from Generate_data import generate_train_data
import datetime
import time

'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device = ', device)

def main(paras):
    data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr, data_if_Van_tr = generate_train_data(num_o=paras.num_o,
                                                                                                     num_ini=paras.num_ini,
                                                                                                     num_b=paras.num_b,
                                                                                                     num_if=paras.num_if,
                                                                                                     beta_p=paras.beta_p,
                                                                                                     beta_n=paras.beta_n,
                                                                                                     device=device)
    model_XI = Vanilla_Net(paras.n_input, paras.n_hidden, paras.n_output, paras.n_layers).to(device)
    model_Vanilla = Vanilla_Net(paras.n_input-1, paras.n_hidden, paras.n_output, paras.n_layers).to(device)
    compute_NTK_XI_PINN(data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr, paras.beta_p, paras.beta_n, model_XI, device)
    compute_NTK_Vanilla_PINN(data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_Van_tr, paras.beta_p, paras.beta_n, model_Vanilla, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_input', type=int, default=4)
    parser.add_argument('--n_hidden', type=int, default=512)
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--num_o', type=int, default=1000)
    parser.add_argument('--num_b', type=int, default=100)
    parser.add_argument('--num_ini', type=int, default=200)
    parser.add_argument('--num_if', type=int, default=400)
    parser.add_argument('--beta_p', type=int, default=10)
    parser.add_argument('--beta_n', type=int, default=1)
    parser.add_argument('--Adam_lr', type=float, default=1e-3)
    parser.add_argument('--Adam_epochs', type=int, default=50000)
    parser.add_argument('--LM_epochs', type=int, default=2000)
    paras = parser.parse_args()
    main(paras)