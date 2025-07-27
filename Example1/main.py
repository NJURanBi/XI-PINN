import torch
import torch.nn as nn
import numpy as np
import argparse
from Train_NN import train_neural_network
from Network import Vanilla_Net
from Error_plot import error_plot
import datetime
import time

'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device = ', device)

def main(paras):
    '''
    cnt_start = time.time()
    model, func_params = train_neural_network(n_input=paras.n_input,
                                              n_hidden=paras.n_hidden,
                                              n_output=paras.n_output,
                                              n_layers=paras.n_layers,
                                              num_o=paras.num_o,
                                              num_b=paras.num_b,
                                              num_ini=paras.num_ini,
                                              num_if=paras.num_if,
                                              beta_p=paras.beta_p,
                                              beta_n=paras.beta_n,
                                              Adam_lr=paras.Adam_lr,
                                              Adam_epochs=paras.Adam_epochs,
                                              LM_epochs=paras.LM_epochs,
                                              device=device)
    end_start = time.time()
    total_T = str(datetime.timedelta(seconds=end_start - cnt_start))
    print(f"total time : {total_T}")
    '''
    model = Vanilla_Net(4, 64, 1, 3).to(device)
    model.load_state_dict(torch.load('best_model.mdl', map_location=device))
    func_params = dict(model.named_parameters())
    error_plot(paras.beta_p, paras.beta_n, model, func_params, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_input', type=int, default=4)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--num_o', type=int, default=15000)
    parser.add_argument('--num_b', type=int, default=1000)
    parser.add_argument('--num_ini', type=int, default=500)
    parser.add_argument('--num_if', type=int, default=3000)
    parser.add_argument('--beta_p', type=int, default=10)
    parser.add_argument('--beta_n', type=int, default=1)
    parser.add_argument('--Adam_lr', type=float, default=1e-3)
    parser.add_argument('--Adam_epochs', type=int, default=50000)
    parser.add_argument('--LM_epochs', type=int, default=5000)
    paras = parser.parse_args()
    main(paras)