import torch
import torch.nn as nn
import numpy as np
import argparse
from Train_NN import train_neural_network
from Network import Vanilla_Net
#from Error_plot import error_plot
import datetime
import time

'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device = ', device)

def main(paras):

    cnt_start = time.time()
    model, func_params = train_neural_network(n_input=paras.n_input,
                                              n_hidden=paras.n_hidden,
                                              n_output=paras.n_output,
                                              n_layers=paras.n_layers,
                                              num_op=paras.num_op,
                                              num_on=paras.num_on,
                                              num_b=paras.num_b,
                                              num_ini=paras.num_ini,
                                              num_if=paras.num_if,
                                              num_t=paras.num_t,
                                              v_p=paras.v_p,
                                              v_n=paras.v_n,
                                              SOAP_lr=paras.SOAP_lr,
                                              SOAP_epochs=paras.SOAP_epochs,
                                              LM_epochs=paras.LM_epochs,
                                              device=device)
    end_start = time.time()
    total_T = str(datetime.timedelta(seconds=end_start - cnt_start))
    print(f"total time : {total_T}")
    '''
    model = Vanilla_Net(4, 64, 3, 3).to(device)
    model.load_state_dict(torch.load('best_model.mdl'))
    func_params = dict(model.named_parameters())
    '''
    #error_plot(paras.beta_p, paras.beta_n, model, func_params, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_input', type=int, default=4)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_output', type=int, default=3)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--num_op', type=int, default=40)
    parser.add_argument('--num_on', type=int, default=10)
    parser.add_argument('--num_b', type=int, default=1000)
    parser.add_argument('--num_ini', type=int, default=500)
    parser.add_argument('--num_if', type=int, default=50)
    parser.add_argument('--num_t', type=int, default=51)
    parser.add_argument('--v_p', type=float, default=1e-3)
    parser.add_argument('--v_n', type=float, default=1)
    parser.add_argument('--SOAP_lr', type=float, default=1e-3)
    parser.add_argument('--SOAP_epochs', type=int, default=100000)
    parser.add_argument('--LM_epochs', type=int, default=500)
    paras = parser.parse_args()
    main(paras)