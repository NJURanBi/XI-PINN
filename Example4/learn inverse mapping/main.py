import torch
import argparse
from Train_NN import train_neural_network
from Network import Vanilla_Net
import datetime
import time

'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.double)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device = ', device)

def main(paras):
    cnt_start = time.time()
    model = train_neural_network(n_input=paras.n_input,
                                 n_hidden=paras.n_hidden,
                                 n_output=paras.n_output,
                                 n_layers=paras.n_layers,
                                 num_if=paras.num_if,
                                 num_b=paras.num_b,
                                 t_ini=paras.t_ini,
                                 t_end=paras.t_end,
                                 tau=paras.tau,
                                 delta=paras.delta,
                                 optimizer=paras.optimizer,
                                 device=device)
    end_start = time.time()
    total_T = str(datetime.timedelta(seconds=end_start - cnt_start))
    print(f"total time : {total_T}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Neural network parameters
    parser.add_argument('--n_input', type=int, default=3)
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--n_output', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=3)
    # Number of sampling points per step
    parser.add_argument('--num_if', type=int, default=400)
    parser.add_argument('--num_b', type=float, default=20)
    # time interval
    parser.add_argument('--t_ini', type=float, default=0.)
    parser.add_argument('--t_end', type=float, default=1.)
    # time step
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--delta', type=float, default=0.2)
    # Optimization tool: LM, SOAP, Adam
    parser.add_argument('--optimizer', type=str, default='LM')
    paras = parser.parse_args()
    main(paras)