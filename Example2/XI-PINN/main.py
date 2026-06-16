import torch
import argparse
from Train_NN import train_neural_network
from Network import Vanilla_Net
import datetime
import time

'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device = ', device)

def main(paras):
    cnt_start = time.time()
    model = train_neural_network(n_input=paras.n_input,
                                 n_hidden=paras.n_hidden,
                                 n_output=paras.n_output,
                                 n_layers=paras.n_layers,
                                 num_o=paras.num_o,
                                 num_b=paras.num_b,
                                 num_ini=paras.num_ini,
                                 num_if=paras.num_if,
                                 beta_p=paras.beta_p,
                                 beta_n=paras.beta_n,
                                 optimizer=paras.optimizer,
                                 device=device,
                                 n_runs=paras.run_times)
    end_start = time.time()
    total_T = str(datetime.timedelta(seconds=end_start - cnt_start))
    print(f"total time : {total_T}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Neural network parameters
    parser.add_argument('--n_input', type=int, default=5)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=2)
    # Number of collocation points
    parser.add_argument('--num_o', type=int, default=20000)
    parser.add_argument('--num_b', type=int, default=1000)
    parser.add_argument('--num_ini', type=int, default=600)
    parser.add_argument('--num_if', type=int, default=4000)
    # PDEs parameters
    parser.add_argument('--beta_p', type=float, default=10.)
    parser.add_argument('--beta_n', type=float, default=1.)
    # Optimization tool: LM, SOAP, Adam
    parser.add_argument('--optimizer', type=str, default='LM')
    parser.add_argument('--run_times', type=int, default=1)
    paras = parser.parse_args()
    main(paras)