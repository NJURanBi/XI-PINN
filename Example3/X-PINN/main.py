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
    model = train_neural_network(V_n_input=paras.V_n_input,
                                 V_n_hidden=paras.V_n_hidden,
                                 V_n_output=paras.V_n_output,
                                 V_n_layers=paras.V_n_layers,
                                 P_n_input=paras.P_n_input,
                                 P_n_hidden=paras.P_n_hidden,
                                 P_n_output=paras.P_n_output,
                                 P_n_layers=paras.P_n_layers,
                                 num_o=paras.num_o,
                                 num_b=paras.num_b,
                                 num_ini=paras.num_ini,
                                 num_if=paras.num_if,
                                 nu_p=paras.nu_p,
                                 nu_n=paras.nu_n,
                                 optimizer=paras.optimizer,
                                 device=device,
                                 n_runs=paras.run_times)
    end_start = time.time()
    total_T = str(datetime.timedelta(seconds=end_start - cnt_start))
    print(f"total time : {total_T}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Neural network parameters: velocity nural network
    parser.add_argument('--V_n_input', type=int, default=3)
    parser.add_argument('--V_n_hidden', type=int, default=45)
    parser.add_argument('--V_n_output', type=int, default=2)
    parser.add_argument('--V_n_layers', type=int, default=3)
    # Neural network parameters: pressure nural network
    parser.add_argument('--P_n_input', type=int, default=3)
    parser.add_argument('--P_n_hidden', type=int, default=24)
    parser.add_argument('--P_n_output', type=int, default=1)
    parser.add_argument('--P_n_layers', type=int, default=2)
    # Number of collocation points
    parser.add_argument('--num_o', type=int, default=10000)
    parser.add_argument('--num_b', type=int, default=3000)
    parser.add_argument('--num_ini', type=int, default=300)
    parser.add_argument('--num_if', type=int, default=2000)
    # PDEs parameters
    parser.add_argument('--nu_p', type=float, default=1.)
    parser.add_argument('--nu_n', type=float, default=.1)
    # Optimization tool: LM, SOAP, Adam
    parser.add_argument('--optimizer', type=str, default='LM')
    parser.add_argument('--run_times', type=int, default=1)
    paras = parser.parse_args()
    main(paras)


