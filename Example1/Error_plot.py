# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 下午3:48
# @Author  : NJU_RanBi
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch import optim, autograd
import torch.nn as nn
from functools import wraps

def map_elementwise(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        container, idx = None, None
        for arg in args:
            if type(arg) in (list, tuple, dict):
                container, idx = type(arg), arg.keys() if type(arg) == dict else len(arg)
                break
        if container is None:
            for value in kwargs.values():
                if type(value) in (list, tuple, dict):
                    container, idx = type(value), value.keys() if type(value) == dict else len(value)
                    break
        if container is None:
            return func(*args, **kwargs)
        elif container in (list, tuple):
            get = lambda element, i: element[i] if type(element) is container else element
            return container(wrapper(*[get(arg, i) for arg in args],
                                     **{key: get(value, i) for key, value in kwargs.items()})
                             for i in range(idx))
        elif container is dict:
            get = lambda element, key: element[key] if type(element) is dict else element
            return {key: wrapper(*[get(arg, key) for arg in args],
                                 **{key_: get(value_, key) for key_, value_ in kwargs.items()})
                    for key in idx}

    return wrapper


@map_elementwise
def To_tensor_grad(x, device):
    x = torch.tensor(x).double().requires_grad_(True).to(device)
    return x


@map_elementwise
def To_tensor(x, device):
    x = torch.tensor(x).double().to(device)
    return x


def error_plot(beta_p, beta_n, model, func_params, device):
    def level_set_function(data):
        x1 = data[:, 0]
        x2 = data[:, 1]
        t = data[:, 2]
        lf = (x1 - 0.3 * np.cos(np.pi * t)) ** 2 + (x2 - 0.3 * np.sin(np.pi * t)) ** 2 - (np.pi / 6) ** 2
        return lf[:, None]

    def add_dimension(data):
        lf = level_set_function(data)
        data_add = np.hstack((data, np.abs(lf)))
        return data_add


    def get_u(data, beta_p, beta_n):
        x1 = data[:, 0]
        x2 = data[:, 1]
        t = data[:, 2]
        lf = level_set_function(data)[:, 0]
        F = np.sqrt((x1 - 0.3 * np.cos(np.pi * t)) ** 2 + (x2 - 0.3 * np.sin(np.pi * t)) ** 2)
        u_p = 6 / beta_p / np.pi * F ** 5 + (np.pi / 6) ** 4 * (1 / beta_n - 1 / beta_p)
        u_n = 6 / beta_n / np.pi * F ** 5
        u_x = np.where(lf >= 0, u_p, u_n)
        return u_x[:, None]

    x1 = np.linspace(-1, 1, 201)
    x2 = np.linspace(-1, 1, 201)
    X, Y = np.meshgrid(x1, x2)
    Z = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    T = 0.9*np.ones((len(Z), 1))
    data = np.hstack((Z, T))
    ur = get_u(data, beta_p, beta_n)
    data_add = add_dimension(data)
    data_add = To_tensor_grad(data_add, device)
    pred = model(data_add)
    pred = pred.cpu().detach().numpy()
    l2_error = np.sqrt(np.mean((pred - ur) ** 2))
    relative_l2_error = l2_error / np.sqrt(np.mean((ur**2)))
    pred = pred.reshape(201, 201)
    ur = ur.reshape(201, 201)
    print(l2_error)
    print(relative_l2_error)
    plt.figure(1)
    h = plt.imshow(np.abs(pred - ur), interpolation='nearest', cmap='coolwarm', extent=[-1, 1, -1, 1], origin='lower', aspect='auto')
    plt.title('Error distribution', fontsize=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(h)
    plt.savefig('error.jpg', bbox_inches='tight', dpi=300)
    plt.show()

    plt.figure(2)
    h = plt.imshow(ur, interpolation='nearest', cmap='coolwarm', extent=[-1, 1, -1, 1], origin='lower', aspect='auto')
    plt.title('Approximate solution', fontsize=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(h)
    plt.savefig('pred.jpg', bbox_inches='tight', dpi=300)
    plt.show()

    # 计算误差
    tao = 0.1
    N_t = 10
    x1 = np.linspace(-1, 1, 101)
    x2 = np.linspace(-1, 1, 101)
    X, Y = np.meshgrid(x1, x2)
    Z = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    T = 0. * np.ones((len(Z), 1))
    data = np.hstack((Z, T))
    for i in range(1, N_t+1):
        data_i = np.hstack((Z, tao * i * np.ones((len(Z), 1))))
        data = np.vstack((data, data_i))
    data_add = add_dimension(data)
    data_add = To_tensor_grad(data_add, device)


    def level_set_function_torch(data):
        x1 = data[:, 0]
        x2 = data[:, 1]
        t = data[:, 2]
        lf = (x1 - 0.3 * torch.cos(np.pi * t)) ** 2 + (x2 - 0.3 * torch.sin(np.pi * t)) ** 2 - (np.pi / 6) ** 2
        return lf[:, None]


    def real_u(data):
        x = data[:, 0]
        y = data[:, 1]
        t = data[:, 2]
        lf = level_set_function_torch(data)[:, 0]
        F = torch.sqrt((x - 0.3 * torch.cos(np.pi * t)) ** 2 + (y - 0.3 * torch.sin(np.pi * t)) ** 2)
        up = 6 / beta_p / torch.pi * F ** 5 + (torch.pi / 6) ** 4 * (1 / beta_n - 1 / beta_p)
        un = 6 / beta_n / torch.pi * F ** 5
        u = torch.where(lf >= 0, up, un)
        return u[:, None]

    def get_dphi_dx(data):
        x1 = data[:, 0]
        t = data[:, 2]
        c_x = 0.3 * torch.cos(np.pi * t)
        phix = 2 * (x1 - c_x)
        return phix[:, None]

    def get_dphi_dy(data):
        x2 = data[:, 1]
        t = data[:, 2]
        c_y = 0.3 * torch.sin(np.pi * t)
        phiy = 2 * (x2 - c_y)
        return phiy[:, None]


    r_u = real_u(data_add)
    p_u = model(data_add)
    lf = level_set_function_torch(data_add)[:, 0]
    grad_r_u = autograd.grad(outputs=r_u, inputs=data_add, grad_outputs=torch.ones_like(r_u), create_graph=True, retain_graph=True)
    dr_udx = grad_r_u[0][:, 0]
    dr_udy = grad_r_u[0][:, 1]
    grad_p_u = autograd.grad(outputs=p_u, inputs=data_add, grad_outputs=torch.ones_like(r_u), create_graph=True, retain_graph=True)
    dp_udx = grad_p_u[0][:, 0]
    dp_udy = grad_p_u[0][:, 1]
    dp_udz = grad_p_u[0][:, 3]
    dlf_dx = get_dphi_dx(data_add)[:, 0]
    dlf_dy = get_dphi_dy(data_add)[:, 0]
    dp_udx = torch.where(lf >= 0, dp_udx + dlf_dx * dp_udz, dp_udx - dlf_dx * dp_udz)
    dp_udy = torch.where(lf >= 0, dp_udy + dlf_dy * dp_udz, dp_udy - dlf_dy * dp_udz)

    l2_error = torch.sqrt(torch.mean((r_u - p_u) ** 2))
    h1_error = torch.sqrt(torch.mean((r_u - p_u) ** 2) + torch.mean((dr_udx - dp_udx) ** 2) + torch.mean((dr_udy - dp_udy) ** 2))
    print('l2_error: ', l2_error.item())
    print('h1_error: ', h1_error.item())
