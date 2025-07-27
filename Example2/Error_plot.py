# -*- coding: utf-8 -*-
# @Time    : 2025/5/28 下午1:46
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


def error_plot(v_p, v_n, model, func_params, device):
    def real_u(data):
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        t = data[:, 3]
        up = 0.1 * np.sin(x) * np.cos(y) * np.exp(z) * np.exp(-t)
        un = np.exp(x**2 + y**2 + z**2) * np.cos(t)
        index = data[:, 4]
        u = np.where(index > 0, up, un)
        return u[:, None]

    def level_set_function(data):
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        t = data[:, 3]
        x0 = x * np.cos(np.pi * t / 2) + y * np.sin(np.pi * t / 2)
        y0 = -x * np.sin(np.pi * t / 2) + y * np.cos(np.pi * t / 2)
        z0 = z - 0.5 * t
        lf = x0 ** 2 / 0.7 ** 2 + y0 ** 2 / 0.5 ** 2 + (z0 + 0.25) ** 2 / 0.5 ** 2 - 1
        return lf[:, None]

    def add_dimension(data):
        lf = level_set_function(data)
        add_x = np.ones_like(lf)
        index_n = np.where(lf < 0)[0]
        add_x[index_n] = -1
        data = np.hstack((data, add_x))
        return data

    x1 = np.linspace(-1, 1, 501)
    x2 = np.linspace(-1, 1, 501)
    X, Y = np.meshgrid(x1, x2)
    data = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    Z = 0. * np.ones((len(data[:, 0]), 1))
    T = 0.9 * np.ones((len(data[:, 0]), 1))
    data = np.hstack((data, Z, T))
    data = add_dimension(data)
    ur = real_u(data)

    data_model = To_tensor_grad(data, device)
    pred = model(data_model)
    pred = pred.cpu().detach().numpy()
    l2_error = np.sqrt(np.mean((pred - ur) ** 2))
    pred = pred.reshape(501, 501)
    ur = ur.reshape(501, 501)
    print(l2_error)
    h = plt.imshow(np.abs(pred - ur), interpolation='nearest', cmap='coolwarm', extent=[-1, 1, -1, 1], origin='lower',
                   aspect='auto')
    plt.title('Error distribution', fontsize=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(h)
    plt.savefig('error.jpg', bbox_inches='tight', dpi=300)
    plt.show()

    plt.figure(2)
    h = plt.imshow(pred, interpolation='nearest', cmap='coolwarm', extent=[-1, 1, -1, 1], origin='lower', aspect='auto')
    plt.title('Approximate solution', fontsize=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(h)
    plt.savefig('pred.jpg', bbox_inches='tight', dpi=300)
    plt.show()

    def omega_points(num):
        x1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
        x2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
        x3 = np.random.uniform(low=-1, high=1, size=num)[:, None]
        #t = 0.9*np.ones_like(x1)
        t = np.random.uniform(low=0, high=1, size=num)[:, None]
        data = np.hstack((x1, x2, x3, t))
        data_add = add_dimension(data)
        return data_add

    data_add = omega_points(10000)
    data_add = To_tensor_grad(data_add, device)

    def real_u(data):
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        t = data[:, 3]
        up = 0.1 * torch.sin(x) * torch.cos(y) * torch.exp(z) * torch.exp(-t)
        un = torch.exp(x ** 2 + y ** 2 + z ** 2) * torch.cos(t)
        index = data[:, 4]
        u = torch.where(index > 0, up, un)
        return u[:, None]

    r_u = real_u(data_add)
    p_u = model(data_add)
    grad_r_u = autograd.grad(outputs=r_u, inputs=data_add, grad_outputs=torch.ones_like(r_u), create_graph=True,
                             retain_graph=True)
    dr_udx = grad_r_u[0][:, 0]
    dr_udy = grad_r_u[0][:, 1]
    grad_p_u = autograd.grad(outputs=p_u, inputs=data_add, grad_outputs=torch.ones_like(r_u), create_graph=True,
                             retain_graph=True)
    dp_udx = grad_p_u[0][:, 0]
    dp_udy = grad_p_u[0][:, 1]

    l2_error = torch.sqrt(torch.mean((r_u - p_u) ** 2))
    h1_error = torch.sqrt(torch.mean((r_u - p_u) ** 2) + torch.mean((dr_udx - dp_udx) ** 2) + torch.mean((dr_udy - dp_udy) ** 2))
    print('l2_error: ', l2_error.item())
    print('h1_error: ', h1_error.item())
    return