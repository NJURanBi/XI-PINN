# -*- coding: utf-8 -*-
# @Time    : 2025/2/26 下午3:58
# @Author  : NJU_RanBi
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io as sio
from functools import wraps
from IPython.utils import data

from Network import Vanilla_Net
torch.set_default_dtype(torch.double)

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

X_mapping = Vanilla_Net(3, 64, 2, 3).double()
inverse_mapping = Vanilla_Net(3, 64, 2, 3).double()

X_mapping.load_state_dict(torch.load('best_model_interface.mdl'))
inverse_mapping.load_state_dict(torch.load('best_model_inverse.mdl'))



def get_omega_points(N_op, N_on, N_t):
    t = np.linspace(0, 1, N_t)[:, None]
    x = np.random.uniform(low=0, high=1, size=10*N_on)[:, None]
    y = np.random.uniform(low=0, high=1, size=10*N_on)[:, None]
    r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    index = (r <= 0.5)[:, 0]
    data = np.hstack((x, y, t[0] * np.ones_like(x)))
    data = data[index, :]
    data = torch.tensor(data, requires_grad=True)
    lf = level_set_function(data, t[0])
    index_p = torch.where(lf >= 0)[0]
    index_n = torch.where(lf < 0)[0]
    data_p = data[index_p, :].detach().numpy()
    data_n = data[index_n, :].detach().numpy()
    data_p = np.hstack((data_p[:N_op, :], np.ones((N_op, 1))))
    data_n = np.hstack((data_n[:N_on, :], -np.ones((N_on, 1))))
    for i in range(1, N_t):
        x = np.random.uniform(low=0, high=1, size=10 * N_on)[:, None]
        y = np.random.uniform(low=0, high=1, size=10 * N_on)[:, None]
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        index = (r <= 0.5)[:, 0]
        data = np.hstack((x, y, t[i] * np.ones_like(x)))
        data = data[index, :]
        data = torch.tensor(data, requires_grad=True)
        lf = level_set_function(data, t[i])
        index_p = torch.where(lf >= 0)[0]
        index_n = torch.where(lf < 0)[0]
        data_p_i = data[index_p, :].detach().numpy()
        data_n_i = data[index_n, :].detach().numpy()
        data_p_i = np.hstack((data_p_i[:N_op, :], np.ones((N_op, 1))))
        data_n_i = np.hstack((data_n_i[:N_on, :], -np.ones((N_on, 1))))
        data_p = np.vstack((data_p, data_p_i))
        data_n = np.vstack((data_n, data_n_i))
    return data_p, data_n



def get_interface_points(N_if, N_t):
    t = np.linspace(0, 1, N_t)[:, None]
    c_x = 0.5
    c_y = 0.5
    theta = np.linspace(0, 2 * np.pi, N_if, endpoint=False)[:, None]
    r = 3 / 10 * np.power((2.5 + 1.5 * np.sin(5 * theta + 5 * np.pi / 36)), -1 / 4)
    x = r * np.cos(theta) + c_x
    y = r * np.sin(theta) + c_y
    data = np.hstack((x, y, t[0] * np.ones_like(x)))
    data = torch.tensor(data, requires_grad=True)
    data = X_mapping(data).detach().numpy()
    data_p = np.hstack((data, t[0] * np.ones((N_if, 1)), np.ones((N_if, 1))))
    data_n = np.hstack((data, t[0] * np.ones((N_if, 1)), -np.ones((N_if, 1))))
    for i in range(1, N_t):
        data = np.hstack((x, y, t[i] * np.ones_like(x)))
        data = torch.tensor(data, requires_grad=True)
        data = X_mapping(data).detach().numpy()
        data_p_i = np.hstack((data, t[i] * np.ones((N_if, 1)), np.ones((N_if, 1))))
        data_n_i = np.hstack((data, t[i] * np.ones((N_if, 1)), -np.ones((N_if, 1))))
        data_p = np.vstack((data_p, data_p_i))
        data_n = np.vstack((data_n, data_n_i))
    return data_p, data_n


def get_boundary_points(N_b):
    c_x = 0.5
    c_y = 0.5
    theta = np.linspace(0, 2 * np.pi, N_b, endpoint=False)[:, None]
    x = 0.5 * np.cos(theta) + c_x
    y = 0.5 * np.sin(theta) + c_y
    t = np.random.uniform(low=0, high=1, size=N_b)[:, None]
    data = np.hstack((x, y, t, np.ones((N_b, 1))))
    return data


def get_initial_points(N_ini):
    x = np.random.uniform(low=0, high=1, size=2*N_ini)[:, None]
    y = np.random.uniform(low=0, high=1, size=2*N_ini)[:, None]
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    index = (r <= 0.5)[:, 0]
    data = np.hstack((x, y, np.zeros_like(x)))
    data = data[index]
    data = torch.tensor(data, requires_grad=True)
    lf = level_set_function(data, 0.)
    index_p = torch.where(lf >= 0)[0]
    index_n = torch.where(lf < 0)[0]
    data_p = data[index_p, :].detach().numpy()
    data_n = data[index_n, :].detach().numpy()
    data_p = np.hstack((data_p, np.ones((len(data_p), 1))))
    data_n = np.hstack((data_n, -np.ones((len(data_n), 1))))
    data = np.vstack((data_p, data_n))
    return data

def get_u_p(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    u_p_1 = np.exp(x) * np.sin(np.pi * y + np.pi * t)
    u_p_2 = 1 / np.pi * np.exp(x) * np.cos(np.pi * y + np.pi * t)
    return u_p_1, u_p_2


def get_u_n(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    u_n_1 = np.cos(t) * np.cos(np.pi * x) * np.sin(np.pi * y)
    u_n_2 = - np.cos(t) * np.sin(np.pi * x) * np.cos(np.pi * y)
    return u_n_1, u_n_2


def get_p_p(data):
    x = data[:, 0]
    y = data[:, 1]
    p = np.sin(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y)
    return p


def get_p_n(data):
    x = data[:, 0]
    y = data[:, 1]
    p = np.cos(0.5 * np.pi * x) * np.sin(0.5 * np.pi * y)
    return p


def get_w(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    w_1 = np.cos(np.pi * t / 3) * np.sin(np.pi * x)**2 * np.sin(2 * np.pi * y)
    w_2 = -np.cos(np.pi * t / 3) * np.sin(np.pi * y)**2 * np.sin(2 * np.pi * x)
    return w_1[:, None], w_2[:, None]

def get_du_pdx(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_p_1dx = np.exp(x) * np.sin(np.pi * y + np.pi * t)
    du_p_2dx = 1 / np.pi * np.exp(x) * np.cos(np.pi * y + np.pi * t)
    return du_p_1dx, du_p_2dx

def get_du_pdy(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_p_1dy = np.pi * np.exp(x) * np.cos(np.pi * y + np.pi * t)
    du_p_2dy = -np.exp(x) * np.sin(np.pi * y + np.pi * t)
    return du_p_1dy, du_p_2dy

def get_du_pdt(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_p_1dt = np.pi * np.exp(x) * np.cos(np.pi * y + np.pi * t)
    du_p_2dt = -np.exp(x) * np.sin(np.pi * y + np.pi * t)
    return du_p_1dt, du_p_2dt

def get_d2u_pdx2(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    d2u_p_1dx2 = np.exp(x) * np.sin(np.pi * y + np.pi * t)
    d2u_p_2dx2 = 1 / np.pi * np.exp(x) * np.cos(np.pi * y + np.pi * t)
    return d2u_p_1dx2, d2u_p_2dx2

def get_d2u_pdy2(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    d2u_p_1dy2 = -np.pi**2 * np.exp(x) * np.sin(np.pi * y + np.pi * t)
    d2u_p_2dy2 = -np.pi * np.exp(x) * np.cos(np.pi * y + np.pi * t)
    return d2u_p_1dy2, d2u_p_2dy2

def get_du_ndx(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_n_1dx = -np.pi * np.cos(t) * np.sin(np.pi * x) * np.sin(np.pi * y)
    du_n_1dy = -np.pi * np.cos(t) * np.cos(np.pi * x) * np.cos(np.pi * y)
    return du_n_1dx, du_n_1dy

def get_du_ndy(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_n_1dy = np.pi * np.cos(t) * np.cos(np.pi * x) * np.cos(np.pi * y)
    du_n_2dy = np.pi * np.cos(t) * np.sin(np.pi * x) * np.sin(np.pi * y)
    return du_n_1dy, du_n_2dy

def get_du_ndt(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_n_1dt = -np.sin(t) * np.cos(np.pi * x) * np.sin(np.pi * y)
    du_n_2dt = np.sin(t) * np.sin(np.pi * x) * np.cos(np.pi * y)
    return du_n_1dt, du_n_2dt

def get_d2u_ndx2(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    d2u_n_1dx2 = -np.pi**2 * np.cos(t) * np.cos(np.pi * x) * np.sin(np.pi * y)
    d2u_n_2dx2 = np.pi**2 * np.cos(t) * np.sin(np.pi * x) * np.cos(np.pi * y)
    return d2u_n_1dx2, d2u_n_2dx2

def get_d2u_ndy2(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    d2u_n_1dy2 = -np.pi**2 * np.cos(t) * np.cos(np.pi * x) * np.sin(np.pi * y)
    d2u_n_2dy2 = np.pi**2 * np.cos(t) * np.sin(np.pi * x) * np.cos(np.pi * y)
    return d2u_n_1dy2, d2u_n_2dy2

def get_dp_pdx(data):
    x = data[:, 0]
    y = data[:, 1]
    dp_pdx = 0.5 * np.pi * np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y)
    return dp_pdx

def get_dp_pdy(data):
    x = data[:, 0]
    y = data[:, 1]
    dp_pdy = -0.5 * np.pi * np.sin(0.5 * np.pi * x) * np.sin(0.5 * np.pi * y)
    return dp_pdy

def get_dp_ndx(data):
    x = data[:, 0]
    y = data[:, 1]
    dp_ndx = -0.5 * np.pi * np.sin(0.5 * np.pi * x) * np.sin(0.5 * np.pi * y)
    return dp_ndx

def get_dp_ndy(data):
    x = data[:, 0]
    y = data[:, 1]
    dp_ndy = 0.5 * np.pi * np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y)
    return dp_ndy

def original_lf_function(data):
    x = data[:, 0] - 0.5
    y = data[:, 1] - 0.5
    theta = torch.arctan2(y, x)
    R = 3 / 10 * torch.pow((2.5 + 1.5 * torch.sin(5 * theta + 5 * np.pi / 36)), -1 / 4)
    r = torch.sqrt(x**2 + y**2)
    lf = r - R
    return lf[:, None]


def level_set_function(data, t_i):
    x = data[:, :2]
    output = inverse_mapping(data)
    xi = x + output  # 参考坐标
    lf = original_lf_function(xi)
    return lf

# 源项
def get_f_p(data, v_p):
    du_p_1dx, du_p_2dx = get_du_pdx(data)
    du_p_1dy, du_p_2dy = get_du_pdy(data)
    du_p_1dt, du_p_2dt = get_du_pdt(data)
    d2u_p_1dx2, d2u_p_2dx2 = get_d2u_pdx2(data)
    d2u_p_1dy2, d2u_p_2dy2 = get_d2u_pdy2(data)
    dp_pdx = get_dp_pdx(data)
    dp_pdy = get_dp_pdy(data)
    w1, w2 = get_w(data)
    w1 = w1[:, 0]
    w2 = w2[:, 0]
    f_p_1 = du_p_1dt + w1 * du_p_1dx + w2 * du_p_1dy - v_p * (d2u_p_1dx2 + d2u_p_1dy2) + dp_pdx
    f_p_2 = du_p_2dt + w1 * du_p_2dx + w2 * du_p_2dy - v_p * (d2u_p_2dx2 + d2u_p_2dy2) + dp_pdy
    return f_p_1[:, None], f_p_2[:, None]


def get_f_n(data, v_n):
    du_n_1dx, du_n_2dx = get_du_ndx(data)
    du_n_1dy, du_n_2dy = get_du_ndy(data)
    du_n_1dt, du_n_2dt = get_du_ndt(data)
    d2u_n_1dx2, d2u_n_2dx2 = get_d2u_ndx2(data)
    d2u_n_1dy2, d2u_n_2dy2 = get_d2u_ndy2(data)
    dp_ndx = get_dp_ndx(data)
    dp_ndy = get_dp_ndy(data)
    w1, w2 = get_w(data)
    w1 = w1[:, 0]
    w2 = w2[:, 0]
    f_n_1 = du_n_1dt + w1 * du_n_1dx + w2 * du_n_1dy - v_n * (d2u_n_1dx2 + d2u_n_1dy2) + dp_ndx
    f_n_2 = du_n_2dt + w1 * du_n_2dx + w2 * du_n_2dy - v_n * (d2u_n_2dx2 + d2u_n_2dy2) + dp_ndy
    return f_n_1[:, None], f_n_2[:, None]

# 边界条件
def get_g(data):
    g_1, g_2 = get_u_p(data)
    return g_1[:, None], g_2[:, None]

# 界面跳量条件
def get_phi(data):
    u_p_1, u_p_2 = get_u_p(data)
    u_n_1, u_n_2 = get_u_n(data)
    phi_1 = u_p_1 - u_n_1
    phi_2 = u_p_2 - u_n_2
    return phi_1[:, None], phi_2[:, None]

# 界面导数跳量条件
def get_psi(data, v_p, v_n):
    du_p_1dx, du_p_2dx = get_du_pdx(data)
    du_p_1dy, du_p_2dy = get_du_pdy(data)
    du_n_1dx, du_n_2dx = get_du_ndx(data)
    du_n_1dy, du_n_2dy = get_du_ndy(data)
    p_p = get_p_p(data)
    p_n = get_p_n(data)
    psi_1 = v_p * (du_p_1dx + du_p_1dy) - p_p - v_n * (du_n_1dx + du_n_1dy) + p_n
    psi_2 = v_p * (du_p_2dx + du_p_2dy) - p_p - v_n * (du_n_2dx + du_n_2dy) + p_n
    return psi_1[:, None], psi_2[:, None]

# 初始条件
def get_u0(data):
    u_p_1, u_p_2 = get_u_p(data)
    u_n_1, u_n_2 = get_u_n(data)
    index = data[:, 3]
    u0_1 = np.where(index > 0, u_p_1, u_n_1)
    u0_2 = np.where(index > 0, u_p_2, u_n_2)
    return u0_1[:, None], u0_2[:, None]


def generate_train_data(num_op, num_on, num_b, num_if, num_ini, num_t, v_p, v_n, device):
    xop, xon = get_omega_points(num_op, num_on, num_t)
    xb = get_boundary_points(num_b)
    xif_p, xif_n = get_interface_points(num_if, num_t)
    xini = get_initial_points(num_ini)
    f_p_1, f_p_2 = get_f_p(xop, v_p)
    f_n_1, f_n_2 = get_f_n(xon, v_n)
    w_p_1, w_p_2 = get_w(xop)
    w_n_1, w_n_2 = get_w(xon)
    g_1, g_2 = get_g(xb)
    phi_1, phi_2 = get_phi(xif_p)
    psi_1, psi_2 = get_psi(xif_p, v_p, v_n)
    u0_1, u0_2 = get_u0(xini)


    xop = To_tensor_grad(xop, device)
    xon = To_tensor_grad(xon, device)
    xb = To_tensor_grad(xb, device)
    xif_p = To_tensor_grad(xif_p, device)
    xif_n = To_tensor_grad(xif_n, device)
    xini = To_tensor_grad(xini, device)

    f_p_1 = To_tensor(f_p_1, device)
    f_p_2 = To_tensor(f_p_2, device)
    f_n_1 = To_tensor(f_n_1, device)
    f_n_2 = To_tensor(f_n_2, device)
    w_p_1 = To_tensor(w_p_1, device)
    w_p_2 = To_tensor(w_p_2, device)
    w_n_1 = To_tensor(w_n_1, device)
    w_n_2 = To_tensor(w_n_2, device)
    g_1 = To_tensor(g_1, device)
    g_2 = To_tensor(g_2, device)
    phi_1 = To_tensor(phi_1, device)
    phi_2 = To_tensor(phi_2, device)
    psi_1 = To_tensor(psi_1, device)
    psi_2 = To_tensor(psi_2, device)
    u0_1 = To_tensor(u0_1, device)
    u0_2 = To_tensor(u0_2, device)

    data_op_tr = (xop, f_p_1, f_p_2, w_p_1, w_p_2)
    data_on_tr = (xon, f_n_1, f_n_2, w_n_1, w_n_2)
    data_b_tr = (xb, g_1, g_2)
    data_if_phi_tr = (xif_p, xif_n, phi_1, phi_2)
    data_if_psi_tr = (xif_p, xif_n, psi_1, psi_2)
    data_ini_tr = (xini, u0_1, u0_2)
    return data_op_tr, data_on_tr, data_b_tr, data_if_phi_tr, data_if_psi_tr, data_ini_tr
