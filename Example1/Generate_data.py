# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 上午10:17
# @Author  : NJU_RanBi
import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import wraps
import scipy.io as sio

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


def omega_points(num):
    x1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    t = np.random.uniform(low=0, high=1, size=num)[:, None]
    data = np.hstack((x1, x2, t))
    phi = level_set_function(data)
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    datap = data[index_p, :]
    datan = data[index_n, :]
    return datap, datan


def ini_points(num):
    x1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    t = np.zeros_like(x1)
    data = np.hstack((x1, x2, t))
    return data


def boundary_points(num):
    index = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb1 = np.hstack((index, np.ones_like(index)))
    index = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb2 = np.hstack((index, -np.ones_like(index)))
    index = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb3 = np.hstack((np.ones_like(index), index))
    index = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb4 = np.hstack((-np.ones_like(index), index))
    xb = np.vstack((xb1, xb2, xb3, xb4))
    t = np.random.uniform(low=0, high=1, size=4*num)[:, None]
    data = np.hstack((xb, t))
    return data


def interface_points(num):
    t = np.random.uniform(low=0, high=1, size=num)[:, None]
    c_x = 0.3 * np.cos(np.pi * t)
    c_y = 0.3 * np.sin(np.pi * t)
    theta = np.random.uniform(low=0, high=2*np.pi, size=num)[:, None]
    x1 = c_x + np.pi / 6 * np.cos(theta)
    x2 = c_y + np.pi / 6 * np.sin(theta)
    data = np.hstack((x1, x2, t))
    return data


def level_set_function(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    lf = (x1 - 0.3 * np.cos(np.pi * t))**2 + (x2 - 0.3 * np.sin(np.pi * t))**2 - (np.pi / 6)**2
    return lf[:, None]


def add_dimension(data):
    lf = level_set_function(data)
    data_add = np.hstack((data, np.abs(lf)))
    return data_add


def dudt(data, beta):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    F = np.sqrt((x1 - 0.3 * np.cos(np.pi * t))**2 + (x2 - 0.3 * np.sin(np.pi * t))**2)
    ut = 15 / np.pi / beta * F**3 * (0.6 * np.pi * np.sin(np.pi * t) * (x1 - 0.3 * np.cos(np.pi * t)) - 0.6 * np.pi * np.cos(np.pi * t) * (x2 - 0.3 * np.sin(np.pi * t)))
    return ut[:, None]


def d2udx2(data, beta):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    F = np.sqrt((x1 - 0.3 * np.cos(np.pi * t))**2 + (x2 - 0.3 * np.sin(np.pi * t))**2)
    uxx = 90 / beta / np.pi * F * (x1 - 0.3 * np.cos(np.pi * t))**2 + 30 / beta / np.pi * F**3
    return uxx[:, None]


def d2udy2(data, beta):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    F = np.sqrt((x1 - 0.3 * np.cos(np.pi * t)) ** 2 + (x2 - 0.3 * np.sin(np.pi * t)) ** 2)
    uyy = 90 / beta / np.pi * F * (x2 - 0.3 * np.sin(np.pi * t))**2 + 30 / beta / np.pi * F**3
    return uyy[:, None]


def get_g(data, beta_p, beta_n):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    F = np.sqrt((x1 - 0.3 * np.cos(np.pi * t)) ** 2 + (x2 - 0.3 * np.sin(np.pi * t)) ** 2)
    g = 6 / beta_p / np.pi * F**5 + (np.pi / 6)**4 * (1 / beta_n - 1 / beta_p)
    return g[:, None]


def get_f(data, beta):
    ut = dudt(data, beta)
    uxx = d2udx2(data, beta)
    uyy = d2udy2(data, beta)
    f = ut - beta * (uxx + uyy)
    return f


def get_u(data, beta_p, beta_n):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    lf = level_set_function(data)[:, 0]
    F = np.sqrt((x1 - 0.3 * np.cos(np.pi * t)) ** 2 + (x2 - 0.3 * np.sin(np.pi * t)) ** 2)
    u_p = 6 / beta_p / np.pi * F**5 + (np.pi / 6)**4 * (1 / beta_n - 1 / beta_p)
    u_n = 6 / beta_n / np.pi * F**5
    u_x = np.where(lf >= 0, u_p, u_n)
    return u_x[:, None]


def get_normal_vector(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    c_x = 0.3 * np.cos(np.pi * t)
    c_y = 0.3 * np.sin(np.pi * t)
    n1 = 2 * (x1 - c_x)
    n2 = 2 * (x2 - c_y)
    e1 = n1 / np.sqrt(n1 * n1 + n2 * n2)
    e2 = n2 / np.sqrt(n1 * n1 + n2 * n2)
    n = np.hstack((e1[:, None], e2[:, None]))
    return n


def get_dphi_dx(data):
    x1 = data[:, 0]
    t = data[:, 2]
    c_x = 0.3 * np.cos(np.pi * t)
    phix = 2 * (x1 - c_x)
    return phix[:, None]


def get_dphi_dy(data):
    x2 = data[:, 1]
    t = data[:, 2]
    c_y = 0.3 * np.sin(np.pi * t)
    phiy = 2 * (x2 - c_y)
    return phiy[:, None]


def get_dphi_dt(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    phit = 0.6 * np.pi * (x1 - 0.3 * np.cos(np.pi * t)) * np.sin(np.pi * t) - 0.6 * np.pi * (x2 - 0.3 * np.sin(np.pi * t)) * np.cos(np.pi * t)
    return phit[:, None]



def generate_train_data(num_o, num_ini, num_b, num_if, beta_p, beta_n, device):
    xop, xon = omega_points(num_o)
    xini = ini_points(num_ini)
    xb = boundary_points(num_b)
    xif = interface_points(num_if)
    f_p = get_f(xop, beta_p)
    f_n = get_f(xon, beta_n)
    u0 = get_u(xini, beta_p, beta_n)
    g = get_g(xb, beta_p, beta_n)

    nor = get_normal_vector(xif)
    phix_xop = get_dphi_dx(xop)
    phiy_xop = get_dphi_dy(xop)
    phit_xop = get_dphi_dt(xop)
    phix_xon = get_dphi_dx(xon)
    phiy_xon = get_dphi_dy(xon)
    phit_xon = get_dphi_dt(xon)
    phix_xif = get_dphi_dx(xif)
    phiy_xif = get_dphi_dy(xif)

    xop_add = add_dimension(xop)
    xon_add = add_dimension(xon)
    xb_add = add_dimension(xb)
    xini_add = add_dimension(xini)
    xif_add = add_dimension(xif)

    xop_add = To_tensor_grad(xop_add, device)
    xon_add = To_tensor_grad(xon_add, device)
    xb_add = To_tensor_grad(xb_add, device)
    xif_add = To_tensor_grad(xif_add, device)
    xini_add = To_tensor_grad(xini_add, device)

    f_p = To_tensor(f_p, device)
    f_n = To_tensor(f_n, device)
    u0 = To_tensor(u0, device)
    g = To_tensor(g, device)

    nor = To_tensor(nor, device)
    phix_xop = To_tensor(phix_xop, device)
    phiy_xop = To_tensor(phiy_xop, device)
    phit_xop = To_tensor(phit_xop, device)
    phix_xon = To_tensor(phix_xon, device)
    phiy_xon = To_tensor(phiy_xon, device)
    phit_xon = To_tensor(phit_xon, device)
    phix_xif = To_tensor(phix_xif, device)
    phiy_xif = To_tensor(phiy_xif, device)


    data_op_tr = (xop_add, f_p, phix_xop, phiy_xop, phit_xop)
    data_on_tr = (xon_add, f_n, phix_xon, phiy_xon, phit_xon)
    data_b_tr = (xb_add, g)
    data_ini_tr = (xini_add, u0)
    data_if_tr = (xif_add, nor, phix_xif, phiy_xif)


    return data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr

