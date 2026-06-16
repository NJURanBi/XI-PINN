# -*- coding: utf-8 -*-
# @Time    : 2026/5/24 下午3:20
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
    x3 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    t = np.random.uniform(low=0, high=1, size=num)[:, None]
    data = np.hstack((x1, x2, x3, t))
    phi = level_set_function(data)
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    datap = data[index_p, :]
    datan = data[index_n, :]
    return datap, datan


def ini_points(num):
    x1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x3 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    t = np.zeros_like(x1)
    data = np.hstack((x1, x2, x3, t))
    phi = level_set_function(data)
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    datap = data[index_p, :]
    datan = data[index_n, :]
    return datap, datan


def boundary_points(num):
    x_face_b1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x_face_b2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb1 = np.hstack((x_face_b1, x_face_b2, np.ones_like(x_face_b1)))
    x_face_b1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x_face_b2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb2 = np.hstack((x_face_b1, x_face_b2, -np.ones_like(x_face_b1)))
    x_face_b1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x_face_b2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb3 = np.hstack((x_face_b1, np.ones_like(x_face_b1), x_face_b2))
    x_face_b1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x_face_b2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb4 = np.hstack((x_face_b1, -np.ones_like(x_face_b1), x_face_b2))
    x_face_b1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x_face_b2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb5 = np.hstack((np.ones_like(x_face_b1), x_face_b1, x_face_b2))
    x_face_b1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x_face_b2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb6 = np.hstack((-np.ones_like(x_face_b1), x_face_b1, x_face_b2))
    xb = np.vstack((xb1, xb2, xb3, xb4, xb5, xb6))
    t = np.random.uniform(low=0, high=1, size=6*num)[:, None]
    data = np.hstack((xb, t))
    return data


def interface_points(num, N_s=400, N_t=51):
    a = 0.9
    b = 0.7
    c = 0.5
    t = np.linspace(0, 1, N_t)
    data = []

    if num > N_s*N_t:
        ValueError('error!')
    else:
        for ti in t:
            i = np.arange(N_s)
            phi = np.arccos(1 - 2 * (i + 0.5) / N_s)
            theta_s = np.pi * (1 + 5 ** 0.5) * i

            su = np.sin(phi) * np.cos(theta_s)
            sv = np.sin(phi) * np.sin(theta_s)
            sw = np.cos(phi)

            u = a * su
            v = b * sv
            w = c * sw

            theta_t = np.pi * ti / 2
            cos_t, sin_t = np.cos(theta_t), np.sin(theta_t)
            x1 = u * cos_t - v * sin_t
            x2 = u * sin_t + v * cos_t
            x3 = w + 0.5 * ti - 0.25

            ti_arr = np.full_like(x1, ti)
            pts = np.stack([x1, x2, x3, ti_arr], axis=1)
            data.append(pts)

        data = np.vstack(data)
        indices = np.random.choice(data.shape[0], size=num, replace=False)
        data = data[indices]
    return data


def level_set_function(data):
    a = 0.9
    b = 0.7
    c = 0.5
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]
    F1 = (x1*np.cos(np.pi*t/2) + x2*np.sin(np.pi*t/2))**2
    F2 = (-x1*np.sin(np.pi*t/2) + x2*np.cos(np.pi*t/2))**2
    F3 = (x3 - 0.5*t + 0.25)**2
    lf = F1/a**2 + F2/b**2 + F3/c**2 - 1
    return lf[:, None]

def get_u(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]
    lf = level_set_function(data)[:, 0]
    u_p = np.exp(-(x1 ** 2 + x2 ** 2 + x3 ** 2)) * np.cos(t)
    u_n = (np.exp(-t) * np.sin(np.pi * x1) * np.sin(np.pi * x2) * np.exp(x3) + 1) / 2
    u_x = np.where(lf >= 0, u_p, u_n)
    return u_x[:, None]


def get_up(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]
    u_p = np.exp(-(x1 ** 2 + x2 ** 2 + x3 ** 2)) * np.cos(t)
    return u_p[:, None]


def get_un(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]
    u_n = (np.exp(-t) * np.sin(np.pi * x1) * np.sin(np.pi * x2) * np.exp(x3) + 1) / 2
    return u_n[:, None]

def get_normal_vector(data):
    a = 0.9
    b = 0.7
    c = 0.5

    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]

    theta = np.pi * t / 2.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    u = x1 * cos_t + x2 * sin_t
    v = -x1 * sin_t + x2 * cos_t
    w = x3 - 0.5 * t + 0.25

    dphi_dx1 = u * cos_t / a ** 2 - v * sin_t / b ** 2
    dphi_dx2 = u * sin_t / a ** 2 + v * cos_t / b ** 2
    dphi_dx3 = w / c ** 2

    grad = np.stack([dphi_dx1, dphi_dx2, dphi_dx3], axis=1)

    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    n = grad / norm
    return n


def get_f_p(data, beta):
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]
    r2 = x1 ** 2 + x2 ** 2 + x3 ** 2
    f = -np.exp(-r2) * (np.sin(t) + beta * (4.0 * r2 - 6.0) * np.cos(t))
    return f[:, None]


def get_f_n(data, beta):
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]
    A = np.sin(np.pi * x1) * np.sin(np.pi * x2) * np.exp(x3)
    coeff = np.pi ** 2 * beta - 0.5 * (1 + beta)
    f = np.exp(-t) * A * coeff
    return f[:, None]


def get_psi_D(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]

    up = np.exp(-(x1 ** 2 + x2 ** 2 + x3 ** 2)) * np.cos(t)
    un = (np.exp(-t) * np.sin(np.pi * x1) * np.sin(np.pi * x2) * np.exp(x3) + 1) / 2
    psi = up - un
    return psi[:, None]


def get_psi_N(data, n, beta_p, beta_n):
    x1, x2, x3, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    n1, n2, n3 = n[:, 0], n[:, 1], n[:, 2]

    # term_p: ∇u_p · n
    r2 = x1**2 + x2**2 + x3**2
    x_dot_n = x1 * n1 + x2 * n2 + x3 * n3
    term_p = -2.0 * np.exp(-r2) * np.cos(t) * x_dot_n

    # term_n: ∇u_n · n
    sin1 = np.sin(np.pi * x1)
    sin2 = np.sin(np.pi * x2)
    cos1 = np.cos(np.pi * x1)
    cos2 = np.cos(np.pi * x2)
    exp_x3 = np.exp(x3)

    grad_un_dot_n = 0.5 * np.exp(-t) * exp_x3 * (
        np.pi * cos1 * sin2 * n1 +
        np.pi * sin1 * cos2 * n2 +
        sin1 * sin2 * n3
    )
    term_n = grad_un_dot_n

    psi = beta_p * term_p - beta_n * term_n
    return psi[:, None]

def generate_train_data(num_o, num_ini, num_b, num_if, beta_p, beta_n, device):
    xop, xon = omega_points(num_o)
    xinip, xinin = ini_points(num_ini)
    xb = boundary_points(num_b)
    xif = interface_points(num_if)
    nv = get_normal_vector(xif)
    f_p = get_f_p(xop, beta_p)
    f_n = get_f_n(xon, beta_n)
    u0_p = get_up(xinip)
    u0_n = get_un(xinin)
    g = get_u(xb)
    psi_D = get_psi_D(xif)
    psi_N = get_psi_N(xif, nv, beta_p, beta_n)

    xop = To_tensor_grad(xop, device)
    xon = To_tensor_grad(xon, device)
    xb = To_tensor_grad(xb, device)
    xif = To_tensor_grad(xif, device)
    xinip = To_tensor_grad(xinip, device)
    xinin = To_tensor_grad(xinin, device)

    nv = To_tensor(nv, device)
    f_p = To_tensor(f_p, device)
    f_n = To_tensor(f_n, device)
    u0_p = To_tensor(u0_p, device)
    u0_n = To_tensor(u0_n, device)
    g = To_tensor(g, device)
    psi_D = To_tensor(psi_D, device)
    psi_N = To_tensor(psi_N, device)

    data_op_tr = (xop, f_p)
    data_on_tr = (xon, f_n)
    data_b_tr = (xb, g)
    data_inip_tr = (xinip, u0_p)
    data_inin_tr = (xinin, u0_n)
    data_if_tr = (xif, nv, psi_D, psi_N)
    return data_op_tr, data_on_tr, data_b_tr, data_inip_tr, data_inin_tr, data_if_tr


def generate_test_data(num_test, beta_p, beta_n, device):
    xop, xon = omega_points(num_test)
    up = get_up(xop)
    un = get_un(xon)
    xop = To_tensor(xop, device)
    xon = To_tensor(xon, device)
    up = To_tensor(up, device)
    un = To_tensor(un, device)
    data_op_test = (xop, up)
    data_on_test = (xon, un)
    return data_op_test, data_on_test