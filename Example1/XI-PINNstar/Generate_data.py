# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 上午10:17
# @Author  : NJU_RanBi
import numpy as np
import torch
from functools import wraps
from Network import Vanilla_Net

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
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().double().to(device)
    else:
        x = torch.tensor(x, dtype=torch.float64, device=device)
    x.requires_grad_(True)
    return x


@map_elementwise
def To_tensor(x, device):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().double().to(device)
    else:
        x = torch.tensor(x, dtype=torch.float64, device=device)
    return x


def level_set_function(data, model):
    t = data[:, 2]
    output = model(data)
    X = data[:, 0] + t * output[:, 0]
    Y = data[:, 1] + t * output[:, 1]
    lf = (X - 0.3) ** 2 + Y ** 2 - (torch.pi / 6) ** 2
    return lf[:, None]


def phi_analytical(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    phi = (x1 - 0.3 * torch.cos(torch.pi * t)) ** 2 + (x2 - 0.3 * torch.sin(torch.pi * t)) ** 2 - (torch.pi / 6) ** 2
    return phi[:, None]


def compute_phi_derivative(data, model):
    x = data.clone().detach().requires_grad_(True)
    phi = level_set_function(x, model)
    grad = torch.autograd.grad(phi.sum(), x, retain_graph=True, create_graph=True)[0]
    dphi_dx = grad[:, 0][:, None]
    dphi_dy = grad[:, 1][:, None]
    dphi_dt = grad[:, 2][:, None]
    d2phi_dx2 = torch.autograd.grad(dphi_dx.sum(), x, retain_graph=True, create_graph=False)[0][:, 0][:, None]
    d2phi_dy2 = torch.autograd.grad(dphi_dy.sum(), x, retain_graph=False, create_graph=False)[0][:, 1][:, None]
    return dphi_dx, dphi_dy, dphi_dt, d2phi_dx2, d2phi_dy2


def get_normal_vector(data, model):
    gx, gy, _, _, _= compute_phi_derivative(data, model)
    norm = torch.sqrt(gx ** 2 + gy ** 2)
    n1 = gx / norm
    n2 = gy / norm
    return torch.cat([n1, n2], dim=1)


def omega_points(num, model, device):
    x1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    t = np.random.uniform(low=0, high=1, size=num)[:, None]
    data = np.hstack((x1, x2, t))
    data = To_tensor(data, device)
    phi = level_set_function(data, model)
    index_p = torch.where(phi >= 0)[0]
    index_n = torch.where(phi < 0)[0]
    datap = data[index_p, :]
    datan = data[index_n, :]
    return datap, datan


def ini_points(num, device):
    x1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    t = np.zeros_like(x1)
    data = np.hstack((x1, x2, t))
    data = To_tensor(data, device)
    return data


def boundary_points(num, device):
    index = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb1 = np.hstack((index, np.ones_like(index)))
    index = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb2 = np.hstack((index, -np.ones_like(index)))
    index = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb3 = np.hstack((np.ones_like(index), index))
    index = np.random.uniform(low=-1, high=1, size=num)[:, None]
    xb4 = np.hstack((-np.ones_like(index), index))
    xb = np.vstack((xb1, xb2, xb3, xb4))
    t = np.random.uniform(low=0, high=1, size=4 * num)[:, None]
    data = np.hstack((xb, t))
    data = To_tensor(data, device)
    return data


def interface_points(num, device):
    t = np.random.uniform(low=0, high=1, size=num)[:, None]
    c_x = 0.3 * np.cos(np.pi * t)
    c_y = 0.3 * np.sin(np.pi * t)
    theta = np.random.uniform(low=0, high=2 * np.pi, size=num)[:, None]
    x1 = c_x + np.pi / 6 * np.cos(theta)
    x2 = c_y + np.pi / 6 * np.sin(theta)
    data = np.hstack((x1, x2, t))
    data = To_tensor(data, device)
    return data


def add_dimension(data, model):
    lf = level_set_function(data, model)
    data_add = torch.hstack((data, torch.abs(lf)))
    return data_add


def dudt(data, beta):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    F = torch.sqrt((x1 - 0.3 * torch.cos(torch.pi * t)) ** 2 + (x2 - 0.3 * torch.sin(torch.pi * t)) ** 2)
    ut = 15.0 / (torch.pi * beta) * F ** 3 * (
            0.6 * torch.pi * torch.sin(torch.pi * t) * (x1 - 0.3 * torch.cos(torch.pi * t)) -
            0.6 * torch.pi * torch.cos(torch.pi * t) * (x2 - 0.3 * torch.sin(torch.pi * t))
    )
    return ut[:, None]


def d2udx2(data, beta):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    F = torch.sqrt((x1 - 0.3 * torch.cos(torch.pi * t)) ** 2 + (x2 - 0.3 * torch.sin(torch.pi * t)) ** 2)
    uxx = 90.0 / (beta * torch.pi) * F * (x1 - 0.3 * torch.cos(torch.pi * t)) ** 2 + 30.0 / (beta * torch.pi) * F ** 3
    return uxx[:, None]


def d2udy2(data, beta):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    F = torch.sqrt((x1 - 0.3 * torch.cos(torch.pi * t)) ** 2 + (x2 - 0.3 * torch.sin(torch.pi * t)) ** 2)
    uyy = 90.0 / (beta * torch.pi) * F * (x2 - 0.3 * torch.sin(torch.pi * t)) ** 2 + 30.0 / (beta * torch.pi) * F ** 3
    return uyy[:, None]


def get_g(data, beta_p, beta_n):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    F = torch.sqrt((x1 - 0.3 * torch.cos(torch.pi * t)) ** 2 + (x2 - 0.3 * torch.sin(torch.pi * t)) ** 2)
    g = 6.0 / (beta_p * torch.pi) * F ** 5 + (torch.pi / 6) ** 4 * (1.0 / beta_n - 1.0 / beta_p)
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
    lf = phi_analytical(data)[:, 0]
    F = torch.sqrt((x1 - 0.3 * torch.cos(torch.pi * t)) ** 2 + (x2 - 0.3 * torch.sin(torch.pi * t)) ** 2)
    u_p = 6.0 / (beta_p * torch.pi) * F ** 5 + (torch.pi / 6) ** 4 * (1.0 / beta_n - 1.0 / beta_p)
    u_n = 6.0 / (beta_n * torch.pi) * F ** 5
    u_x = torch.where(lf >= 0, u_p, u_n)
    return u_x[:, None]


def generate_train_data(num_o, num_ini, num_b, num_if, beta_p, beta_n, device):
    inverse_mapping = Vanilla_Net(3, 64, 2, 1).to(device)
    inverse_mapping.load_state_dict(torch.load('inverse_mapping.mdl'))
    inverse_mapping.eval()

    xop, xon = omega_points(num_o, inverse_mapping, device)
    xini = ini_points(num_ini, device)
    xb = boundary_points(num_b, device)
    xif = interface_points(num_if, device)

    f_p = get_f(xop, beta_p)
    f_n = get_f(xon, beta_n)
    u0 = get_u(xini, beta_p, beta_n)
    g = get_g(xb, beta_p, beta_n)

    nor = get_normal_vector(xif, inverse_mapping)
    phix_xop, phiy_xop, phit_xop, phix2_xop, phiy2_xop = compute_phi_derivative(xop, inverse_mapping)
    phix_xon, phiy_xon, phit_xon, phix2_xon, phiy2_xon = compute_phi_derivative(xon, inverse_mapping)
    phix_xif, phiy_xif, _, _, _ = compute_phi_derivative(xif, inverse_mapping)

    xop_add = add_dimension(xop, inverse_mapping)
    xon_add = add_dimension(xon, inverse_mapping)
    xb_add = add_dimension(xb, inverse_mapping)
    xini_add = add_dimension(xini, inverse_mapping)
    xif_add = add_dimension(xif, inverse_mapping)

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
    phix2_xop = To_tensor(phix2_xop, device)
    phiy2_xop = To_tensor(phiy2_xop, device)
    phix2_xon = To_tensor(phix2_xon, device)
    phiy2_xon = To_tensor(phiy2_xon, device)
    data_op_tr = (xop_add, f_p, phix_xop, phiy_xop, phit_xop, phix2_xop, phiy2_xop)
    data_on_tr = (xon_add, f_n, phix_xon, phiy_xon, phit_xon, phix2_xon, phiy2_xon)
    data_b_tr = (xb_add, g)
    data_ini_tr = (xini_add, u0)
    data_if_tr = (xif_add, nor, phix_xif, phiy_xif)

    return data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr



def generate_test_data(num_test, beta_p, beta_n, device):
    inverse_mapping = Vanilla_Net(3, 64, 2, 1).to(device)
    inverse_mapping.load_state_dict(torch.load('inverse_mapping.mdl'))
    inverse_mapping.eval()
    xop, xon = omega_points(num_test, inverse_mapping, device)
    x = torch.vstack((xop, xon))
    u = get_u(x, beta_p, beta_n)
    x_add = add_dimension(x, inverse_mapping)
    x_add = To_tensor(x_add, device)
    u = To_tensor(u, device)
    data_test = (x_add, u)
    return data_test