# -*- coding: utf-8 -*-
# @Time    : 2026/5/26 下午3:07
# @Author  : NJU_RanBi
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io as sio
from functools import wraps
from IPython.utils import data
from scipy.optimize import root_scalar
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
    lf = X**2 + Y**2 - 0.4 - (1.0/np.pi) * torch.sin(torch.pi * X) * torch.cos(torch.pi * Y)
    return lf[:, None]


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
    x1 = np.random.uniform(low=-1, high=1, size=2 * num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=2 * num)[:, None]
    t = np.random.uniform(low=0, high=1, size=2 * num)[:, None]
    r2 = x1 ** 2 + x2 ** 2
    data = np.hstack((x1, x2, t))[(r2 <= 1)[:, 0], :][:num, :]
    data = To_tensor(data, device)
    phi = level_set_function(data, model)
    index_p = torch.where(phi >= 0)[0]
    index_n = torch.where(phi < 0)[0]
    datap = data[index_p, :]
    datan = data[index_n, :]
    return datap, datan


def ini_points(num, device):
    x1 = np.random.uniform(low=-1, high=1, size=2 * num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=2 * num)[:, None]
    t = np.zeros_like(x1)
    r2 = x1 ** 2 + x2 ** 2
    data = np.hstack((x1, x2, t))[(r2 <= 1)[:, 0], :][:num, :]
    data = To_tensor(data, device)
    return data


def boundary_points(num, device):
    theta = np.random.uniform(low=0, high=2 * np.pi, size=num)[:, None]
    x1 = np.cos(theta)
    x2 = np.sin(theta)
    t = np.random.uniform(low=0, high=1, size=num)[:, None]
    data = np.hstack((x1, x2, t))
    data = To_tensor(data, device)
    return data

def interface_points(num, device):
    theta = np.random.uniform(0, 2 * np.pi, num)
    t = np.random.uniform(0, 1, num)
    r = np.zeros(num)
    for i in range(num):
        th = theta[i]
        def f(r_val):
            xi = r_val * np.cos(th)
            eta = r_val * np.sin(th)
            return r_val**2 - 0.4 - (1.0/np.pi) * np.sin(np.pi * xi) * np.cos(np.pi * eta)
        try:
            sol = root_scalar(f, bracket=[0.2, 0.9], method='bisect')
            r[i] = sol.root
        except ValueError:
            r[i] = 0.6
    xi = r * np.cos(theta)
    eta = r * np.sin(theta)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    x1 = xi * cos_t - eta * sin_t
    x2 = xi * sin_t + eta * cos_t
    data = np.hstack((x1[:, None], x2[:, None], t[:, None]))
    data = To_tensor(data, device)
    return data


def add_dimension_lf(data, model):
    lf = level_set_function(data, model)
    data_add = torch.hstack((data, torch.abs(lf)))
    return data_add


def add_dimension_sign_p(data, device):
    return torch.hstack((data, torch.ones((len(data[:, 0]), 1), device=device)))


def add_dimension_sign_n(data, device):
    return torch.hstack((data, -torch.ones((len(data[:, 0]), 1), device=device)))


def get_u_p(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]

    xi = x1 * torch.cos(t) + x2 * torch.sin(t)
    eta = -x1 * torch.sin(t) + x2 * torch.cos(t)

    phi = xi ** 2 + eta ** 2 - 0.4 - (1.0 / torch.pi) * torch.sin(torch.pi * xi) * torch.cos(torch.pi * eta)

    phi_xi = 2 * xi - torch.cos(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_eta = 2 * eta + torch.sin(torch.pi * xi) * torch.sin(torch.pi * eta)

    phi_x1 = phi_xi * torch.cos(t) - phi_eta * torch.sin(t)
    phi_x2 = phi_xi * torch.sin(t) + phi_eta * torch.cos(t)

    u_base1 = torch.exp(x1) * torch.sin(x2 + t)
    u_base2 = torch.exp(x1) * torch.cos(x2 + t)

    u_p_1 = u_base1 + phi * phi_x2
    u_p_2 = u_base2 - phi * phi_x1
    return u_p_1, u_p_2


def get_u_n(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    u_n_1 = torch.exp(x1) * torch.sin(x2 + t)
    u_n_2 = torch.exp(x1) * torch.cos(x2 + t)
    return u_n_1, u_n_2


def get_p_p(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    p = torch.exp(t) * torch.sin(x1) * torch.sin(x2)
    return p


def get_p_n(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    p = torch.cos(t) * torch.sin(x1 + x2)
    return p


def get_fluid_velocity(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    v1 = -x2
    v2 = x1
    return v1[:, None], v2[:, None]


def _phi_derivatives(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)

    xi = x1 * cos_t + x2 * sin_t
    eta = -x1 * sin_t + x2 * cos_t
    phi = xi**2 + eta**2 - 0.4 - (1.0/torch.pi) * torch.sin(torch.pi * xi) * torch.cos(torch.pi * eta)

    phi_xi = 2*xi - torch.cos(torch.pi*xi) * torch.cos(torch.pi*eta)
    phi_eta = 2*eta + torch.sin(torch.pi*xi) * torch.sin(torch.pi*eta)

    phi_x1 = phi_xi * cos_t - phi_eta * sin_t
    phi_x2 = phi_xi * sin_t + phi_eta * cos_t
    phi_t = phi_xi * eta - phi_eta * xi

    phi_xi_xi = 2 + torch.pi * torch.sin(torch.pi*xi) * torch.cos(torch.pi*eta)
    phi_eta_eta = phi_xi_xi
    phi_xi_eta = torch.pi * torch.cos(torch.pi*xi) * torch.sin(torch.pi*eta)

    phi_x1x1 = (phi_xi_xi * cos_t**2 - 2*phi_xi_eta * sin_t*cos_t + phi_eta_eta * sin_t**2)
    phi_x2x2 = (phi_xi_xi * sin_t**2 + 2*phi_xi_eta * sin_t*cos_t + phi_eta_eta * cos_t**2)
    phi_x1x2 = ((phi_xi_xi - phi_eta_eta) * sin_t*cos_t + phi_xi_eta * (cos_t**2 - sin_t**2))

    return xi, eta, phi, phi_x1, phi_x2, phi_t, phi_x1x1, phi_x2x2, phi_x1x2

def get_du_pdx(data):
    du_p_1dx, du_p_2dx = get_du_ndx(data)
    _, _, phi, phi_x1, phi_x2, _, phi_x1x1, phi_x2x2, phi_x1x2 = _phi_derivatives(data)
    du_p_1dx = du_p_1dx + phi_x1 * phi_x2 + phi * phi_x1x2
    du_p_2dx = du_p_2dx - (phi_x1 * phi_x1 + phi * phi_x1x1)
    return du_p_1dx, du_p_2dx


def get_du_pdy(data):
    du_p_1dy, du_p_2dy = get_du_ndy(data)
    _, _, phi, phi_x1, phi_x2, _, phi_x1x1, phi_x2x2, phi_x1x2 = _phi_derivatives(data)
    du_p_1dy = du_p_1dy + phi_x2 * phi_x2 + phi * phi_x2x2
    du_p_2dy = du_p_2dy - (phi_x1 * phi_x2 + phi * phi_x1x2)
    return du_p_1dy, du_p_2dy


def get_du_pdt(data):
    du_p_1dt, du_p_2dt = get_du_ndt(data)
    _, _, phi, phi_x1, phi_x2, phi_t, _, _, _ = _phi_derivatives(data)

    x1, x2, t = data[:, 0], data[:, 1], data[:, 2]
    cos_t, sin_t = torch.cos(t), torch.sin(t)
    xi = x1 * cos_t + x2 * sin_t
    eta = -x1 * sin_t + x2 * cos_t
    phi_xi = 2 * xi - torch.cos(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_eta = 2 * eta + torch.sin(torch.pi * xi) * torch.sin(torch.pi * eta)

    phi_xi_t = (2 + torch.pi * torch.sin(torch.pi * xi) * torch.cos(torch.pi * eta)) * eta + (
                torch.pi * torch.cos(torch.pi * xi) * torch.sin(torch.pi * eta)) * (-xi)
    phi_eta_t = (torch.pi * torch.cos(torch.pi * xi) * torch.sin(torch.pi * eta)) * eta + (
                2 + torch.pi * torch.sin(torch.pi * xi) * torch.cos(torch.pi * eta)) * (-xi)

    phi_x2_t = phi_xi_t * sin_t + phi_xi * cos_t + phi_eta_t * cos_t - phi_eta * sin_t
    phi_x1_t = phi_xi_t * cos_t - phi_xi * sin_t - phi_eta_t * sin_t - phi_eta * cos_t

    du_p_1dt = du_p_1dt + phi_t * phi_x2 + phi * phi_x2_t
    du_p_2dt = du_p_2dt - (phi_t * phi_x1 + phi * phi_x1_t)
    return du_p_1dt, du_p_2dt


def get_d2u_pdx2(data):
    d2u_p_1dx2, d2u_p_2dx2 = get_d2u_ndx2(data)
    x1, x2, t = data[:, 0], data[:, 1], data[:, 2]
    cos_t, sin_t = torch.cos(t), torch.sin(t)
    xi = x1 * cos_t + x2 * sin_t
    eta = -x1 * sin_t + x2 * cos_t
    phi = xi ** 2 + eta ** 2 - 0.4 - (1.0 / torch.pi) * torch.sin(torch.pi * xi) * torch.cos(torch.pi * eta)

    phi_xi = 2 * xi - torch.cos(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_eta = 2 * eta + torch.sin(torch.pi * xi) * torch.sin(torch.pi * eta)
    phi_xi_xi = 2 + torch.pi * torch.sin(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_eta_eta = phi_xi_xi
    phi_xi_eta = torch.pi * torch.cos(torch.pi * xi) * torch.sin(torch.pi * eta)

    phi_x1 = phi_xi * cos_t - phi_eta * sin_t
    phi_x2 = phi_xi * sin_t + phi_eta * cos_t
    phi_x1x1 = phi_xi_xi * cos_t ** 2 - 2 * phi_xi_eta * sin_t * cos_t + phi_eta_eta * sin_t ** 2
    phi_x1x2 = (phi_xi_xi - phi_eta_eta) * sin_t * cos_t + phi_xi_eta * (cos_t ** 2 - sin_t ** 2)

    phi_xi_xi_xi = torch.pi ** 2 * torch.cos(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_xi_xi_eta = -torch.pi ** 2 * torch.sin(torch.pi * xi) * torch.sin(torch.pi * eta)
    phi_xi_eta_eta = torch.pi ** 2 * torch.cos(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_eta_eta_eta = -torch.pi ** 2 * torch.sin(torch.pi * xi) * torch.sin(torch.pi * eta)

    phi_x1x1x1 = (phi_xi_xi_xi * cos_t ** 3 - 3 * phi_xi_xi_eta * cos_t ** 2 * sin_t +
                  3 * phi_xi_eta_eta * cos_t * sin_t ** 2 - phi_eta_eta_eta * sin_t ** 3)
    phi_x2x1x1 = (phi_xi_xi_xi * cos_t ** 2 * sin_t + phi_xi_xi_eta * cos_t * (cos_t ** 2 - 2 * sin_t ** 2) -
                  phi_xi_eta_eta * sin_t * (2 * cos_t ** 2 - sin_t ** 2) + phi_eta_eta_eta * sin_t ** 2 * cos_t)

    d2u_p_1dx2 = d2u_p_1dx2 + phi_x1x1 * phi_x2 + 2 * phi_x1 * phi_x1x2 + phi * phi_x2x1x1
    d2u_p_2dx2 = d2u_p_2dx2 - (phi_x1x1 * phi_x1 + 2 * phi_x1 * phi_x1x1 + phi * phi_x1x1x1)
    return d2u_p_1dx2, d2u_p_2dx2


def get_d2u_pdy2(data):
    d2u_p_1dy2, d2u_p_2dy2 = get_d2u_ndy2(data)
    x1, x2, t = data[:, 0], data[:, 1], data[:, 2]
    cos_t, sin_t = torch.cos(t), torch.sin(t)
    xi = x1 * cos_t + x2 * sin_t
    eta = -x1 * sin_t + x2 * cos_t
    phi = xi ** 2 + eta ** 2 - 0.4 - (1.0 / torch.pi) * torch.sin(torch.pi * xi) * torch.cos(torch.pi * eta)

    phi_xi = 2 * xi - torch.cos(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_eta = 2 * eta + torch.sin(torch.pi * xi) * torch.sin(torch.pi * eta)
    phi_xi_xi = 2 + torch.pi * torch.sin(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_eta_eta = phi_xi_xi
    phi_xi_eta = torch.pi * torch.cos(torch.pi * xi) * torch.sin(torch.pi * eta)

    phi_x2 = phi_xi * sin_t + phi_eta * cos_t
    phi_x1 = phi_xi * cos_t - phi_eta * sin_t
    phi_x2x2 = phi_xi_xi * sin_t ** 2 + 2 * phi_xi_eta * sin_t * cos_t + phi_eta_eta * cos_t ** 2
    phi_x1x2 = (phi_xi_xi - phi_eta_eta) * sin_t * cos_t + phi_xi_eta * (cos_t ** 2 - sin_t ** 2)

    phi_xi_xi_xi = torch.pi ** 2 * torch.cos(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_xi_xi_eta = -torch.pi ** 2 * torch.sin(torch.pi * xi) * torch.sin(torch.pi * eta)
    phi_xi_eta_eta = torch.pi ** 2 * torch.cos(torch.pi * xi) * torch.cos(torch.pi * eta)
    phi_eta_eta_eta = -torch.pi ** 2 * torch.sin(torch.pi * xi) * torch.sin(torch.pi * eta)

    phi_x2y2y2 = (phi_xi_xi_xi * sin_t ** 3 + 3 * phi_xi_xi_eta * sin_t ** 2 * cos_t +
                  3 * phi_xi_eta_eta * sin_t * cos_t ** 2 + phi_eta_eta_eta * cos_t ** 3)
    phi_x1y2y2 = (phi_xi_xi_xi * sin_t ** 2 * cos_t + phi_xi_xi_eta * sin_t * (2 * cos_t ** 2 - sin_t ** 2) +
                  phi_xi_eta_eta * cos_t * (cos_t ** 2 - 2 * sin_t ** 2) - phi_eta_eta_eta * sin_t * cos_t ** 2)

    d2u_p_1dy2 = d2u_p_1dy2 + phi_x2x2 * phi_x2 + 2 * phi_x2 * phi_x2x2 + phi * phi_x2y2y2
    d2u_p_2dy2 = d2u_p_2dy2 - (phi_x2x2 * phi_x1 + 2 * phi_x2 * phi_x1x2 + phi * phi_x1y2y2)
    return d2u_p_1dy2, d2u_p_2dy2


def get_du_ndx(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    du_n1dx = torch.exp(x1) * torch.sin(x2 + t)
    du_n2dx = torch.exp(x1) * torch.cos(x2 + t)
    return du_n1dx, du_n2dx


def get_du_ndy(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    du_n1dy = torch.exp(x1) * torch.cos(x2 + t)
    du_n2dy = -torch.exp(x1) * torch.sin(x2 + t)
    return du_n1dy, du_n2dy


def get_du_ndt(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    du_n1dt = torch.exp(x1) * torch.cos(x2 + t)
    du_n2dt = -torch.exp(x1) * torch.sin(x2 + t)
    return du_n1dt, du_n2dt


def get_d2u_ndx2(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    d2u_n1dx2 = torch.exp(x1) * torch.sin(x2 + t)
    d2u_n2dx2 = torch.exp(x1) * torch.cos(x2 + t)
    return d2u_n1dx2, d2u_n2dx2


def get_d2u_ndy2(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    d2u_n1dy2 = -torch.exp(x1) * torch.sin(x2 + t)
    d2u_n2dy2 = -torch.exp(x1) * torch.cos(x2 + t)
    return d2u_n1dy2, d2u_n2dy2


def get_dp_pdx(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    dp_pdx = torch.exp(t) * torch.cos(x1) * torch.sin(x2)
    return dp_pdx


def get_dp_pdy(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    dp_pdy = torch.exp(t) * torch.sin(x1) * torch.cos(x2)
    return dp_pdy


def get_dp_ndx(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    dp_ndx = torch.cos(t) * torch.cos(x1 + x2)
    return dp_ndx


def get_dp_ndy(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    dp_ndy = torch.cos(t) * torch.cos(x1 + x2)
    return dp_ndy


def get_f_p(data, nu_p):
    du_p_1dx, du_p_2dx = get_du_pdx(data)
    du_p_1dy, du_p_2dy = get_du_pdy(data)
    du_p_1dt, du_p_2dt = get_du_pdt(data)
    d2u_p_1dx2, d2u_p_2dx2 = get_d2u_pdx2(data)
    d2u_p_1dy2, d2u_p_2dy2 = get_d2u_pdy2(data)
    dp_pdx = get_dp_pdx(data)
    dp_pdy = get_dp_pdy(data)
    v1, v2 = get_fluid_velocity(data)
    f_p_1 = du_p_1dt + v1[:, 0] * du_p_1dx + v2[:, 0] * du_p_1dy - nu_p * (d2u_p_1dx2 + d2u_p_1dy2) + dp_pdx
    f_p_2 = du_p_2dt + v1[:, 0] * du_p_2dx + v2[:, 0] * du_p_2dy - nu_p * (d2u_p_2dx2 + d2u_p_2dy2) + dp_pdy
    return f_p_1[:, None], f_p_2[:, None]


def get_f_n(data, nu_n):
    du_n_1dx, du_n_2dx = get_du_ndx(data)
    du_n_1dy, du_n_2dy = get_du_ndy(data)
    du_n_1dt, du_n_2dt = get_du_ndt(data)
    d2u_n_1dx2, d2u_n_2dx2 = get_d2u_ndx2(data)
    d2u_n_1dy2, d2u_n_2dy2 = get_d2u_ndy2(data)
    dp_ndx = get_dp_ndx(data)
    dp_ndy = get_dp_ndy(data)
    v1, v2 = get_fluid_velocity(data)
    f_n_1 = du_n_1dt + v1[:, 0] * du_n_1dx + v2[:, 0] * du_n_1dy - nu_n * (d2u_n_1dx2 + d2u_n_1dy2) + dp_ndx
    f_n_2 = du_n_2dt + v1[:, 0] * du_n_2dx + v2[:, 0] * du_n_2dy - nu_n * (d2u_n_2dx2 + d2u_n_2dy2) + dp_ndy
    return f_n_1[:, None], f_n_2[:, None]


def get_g(data):
    g_1, g_2 = get_u_p(data)
    return g_1[:, None], g_2[:, None]


def get_psi(data, nor, nu_p, nu_n):
    du_p_1dx, du_p_2dx = get_du_pdx(data)
    du_p_1dy, du_p_2dy = get_du_pdy(data)
    du_n_1dx, du_n_2dx = get_du_ndx(data)
    du_n_1dy, du_n_2dy = get_du_ndy(data)
    p_p = get_p_p(data)
    p_n = get_p_n(data)
    n1 = nor[:, 0]
    n2 = nor[:, 1]

    psi_1 = (nu_p * (du_p_1dx * n1 + du_p_2dx * n2 + du_p_1dx * n1 + du_p_1dy * n2)
             - p_p * n1
             - nu_n * (du_n_1dx * n1 + du_n_2dx * n2 + du_n_1dx * n1 + du_n_1dy * n2)
             + p_n * n1)

    psi_2 = (nu_p * (du_p_1dy * n1 + du_p_2dy * n2 + du_p_2dx * n1 + du_p_2dy * n2)
             - p_p * n2
             - nu_n * (du_n_1dy * n1 + du_n_2dy * n2 + du_n_2dx * n1 + du_n_2dy * n2)
             + p_n * n2)

    return psi_1[:, None], psi_2[:, None]


def get_u(data, model):
    u_p_1, u_p_2 = get_u_p(data)
    u_n_1, u_n_2 = get_u_n(data)
    lf = level_set_function(data, model)[:, 0]
    u_1 = torch.where(lf > 0, u_p_1, u_n_1)
    u_2 = torch.where(lf > 0, u_p_2, u_n_2)
    return u_1[:, None], u_2[:, None]

def generate_train_data(num_o, num_ini, num_b, num_if, nu_p, nu_n, device):
    inverse_mapping = Vanilla_Net(3, 64, 2, 1).to(device)
    inverse_mapping.load_state_dict(torch.load('inverse_mapping.mdl'))
    inverse_mapping.eval()

    xop, xon = omega_points(num_o, inverse_mapping, device)
    xb = boundary_points(num_b, device)
    xif = interface_points(num_if, device)
    xini = ini_points(num_ini, device)
    nor = get_normal_vector(xif, inverse_mapping)

    f_p_1, f_p_2 = get_f_p(xop, nu_p)
    f_n_1, f_n_2 = get_f_n(xon, nu_n)
    v_p_1, v_p_2 = get_fluid_velocity(xop)
    v_n_1, v_n_2 = get_fluid_velocity(xon)
    g_1, g_2 = get_g(xb)
    psi_1, psi_2 = get_psi(xif, nor, nu_p, nu_n)
    u0_1, u0_2 = get_u(xini, inverse_mapping)

    phix_xop, phiy_xop, phit_xop, phix2_xop, phiy2_xop = compute_phi_derivative(xop, inverse_mapping)
    phix_xon, phiy_xon, phit_xon, phix2_xon, phiy2_xon = compute_phi_derivative(xon, inverse_mapping)
    phix_xif, phiy_xif, _, _, _ = compute_phi_derivative(xif, inverse_mapping)

    xop_add_lf = add_dimension_lf(xop, inverse_mapping)
    xon_add_lf = add_dimension_lf(xon, inverse_mapping)
    xb_add_lf = add_dimension_lf(xb, inverse_mapping)
    xini_add_lf = add_dimension_lf(xini, inverse_mapping)
    xif_add_lf = add_dimension_lf(xif, inverse_mapping)

    xop_add_sign = add_dimension_sign_p(xop, device)
    xon_add_sign = add_dimension_sign_n(xon, device)
    xif_add_sign_p = add_dimension_sign_p(xif, device)
    xif_add_sign_n = add_dimension_sign_n(xif, device)

    xop_add_lf = To_tensor_grad(xop_add_lf, device)
    xon_add_lf = To_tensor_grad(xon_add_lf, device)
    xb_add_lf = To_tensor_grad(xb_add_lf, device)
    xif_add_lf = To_tensor_grad(xif_add_lf, device)
    xini_add_lf = To_tensor_grad(xini_add_lf, device)

    xop_add_sign = To_tensor_grad(xop_add_sign, device)
    xon_add_sign = To_tensor_grad(xon_add_sign, device)
    xif_add_sign_p = To_tensor_grad(xif_add_sign_p, device)
    xif_add_sign_n = To_tensor_grad(xif_add_sign_n, device)

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
    f_p_1 = To_tensor(f_p_1, device)
    f_p_2 = To_tensor(f_p_2, device)
    f_n_1 = To_tensor(f_n_1, device)
    f_n_2 = To_tensor(f_n_2, device)
    v_p_1 = To_tensor(v_p_1, device)
    v_p_2 = To_tensor(v_p_2, device)
    v_n_1 = To_tensor(v_n_1, device)
    v_n_2 = To_tensor(v_n_2, device)
    g_1 = To_tensor(g_1, device)
    g_2 = To_tensor(g_2, device)
    psi_1 = To_tensor(psi_1, device)
    psi_2 = To_tensor(psi_2, device)
    u0_1 = To_tensor(u0_1, device)
    u0_2 = To_tensor(u0_2, device)

    data_op_tr = (xop_add_lf, xop_add_sign, f_p_1, f_p_2, v_p_1, v_p_2, phix_xop, phiy_xop, phit_xop, phix2_xop, phiy2_xop)
    data_on_tr = (xon_add_lf, xon_add_sign, f_n_1, f_n_2, v_n_1, v_n_2, phix_xon, phiy_xon, phit_xon, phix2_xon, phiy2_xon)
    data_b_tr = (xb_add_lf, g_1, g_2)
    data_ini_tr = (xini_add_lf, u0_1, u0_2)
    data_if_tr = (xif_add_lf, xif_add_sign_p, xif_add_sign_n, nor, psi_1, psi_2, phix_xif, phiy_xif)
    return data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr


def generate_test_data(num_test, device):
    inverse_mapping = Vanilla_Net(3, 64, 2, 1).to(device)
    inverse_mapping.load_state_dict(torch.load('inverse_mapping.mdl'))
    inverse_mapping.eval()
    xop, xon = omega_points(num_test, inverse_mapping, device)
    x = torch.vstack((xop, xon))
    u1, u2 = get_u(x, inverse_mapping)
    x_add = add_dimension_lf(x, inverse_mapping)
    x_add = To_tensor(x_add, device)
    u1 = To_tensor(u1, device)
    u2 = To_tensor(u2, device)
    data_test = (x_add, u1, u2)
    return data_test
