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
    x = torch.tensor(x).double().requires_grad_(True).to(device)
    return x


@map_elementwise
def To_tensor(x, device):
    x = torch.tensor(x).double().to(device)
    return x


def omega_points(num):
    x1 = np.random.uniform(low=-1, high=1, size=2 * num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=2 * num)[:, None]
    t = np.random.uniform(low=0, high=1, size=2 * num)[:, None]
    r2 = x1 ** 2 + x2 ** 2
    data = np.hstack((x1, x2, t))[(r2 <= 1)[:, 0], :][:num, :]
    phi = level_set_function(data)
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    datap = data[index_p, :]
    datan = data[index_n, :]
    return datap, datan


def ini_points(num):
    x1 = np.random.uniform(low=-1, high=1, size=2 * num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=2 * num)[:, None]
    t = np.zeros_like(x1)
    r2 = x1 ** 2 + x2 ** 2
    data = np.hstack((x1, x2, t))[(r2 <= 1)[:, 0], :][:num, :]
    return data


def boundary_points(num):
    theta = np.random.uniform(low=0, high=2 * np.pi, size=num)[:, None]
    x1 = np.cos(theta)
    x2 = np.sin(theta)
    t = np.random.uniform(low=0, high=1, size=num)[:, None]
    data = np.hstack((x1, x2, t))
    return data


def interface_points(num):
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
    return data


def level_set_function(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    xi = x1 * np.cos(t) + x2 * np.sin(t)
    eta = -x1 * np.sin(t) + x2 * np.cos(t)
    lf = xi**2 + eta**2 - 0.4 - (1.0/np.pi) * np.sin(np.pi * xi) * np.cos(np.pi * eta)
    return lf[:, None]


def add_dimension_lf(data):
    lf = level_set_function(data)
    data_add = np.hstack((data, np.abs(lf)))
    return data_add


def add_dimension_sign(data):
    lf = level_set_function(data)
    add_x = np.ones_like(lf)
    index_n = np.where(lf < 0)[0]
    add_x[index_n] = -1
    data = np.hstack((data, add_x))
    return data


def add_dimension_sign_p(data):
    return np.hstack((data, np.ones((len(data[:, 0]), 1))))


def add_dimension_sign_n(data):
    return np.hstack((data, -np.ones((len(data[:, 0]), 1))))


def get_u_p(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]

    xi = x1 * np.cos(t) + x2 * np.sin(t)
    eta = -x1 * np.sin(t) + x2 * np.cos(t)

    phi = xi ** 2 + eta ** 2 - 0.4 - (1.0 / np.pi) * np.sin(np.pi * xi) * np.cos(np.pi * eta)

    phi_xi = 2 * xi - np.cos(np.pi * xi) * np.cos(np.pi * eta)
    phi_eta = 2 * eta + np.sin(np.pi * xi) * np.sin(np.pi * eta)

    phi_x1 = phi_xi * np.cos(t) - phi_eta * np.sin(t)
    phi_x2 = phi_xi * np.sin(t) + phi_eta * np.cos(t)

    u_base1 = np.exp(x1) * np.sin(x2 + t)
    u_base2 = np.exp(x1) * np.cos(x2 + t)

    u_p_1 = u_base1 + phi * phi_x2
    u_p_2 = u_base2 - phi * phi_x1
    return u_p_1, u_p_2


def get_u_n(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    u_n_1 = np.exp(x1) * np.sin(x2 + t)
    u_n_2 = np.exp(x1) * np.cos(x2 + t)
    return u_n_1, u_n_2


def get_p_p(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    p = np.exp(t) * np.sin(x1) * np.sin(x2)
    return p


def get_p_n(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    p = np.cos(t) * np.sin(x1 + x2)
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
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    xi = x1 * cos_t + x2 * sin_t
    eta = -x1 * sin_t + x2 * cos_t
    phi = xi**2 + eta**2 - 0.4 - (1.0/np.pi) * np.sin(np.pi * xi) * np.cos(np.pi * eta)

    phi_xi = 2*xi - np.cos(np.pi*xi) * np.cos(np.pi*eta)
    phi_eta = 2*eta + np.sin(np.pi*xi) * np.sin(np.pi*eta)

    phi_x1 = phi_xi * cos_t - phi_eta * sin_t
    phi_x2 = phi_xi * sin_t + phi_eta * cos_t
    phi_t = phi_xi * eta - phi_eta * xi

    phi_xi_xi = 2 + np.pi * np.sin(np.pi*xi) * np.cos(np.pi*eta)
    phi_eta_eta = phi_xi_xi
    phi_xi_eta = np.pi * np.cos(np.pi*xi) * np.sin(np.pi*eta)

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
    cos_t, sin_t = np.cos(t), np.sin(t)
    xi = x1 * cos_t + x2 * sin_t
    eta = -x1 * sin_t + x2 * cos_t
    phi_xi = 2 * xi - np.cos(np.pi * xi) * np.cos(np.pi * eta)
    phi_eta = 2 * eta + np.sin(np.pi * xi) * np.sin(np.pi * eta)

    phi_xi_t = (2 + np.pi * np.sin(np.pi * xi) * np.cos(np.pi * eta)) * eta + (
                np.pi * np.cos(np.pi * xi) * np.sin(np.pi * eta)) * (-xi)
    phi_eta_t = (np.pi * np.cos(np.pi * xi) * np.sin(np.pi * eta)) * eta + (
                2 + np.pi * np.sin(np.pi * xi) * np.cos(np.pi * eta)) * (-xi)

    phi_x2_t = phi_xi_t * sin_t + phi_xi * cos_t + phi_eta_t * cos_t - phi_eta * sin_t
    phi_x1_t = phi_xi_t * cos_t - phi_xi * sin_t - phi_eta_t * sin_t - phi_eta * cos_t

    du_p_1dt = du_p_1dt + phi_t * phi_x2 + phi * phi_x2_t
    du_p_2dt = du_p_2dt - (phi_t * phi_x1 + phi * phi_x1_t)
    return du_p_1dt, du_p_2dt


def get_d2u_pdx2(data):
    d2u_p_1dx2, d2u_p_2dx2 = get_d2u_ndx2(data)
    x1, x2, t = data[:, 0], data[:, 1], data[:, 2]
    cos_t, sin_t = np.cos(t), np.sin(t)
    xi = x1 * cos_t + x2 * sin_t
    eta = -x1 * sin_t + x2 * cos_t
    phi = xi ** 2 + eta ** 2 - 0.4 - (1.0 / np.pi) * np.sin(np.pi * xi) * np.cos(np.pi * eta)

    phi_xi = 2 * xi - np.cos(np.pi * xi) * np.cos(np.pi * eta)
    phi_eta = 2 * eta + np.sin(np.pi * xi) * np.sin(np.pi * eta)
    phi_xi_xi = 2 + np.pi * np.sin(np.pi * xi) * np.cos(np.pi * eta)
    phi_eta_eta = phi_xi_xi
    phi_xi_eta = np.pi * np.cos(np.pi * xi) * np.sin(np.pi * eta)

    phi_x1 = phi_xi * cos_t - phi_eta * sin_t
    phi_x2 = phi_xi * sin_t + phi_eta * cos_t
    phi_x1x1 = phi_xi_xi * cos_t ** 2 - 2 * phi_xi_eta * sin_t * cos_t + phi_eta_eta * sin_t ** 2
    phi_x1x2 = (phi_xi_xi - phi_eta_eta) * sin_t * cos_t + phi_xi_eta * (cos_t ** 2 - sin_t ** 2)

    phi_xi_xi_xi = np.pi ** 2 * np.cos(np.pi * xi) * np.cos(np.pi * eta)
    phi_xi_xi_eta = -np.pi ** 2 * np.sin(np.pi * xi) * np.sin(np.pi * eta)
    phi_xi_eta_eta = np.pi ** 2 * np.cos(np.pi * xi) * np.cos(np.pi * eta)
    phi_eta_eta_eta = -np.pi ** 2 * np.sin(np.pi * xi) * np.sin(np.pi * eta)

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
    cos_t, sin_t = np.cos(t), np.sin(t)
    xi = x1 * cos_t + x2 * sin_t
    eta = -x1 * sin_t + x2 * cos_t
    phi = xi ** 2 + eta ** 2 - 0.4 - (1.0 / np.pi) * np.sin(np.pi * xi) * np.cos(np.pi * eta)

    phi_xi = 2 * xi - np.cos(np.pi * xi) * np.cos(np.pi * eta)
    phi_eta = 2 * eta + np.sin(np.pi * xi) * np.sin(np.pi * eta)
    phi_xi_xi = 2 + np.pi * np.sin(np.pi * xi) * np.cos(np.pi * eta)
    phi_eta_eta = phi_xi_xi
    phi_xi_eta = np.pi * np.cos(np.pi * xi) * np.sin(np.pi * eta)

    phi_x2 = phi_xi * sin_t + phi_eta * cos_t
    phi_x1 = phi_xi * cos_t - phi_eta * sin_t
    phi_x2x2 = phi_xi_xi * sin_t ** 2 + 2 * phi_xi_eta * sin_t * cos_t + phi_eta_eta * cos_t ** 2
    phi_x1x2 = (phi_xi_xi - phi_eta_eta) * sin_t * cos_t + phi_xi_eta * (cos_t ** 2 - sin_t ** 2)

    phi_xi_xi_xi = np.pi ** 2 * np.cos(np.pi * xi) * np.cos(np.pi * eta)
    phi_xi_xi_eta = -np.pi ** 2 * np.sin(np.pi * xi) * np.sin(np.pi * eta)
    phi_xi_eta_eta = np.pi ** 2 * np.cos(np.pi * xi) * np.cos(np.pi * eta)
    phi_eta_eta_eta = -np.pi ** 2 * np.sin(np.pi * xi) * np.sin(np.pi * eta)

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
    du_n1dx = np.exp(x1) * np.sin(x2 + t)
    du_n2dx = np.exp(x1) * np.cos(x2 + t)
    return du_n1dx, du_n2dx


def get_du_ndy(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    du_n1dy = np.exp(x1) * np.cos(x2 + t)
    du_n2dy = -np.exp(x1) * np.sin(x2 + t)
    return du_n1dy, du_n2dy


def get_du_ndt(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    du_n1dt = np.exp(x1) * np.cos(x2 + t)
    du_n2dt = -np.exp(x1) * np.sin(x2 + t)
    return du_n1dt, du_n2dt


def get_d2u_ndx2(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    d2u_n1dx2 = np.exp(x1) * np.sin(x2 + t)
    d2u_n2dx2 = np.exp(x1) * np.cos(x2 + t)
    return d2u_n1dx2, d2u_n2dx2


def get_d2u_ndy2(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    d2u_n1dy2 = -np.exp(x1) * np.sin(x2 + t)
    d2u_n2dy2 = -np.exp(x1) * np.cos(x2 + t)
    return d2u_n1dy2, d2u_n2dy2


def get_dp_pdx(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    dp_pdx = np.exp(t) * np.cos(x1) * np.sin(x2)
    return dp_pdx


def get_dp_pdy(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    dp_pdy = np.exp(t) * np.sin(x1) * np.cos(x2)
    return dp_pdy


def get_dp_ndx(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    dp_ndx = np.cos(t) * np.cos(x1 + x2)
    return dp_ndx


def get_dp_ndy(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    dp_ndy = np.cos(t) * np.cos(x1 + x2)
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


def get_normal_vector(data):
    _, _, _, phi_x1, phi_x2, _, _, _, _ = _phi_derivatives(data)
    norm = np.sqrt(phi_x1**2 + phi_x2**2)
    n1 = phi_x1 / norm
    n2 = phi_x2 / norm
    return np.hstack((n1[:, None], n2[:, None]))


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


def get_u(data):
    u_p_1, u_p_2 = get_u_p(data)
    u_n_1, u_n_2 = get_u_n(data)
    lf = level_set_function(data)[:, 0]
    u_1 = np.where(lf > 0, u_p_1, u_n_1)
    u_2 = np.where(lf > 0, u_p_2, u_n_2)
    return u_1[:, None], u_2[:, None]


def get_dphi_dx(data):
    _, _, _, phi_x1, _, _, _, _, _ = _phi_derivatives(data)
    return phi_x1[:, None]

def get_dphi_dy(data):
    _, _, _, _, phi_x2, _, _, _, _ = _phi_derivatives(data)
    return phi_x2[:, None]

def get_dphi_dt(data):
    _, _, _, _, _, phi_t, _, _, _ = _phi_derivatives(data)
    return phi_t[:, None]

def get_d2phi_dx2(data):
    _, _, _, _, _, _, phi_x1x1, _, _ = _phi_derivatives(data)
    return phi_x1x1[:, None]

def get_d2phi_dy2(data):
    _, _, _, _, _, _, _, phi_x2x2, _ = _phi_derivatives(data)
    return phi_x2x2[:, None]


def generate_train_data(num_o, num_ini, num_b, num_if, nu_p, nu_n, device):
    xop, xon = omega_points(num_o)
    xb = boundary_points(num_b)
    xif = interface_points(num_if)
    xini = ini_points(num_ini)
    nor = get_normal_vector(xif)

    phix_xop = get_dphi_dx(xop)
    phiy_xop = get_dphi_dy(xop)
    phit_xop = get_dphi_dt(xop)
    phix_xon = get_dphi_dx(xon)
    phiy_xon = get_dphi_dy(xon)
    phit_xon = get_dphi_dt(xon)
    phix_xif = get_dphi_dx(xif)
    phiy_xif = get_dphi_dy(xif)
    phix2_xop = get_d2phi_dx2(xop)
    phiy2_xop = get_d2phi_dy2(xop)
    phix2_xon = get_d2phi_dx2(xon)
    phiy2_xon = get_d2phi_dy2(xon)

    f_p_1, f_p_2 = get_f_p(xop, nu_p)
    f_n_1, f_n_2 = get_f_n(xon, nu_n)
    v_p_1, v_p_2 = get_fluid_velocity(xop)
    v_n_1, v_n_2 = get_fluid_velocity(xon)
    g_1, g_2 = get_g(xb)
    psi_1, psi_2 = get_psi(xif, nor, nu_p, nu_n)
    u0_1, u0_2 = get_u(xini)

    xop_add_lf = add_dimension_lf(xop)
    xon_add_lf = add_dimension_lf(xon)
    xb_add_lf = add_dimension_lf(xb)
    xini_add_lf = add_dimension_lf(xini)
    xif_add_lf = add_dimension_lf(xif)

    xop_add_sign = add_dimension_sign_p(xop)
    xon_add_sign = add_dimension_sign_n(xon)
    xif_add_sign_p = add_dimension_sign_p(xif)
    xif_add_sign_n = add_dimension_sign_n(xif)

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

    data_op_tr = (
    xop_add_lf, xop_add_sign, f_p_1, f_p_2, v_p_1, v_p_2, phix_xop, phiy_xop, phit_xop, phix2_xop, phiy2_xop)
    data_on_tr = (
    xon_add_lf, xon_add_sign, f_n_1, f_n_2, v_n_1, v_n_2, phix_xon, phiy_xon, phit_xon, phix2_xon, phiy2_xon)
    data_b_tr = (xb_add_lf, g_1, g_2)
    data_ini_tr = (xini_add_lf, u0_1, u0_2)
    data_if_tr = (xif_add_lf, xif_add_sign_p, xif_add_sign_n, nor, psi_1, psi_2, phix_xif, phiy_xif)
    return data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr


def generate_test_data(num_test, device):
    xop, xon = omega_points(num_test)
    x = np.vstack((xop, xon))
    u1, u2 = get_u(x)
    x_add = add_dimension_lf(x)
    x_add = To_tensor(x_add, device)
    u1 = To_tensor(u1, device)
    u2 = To_tensor(u2, device)
    data_test = (x_add, u1, u2)
    return data_test
