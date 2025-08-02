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


def omega_points(num_p, num_n):
    x1 = np.random.uniform(low=-1, high=1, size=20*num_n)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=20*num_n)[:, None]
    x3 = np.random.uniform(low=-1, high=1, size=20*num_n)[:, None]
    t = np.random.uniform(low=0, high=1, size=20*num_n)[:, None]
    data = np.hstack((x1, x2, x3, t))
    phi = level_set_function(data)
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    datap = data[index_p, :]
    datap = add_dimension(datap)
    datan = data[index_n, :]
    datan = add_dimension(datan)
    datap = datap[:num_p, :]
    datan = datan[:num_n, :]
    return datap, datan


def ini_points(num):
    x1 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x2 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    x3 = np.random.uniform(low=-1, high=1, size=num)[:, None]
    t = np.zeros_like(x1)
    data = np.hstack((x1, x2, x3, t))
    data = add_dimension(data)
    return data


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
    data = add_dimension(data)
    return data

def runge_kutta_4_system(f, data, i, tao):
    y0 = data[i][:, :3]
    t0 = data[i][0][3]
    k1 = f(t0, y0)
    k2 = f(t0 + tao / 2, y0 + tao * k1 / 2)
    k3 = f(t0 + tao / 2, y0 + tao * k2 / 2)
    k4 = f(t0 + tao, y0 + tao * k3)
    y1 = y0 + (tao / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y1


def ode_func(t, y):
    dxdt = -np.pi*y[:, 1] / 2
    dydt = np.pi*y[:, 0] / 2
    dzdt = 0.5*np.ones_like(y[:, 2])
    return np.hstack((dxdt[:, None], dydt[:, None], dzdt[:, None]))


def get_time_step_data(data_0, n_steps, tao):
    data = data_0
    data1 = runge_kutta_4_system(ode_func, data_0, 0, tao)
    data1 = np.hstack((data1, data[0][:, 3][:, None] + tao))
    data = data + (data1,)
    for i in range(1, n_steps):
        datai = runge_kutta_4_system(ode_func, data, i, tao)
        datai = np.hstack((datai, data[i][:, 3][:, None] + tao))
        data = data + (datai,)
    return data


def interface_points(num):
    tao = 0.01
    u = np.random.uniform(low=0, high=1, size=200)
    v = np.random.uniform(low=0, high=1, size=200)
    theta = np.arccos(2 * u - 1)
    phi = 2 * np.pi * v
    x = 0.7 * np.sin(theta) * np.cos(phi)
    y = 0.5 * np.sin(theta) * np.sin(phi)
    z = 0.5 * np.cos(theta) - 0.25
    t = np.zeros((200, 1))
    data = np.hstack((x[:, None], y[:, None], z[:, None], t))
    data = (data,)
    data = get_time_step_data(data, 100, tao)
    data = list_data(data)
    indices = np.random.choice(data.shape[0], size=num, replace=False)
    data = data[indices]
    datap = np.hstack((data, np.ones((len(data[:, 0]), 1))))
    datan = np.hstack((data, -np.ones((len(data[:, 0]), 1))))
    return datap, datan


def list_data(data):
    data_list = data[0]
    for i in range(1, len(data)):
        data_list = np.vstack((data_list, data[i]))
    return data_list

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

def get_u(data):
    u_p = get_u_p(data)[:, 0]
    u_n = get_u_n(data)[:, 0]
    index = data[:, 4]
    u = np.where(index >= 0, u_p, u_n)
    return u[:, None]

def get_u_p(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    t = data[:, 3]
    u_p = 0.1 * np.sin(x) * np.cos(y) * np.exp(z) * np.exp(-t)
    return u_p[:, None]

def get_u_n(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    t = data[:, 3]
    u_n = np.exp(x**2 + y**2 + z**2) * np.cos(t)
    return u_n[:, None]

def get_normal_vector(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    t = data[:, 3]
    n1 = 2 * np.cos(np.pi * t / 2) * (x * np.cos(np.pi * t / 2) + y * np.sin(np.pi * t / 2)) / 0.7**2 - 2 * np.sin(np.pi * t / 2) * (-x * np.sin(np.pi * t / 2) + y * np.cos(np.pi * t / 2)) / 0.5**2
    n2 = 2 * np.sin(np.pi * t / 2) * (x * np.cos(np.pi * t / 2) + y * np.sin(np.pi * t / 2)) / 0.7**2 + 2 * np.cos(np.pi * t / 2) * (-x * np.sin(np.pi * t / 2) + y * np.cos(np.pi * t / 2)) / 0.5**2
    n3 = 2 * (z - 0.5 * t + 0.25) / 0.5**2
    s = np.sqrt(n1**2 + n2**2 + n3**2)
    n1 = n1 / s
    n2 = n2 / s
    n3 = n3 / s
    n = np.hstack((n1[:, None], n2[:, None], n3[:, None]))
    return n

def get_phi(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    t = data[:, 3]
    u_p = 0.1 * np.sin(x) * np.cos(y) * np.exp(z) * np.exp(-t)
    u_n = np.exp(x**2 + y**2 + z**2) * np.cos(t)
    phi = u_p - u_n
    return phi[:, None]

def get_psi(data, n):
    n1 = n[:, 0]
    n2 = n[:, 1]
    n3 = n[:, 2]
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    t = data[:, 3]
    dup_dx = np.cos(x) * np.cos(y) * np.exp(z) * np.exp(-t)
    dun_dx = 2 * np.exp(x**2 + y**2 + z**2) * x * np.cos(t)
    dup_dy = -np.sin(x) * np.sin(y) * np.exp(z) * np.exp(-t)
    dun_dy = 2 * np.exp(x**2 + y**2 + z**2) * y * np.cos(t)
    dup_dz = np.sin(x) * np.cos(y) * np.exp(z) * np.exp(-t)
    dun_dz = 2 * np.exp(x**2 + y**2 + z**2) * z * np.cos(t)
    psi = n1 * (dup_dx - dun_dx) + n2 * (dup_dy - dun_dy) + n3 * (dup_dz - dun_dz)
    return psi[:, None]

def get_f_p(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    t = data[:, 3]
    du_dt = 0.1*(-np.sin(x) * np.cos(y) * np.exp(z) * np.exp(-t))
    laplace_x = -np.sin(x) * np.cos(y) * np.exp(z) * np.exp(-t)
    laplace_y = -np.sin(x) * np.cos(y) * np.exp(z) * np.exp(-t)
    laplace_z = np.sin(x) * np.cos(y) * np.exp(z) * np.exp(-t)
    f_p = du_dt - (laplace_x + laplace_y + laplace_z)
    return f_p[:, None]

def get_f_n(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    t = data[:, 3]
    r = np.sqrt(x**2 + y**2 + z**2)
    du_dt = -np.exp(r**2) * np.sin(t)
    laplace_x = (2 * np.exp(r**2) + 4 * x**2 * np.exp(r**2)) * np.cos(t)
    laplace_y = (2 * np.exp(r**2) + 4 * y**2 * np.exp(r**2)) * np.cos(t)
    laplace_z = (2 * np.exp(r**2) + 4 * z**2 * np.exp(r**2)) * np.cos(t)
    f_n = du_dt - (laplace_x + laplace_y + laplace_z)
    return f_n[:, None]


def generate_train_data(num_op, num_on, num_ini, num_b, num_if, device):
    xop, xon = omega_points(num_op, num_on)
    xifp, xifn = interface_points(num_if)
    xini = ini_points(num_ini)
    xb = boundary_points(num_b)

    f_p = get_f_p(xop)
    f_n = get_f_n(xon)
    u_0 = get_u(xini)
    u_b = get_u(xb)
    nor = get_normal_vector(xifp)
    phi = get_phi(xifp)
    psi = get_psi(xifp, nor)

    xop = To_tensor_grad(xop, device)
    xon = To_tensor_grad(xon, device)
    xifp = To_tensor_grad(xifp, device)
    xifn = To_tensor_grad(xifn, device)
    xini = To_tensor_grad(xini, device)
    xb = To_tensor_grad(xb, device)

    f_p = To_tensor(f_p, device)
    f_n = To_tensor(f_n, device)
    u_0 = To_tensor(u_0, device)
    u_b = To_tensor(u_b, device)
    nor = To_tensor(nor, device)
    phi = To_tensor(phi, device)
    psi = To_tensor(psi, device)

    data_tr_op = (xop, f_p)
    data_tr_on = (xon, f_n)
    data_tr_b = (xb, u_b)
    data_tr_ini = (xini, u_0)
    data_tr_if_d = (xifp, xifn, phi)
    data_tr_if_n = (xifp, xifn, nor, psi)

    return data_tr_op, data_tr_on, data_tr_b, data_tr_if_d, data_tr_if_n, data_tr_ini



