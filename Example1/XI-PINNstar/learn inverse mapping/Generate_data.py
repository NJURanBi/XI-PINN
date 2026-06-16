# -*- coding: utf-8 -*-
# @Time    : 2026/5/22 下午12:15
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

def initial_interface_points(num, t_ini):
    c_x = 0.3 * np.cos(np.pi*t_ini)
    c_y = np.sin(np.pi*t_ini)
    r = np.pi / 6
    theta = np.random.uniform(low=0., high=2 * np.pi, size=num)[:, None]
    x1 = r * np.cos(theta) + c_x
    x2 = r * np.sin(theta) + c_y
    data = np.hstack((x1, x2))
    return data


def velocity(t):
    u = -0.3 * np.pi * np.sin(np.pi * t)
    v = 0.3 * np.pi * np.cos(np.pi * t)
    return np.array([u, v])


def rk4_integrate(points, t_target, t_ini, dt=0.01):
    x = points.copy()
    t = t_ini
    while t < t_target - 1e-12:
        dt_step = min(dt, t_target - t)
        k1 = velocity(t)
        k2 = velocity(t + dt_step / 2)
        k3 = velocity(t + dt_step / 2)
        k4 = velocity(t + dt_step)
        x = x + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt_step
    return x


def generate_train_data(num, T, t_ini, data_tr, data_tr_empty, device):
    initial_points = initial_interface_points(num, t_ini)
    T_points = np.hstack((rk4_integrate(initial_points, T, t_ini), T*np.ones((num, 1))))
    T_points = To_tensor(T_points, device)
    initial_points = To_tensor(initial_points, device)
    if data_tr_empty:
        data_tr = (T_points, initial_points)
    else:
        T_points = torch.vstack((T_points, data_tr[0]))
        initial_points = torch.vstack((initial_points, data_tr[1]))
        data_tr = (T_points, initial_points)
    return data_tr
