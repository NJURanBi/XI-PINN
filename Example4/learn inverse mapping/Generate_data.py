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
    if t_ini == 0:
        theta = np.random.uniform(0, 2 * np.pi, num)[:, None]
        x1 = 0.15 * np.cos(theta) + 0.5
        x2 = 0.15 * np.sin(theta) + 0.75
        data = np.hstack((x1, x2))
    else:
        theta = np.random.uniform(0, 2 * np.pi, num)[:, None]
        x1 = 0.15 * np.cos(theta) + 0.5
        x2 = 0.15 * np.sin(theta) + 0.75
        initial_points = np.hstack((x1, x2))
        data = rk4_integrate(initial_points, t_ini, 0)
    return data

def initial_boundary_points(num):
    index = np.random.uniform(low=0, high=1, size=num)[:, None]
    xb1 = np.hstack((index, np.ones_like(index)))
    index = np.random.uniform(low=0, high=1, size=num)[:, None]
    xb2 = np.hstack((index, np.zeros_like(index)))
    index = np.random.uniform(low=0, high=1, size=num)[:, None]
    xb3 = np.hstack((np.ones_like(index), index))
    index = np.random.uniform(low=0, high=1, size=num)[:, None]
    xb4 = np.hstack((np.zeros_like(index), index))
    xb = np.vstack((xb1, xb2, xb3, xb4))
    return xb


def velocity(points, t):
    x = points[:, 0]
    y = points[:, 1]
    factor = np.cos(np.pi * t / 3.0)
    u = factor * np.sin(np.pi * x)**2 * np.sin(2 * np.pi * y)
    v = -factor * np.sin(np.pi * y)**2 * np.sin(2 * np.pi * x)
    return np.column_stack([u, v])


def rk4_integrate(points, t_target, t_ini, dt=1e-3):
    x = points.copy()
    t = t_ini
    while t < t_target - 1e-12:
        dt_step = min(dt, t_target - t)
        k1 = velocity(x, t)
        k2 = velocity(x + 0.5 * dt_step * k1, t + 0.5 * dt_step)
        k3 = velocity(x + 0.5 * dt_step * k2, t + 0.5 * dt_step)
        k4 = velocity(x + dt_step * k3, t + dt_step)
        x = x + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt_step
    return x


def generate_train_data(num_if, num_b, T, t_ini, data_tr, data_tr_empty, device):
    interface_points = initial_interface_points(num_if, t_ini)
    boundary_points = initial_boundary_points(num_b)
    T_points_if = np.hstack((rk4_integrate(interface_points, T, t_ini), T*np.ones((num_if, 1))))
    T_points_b = np.hstack((boundary_points, T*np.ones((4*num_b, 1))))

    T_points_if = To_tensor(T_points_if, device)
    T_points_b = To_tensor(T_points_b, device)
    interface_points = To_tensor(interface_points, device)
    boundary_points = To_tensor(boundary_points, device)
    if data_tr_empty:
        T_points = torch.vstack((T_points_if, T_points_b))
        initial_points = torch.vstack((interface_points, boundary_points))
        data_tr = (T_points, initial_points)
    else:
        T_points = torch.vstack((T_points_if, T_points_b, data_tr[0]))
        initial_points = torch.vstack((interface_points, boundary_points, data_tr[1]))
        data_tr = (T_points, initial_points)
    return data_tr

