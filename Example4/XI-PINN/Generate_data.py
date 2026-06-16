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


def level_set_function(data, models):
    """
    data: Tensor, shape (N, 3), 列为 (x, y, t)
    返回: 水平集函数值 Tensor, shape (N, 1)
    """
    t_nodes = [0.0, 0.32, 0.52, 0.73]

    t = data[:, 2]
    N = data.shape[0]
    device = data.device

    # 初始化最终坐标的占位符
    X_final = torch.empty(N, 1, device=device)
    Y_final = torch.empty(N, 1, device=device)

    # 按时间区间分组，依次处理
    # 区间1:
    mask = t < t_nodes[1]
    if mask.any():
        sub_data = data[mask]
        cur = sub_data
        # 从当前时间直接回溯到 t=0 (model1)
        out = models[0](cur)
        dt = cur[:, 2:3] - t_nodes[0]  # t - 0
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        X_final[mask] = x
        Y_final[mask] = y

    # 区间2:
    mask = (t >= t_nodes[1]) & (t < t_nodes[2])
    if mask.any():
        sub_data = data[mask]
        # 第一步: (model2)
        out = models[1](sub_data)
        dt = sub_data[:, 2:3] - t_nodes[1]
        x = sub_data[:, 0:1] + dt * out[:, 0:1]
        y = sub_data[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[1] * torch.ones_like(x)], dim=1)
        # 第二步: (model1)
        out = models[0](cur)
        dt = t_nodes[1] - t_nodes[0]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        X_final[mask] = x
        Y_final[mask] = y

    # 区间3:
    mask = (t >= t_nodes[2]) & (t < t_nodes[3])
    if mask.any():
        sub_data = data[mask]
        # (model3)
        out = models[2](sub_data)
        dt = sub_data[:, 2:3] - t_nodes[2]
        x = sub_data[:, 0:1] + dt * out[:, 0:1]
        y = sub_data[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[2] * torch.ones_like(x)], dim=1)
        # (model2)
        out = models[1](cur)
        dt = t_nodes[2] - t_nodes[1]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[1] * torch.ones_like(x)], dim=1)
        # (model1)
        out = models[0](cur)
        dt = t_nodes[1] - t_nodes[0]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        X_final[mask] = x
        Y_final[mask] = y

    # 区间4:
    mask = t >= t_nodes[3]
    if mask.any():
        sub_data = data[mask]
        # (model4)
        out = models[3](sub_data)
        dt = sub_data[:, 2:3] - t_nodes[3]
        x = sub_data[:, 0:1] + dt * out[:, 0:1]
        y = sub_data[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[3] * torch.ones_like(x)], dim=1)
        # (model3)
        out = models[2](cur)
        dt = t_nodes[3] - t_nodes[2]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[2] * torch.ones_like(x)], dim=1)
        # (model2)
        out = models[1](cur)
        dt = t_nodes[2] - t_nodes[1]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[1] * torch.ones_like(x)], dim=1)
        # (model1)
        out = models[0](cur)
        dt = t_nodes[1] - t_nodes[0]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        X_final[mask] = x
        Y_final[mask] = y

    # 最终计算初始圆形界面的水平集值
    lf = (X_final - 0.5) ** 2 + (Y_final - 0.75) ** 2 - 0.15 ** 2
    return lf


def compute_phi_derivative(data, model):
    x = data.clone().detach().requires_grad_(True)
    phi = level_set_function(x, model)
    grad = torch.autograd.grad(phi.sum(), x, retain_graph=True, create_graph=True)[0]
    dphi_dx = grad[:, 0][:, None]
    dphi_dy = grad[:, 1][:, None]
    return dphi_dx, dphi_dy


def get_normal_vector(data, model):
    gx, gy = compute_phi_derivative(data, model)
    norm = torch.sqrt(gx ** 2 + gy ** 2)
    n1 = gx / norm
    n2 = gy / norm
    return torch.cat([n1, n2], dim=1)


def omega_points(num, models, device):
    x1 = np.random.uniform(low=0, high=1, size=num)[:, None]
    x2 = np.random.uniform(low=0, high=1, size=num)[:, None]
    t = np.random.uniform(low=0, high=1, size=num)[:, None]
    data = np.hstack((x1, x2, t))
    data = To_tensor(data, device)
    phi = level_set_function(data, models)
    index_p = torch.where(phi >= 0)[0]
    index_n = torch.where(phi < 0)[0]
    datap = data[index_p, :]
    datan = data[index_n, :]
    return datap, datan


def ini_points(num, device):
    x1 = np.random.uniform(low=0, high=1, size=num)[:, None]
    x2 = np.random.uniform(low=0, high=1, size=num)[:, None]
    t = np.zeros_like(x1)
    data = np.hstack((x1, x2, t))
    data = To_tensor(data, device)
    return data


def boundary_points(num, device):
    index = np.random.uniform(low=0, high=1, size=num)[:, None]
    xb1 = np.hstack((index, np.ones_like(index)))
    index = np.random.uniform(low=0, high=1, size=num)[:, None]
    xb2 = np.hstack((index, -np.ones_like(index)))
    index = np.random.uniform(low=0, high=1, size=num)[:, None]
    xb3 = np.hstack((np.ones_like(index), index))
    index = np.random.uniform(low=0, high=1, size=num)[:, None]
    xb4 = np.hstack((-np.ones_like(index), index))
    xb = np.vstack((xb1, xb2, xb3, xb4))
    t = np.random.uniform(low=0, high=1, size=4 * num)[:, None]
    data = np.hstack((xb, t))
    data = To_tensor(data, device)
    return data


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

def interface_points(num, device, N_s=200, N_t=51):
    c_x = 0.5
    c_y = 0.75
    r = 0.15
    t = np.linspace(0, 1, N_t)[1:]
    data = []

    if num > N_s*N_t:
        ValueError('error!')
    else:
        theta = np.random.uniform(low=0, high=2 * np.pi, size=N_s)[:, None]
        x1 = c_x + np.cos(theta) * r
        x2 = c_y + np.sin(theta) * r
        ti_arr = np.zeros_like(x1)
        pts = np.hstack((x1, x2, ti_arr))
        data.append(pts)
        for ti in t:
            theta = np.random.uniform(low=0, high=2 * np.pi, size=N_s)[:, None]
            x1_0 = c_x + np.cos(theta) * r
            x2_0 = c_y + np.sin(theta) * r
            pts_ini = np.hstack((x1_0, x2_0))
            pts = rk4_integrate(pts_ini, ti, 0)
            ti_arr = np.full_like(x1_0, ti)
            pts = np.hstack((pts, ti_arr))
            data.append(pts)
    data = np.vstack(data)
    indices = np.random.choice(data.shape[0], size=num, replace=False)
    data = data[indices]
    data = To_tensor(data, device)
    return data



def add_dimension(data, models):
    lf = level_set_function(data, models)
    add_x = torch.ones_like(lf)
    index_n = torch.where(lf < 0)[0]
    add_x[index_n] = -1
    data = torch.hstack((data, add_x))
    return data


def add_dimension_p(data, device):
    return torch.hstack((data, torch.ones((len(data[:, 0]), 1), device=device)))


def add_dimension_n(data, device):
    return torch.hstack((data, -torch.ones((len(data[:, 0]), 1), device=device)))


def get_u_p(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    u_p_1 = torch.exp(x) * torch.sin(np.pi * y + np.pi * t)
    u_p_2 = 1 / np.pi * torch.exp(x) * torch.cos(np.pi * y + np.pi * t)
    return u_p_1, u_p_2


def get_u_n(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    u_n_1 = torch.cos(t) * torch.cos(np.pi * x) * torch.sin(np.pi * y)
    u_n_2 = -torch.cos(t) * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    return u_n_1, u_n_2


def get_p_p(data):
    x = data[:, 0]
    y = data[:, 1]
    p = torch.sin(0.5 * np.pi * x) * torch.cos(0.5 * np.pi * y)
    return p


def get_p_n(data):
    x = data[:, 0]
    y = data[:, 1]
    p = torch.cos(0.5 * np.pi * x) * torch.sin(0.5 * np.pi * y)
    return p


def get_fluid_velocity(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    w_1 = torch.cos(np.pi * t / 3) * torch.sin(np.pi * x)**2 * torch.sin(2 * np.pi * y)
    w_2 = -torch.cos(np.pi * t / 3) * torch.sin(np.pi * y)**2 * torch.sin(2 * np.pi * x)
    return w_1[:, None], w_2[:, None]

def get_du_pdx(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_p_1dx = torch.exp(x) * torch.sin(np.pi * y + np.pi * t)
    du_p_2dx = 1 / np.pi * torch.exp(x) * torch.cos(np.pi * y + np.pi * t)
    return du_p_1dx, du_p_2dx

def get_du_pdy(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_p_1dy = np.pi * torch.exp(x) * torch.cos(np.pi * y + np.pi * t)
    du_p_2dy = -torch.exp(x) * torch.sin(np.pi * y + np.pi * t)
    return du_p_1dy, du_p_2dy

def get_du_pdt(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_p_1dt = np.pi * torch.exp(x) * torch.cos(np.pi * y + np.pi * t)
    du_p_2dt = -torch.exp(x) * torch.sin(np.pi * y + np.pi * t)
    return du_p_1dt, du_p_2dt

def get_d2u_pdx2(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    d2u_p_1dx2 = torch.exp(x) * torch.sin(np.pi * y + np.pi * t)
    d2u_p_2dx2 = 1 / np.pi * torch.exp(x) * torch.cos(np.pi * y + np.pi * t)
    return d2u_p_1dx2, d2u_p_2dx2

def get_d2u_pdy2(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    d2u_p_1dy2 = -np.pi**2 * torch.exp(x) * torch.sin(np.pi * y + np.pi * t)
    d2u_p_2dy2 = -np.pi * torch.exp(x) * torch.cos(np.pi * y + np.pi * t)
    return d2u_p_1dy2, d2u_p_2dy2

def get_du_ndx(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_n_1dx = -np.pi * torch.cos(t) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    du_n_2dx = -np.pi * torch.cos(t) * torch.cos(np.pi * x) * torch.cos(np.pi * y)
    return du_n_1dx, du_n_2dx

def get_du_ndy(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_n_1dy = np.pi * torch.cos(t) * torch.cos(np.pi * x) * torch.cos(np.pi * y)
    du_n_2dy = np.pi * torch.cos(t) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    return du_n_1dy, du_n_2dy

def get_du_ndt(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    du_n_1dt = -torch.sin(t) * torch.cos(np.pi * x) * torch.sin(np.pi * y)
    du_n_2dt = torch.sin(t) * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    return du_n_1dt, du_n_2dt

def get_d2u_ndx2(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    d2u_n_1dx2 = -np.pi**2 * torch.cos(t) * torch.cos(np.pi * x) * torch.sin(np.pi * y)
    d2u_n_2dx2 = np.pi**2 * torch.cos(t) * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    return d2u_n_1dx2, d2u_n_2dx2

def get_d2u_ndy2(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    d2u_n_1dy2 = -np.pi**2 * torch.cos(t) * torch.cos(np.pi * x) * torch.sin(np.pi * y)
    d2u_n_2dy2 = np.pi**2 * torch.cos(t) * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    return d2u_n_1dy2, d2u_n_2dy2

def get_dp_pdx(data):
    x = data[:, 0]
    y = data[:, 1]
    dp_pdx = 0.5 * np.pi * torch.cos(0.5 * np.pi * x) * torch.cos(0.5 * np.pi * y)
    return dp_pdx

def get_dp_pdy(data):
    x = data[:, 0]
    y = data[:, 1]
    dp_pdy = -0.5 * np.pi * torch.sin(0.5 * np.pi * x) * torch.sin(0.5 * np.pi * y)
    return dp_pdy

def get_dp_ndx(data):
    x = data[:, 0]
    y = data[:, 1]
    dp_ndx = -0.5 * np.pi * torch.sin(0.5 * np.pi * x) * torch.sin(0.5 * np.pi * y)
    return dp_ndx

def get_dp_ndy(data):
    x = data[:, 0]
    y = data[:, 1]
    dp_ndy = 0.5 * np.pi * torch.cos(0.5 * np.pi * x) * torch.cos(0.5 * np.pi * y)
    return dp_ndy


def get_f_p(data, nu_p):
    du_p_1dx, du_p_2dx = get_du_pdx(data)
    du_p_1dy, du_p_2dy = get_du_pdy(data)
    du_p_1dt, du_p_2dt = get_du_pdt(data)
    d2u_p_1dx2, d2u_p_2dx2 = get_d2u_pdx2(data)
    d2u_p_1dy2, d2u_p_2dy2 = get_d2u_pdy2(data)
    dp_pdx = get_dp_pdx(data)
    dp_pdy = get_dp_pdy(data)
    w1, w2 = get_fluid_velocity(data)
    w1 = w1[:, 0]
    w2 = w2[:, 0]
    f_p_1 = du_p_1dt + w1 * du_p_1dx + w2 * du_p_1dy - nu_p * (d2u_p_1dx2 + d2u_p_1dy2) + dp_pdx
    f_p_2 = du_p_2dt + w1 * du_p_2dx + w2 * du_p_2dy - nu_p * (d2u_p_2dx2 + d2u_p_2dy2) + dp_pdy
    return f_p_1[:, None], f_p_2[:, None]


def get_f_n(data, nu_n):
    du_n_1dx, du_n_2dx = get_du_ndx(data)
    du_n_1dy, du_n_2dy = get_du_ndy(data)
    du_n_1dt, du_n_2dt = get_du_ndt(data)
    d2u_n_1dx2, d2u_n_2dx2 = get_d2u_ndx2(data)
    d2u_n_1dy2, d2u_n_2dy2 = get_d2u_ndy2(data)
    dp_ndx = get_dp_ndx(data)
    dp_ndy = get_dp_ndy(data)
    w1, w2 = get_fluid_velocity(data)
    w1 = w1[:, 0]
    w2 = w2[:, 0]
    f_n_1 = du_n_1dt + w1 * du_n_1dx + w2 * du_n_1dy - nu_n * (d2u_n_1dx2 + d2u_n_1dy2) + dp_ndx
    f_n_2 = du_n_2dt + w1 * du_n_2dx + w2 * du_n_2dy - nu_n * (d2u_n_2dx2 + d2u_n_2dy2) + dp_ndy
    return f_n_1[:, None], f_n_2[:, None]


def get_g(data):
    g_1, g_2 = get_u_p(data)
    return g_1[:, None], g_2[:, None]

def get_u(data, models):
    up1, up2 = get_u_p(data)
    un1, un2 = get_u_n(data)
    lf = level_set_function(data, models)[:, 0]
    u1 = torch.where(lf > 0, up1, un1)
    u2 = torch.where(lf > 0, up2, un2)
    return u1[:, None], u2[:, None]


def get_phi(data):
    u_p_1, u_p_2 = get_u_p(data)
    u_n_1, u_n_2 = get_u_n(data)
    phi_1 = u_p_1 - u_n_1
    phi_2 = u_p_2 - u_n_2
    return phi_1[:, None], phi_2[:, None]


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


def generate_train_data(num_o, num_ini, num_b, num_if, nu_p, nu_n, device):
    inverse_mapping1 = Vanilla_Net(3, 32, 2, 3).to(device)
    inverse_mapping2 = Vanilla_Net(3, 32, 2, 3).to(device)
    inverse_mapping3 = Vanilla_Net(3, 32, 2, 3).to(device)
    inverse_mapping4 = Vanilla_Net(3, 32, 2, 3).to(device)
    inverse_mapping1.load_state_dict(torch.load('inverse_mapping_0.32.mdl'))
    inverse_mapping2.load_state_dict(torch.load('inverse_mapping_0.52.mdl'))
    inverse_mapping3.load_state_dict(torch.load('inverse_mapping_0.73.mdl'))
    inverse_mapping4.load_state_dict(torch.load('inverse_mapping_1.mdl'))
    inverse_mapping1.eval()
    inverse_mapping2.eval()
    inverse_mapping3.eval()
    inverse_mapping4.eval()
    models = [inverse_mapping1, inverse_mapping2, inverse_mapping3, inverse_mapping4]

    xop, xon = omega_points(num_o, models, device)
    xini = ini_points(num_ini, device)
    xb = boundary_points(num_b, device)
    xif = interface_points(num_if, device)
    nor = get_normal_vector(xif, models)

    xop_add = add_dimension_p(xop, device)
    xon_add = add_dimension_n(xon, device)
    xini_add = add_dimension(xini, models)
    xb_add = add_dimension(xb, models)
    xif_p_add = add_dimension_p(xif, device)
    xif_n_add = add_dimension_n(xif, device)

    f_p_1, f_p_2 = get_f_p(xop, nu_p)
    f_n_1, f_n_2 = get_f_n(xon, nu_n)
    v_p_1, v_p_2 = get_fluid_velocity(xop)
    v_n_1, v_n_2 = get_fluid_velocity(xon)
    g_1, g_2 = get_g(xb)
    phi_1, phi_2 = get_phi(xif)
    psi_1, psi_2 = get_psi(xif, nor, nu_p, nu_n)
    u0_1, u0_2 = get_u(xini, models)

    xop_add = To_tensor_grad(xop_add, device)
    xon_add = To_tensor_grad(xon_add, device)
    xb_add = To_tensor_grad(xb_add, device)
    xif_p_add = To_tensor_grad(xif_p_add, device)
    xif_n_add = To_tensor_grad(xif_n_add, device)
    xini_add = To_tensor_grad(xini_add, device)

    nor = To_tensor(nor, device)
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
    phi_1 = To_tensor(phi_1, device)
    phi_2 = To_tensor(phi_2, device)
    psi_1 = To_tensor(psi_1, device)
    psi_2 = To_tensor(psi_2, device)
    u0_1 = To_tensor(u0_1, device)
    u0_2 = To_tensor(u0_2, device)


    data_op_tr = (xop_add, f_p_1, f_p_2, v_p_1, v_p_2)
    data_on_tr = (xon_add, f_n_1, f_n_2, v_n_1, v_n_2)
    data_b_tr = (xb_add, g_1, g_2)
    data_ini_tr = (xini_add, u0_1, u0_2)
    data_if_tr = (xif_p_add, xif_n_add, nor, phi_1, phi_2, psi_1, psi_2)
    return data_op_tr, data_on_tr, data_b_tr, data_ini_tr, data_if_tr



def generate_test_data(num_test, device):
    inverse_mapping1 = Vanilla_Net(3, 32, 2, 3).to(device)
    inverse_mapping2 = Vanilla_Net(3, 32, 2, 3).to(device)
    inverse_mapping3 = Vanilla_Net(3, 32, 2, 3).to(device)
    inverse_mapping4 = Vanilla_Net(3, 32, 2, 3).to(device)
    inverse_mapping1.load_state_dict(torch.load('inverse_mapping_0.32.mdl'))
    inverse_mapping2.load_state_dict(torch.load('inverse_mapping_0.52.mdl'))
    inverse_mapping3.load_state_dict(torch.load('inverse_mapping_0.73.mdl'))
    inverse_mapping4.load_state_dict(torch.load('inverse_mapping_1.mdl'))
    inverse_mapping1.eval()
    inverse_mapping2.eval()
    inverse_mapping3.eval()
    inverse_mapping4.eval()
    models = [inverse_mapping1, inverse_mapping2, inverse_mapping3, inverse_mapping4]
    xop, xon = omega_points(num_test, models, device)
    x = torch.vstack((xop, xon))
    up1, up2 = get_u_p(xop)
    un1, un2 = get_u_n(xon)
    u1 = torch.vstack((up1[:, None], un1[:, None]))
    u2 = torch.vstack((up2[:, None], un2[:, None]))
    x_add = add_dimension(x, models)
    x_add = To_tensor(x_add, device)
    u1 = To_tensor(u1, device)
    u2 = To_tensor(u2, device)
    data_test = (x_add, u1, u2)
    return data_test


