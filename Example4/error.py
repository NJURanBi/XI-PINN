# -*- coding: utf-8 -*-
# @Time    : 2025/2/27 下午2:46
# @Author  : NJU_RanBi
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io as sio
from functools import wraps
from torch import optim, autograd
from Network import Vanilla_Net

torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def original_lf_function(data):
    x = data[:, 0]
    y = data[:, 1]
    lf = (x - 0.5) ** 2 + (y - 0.75) ** 2 - 0.15**2
    return lf[:, None]


def level_set_function(data, t_i):
    x = data[:, :2]
    if 0 <= t_i < 0.2:
        output = inverse_mapping_1(data)
        xi = x + output  # 参考坐标
        lf = original_lf_function(xi)

    if 0.2 <= t_i < 0.4:
        output = inverse_mapping_2(data)
        output = x + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.2 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_1(data)
        xi = x_new + output  # 得到新的参考坐标
        lf = original_lf_function(xi)

    if 0.4 <= t_i < 0.6:
        output = inverse_mapping_3(data)
        output = x + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.4 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_2(data)
        output = x_new + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.2 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_1(data)
        xi = x_new + output  # 得到新的参考坐标
        lf = original_lf_function(xi)

    if 0.6 <= t_i < 0.8:
        output = inverse_mapping_4(data)
        output = x + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.6 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_3(data)
        output = x_new + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.4 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_2(data)
        output = x_new + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.2 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_1(data)
        xi = x_new + output  # 得到新的参考坐标
        lf = original_lf_function(xi)

    if 0.8 <= t_i <= 1:
        output = inverse_mapping_5(data)
        output = x + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.8 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_4(data)
        output = x_new + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.6 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_3(data)
        output = x_new + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.4 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_2(data)
        output = x_new + output  # 上一时间层的参考坐标
        x_new = output  # 作为新时间层的物理坐标
        data = torch.cat((x_new, 0.2 * torch.ones((len(output), 1))), dim=1)

        output = inverse_mapping_1(data)
        xi = x_new + output  # 得到新的参考坐标
        lf = original_lf_function(xi)
    return lf

def get_omega_points(num, t_i):
    x = np.linspace(0, 1, num)
    y = np.linspace(0, 1, num)
    X, Y = np.meshgrid(x, y)
    Z = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    T = t_i * np.ones((len(Z), 1))
    data = np.hstack((Z, T))
    data = torch.tensor(data, requires_grad=True)
    lf = level_set_function(data, t_i)[:, 0]
    index_p = lf >= 0
    index_n = lf < 0
    index_p = index_p.float()
    index_n = index_n.float()
    index = index_p - index_n
    data = torch.cat((data, index[:, None]), dim=1)
    return data

def get_u(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    up_1 = torch.exp(x) * torch.sin(np.pi * y + np.pi * t)
    up_2 = 1 / np.pi * torch.exp(x) * torch.cos(np.pi * y + np.pi * t)
    un_1 = torch.cos(t) * torch.cos(np.pi * x) * torch.sin(np.pi * y)
    un_2 = -torch.cos(t) * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    pp = torch.sin(0.5 * np.pi * x) * torch.cos(0.5 * np.pi * y)
    pn = torch.cos(0.5 * np.pi * x) * torch.sin(0.5 * np.pi * y)
    index = data[:, 3]
    u_1 = torch.where(index > 0, up_1, un_1)
    u_2 = torch.where(index > 0, up_2, un_2)
    p = torch.where(index > 0, pp, pn)
    u = torch.cat((u_1[:, None], u_2[:, None], p[:, None]), dim=1)
    return u

X_mapping = Vanilla_Net(3, 64, 2, 3).double()
inverse_mapping_1 = Vanilla_Net(3, 64, 2, 3).double()
inverse_mapping_2 = Vanilla_Net(3, 64, 2, 3).double()
inverse_mapping_3 = Vanilla_Net(3, 64, 2, 3).double()
inverse_mapping_4 = Vanilla_Net(3, 64, 2, 3).double()
inverse_mapping_5 = Vanilla_Net(3, 64, 2, 3).double()

X_mapping.load_state_dict(torch.load('best_model_1e-15.mdl', map_location=device))
inverse_mapping_1.load_state_dict(torch.load('best_model_0_0.2.mdl', map_location=device))
inverse_mapping_2.load_state_dict(torch.load('best_model_0.2_0.4.mdl', map_location=device))
inverse_mapping_3.load_state_dict(torch.load('best_model_0.4_0.6.mdl', map_location=device))
inverse_mapping_4.load_state_dict(torch.load('best_model_0.6_0.8.mdl', map_location=device))
inverse_mapping_5.load_state_dict(torch.load('best_model_0.8_1.mdl', map_location=device))

data = get_omega_points(1001, 1)
model = Vanilla_Net(4, 64, 3, 3).double()
model.load_state_dict(torch.load('best_model.mdl', map_location=device))
real_u = get_u(data).detach().numpy()
pred_u = model(data).detach().numpy()
# 速度场
pred = np.sqrt(pred_u[:, 0]**2 + pred_u[:, 1]**2)
real = np.sqrt(real_u[:, 0]**2 + real_u[:, 1]**2)
# 压力场
#pred = pred_u[:, 2]
#real = real_u[:, 2]
delta = np.mean(pred - real)
pred = pred - delta
plt.figure(1)
pred = pred.reshape(1001, 1001)
real = real.reshape(1001, 1001)
h = plt.imshow(np.abs(pred - real), interpolation='nearest', cmap='coolwarm', extent=[0, 1, 0, 1], origin='lower', aspect='auto')
plt.title('Error distribution:t=1', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(h)
plt.savefig('Ex4_error_u_t=1.jpg', bbox_inches='tight', dpi=300)
plt.show()

plt.figure(2)
h = plt.imshow(pred, interpolation='nearest', cmap='coolwarm', extent=[0, 1, 0, 1], origin='lower', aspect='auto')
plt.title('Approximate solution:t=1', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(h)
plt.savefig('Ex4_pred_u_t=1.jpg', bbox_inches='tight', dpi=300)
plt.show()


tao = 0.1
N_t = 10
data = get_omega_points(101, 0)
for i in range(1, N_t+1):
    data_i = get_omega_points(101, i*tao)
    data = torch.vstack((data, data_i))

r_u = get_u(data)
r_u1 = r_u[:, 0][:, None]
r_u2 = r_u[:, 1][:, None]
p_u = model(data)
p_u1 = p_u[:, 0][:, None]
p_u2 = p_u[:, 1][:, None]

gradr_u1 = autograd.grad(outputs=r_u1, inputs=data, grad_outputs=torch.ones_like(r_u1), create_graph=True, retain_graph=True)
dr_u1dx = gradr_u1[0][:, 0][:, None]
dr_u1dy = gradr_u1[0][:, 1][:, None]
gradr_u2 = autograd.grad(outputs=r_u2, inputs=data, grad_outputs=torch.ones_like(r_u2), create_graph=True, retain_graph=True)
dr_u2dx = gradr_u2[0][:, 0][:, None]
dr_u2dy = gradr_u2[0][:, 1][:, None]

gradp_u1 = autograd.grad(outputs=p_u1, inputs=data, grad_outputs=torch.ones_like(p_u1), create_graph=True, retain_graph=True)
dp_u1dx = gradp_u1[0][:, 0][:, None]
dp_u1dy = gradp_u1[0][:, 1][:, None]
gradp_u2 = autograd.grad(outputs=p_u2, inputs=data, grad_outputs=torch.ones_like(p_u2), create_graph=True, retain_graph=True)
dp_u2dx = gradp_u2[0][:, 0][:, None]
dp_u2dy = gradp_u2[0][:, 1][:, None]

l2_error = torch.sqrt(torch.mean((r_u1 - p_u1)**2 + (r_u2 - p_u2)**2))
h1_error = torch.sqrt(torch.mean((r_u1 - p_u1)**2 + (r_u2 - p_u2)**2) + torch.mean((dr_u1dx - dp_u1dx)**2 + (dr_u2dx - dp_u2dx)**2) + torch.mean((dr_u1dy - dp_u1dy)**2 + (dr_u2dy - dp_u2dy)**2))
print('l2_error: ', l2_error.item())
print('h1_error: ', h1_error.item())