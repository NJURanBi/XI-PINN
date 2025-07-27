# -*- coding: utf-8 -*-
# @Time    : 2025/6/13 下午9:08
# @Author  : NJU_RanBi
import torch
import functools
import numpy as np
from torch import optim, autograd
import torch.nn as nn
import matplotlib.pyplot as plt


def compute_NTK_XI_PINN(data_op, data_on, data_b, data_ini, data_if, beta_p, beta_n, model, device):
    func_params = dict(model.named_parameters())
    def u_NN(data, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)

    def res_omega_p(func_params, data, f, phix, phiy, phit):
        phix = phix[0]
        phiy = phiy[0]
        phit = phit[0]
        f = f[0]
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params)
        dudt = gradu[2]
        dudz = gradu[3]
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data, func_params)
        d2udx2 = grad2u[0][0]
        d2udxdz = grad2u[0][3]
        d2udy2 = grad2u[1][1]
        d2udydz = grad2u[1][3]
        d2udz2 = grad2u[3][3]
        Ut = dudt + dudz * phit
        Uxx = d2udx2 + 2 * phix * d2udxdz + phix * phix * d2udz2 + 2 * dudz
        Uyy = d2udy2 + 2 * phiy * d2udydz + phiy * phiy * d2udz2 + 2 * dudz
        res = Ut - beta_p * (Uxx + Uyy) - f
        return res

    def res_omega_n(func_params, data, f, phix, phiy, phit):
        phix = -phix[0]
        phiy = -phiy[0]
        phit = -phit[0]
        f = f[0]
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params)
        dudt = gradu[2]
        dudz = gradu[3]
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data, func_params)
        d2udx2 = grad2u[0][0]
        d2udxdz = grad2u[0][3]
        d2udy2 = grad2u[1][1]
        d2udydz = grad2u[1][3]
        d2udz2 = grad2u[3][3]
        Ut = dudt + dudz * phit
        Uxx = d2udx2 + 2 * phix * d2udxdz + phix * phix * d2udz2 - 2 * dudz
        Uyy = d2udy2 + 2 * phiy * d2udydz + phiy * phiy * d2udz2 - 2 * dudz
        res = Ut - beta_n * (Uxx + Uyy) - f
        return res

    def res_ini(func_params, data, u0):
        u = u_NN(data, func_params)
        res = u - u0[0]
        return res

    def res_boundary(func_params, data, f_b):
        u = u_NN(data, func_params)
        res = u - f_b[0]
        return res

    def res_interface(func_params, data, nor, phix, phiy):
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params)
        n1 = nor[0]
        n2 = nor[1]
        phix = phix[0]
        phiy = phiy[0]
        dudx = gradu[0]
        dudy = gradu[1]
        dudz = gradu[3]
        res = ((beta_p - beta_n) * dudx + (beta_p + beta_n) * phix * dudz) * n1 + ((beta_p - beta_n) * dudy + (beta_p + beta_n) * phiy * dudz) * n2
        return res

    per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_p), (None, 0, 0, 0, 0, 0))(func_params, data_op[0], data_op[1], data_op[2], data_op[3], data_op[4])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_op = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_op, g.reshape(len(g), -1)])
        cnt = 1

    per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_n), (None, 0, 0, 0, 0, 0))(func_params, data_on[0], data_on[1], data_on[2], data_on[3], data_on[4])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_on = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_on, g.reshape(len(g), -1)])
        cnt = 1

    per_sample_grads = torch.vmap(torch.func.jacrev(res_boundary), (None, 0, 0))(func_params, data_b[0], data_b[1])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_b = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_b, g.reshape(len(g), -1)])
        cnt = 1

    per_sample_grads = torch.vmap(torch.func.jacrev(res_ini), (None, 0, 0))(func_params, data_ini[0], data_ini[1])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_ini = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_ini, g.reshape(len(g), -1)])
        cnt = 1

    per_sample_grads = torch.vmap(torch.func.jacrev(res_interface), (None, 0, 0, 0, 0))(func_params, data_if[0], data_if[1], data_if[2], data_if[3])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_if = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if, g.reshape(len(g), -1)])
        cnt = 1

    J = torch.cat((J_op, J_on, J_b, J_ini, J_if))
    J_o = torch.cat((J_op, J_on))
    NTK = J @ J.t()
    NTK_1 = J_o @ J_o.t()
    NTK_2 = J_b @ J_b.t()
    NTK_3 = J_ini @ J_ini.t()
    NTK_4 = J_if @ J_if.t()
    np.savetxt('NTK_XI.txt', NTK.cpu().detach().numpy())
    np.savetxt('NTK_XI_O.txt', NTK_1.cpu().detach().numpy())
    np.savetxt('NTK_XI_B.txt', NTK_2.cpu().detach().numpy())
    np.savetxt('NTK_XI_INI.txt', NTK_3.cpu().detach().numpy())
    np.savetxt('NTK_XI_IF.txt', NTK_4.cpu().detach().numpy())
    return



def compute_NTK_Vanilla_PINN(data_op, data_on, data_b, data_ini, data_if, beta_p, beta_n, model, device):
    func_params = dict(model.named_parameters())
    def u_NN(data, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)

    def res_omega_p(func_params, data, f):
        f = f[0]
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params)
        dudt = gradu[2]
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data, func_params)
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        res = dudt - beta_p * (d2udx2 + d2udy2) - f
        return res

    def res_omega_n(func_params, data, f):
        f = f[0]
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params)
        dudt = gradu[2]
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data, func_params)
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        res = dudt - beta_p * (d2udx2 + d2udy2) - f
        return res

    def res_ini(func_params, data, u0):
        u = u_NN(data, func_params)
        res = u - u0[0]
        return res

    def res_boundary(func_params, data, f_b):
        u = u_NN(data, func_params)
        res = u - f_b[0]
        return res

    def res_interface(func_params, data_n, data_p, nor):
        gradu_n = torch.func.jacrev(u_NN, argnums=0)(data_n, func_params)
        gradu_p = torch.func.jacrev(u_NN, argnums=0)(data_p, func_params)
        n1 = nor[0]
        n2 = nor[1]
        du_ndx = gradu_n[0]
        du_ndy = gradu_n[1]
        du_pdx = gradu_p[0]
        du_pdy = gradu_p[1]
        res = (beta_p * du_pdx - beta_n * du_ndx) * n1 + (beta_p * du_pdy - beta_n * du_ndy) * n2
        return res

    per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_p), (None, 0, 0))(func_params, data_op[0][:, :3], data_op[1])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_op = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_op, g.reshape(len(g), -1)])
        cnt = 1

    per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_n), (None, 0, 0))(func_params, data_on[0][:, :3], data_on[1])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_on = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_on, g.reshape(len(g), -1)])
        cnt = 1

    per_sample_grads = torch.vmap(torch.func.jacrev(res_boundary), (None, 0, 0))(func_params, data_b[0][:, :3], data_b[1])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_b = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_b, g.reshape(len(g), -1)])
        cnt = 1

    per_sample_grads = torch.vmap(torch.func.jacrev(res_ini), (None, 0, 0))(func_params, data_ini[0][:, :3], data_ini[1])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_ini = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_ini, g.reshape(len(g), -1)])
        cnt = 1

    per_sample_grads = torch.vmap(torch.func.jacrev(res_interface), (None, 0, 0, 0))(func_params, data_if[0][:, :3], data_if[1], data_if[2])
    cnt = 0
    for g in per_sample_grads:
        g = per_sample_grads[g].detach()
        J_if = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if, g.reshape(len(g), -1)])
        cnt = 1

    J = torch.cat((J_op, J_on, J_b, J_ini, J_if))
    J_o = torch.cat((J_op, J_on))
    NTK = J @ J.t()
    NTK_1 = J_o @ J_o.t()
    NTK_2 = J_b @ J_b.t()
    NTK_3 = J_ini @ J_ini.t()
    NTK_4 = J_if @ J_if.t()
    np.savetxt('NTK_VAN.txt', NTK.cpu().detach().numpy())
    np.savetxt('NTK_VAN_O.txt', NTK_1.cpu().detach().numpy())
    np.savetxt('NTK_VAN_B.txt', NTK_2.cpu().detach().numpy())
    np.savetxt('NTK_VAN_INI.txt', NTK_3.cpu().detach().numpy())
    np.savetxt('NTK_VAN_IF.txt', NTK_4.cpu().detach().numpy())
    return