# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 下午2:06
# @Author  : NJU_RanBi
import torch
import functools
import numpy as np
from torch import optim, autograd
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from soap import SOAP
from Generate_data import generate_test_data
import copy

def error_compute(model_p, model_n, func_params_p, func_params_n, data_p, data_n):
    def u_NN(data, model, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)

    u_numerical_p = u_NN(data_p[0], model_p, func_params_p)
    u_numerical_n = u_NN(data_n[0], model_n, func_params_n)
    relative_l2 = torch.sqrt(((u_numerical_p - data_p[1]).pow(2).sum() + (u_numerical_n - data_n[1]).pow(2).sum()) / (data_p[1].pow(2).sum() + data_n[1].pow(2).sum()))
    return relative_l2

def optimize_parameters_Adam(data_op, data_on, data_b, data_inip, data_inin, data_if, beta_p, beta_n, model_p, model_n, epochs, lr, lossval, lossval_dbg, relative_error, device):
    def loss_omega_p(data):
        u = model_p(data[0])
        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True,
                              retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudz = gradu[0][:, 2][:, None]
        dudt = gradu[0][:, 3][:, None]
        d2udx2 = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True,
                               retain_graph=True)[0][:, 0][:, None]
        d2udy2 = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True,
                               retain_graph=True)[0][:, 1][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True,
                               retain_graph=True)[0][:, 2][:, None]
        res = dudt - beta_p * (d2udx2 + d2udy2 + d2udz2)
        loss = nn.MSELoss()(res, data[1])
        return loss

    def loss_omega_n(data):
        u = model_n(data[0])
        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True,
                              retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudz = gradu[0][:, 2][:, None]
        dudt = gradu[0][:, 3][:, None]
        d2udx2 = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True,
                               retain_graph=True)[0][:, 0][:, None]
        d2udy2 = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True,
                               retain_graph=True)[0][:, 1][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True,
                               retain_graph=True)[0][:, 2][:, None]
        res = dudt - beta_n * (d2udx2 + d2udy2 + d2udz2)
        loss = nn.MSELoss()(res, data[1])
        return loss

    def loss_ini_p(data):
        u = model_p(data[0])
        loss = nn.MSELoss()(u, data[1])
        return loss

    def loss_ini_n(data):
        u = model_n(data[0])
        loss = nn.MSELoss()(u, data[1])
        return loss

    def loss_boundary(data):
        u = model_p(data[0])
        loss = nn.MSELoss()(u, data[1])
        return loss

    def loss_interface_N(data):
        up = model_p(data[0])
        un = model_n(data[0])
        n = data[1]
        gradup = autograd.grad(outputs=up, inputs=data[0], grad_outputs=torch.ones_like(up), create_graph=True,
                               retain_graph=True)
        gradun = autograd.grad(outputs=un, inputs=data[0], grad_outputs=torch.ones_like(un), create_graph=True,
                               retain_graph=True)
        dupdx = gradup[0][:, 0][:, None]
        dupdy = gradup[0][:, 1][:, None]
        dupdz = gradup[0][:, 2][:, None]
        dundx = gradun[0][:, 0][:, None]
        dundy = gradun[0][:, 1][:, None]
        dundz = gradun[0][:, 2][:, None]
        res = ((beta_p * dupdx - beta_n * dundx) * n[:, 0][:, None]
               + (beta_p * dupdy - beta_n * dundy) * n[:, 1][:, None]
               + (beta_p * dupdz - beta_n * dundz) * n[:, 2][:, None]) - data[3]
        loss = torch.mean(res ** 2)
        return loss

    def loss_interface_D(data):
        up = model_p(data[0])
        un = model_n(data[0])
        res = up - un - data[2]
        loss = torch.mean(res ** 2)
        return loss

    params = list(model_p.parameters()) + list(model_n.parameters())
    optimizer = optim.Adam(params, lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)
    best_loss = float('inf')
    num_test = 10000
    data_test_p, data_test_n = generate_test_data(num_test, beta_p, beta_n, device)

    for epoch in range(epochs + 1):
        loss1 = loss_omega_p(data_op)
        loss2 = loss_omega_n(data_on)
        loss3 = loss_boundary(data_b)
        loss4 = loss_ini_p(data_inip) + loss_ini_n(data_inin)
        loss5 = loss_interface_N(data_if)
        loss6 = loss_interface_D(data_if)
        loss = loss1 + loss2 + 100*loss3 + 100*loss4 + 10*loss5 + 100*loss6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            print('epoch:', epoch, 'loss:', loss.item(), 'loss1:', loss1.item(), 'loss2:', loss2.item(), 'loss3:', loss3.item(), 'loss4:', loss4.item(), 'loss5:', loss5.item(), 'loss6:', loss6.item())
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Learning Rate: {current_lr:.2e}")
            params_p = dict(model_p.named_parameters())
            params_n = dict(model_n.named_parameters())
            error = error_compute(model_p, model_n, params_p, params_n, data_test_p, data_test_n)
            relative_error.append(error.item())
            print(f" relative error: {relative_error[-1]:.4e}")
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_epoch = epoch
                best_state_dict_p = copy.deepcopy(model_p.state_dict())
                best_state_dict_n = copy.deepcopy(model_n.state_dict())
            lossval.append(loss.item())
            lossval_dbg.append([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()])

    print('best epoch:', best_epoch, 'best loss:', best_loss)
    model_p.load_state_dict(best_state_dict_p)
    model_n.load_state_dict(best_state_dict_n)
    torch.save(model_p.state_dict(), 'best_model_p_XIPINN_Adam.mdl')
    torch.save(model_p.state_dict(), 'best_model_n_XIPINN_Adam.mdl')
    return model_p, model_n, lossval, lossval_dbg, relative_error


def optimize_parameters_SOAP(data_op, data_on, data_b, data_inip, data_inin, data_if, beta_p, beta_n, model_p, model_n, epochs, lr, lossval, lossval_dbg, relative_error, device):
    def loss_omega_p(data):
        u = model_p(data[0])
        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudz = gradu[0][:, 2][:, None]
        dudt = gradu[0][:, 3][:, None]
        d2udx2 = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2udy2 = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 1][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 2][:, None]
        res = dudt - beta_p * (d2udx2 + d2udy2 + d2udz2)
        loss = nn.MSELoss()(res, data[1])
        return loss

    def loss_omega_n(data):
        u = model_n(data[0])
        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudz = gradu[0][:, 2][:, None]
        dudt = gradu[0][:, 3][:, None]
        d2udx2 = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2udy2 = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 1][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 2][:, None]
        res = dudt - beta_n * (d2udx2 + d2udy2 + d2udz2)
        loss = nn.MSELoss()(res, data[1])
        return loss

    def loss_ini_p(data):
        u = model_p(data[0])
        loss = nn.MSELoss()(u, data[1])
        return loss

    def loss_ini_n(data):
        u = model_n(data[0])
        loss = nn.MSELoss()(u, data[1])
        return loss

    def loss_boundary(data):
        u = model_p(data[0])
        loss = nn.MSELoss()(u, data[1])
        return loss

    def loss_interface_N(data):
        up = model_p(data[0])
        un = model_n(data[0])
        n = data[1]
        gradup = autograd.grad(outputs=up, inputs=data[0], grad_outputs=torch.ones_like(up), create_graph=True, retain_graph=True)
        gradun = autograd.grad(outputs=un, inputs=data[0], grad_outputs=torch.ones_like(un), create_graph=True, retain_graph=True)
        dupdx = gradup[0][:, 0][:, None]
        dupdy = gradup[0][:, 1][:, None]
        dupdz = gradup[0][:, 2][:, None]
        dundx = gradun[0][:, 0][:, None]
        dundy = gradun[0][:, 1][:, None]
        dundz = gradun[0][:, 2][:, None]
        res = ((beta_p * dupdx - beta_n * dundx) * n[:, 0][:, None]
               + (beta_p * dupdy - beta_n * dundy) * n[:, 1][:, None]
               + (beta_p * dupdz - beta_n * dundz) * n[:, 2][:, None]) - data[3]
        loss = torch.mean(res**2)
        return loss

    def loss_interface_D(data):
        up = model_p(data[0])
        un = model_n(data[0])
        res = up - un - data[2]
        loss = torch.mean(res ** 2)
        return loss

    params = list(model_p.parameters()) + list(model_n.parameters())
    optimizer = SOAP(params, lr=lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)
    best_loss = float('inf')
    num_test = 10000
    data_test_p, data_test_n = generate_test_data(num_test, beta_p, beta_n, device)

    for epoch in range(epochs + 1):
        loss1 = loss_omega_p(data_op)
        loss2 = loss_omega_n(data_on)
        loss3 = loss_boundary(data_b)
        loss4 = loss_ini_p(data_inip) + loss_ini_n(data_inin)
        loss5 = loss_interface_N(data_if)
        loss6 = loss_interface_D(data_if)
        loss = loss1 + loss2 + 100*loss3 + 100*loss4 + 10*loss5 + 100*loss6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            print('epoch:', epoch, 'loss:', loss.item(), 'loss1:', loss1.item(), 'loss2:', loss2.item(), 'loss3:',
                  loss3.item(), 'loss4:', loss4.item(), 'loss5:', loss5.item(), 'loss6:', loss6.item())
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Learning Rate: {current_lr:.2e}")
            params_p = dict(model_p.named_parameters())
            params_n = dict(model_n.named_parameters())
            error = error_compute(model_p, model_n, params_p, params_n, data_test_p, data_test_n)
            print(f" relative error: {relative_error[-1]:.4e}")
            relative_error.append(error.item())
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_epoch = epoch
                best_state_dict_p = copy.deepcopy(model_p.state_dict())
                best_state_dict_n = copy.deepcopy(model_n.state_dict())
            lossval.append(loss.item())
            lossval_dbg.append([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()])

    print('best epoch:', best_epoch, 'best loss:', best_loss)
    model_p.load_state_dict(best_state_dict_p)
    model_n.load_state_dict(best_state_dict_n)
    torch.save(model_p.state_dict(), 'best_model_p_XIPINN_SOAP.mdl')
    torch.save(model_p.state_dict(), 'best_model_n_XIPINN_SOAP.mdl')
    return model_p, model_n, lossval, lossval_dbg, relative_error

def optimize_parameters_LM(data_op, data_on, data_b, data_inip, data_inin, data_if, beta_p, beta_n, model_p, model_n, tr_iter_max, mu, lossval, lossval_dbg, relative_error, mu_div, mu_mul, device):
    def get_all_params_vec(model_p, model_n):
        params_p = torch.cat([p.contiguous().view(-1) for p in model_p.parameters()])
        params_n = torch.cat([p.contiguous().view(-1) for p in model_n.parameters()])
        return torch.cat([params_p, params_n]), params_p.numel(), params_n.numel()

    all_params, num_p, num_n = get_all_params_vec(model_p, model_n)

    unflatten_p = []
    offset = 0
    for name, p in model_p.named_parameters():
        n = p.numel()
        unflatten_p.append((name, p.shape, n, offset))
        offset += n

    unflatten_n = []
    offset = 0
    for name, p in model_n.named_parameters():
        n = p.numel()
        unflatten_n.append((name, p.shape, n, offset))
        offset += n

    def vec_to_params(vec):
        vec_p = vec[:num_p]
        pd = {}
        for name, shape, n, st in unflatten_p:
            end = st + n
            pd[name] = vec_p[st:end].reshape(shape)
        vec_n = vec[num_p:]
        nd = {}
        for name, shape, n, st in unflatten_n:
            end = st + n
            nd[name] = vec_n[st:end].reshape(shape)
        return pd, nd

    def u_NN(data, func_params, model):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)

    def res_omega_p(vec, point, f):
        params_p, _ = vec_to_params(vec)
        gradu = torch.func.jacrev(u_NN, argnums=0)(point, params_p, model_p)
        dudt = gradu[3]
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(point, params_p, model_p)
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        d2udz2 = grad2u[2][2]
        f_val = f[0]
        res = dudt - beta_p * (d2udx2 + d2udy2 + d2udz2) - f_val
        return res

    def res_omega_n(vec, point, f):
        _, params_n = vec_to_params(vec)
        gradu = torch.func.jacrev(u_NN, argnums=0)(point, params_n, model_n)
        dudt = gradu[3]
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(point, params_n, model_n)
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        d2udz2 = grad2u[2][2]
        f_val = f[0]
        res = dudt - beta_n * (d2udx2 + d2udy2 + d2udz2) - f_val
        return res

    def res_boundary(vec, point, g_val):
        params_p, _ = vec_to_params(vec)
        u = u_NN(point, params_p, model_p)
        return u - g_val[0]

    def res_ini_p(vec, point, u0):
        params_p, _ = vec_to_params(vec)
        u = u_NN(point, params_p, model_p)
        return u - u0[0]

    def res_ini_n(vec, point, u0):
        _, params_n = vec_to_params(vec)
        u = u_NN(point, params_n, model_n)
        return u - u0[0]

    def res_interface_D(vec, point, psi_D):
        params_p, params_n = vec_to_params(vec)
        u_p = u_NN(point, params_p, model_p)
        u_n = u_NN(point, params_n, model_n)
        res = u_p - u_n - psi_D[0]
        return res

    def res_interface_N(vec, point, nor, psi_N):
        params_p, params_n = vec_to_params(vec)
        grad_p = torch.func.jacrev(u_NN, argnums=0)(point, params_p, model_p)
        dupdx, dupdy, dupdz = grad_p[0], grad_p[1], grad_p[2]
        grad_n = torch.func.jacrev(u_NN, argnums=0)(point, params_n, model_n)
        dundx, dundy, dundz = grad_n[0], grad_n[1], grad_n[2]
        n1, n2, n3 = nor[0], nor[1], nor[2]
        flux_p = beta_p * (dupdx * n1 + dupdy * n2 + dupdz * n3)
        flux_n = beta_n * (dundx * n1 + dundy * n2 + dundz * n3)
        res = flux_p - flux_n - psi_N[0]
        return res

    # tolerence for LM
    tol_main = 10 ** (-13)
    tol_machine = 10 ** (-15)
    mu_max = 10 ** 8
    # iteration check
    ls_check = 10
    ls_check0 = ls_check - 1
    # 残差数量统计
    Nop = len(data_op[0])
    Non = len(data_on[0])
    Nb = len(data_b[0])
    Ninip = len(data_inip[0])
    Ninin = len(data_inin[0])
    Nif = len(data_if[0])
    total_res = Nop + Non + Nb + Ninip + Ninin + 2 * Nif
    NL = [total_res, Nop, Non, Nb, Ninip, Ninin, Nif, Nif]  # 索引1用于omega_p, 2omega_n, 3b, 4inip, 5inin, 6if_D, 7if_N
    NL_sqrt = [np.sqrt(n) if n > 0 else 1.0 for n in NL]
    # 初始化参数向量和辅助变量
    all_params, _, _ = get_all_params_vec(model_p, model_n)
    p_vec_old = all_params.detach().clone()
    total_params = len(p_vec_old)
    dp_old = torch.zeros(total_params, 1).double().to(device)
    L_old = torch.zeros(total_res, 1).double().to(device)
    J_old = torch.zeros(total_res, total_params).double().to(device)
    I_pvec = torch.eye(total_params).to(device)
    criterion = True
    # iteration counts and check
    Comput_old = True
    step = 0

    num_test = 10000
    data_test_p, data_test_n = generate_test_data(num_test, beta_p, beta_n, device)
    try:
        while (lossval[-1] > tol_main) and (step <= tr_iter_max):
            torch.cuda.empty_cache()
            if Comput_old: # need to compute loss_old and J_olds
                ### computation of loss
                Lop = torch.vmap(res_omega_p, (None, 0, 0))(all_params, data_op[0], data_op[1]).flatten().detach()
                Lon = torch.vmap(res_omega_n, (None, 0, 0))(all_params, data_on[0], data_on[1]).flatten().detach()
                Lb = torch.vmap(res_boundary, (None, 0, 0))(all_params, data_b[0], data_b[1]).flatten().detach()
                Linip = torch.vmap(res_ini_p, (None, 0, 0))(all_params, data_inip[0], data_inip[1]).flatten().detach()
                Linin = torch.vmap(res_ini_n, (None, 0, 0))(all_params, data_inin[0], data_inin[1]).flatten().detach()
                Lif_D = torch.vmap(res_interface_D, (None, 0, 0))(all_params, data_if[0], data_if[2]).flatten().detach()
                Lif_N = torch.vmap(res_interface_N, (None, 0, 0, 0))(all_params, data_if[0], data_if[1], data_if[3]).flatten().detach()

                L = torch.cat([
                    Lop / NL_sqrt[1],
                    Lon / NL_sqrt[2],
                    Lb / NL_sqrt[3],
                    Linip / NL_sqrt[4],
                    Linin / NL_sqrt[5],
                    Lif_D / NL_sqrt[6],
                    Lif_N / NL_sqrt[7]
                ]).reshape(-1, 1).detach()

                loss_dbg_old = [
                    (Lop ** 2).mean().item(),
                    (Lon ** 2).mean().item(),
                    (Lb ** 2).mean().item(),
                    (Linip ** 2).mean().item(),
                    (Linin ** 2).mean().item(),
                    (Lif_D ** 2).mean().item(),
                    (Lif_N ** 2).mean().item()
                ]
            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]
            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = all_params.detach().clone()  # get p_vec for p_vec_old if neccessary

            if criterion:
                Jop = torch.vmap(torch.func.jacrev(res_omega_p, argnums=0), (None, 0, 0))(all_params, data_op[0], data_op[1])
                Jop = Jop / NL_sqrt[1]
                Jon = torch.vmap(torch.func.jacrev(res_omega_n, argnums=0), (None, 0, 0))(all_params, data_on[0], data_on[1])
                Jon = Jon / NL_sqrt[2]
                Jb = torch.vmap(torch.func.jacrev(res_boundary, argnums=0), (None, 0, 0))(all_params, data_b[0], data_b[1])
                Jb = Jb / NL_sqrt[3]
                Jinip = torch.vmap(torch.func.jacrev(res_ini_p, argnums=0), (None, 0, 0))(all_params, data_inip[0], data_inip[1])
                Jinip = Jinip / NL_sqrt[4]
                Jinin = torch.vmap(torch.func.jacrev(res_ini_n, argnums=0), (None, 0, 0))(all_params, data_inin[0], data_inin[1])
                Jinin = Jinin / NL_sqrt[5]
                Jif_D = torch.vmap(torch.func.jacrev(res_interface_D, argnums=0), (None, 0, 0))(all_params, data_if[0], data_if[2])
                Jif_D = Jif_D / NL_sqrt[6]
                Jif_N = torch.vmap(torch.func.jacrev(res_interface_N, argnums=0), (None, 0, 0, 0))(all_params, data_if[0], data_if[1], data_if[3])
                Jif_N = Jif_N / NL_sqrt[7]

                J = torch.cat([Jop, Jon, Jb, Jinip, Jinin, Jif_D, Jif_N], dim=0).detach()
                J_product = J.t() @ J
                rhs = - J.t() @ L

            with torch.no_grad():
                ### solve the linear system
                dp = torch.linalg.solve(J_product + mu * I_pvec, rhs)
                all_params += dp.view(-1)

            ### Compute loss_new
            Lop = torch.vmap(res_omega_p, (None, 0, 0))(all_params, data_op[0], data_op[1]).flatten().detach()
            Lon = torch.vmap(res_omega_n, (None, 0, 0))(all_params, data_on[0], data_on[1]).flatten().detach()
            Lb = torch.vmap(res_boundary, (None, 0, 0))(all_params, data_b[0], data_b[1]).flatten().detach()
            Linip = torch.vmap(res_ini_p, (None, 0, 0))(all_params, data_inip[0], data_inip[1]).flatten().detach()
            Linin = torch.vmap(res_ini_n, (None, 0, 0))(all_params, data_inin[0], data_inin[1]).flatten().detach()
            Lif_D = torch.vmap(res_interface_D, (None, 0, 0))(all_params, data_if[0], data_if[2]).flatten().detach()
            Lif_N = torch.vmap(res_interface_N, (None, 0, 0, 0))(all_params, data_if[0], data_if[1], data_if[3]).flatten().detach()

            L = torch.cat([
                Lop / NL_sqrt[1],
                Lon / NL_sqrt[2],
                Lb / NL_sqrt[3],
                Linip / NL_sqrt[4],
                Linin / NL_sqrt[5],
                Lif_D / NL_sqrt[6],
                Lif_N / NL_sqrt[7]
            ]).reshape(-1, 1).detach()
            loss_new = torch.sum(L * L).item()
            loss_dbg_new = [
                (Lop ** 2).mean().item(),
                (Lon ** 2).mean().item(),
                (Lb ** 2).mean().item(),
                (Linip ** 2).mean().item(),
                (Linin ** 2).mean().item(),
                (Lif_D ** 2).mean().item(),
                (Lif_N ** 2).mean().item()
            ]

            # strategy to update mu
            if step > 0:
                with torch.no_grad():
                    # accept update
                    if loss_new < loss_old:
                        p_vec_old = p_vec.detach()
                        dp_old = dp
                        L_old = L
                        J_old = J
                        mu = max(mu / mu_div, tol_machine)
                        criterion = True
                        Comput_old = False
                        lossval.append(loss_new)
                        lossval_dbg.append(loss_dbg_new)

                    else:
                        cosine = torch.cosine_similarity(dp, dp_old, dim=0, eps=1e-15)
                        cosine_check = (1. - cosine) * loss_new > min(lossval)
                        if cosine_check:   # give up the direction
                            all_params -= dp.view(-1)  # 回退参数
                            mu = min(mu_mul * mu, mu_max)
                            criterion = False
                            Comput_old = False
                        else:  # accept
                            p_vec_old = p_vec.detach()
                            dp_old = dp
                            L_old = L
                            J_old = J
                            mu = max(mu / mu_div, tol_machine)
                            criterion = True
                            Comput_old = False
                        lossval.append(loss_old)
                        lossval_dbg.append(loss_dbg_old)
            else:  # for old info.
                with torch.no_grad():
                    p_vec_old = p_vec.detach()
                    dp_old = dp
                    L_old = L
                    J_old = J
                    mu = max(mu / mu_div, tol_machine)
                    criterion = True
                    Comput_old = False
                    lossval.append(loss_new)
                    lossval_dbg.append(loss_dbg_new)

            if step % ls_check == ls_check0:
                print("Step %s: " % (step))
                print(f" training loss: {lossval[-1]:.4e}")
                print(f" relative error: {relative_error[-1]:.4e}")
            step += 1

            params_p, params_n = vec_to_params(all_params)
            error = error_compute(model_p, model_n, params_p, params_n, data_test_p, data_test_n)
            relative_error.append(error.item())

        print(f"Step {step - 1}: training loss = {lossval[-1]:.4e}")
        print("finished")

        with torch.no_grad():
            params_p_dict, params_n_dict = vec_to_params(all_params)
            model_p.load_state_dict(params_p_dict)
            model_n.load_state_dict(params_n_dict)
        torch.save(model_p.state_dict(), 'best_model_p_LM.mdl')
        torch.save(model_n.state_dict(), 'best_model_n_LM.mdl')
        return model_p, model_n, lossval, lossval_dbg, relative_error

    except KeyboardInterrupt:
        print("Interrupted")
        print("steps =", step)
        with torch.no_grad():
            params_p_dict, params_n_dict = vec_to_params(all_params)
            model_p.load_state_dict(params_p_dict)
            model_n.load_state_dict(params_n_dict)
        torch.save(model_p.state_dict(), 'best_model_p_LM.mdl')
        torch.save(model_n.state_dict(), 'best_model_n_LM.mdl')
        return model_p, model_n, lossval, lossval_dbg, relative_error