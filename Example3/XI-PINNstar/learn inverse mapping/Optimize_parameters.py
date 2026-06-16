# -*- coding: utf-8 -*-
# @Time    : 2026/5/22 下午12:54
# @Author  : NJU_RanBi
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from soap import SOAP
import copy
from torch import optim
import numpy as np
import functools


def optimize_parameters_Adam(data_tr, t_ini, model, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)

    best_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    model.train()

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        output = model(data_tr[0])
        t = data_tr[0][:, 2][:, None]
        dt = t - t_ini
        pred = data_tr[0][:, :2] + dt * output
        loss = nn.MSELoss(reduction='mean')(pred, data_tr[1])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            loss_val = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Loss: {loss_val:.2e}, LR: {current_lr:.2e}')

            if loss_val < best_loss:
                best_loss = loss_val
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

    print(f'Best Epoch: {best_epoch}, Best Loss: {best_loss:.6f}')
    model.load_state_dict(best_state_dict)
    return model


def optimize_parameters_SOAP(data_tr, t_ini, model, epochs, lr):
    optimizer = SOAP(model.parameters(), lr=lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
    best_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    model.train()

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        output = model(data_tr[0])
        t = data_tr[0][:, 2][:, None]
        dt = t - t_ini
        pred = data_tr[0][:, :2] + dt * output
        loss = nn.MSELoss(reduction='mean')(pred, data_tr[1])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            loss_val = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Loss: {loss_val:.2e}, LR: {current_lr:.2e}')

            if loss_val < best_loss:
                best_loss = loss_val
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

    print(f'Best Epoch: {best_epoch}, Best Loss: {best_loss:.6f}')
    model.load_state_dict(best_state_dict)
    return model



def optimize_parameters_LM(data_tr, t_ini, model, tr_iter_max, mu, mu_div, mu_mul, device):
    def get_p_vec(func_params):
        p_vec = []
        cnt = 0
        for p in func_params:
            p_vec = func_params[p].contiguous().view(-1) if cnt == 0 else torch.cat([p_vec, func_params[p].contiguous().view(-1)])
            cnt = 1
        return p_vec

    def count_parameters(func_params):
        return sum(x.numel() for x in func_params.values())

    def generate_initial_LM(func_params, data_len):
        # data_length
        data_length = 2 * data_len

        with torch.no_grad():
            p_vec_old = get_p_vec(func_params).double().to(device)

        # dp
        dp_old = torch.zeros([count_parameters(func_params), 1]).double().to(device)

        # Loss
        L_old = torch.zeros([data_length, 1]).double().to(device)

        # Jacobian
        J_old = torch.zeros([data_length, count_parameters(func_params)]).double().to(device)

        return p_vec_old, dp_old, L_old, J_old

    def ux_NN(data, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)[0]

    def uy_NN(data, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)[1]

    def res_map_x(func_params, data, f):
        ux = ux_NN(data, func_params)
        dt = data[2] - t_ini
        res = dt*ux + data[0] - f[0]
        return res

    def res_map_y(func_params, data, f):
        uy = uy_NN(data, func_params)
        dt = data[2] - t_ini
        res = dt*uy + data[1] - f[1]
        return res


    torch.cuda.empty_cache()
    lossval = []
    lossval.append(1.)
    # tolerence for LM
    tol_main = 10 ** (-14)
    tol_machine = 10 ** (-15)
    mu_max = 10 ** 8
    # iteration check
    ls_check = 10
    ls_check0 = ls_check - 1
    # Loss parameters
    NL = [2 * len(data_tr[0]), len(data_tr[0]), len(data_tr[0])]
    NL_sqrt = np.sqrt(NL)
    func_params = dict(model.named_parameters())
    p_vec_o, dp_o, L_o, J_o = generate_initial_LM(func_params, NL[1])
    I_pvec = torch.eye(len(p_vec_o)).to(device)
    criterion = True
    # iteration counts and check
    Comput_old = True
    step = 0
    try:
        while (lossval[-1] > tol_main) and (step <= tr_iter_max):
            torch.cuda.empty_cache()
            if (Comput_old == True):  # need to compute loss_old and J_old
                ### computation of loss 计算各部分损失函数
                Lifux = torch.vmap((res_map_x), (None, 0, 0))(func_params, data_tr[0], data_tr[1]).flatten().detach()
                Lifuy = torch.vmap((res_map_y), (None, 0, 0))(func_params, data_tr[0], data_tr[1]).flatten().detach()
                L = torch.cat((Lifux / NL_sqrt[1], Lifuy / NL_sqrt[2]))
                L = L.reshape(NL[0], 1).detach()

            loss_old = lossval[-1]
            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = get_p_vec(func_params).detach()  # get p_vec for p_vec_old if neccessary

            if criterion:
                per_sample_grads = torch.vmap(torch.func.jacrev(res_map_x), (None, 0, 0))(func_params, data_tr[0], data_tr[1])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_ifux = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_ifux, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_map_y), (None, 0, 0))(func_params, data_tr[0], data_tr[1])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_ifuy = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_ifuy, g.reshape(len(g), -1)])
                    cnt = 1

                J = torch.cat((J_ifux / NL_sqrt[1], J_ifuy / NL_sqrt[2])).detach()
                ### info. normal equation of J
                J_product = J.t() @ J
                rhs = - J.t() @ L

            with torch.no_grad():
                ### solve the linear system
                dp = torch.linalg.solve(J_product + mu * I_pvec, rhs)
                cnt = 0
                for p in func_params:
                    mm = torch.Tensor([func_params[p].shape]).tolist()[0]
                    num = int(functools.reduce(lambda x, y: x * y, mm, 1))
                    func_params[p] += dp[cnt:cnt + num].reshape(func_params[p].shape)
                    cnt += num

            ### Compute loss_new
            Lifux = torch.vmap((res_map_x), (None, 0, 0))(func_params, data_tr[0], data_tr[1]).flatten().detach()
            Lifuy = torch.vmap((res_map_y), (None, 0, 0))(func_params, data_tr[0], data_tr[1]).flatten().detach()
            L = torch.cat((Lifux / NL_sqrt[1], Lifuy / NL_sqrt[2]))
            L = L.reshape(NL[0], 1).detach()
            loss_new = torch.sum(L * L).item()


            # strategy to update mu
            if (step > 0):

                with torch.no_grad():

                    # accept update
                    if loss_new < loss_old:
                        p_vec_old = p_vec.detach()
                        dp_old = dp
                        L_old = L
                        J_old = J
                        mu = max(mu / mu_div, tol_machine)
                        criterion = True  # False
                        Comput_old = False
                        lossval.append(loss_new)
                    else:
                        cosine = torch.cosine_similarity(dp, dp_old, dim=0, eps=1e-15)
                        cosine_check = (1. - cosine) * loss_new > min(lossval)  # loss_old
                        if cosine_check:  # give up the direction
                            cnt = 0
                            for p in func_params:
                                mm = torch.Tensor([func_params[p].shape]).tolist()[0]
                                num = int(functools.reduce(lambda x, y: x * y, mm, 1))
                                func_params[p] -= dp[cnt:cnt + num].reshape(func_params[p].shape)
                                cnt += num
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
            if step % ls_check == ls_check0:
                print("Step %s: " % (step))
                print(f" training loss: {lossval[-1]:.4e}")
            step += 1

        print("Step %s: " % (step - 1))
        print(f" training loss: {lossval[-1]:.4e}")
        print('finished')
        return model

    except KeyboardInterrupt:
        print('Interrupt')
        print('steps = ', step)
        return model