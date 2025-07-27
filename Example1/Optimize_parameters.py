# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 下午2:06
# @Author  : NJU_RanBi
import torch
import functools
import numpy as np
from torch import optim, autograd
import torch.nn as nn
import matplotlib.pyplot as plt


def optimize_parameters_adam(data_op, data_on, data_b, data_ini, data_if,  beta_p, beta_n, model, device, epochs, lr):
    def loss_omega_p(data):
        u = model(data[0])
        phix = data[2]
        phiy = data[3]
        phit = data[4]
        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudt = gradu[0][:, 2][:, None]
        dudz = gradu[0][:, 3][:, None]
        # u对x的一阶导的导数
        gradux = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        d2udx2 = gradux[0][:, 0][:, None]
        d2udxdz = gradux[0][:, 3][:, None]
        # u对y的一阶导的导数
        graduy = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        d2udy2 = graduy[0][:, 1][:, None]
        d2udydz = graduy[0][:, 3][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 3][:, None]
        Ut = dudt + dudz * phit
        Uxx = d2udx2 + 2 * phix * d2udxdz + phix * phix * d2udz2 + 2 * dudz
        Uyy = d2udy2 + 2 * phiy * d2udydz + phiy * phiy * d2udz2 + 2 * dudz
        res = Ut - beta_p * (Uxx + Uyy)
        loss = nn.MSELoss()(res, data[1])
        return loss

    def loss_omega_n(data):
        u = model(data[0])
        phix = -data[2]
        phiy = -data[3]
        phit = -data[4]
        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudt = gradu[0][:, 2][:, None]
        dudz = gradu[0][:, 3][:, None]
        # u对x的一阶导的导数
        gradux = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        d2udx2 = gradux[0][:, 0][:, None]
        d2udxdz = gradux[0][:, 3][:, None]
        # u对y的一阶导的导数
        graduy = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        d2udy2 = graduy[0][:, 1][:, None]
        d2udydz = graduy[0][:, 3][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 3][:, None]
        Ut = dudt + dudz * phit
        Uxx = d2udx2 + 2 * phix * d2udxdz + phix * phix * d2udz2 - 2 * dudz
        Uyy = d2udy2 + 2 * phiy * d2udydz + phiy * phiy * d2udz2 - 2 * dudz
        res = Ut - beta_n * (Uxx + Uyy)
        loss = nn.MSELoss()(res, data[1])
        return loss

    def loss_ini(data):
        u = model(data[0])
        loss = nn.MSELoss()(u, data[1])
        return loss

    def loss_boundary(data):
        u = model(data[0])
        loss = nn.MSELoss()(u, data[1])
        return loss

    def loss_interface(data):
        u = model(data[0])
        n = data[1]
        phix = data[2]
        phiy = data[3]
        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudz = gradu[0][:, 3][:, None]
        res = ((beta_p - beta_n) * dudx + (beta_p + beta_n) * phix * dudz) * n[:, 0][:, None] + ((beta_p - beta_n) * dudy + (beta_p + beta_n) * phiy * dudz) * n[:, 1][:, None]
        loss = torch.mean(res**2)
        return loss


    optimizer = optim.Adam(model.parameters(), lr)
    best_loss = 1000
    for epoch in range(epochs + 1):
        loss1 = loss_omega_p(data_op)
        loss2 = loss_omega_n(data_on)
        loss3 = loss_boundary(data_b)
        loss4 = loss_ini(data_ini)
        loss5 = loss_interface(data_if)
        loss = loss1 + loss2 + 100*loss3 + 100*loss4 + 10*loss5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        func_params = dict(model.named_parameters())
        if (epoch + 1) % 2000 == 0:
            lr = 0.95 * lr
            optimizer = optim.Adam(model.parameters(), lr)
            print('lr:', lr)
        if epoch % 100 == 0:
            print('epoch:', epoch, 'loss:', loss.item(), 'loss1:', loss1.item(), 'loss2:', loss2.item(),
                  'loss3:', loss3.item(), 'loss4:', loss4.item(), 'loss5:', loss5.item(), 'loss6:', loss6.item())
            if epoch > int(4 * epochs / 5):
                if torch.abs(loss) < best_loss:
                    best_loss = torch.abs(loss).item()
                    best_epoch = epoch
                    torch.save(model.state_dict(), 'best_model.mdl')
                    func_params = dict(model.named_parameters())
    print('best epoch:', best_epoch, 'best loss:', best_loss)

    model.load_state_dict(torch.load('best_model.mdl'))
    func_params = dict(model.named_parameters())
    return model, func_params

def optimize_parameters_LM(data_op, data_on, data_b, data_ini, data_if, beta_p, beta_n, model, tr_iter_max, mu, lossval, lossval_dbg, relative_error, mu_div, mu_mul, device):
    def get_p_vec(func_params):
        p_vec = []
        cnt = 0
        for p in func_params:
            p_vec = func_params[p].contiguous().view(-1) if cnt == 0 else torch.cat([p_vec, func_params[p].contiguous().view(-1)])
            cnt = 1
        return p_vec

    def count_parameters(func_params):
        return sum(x.numel() for x in model.parameters())

    def generate_initial_LM(func_params, Xop_len, Xon_len, Xb_len, Xini_len, Xif_len):
        # data_length
        data_length = Xop_len + Xon_len + Xini_len + Xb_len + Xif_len  # 输入数据长度和

        # p_vector p向量自然为model参数
        with torch.no_grad():
            p_vec_old = get_p_vec(func_params).double().to(device)

        # dp 初始所有参量搜索方向设置为0，其size应当和model参数一致
        dp_old = torch.zeros([count_parameters(func_params), 1]).double().to(device)

        # Loss 损失函数值同样设置为0
        L_old = torch.zeros([data_length, 1]).double().to(device)

        # Jacobian J矩阵同样
        J_old = torch.zeros([data_length, count_parameters(func_params)]).double().to(device)

        return p_vec_old, dp_old, L_old, J_old

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

    torch.cuda.empty_cache()
    # tolerence for LM
    tol_main = 10 ** (-13)
    tol_machine = 10 ** (-15)
    mu_max = 10 ** 8
    # iteration check
    ls_check = 10
    ls_check0 = ls_check - 1
    # Loss parameters
    NL = [len(data_op[0]) + len(data_on[0]) + len(data_b[0]) + len(data_ini[0]) + len(data_if[0]), len(data_op[0]), len(data_on[0]), len(data_b[0]), len(data_ini[0]), len(data_if[0])]
    NL_sqrt = np.sqrt(NL)
    func_params = dict(model.named_parameters())
    p_vec_o, dp_o, L_o, J_o = generate_initial_LM(func_params, NL[1], NL[2], NL[3], NL[4], NL[5])
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
                Lop = torch.vmap((res_omega_p), (None, 0, 0, 0, 0, 0))(func_params, data_op[0], data_op[1], data_op[2], data_op[3], data_op[4]).flatten().detach()
                Lon = torch.vmap((res_omega_n), (None, 0, 0, 0, 0, 0))(func_params, data_on[0], data_on[1], data_on[2], data_on[3], data_on[4]).flatten().detach()
                Lb = torch.vmap((res_boundary), (None, 0, 0))(func_params, data_b[0], data_b[1]).flatten().detach()
                Lini = torch.vmap((res_ini), (None, 0, 0))(func_params, data_ini[0], data_ini[1]).flatten().detach()
                Lif = torch.vmap((res_interface), (None, 0, 0, 0, 0))(func_params, data_if[0], data_if[1], data_if[2], data_if[3]).flatten().detach()
                L = torch.cat((Lop / NL_sqrt[1], Lon / NL_sqrt[2], Lb / NL_sqrt[3], Lini / NL_sqrt[4], Lif / NL_sqrt[5]))
                L = L.reshape(NL[0], 1).detach()
                lsop_sum = torch.sum(Lop * Lop) / NL[1]
                lson_sum = torch.sum(Lon * Lon) / NL[2]
                lsb_sum = torch.sum(Lb * Lb) / NL[3]
                lsini_sum = torch.sum(Lini * Lini) / NL[4]
                lsif_sum = torch.sum(Lif * Lif) / NL[5]
                loss_dbg_old = [lsop_sum.item(), lson_sum.item(), lsb_sum.item(), lsini_sum.item(), lsif_sum.item()]

            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]
            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = get_p_vec(func_params).detach()  # get p_vec for p_vec_old if neccessary

            if criterion:
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

                J = torch.cat((J_op / NL_sqrt[1], J_on / NL_sqrt[2], J_b / NL_sqrt[3], J_ini / NL_sqrt[4], J_if / NL_sqrt[5])).detach()
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
            Lop = torch.vmap((res_omega_p), (None, 0, 0, 0, 0, 0))(func_params, data_op[0], data_op[1], data_op[2], data_op[3], data_op[4]).flatten().detach()
            Lon = torch.vmap((res_omega_n), (None, 0, 0, 0, 0, 0))(func_params, data_on[0], data_on[1], data_on[2], data_on[3], data_on[4]).flatten().detach()
            Lb = torch.vmap((res_boundary), (None, 0, 0))(func_params, data_b[0], data_b[1]).flatten().detach()
            Lini = torch.vmap((res_ini), (None, 0, 0))(func_params, data_ini[0], data_ini[1]).flatten().detach()
            Lif = torch.vmap((res_interface), (None, 0, 0, 0, 0))(func_params, data_if[0], data_if[1], data_if[2], data_if[3]).flatten().detach()
            L = torch.cat((Lop / NL_sqrt[1], Lon / NL_sqrt[2], Lb / NL_sqrt[3], Lini / NL_sqrt[4], Lif / NL_sqrt[5]))
            L = L.reshape(NL[0], 1).detach()
            loss_new = torch.sum(L * L).item()
            lsop_sum = torch.sum(Lop * Lop) / NL[1]
            lson_sum = torch.sum(Lon * Lon) / NL[2]
            lsb_sum = torch.sum(Lb * Lb) / NL[3]
            lsini_sum = torch.sum(Lini * Lini) / NL[4]
            lsif_sum = torch.sum(Lif * Lif) / NL[5]
            loss_dbg_new = [lsop_sum.item(), lson_sum.item(), lsb_sum.item(), lsini_sum.item(), lsif_sum.item()]

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
                        lossval_dbg.append(loss_dbg_new)

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

            step += 1
            # 误差计算
            # error = error_compute(model, func_params, device)
            # relative_error.append(error)

        print("Step %s: " % (step - 1))
        print(f" training loss: {lossval[-1]:.4e}")
        print('finished')
        torch.save(model.state_dict(), 'best_model.mdl')
        return model, func_params, lossval, lossval_dbg, relative_error

    except KeyboardInterrupt:
        print('Interrupt')
        print('steps = ', step)
        torch.save(model.state_dict(), 'best_model.mdl')
        return model, func_params, lossval, lossval_dbg, relative_error