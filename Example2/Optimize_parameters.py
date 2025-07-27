# -*- coding: utf-8 -*-
# @Time    : 2025/5/28 下午12:18
# @Author  : NJU_RanBi
import torch
import functools
import numpy as np
from torch import optim, autograd
import torch.nn as nn
import matplotlib.pyplot as plt
from soap import SOAP

def optimize_parameters_soap(data_op, data_on, data_b, data_if_phi, data_if_psi, data_ini, v_p, v_n, model, device, epochs, lr):
    def loss_omega_p(data):
        output = model(data[0])
        u = output
        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudz = gradu[0][:, 2][:, None]
        dudt = gradu[0][:, 3][:, None]
        d2udx2 = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(dudx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2udy2 = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(dudy), create_graph=True, retain_graph=True)[0][:, 1][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(dudz), create_graph=True, retain_graph=True)[0][:, 2][:, None]
        f = data[1]
        res = dudt - v_p * (d2udx2 + d2udy2 + d2udz2)
        loss = nn.MSELoss()(res, f)
        return loss

    def loss_omega_n(data):
        output = model(data[0])
        u = output
        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudz = gradu[0][:, 2][:, None]
        dudt = gradu[0][:, 3][:, None]
        d2udx2 = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(dudx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2udy2 = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(dudy), create_graph=True, retain_graph=True)[0][:, 1][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(dudz), create_graph=True, retain_graph=True)[0][:, 2][:, None]
        f = data[1]
        res = dudt - v_n * (d2udx2 + d2udy2 + d2udz2)
        loss = nn.MSELoss()(res, f)
        return loss

    def loss_boundary(data):
        output = model(data[0])
        u = output
        g = data[1]
        loss = nn.MSELoss()(u, g)
        return loss

    def loss_initial(data):
        output = model(data[0])
        u = output
        u0 = data[1]
        loss = nn.MSELoss()(u, u0)
        return loss

    def loss_interface_phi(data):
        phi = data[2]
        u_p = model(data[0])
        u_n = model(data[1])
        loss = nn.MSELoss()(u_p - u_n, phi)
        return loss

    def loss_interface_psi(data):
        n = data[2]
        psi = data[3]
        u_p = model(data[0])
        u_n = model(data[1])
        gradu_p = autograd.grad(outputs=u_p, inputs=data[0], grad_outputs=torch.ones_like(u_p), create_graph=True, retain_graph=True)
        du_pdx = gradu_p[0][:, 0][:, None]
        du_pdy = gradu_p[0][:, 1][:, None]
        du_pdz = gradu_p[0][:, 2][:, None]
        gradu_n = autograd.grad(outputs=u_n, inputs=data[1], grad_outputs=torch.ones_like(u_n), create_graph=True, retain_graph=True)
        du_ndx = gradu_n[0][:, 0][:, None]
        du_ndy = gradu_n[0][:, 1][:, None]
        du_ndz = gradu_n[0][:, 2][:, None]

        res = n[:, 0][:, None] * (v_p * du_pdx - v_n * du_ndx) + n[:, 1][:, None] * (v_p * du_pdy - du_ndy) + v_n * n[:, 2][:, None] * (v_p * du_pdz - v_n * du_ndz)
        loss = nn.MSELoss()(res, psi)
        return loss

    optimizer = SOAP(params=model.parameters(), lr=lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    best_loss = 1000

    for epoch in range(epochs + 1):
        loss1 = loss_omega_p(data_op)
        loss2 = loss_omega_n(data_on)
        loss3 = loss_boundary(data_b)
        loss4 = loss_initial(data_ini)
        loss5 = loss_interface_phi(data_if_phi)
        loss6 = loss_interface_psi(data_if_psi)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        func_params = dict(model.named_parameters())
        if (epoch + 1) % 2000 == 0:
            lr = 0.95 * lr
            optimizer = SOAP(params=model.parameters(), lr=lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
            print('lr:', lr)
        #if (epoch + 1) % 1000 == 0:
        #    weight = loss_weight(data_op, data_on, data_b, data_ini, data_if, beta_p, beta_n, model, device, eps)
        if epoch % 100 == 0:
            print('epoch:', epoch)
            print('loss:', loss.item(), 'loss1:', loss1.item(), 'loss2:', loss2.item(), 'loss3:', loss3.item(), 'loss4:', loss4.item())
            print('loss5:', loss5.item(), 'loss6:', loss6.item())
            # print(lr)
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


def optimize_parameters_LM(data_op, data_on, data_b, data_if_phi, data_if_psi, data_ini, v_p, v_n, model, tr_iter_max, mu, lossval, lossval_dbg, relative_error, mu_div, mu_mul, device):
    def get_p_vec(func_params):
        p_vec = []
        cnt = 0
        for p in func_params:
            p_vec = func_params[p].contiguous().view(-1) if cnt == 0 else torch.cat([p_vec, func_params[p].contiguous().view(-1)])
            cnt = 1
        return p_vec

    def count_parameters(func_params):
        return sum(x.numel() for x in model.parameters())

    def generate_initial_LM(func_params, Xop_len, Xon_len, Xb_len, Xini_len, Xif_phi_len, Xif_psi_len):
        # data_length
        data_length = Xop_len + Xon_len + Xini_len + Xb_len + Xif_phi_len + Xif_psi_len

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

    def res_omega_p(func_params, data, f):
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data, func_params)
        dudt = gradu[3]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        d2udz2 = grad2u[2][2]
        f = f[0]
        res = dudt - v_p * (d2udx2 + d2udy2 + d2udz2) - f
        return res

    def res_omega_n(func_params, data, f):
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data, func_params)
        dudt = gradu[3]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        d2udz2 = grad2u[2][2]
        f = f[0]
        res = dudt - v_n * (d2udx2 + d2udy2 + d2udz2) - f
        return res

    def res_boundary(func_params, data, g):
        u = u_NN(data, func_params)
        g = g[0]
        res = u - g
        return res


    def res_initial(func_params, data, u0):
        u = u_NN(data, func_params)
        u0 = u0[0]
        res = u - u0
        return res


    def res_interface_phi(func_params, data_p, data_n, phi):
        up = u_NN(data_p, func_params)
        un = u_NN(data_n, func_params)
        phi = phi[0]
        res = up - un - phi
        return res


    def res_interface_psi(func_params, data_p, data_n, nor, psi):
        gradu_p = torch.func.jacrev(u_NN, argnums=0)(data_p, func_params)
        gradu_n = torch.func.jacrev(u_NN, argnums=0)(data_n, func_params)
        du_pdx = gradu_p[0]
        du_pdy = gradu_p[1]
        du_pdz = gradu_p[2]
        du_ndx = gradu_n[0]
        du_ndy = gradu_n[1]
        du_ndz = gradu_n[2]
        psi = psi[0]
        n1 = nor[0]
        n2 = nor[1]
        n3 = nor[2]
        res = n1 * (v_p * du_pdx - v_n * du_ndx) + n2 * (v_p * du_pdy - du_ndy) + v_n * n3 * (v_p * du_pdz - v_n * du_ndz) - psi
        return res


    torch.cuda.empty_cache()
    # tolerence for LM
    tol_main = 10 ** (-14)
    tol_machine = 10 ** (-15)
    mu_max = 10 ** 8
    # iteration check
    ls_check = 10
    ls_check0 = ls_check - 1
    # Loss parameters
    NL = [len(data_op[0]) + len(data_on[0]) + len(data_b[0]) + len(data_ini[0]) + len(data_if_phi[0]) + len(data_if_psi[0]),
          len(data_op[0]), len(data_on[0]), len(data_b[0]), len(data_ini[0]), len(data_if_phi[0]), len(data_if_psi[0])]
    NL_sqrt = np.sqrt(NL)
    func_params = dict(model.named_parameters())
    p_vec_o, dp_o, L_o, J_o = generate_initial_LM(func_params, NL[1], NL[2], NL[3], NL[4], NL[5], NL[6])
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
                Lop = torch.vmap((res_omega_p), (None, 0, 0))(func_params, data_op[0], data_op[1]).flatten().detach()
                Lon = torch.vmap((res_omega_n), (None, 0, 0))(func_params, data_on[0], data_on[1]).flatten().detach()
                Lb = torch.vmap((res_boundary), (None, 0, 0))(func_params, data_b[0], data_b[1]).flatten().detach()
                Lini = torch.vmap((res_initial), (None, 0, 0))(func_params, data_ini[0], data_ini[1]).flatten().detach()
                Lif_phi = torch.vmap((res_interface_phi), (None, 0, 0, 0))(func_params, data_if_phi[0], data_if_phi[1], data_if_phi[2]).flatten().detach()
                Lif_psi = torch.vmap((res_interface_psi), (None, 0, 0, 0, 0))(func_params, data_if_psi[0], data_if_psi[1], data_if_psi[2], data_if_psi[3]).flatten().detach()
                L = torch.cat((Lop / NL_sqrt[1], Lon / NL_sqrt[2], Lb / NL_sqrt[3], Lini / NL_sqrt[4], Lif_phi / NL_sqrt[5], Lif_psi / NL_sqrt[6]))
                L = L.reshape(NL[0], 1).detach()
                lsop_sum = torch.sum(Lop * Lop) / NL[1]
                lson_sum = torch.sum(Lon * Lon) / NL[2]
                lsb_sum = torch.sum(Lb * Lb) / NL[3]
                lsini_sum = torch.sum(Lini * Lini) / NL[4]
                lsif_phi_sum = torch.sum(Lif_phi * Lif_phi) / NL[5]
                lsif_psi_sum = torch.sum(Lif_psi * Lif_psi) / NL[6]
                loss_dbg_old = [lsop_sum.item(), lson_sum.item(), lsb_sum.item(), lsini_sum.item(), lsif_phi_sum.item(), lsif_psi_sum.item()]
            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]
            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = get_p_vec(func_params).detach()  # get p_vec for p_vec_old if neccessary

            if criterion:
                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_p), (None, 0, 0))(func_params, data_op[0], data_op[1])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_op = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_op, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_n), (None, 0, 0))(func_params, data_on[0], data_on[1])
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


                per_sample_grads = torch.vmap(torch.func.jacrev(res_initial), (None, 0, 0))(func_params, data_ini[0], data_ini[1])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_ini = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_ini, g.reshape(len(g), -1)])
                    cnt = 1


                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_phi), (None, 0, 0, 0))(func_params, data_if_phi[0], data_if_phi[1], data_if_phi[2])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_if_phi = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if_phi, g.reshape(len(g), -1)])
                    cnt = 1


                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_psi), (None, 0, 0, 0, 0))(func_params, data_if_psi[0], data_if_psi[1], data_if_psi[2], data_if_psi[3])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_if_psi = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if_psi, g.reshape(len(g), -1)])
                    cnt = 1

                J = torch.cat((J_op / NL_sqrt[1], J_on / NL_sqrt[2], J_b / NL_sqrt[3], J_ini / NL_sqrt[4], J_if_phi / NL_sqrt[5], J_if_psi / NL_sqrt[6])).detach()
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
            Lop = torch.vmap((res_omega_p), (None, 0, 0))(func_params, data_op[0], data_op[1]).flatten().detach()
            Lon = torch.vmap((res_omega_n), (None, 0, 0))(func_params, data_on[0], data_on[1]).flatten().detach()
            Lb = torch.vmap((res_boundary), (None, 0, 0))(func_params, data_b[0], data_b[1]).flatten().detach()
            Lini = torch.vmap((res_initial), (None, 0, 0))(func_params, data_ini[0], data_ini[1]).flatten().detach()
            Lif_phi = torch.vmap((res_interface_phi), (None, 0, 0, 0))(func_params, data_if_phi[0], data_if_phi[1], data_if_phi[2]).flatten().detach()
            Lif_psi = torch.vmap((res_interface_psi), (None, 0, 0, 0, 0))(func_params, data_if_psi[0], data_if_psi[1], data_if_psi[2], data_if_psi[3]).flatten().detach()
            L = torch.cat((Lop / NL_sqrt[1], Lon / NL_sqrt[2], Lb / NL_sqrt[3], Lini / NL_sqrt[4], Lif_phi / NL_sqrt[5],
                           Lif_psi / NL_sqrt[6]))
            L = L.reshape(NL[0], 1).detach()
            loss_new = torch.sum(L * L).item()
            lsop_sum = torch.sum(Lop * Lop) / NL[1]
            lson_sum = torch.sum(Lon * Lon) / NL[2]
            lsb_sum = torch.sum(Lb * Lb) / NL[3]
            lsini_sum = torch.sum(Lini * Lini) / NL[4]
            lsif_phi_sum = torch.sum(Lif_phi * Lif_phi) / NL[5]
            lsif_psi_sum = torch.sum(Lif_psi * Lif_psi) / NL[6]
            loss_dbg_new = [lsop_sum.item(), lson_sum.item(), lsb_sum.item(), lsini_sum.item(), lsif_phi_sum.item(), lsif_psi_sum.item()]

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