# -*- coding: utf-8 -*-
# @Time    : 2025/2/27 上午10:54
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
        u1 = output[:, 0][:, None]
        u2 = output[:, 1][:, None]
        p = output[:, 2][:, None]
        gradu1 = autograd.grad(outputs=u1, inputs=data[0], grad_outputs=torch.ones_like(u1), create_graph=True, retain_graph=True)
        du1dx = gradu1[0][:, 0][:, None]
        du1dy = gradu1[0][:, 1][:, None]
        du1dt = gradu1[0][:, 2][:, None]
        d2u1dx2 = autograd.grad(outputs=du1dx, inputs=data[0], grad_outputs=torch.ones_like(du1dx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2u1dy2 = autograd.grad(outputs=du1dy, inputs=data[0], grad_outputs=torch.ones_like(du1dy), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        gradu2 = autograd.grad(outputs=u2, inputs=data[0], grad_outputs=torch.ones_like(u2), create_graph=True, retain_graph=True)
        du2dx = gradu2[0][:, 0][:, None]
        du2dy = gradu2[0][:, 1][:, None]
        du2dt = gradu2[0][:, 2][:, None]
        d2u2dx2 = autograd.grad(outputs=du2dx, inputs=data[0], grad_outputs=torch.ones_like(du2dx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2u2dy2 = autograd.grad(outputs=du2dy, inputs=data[0], grad_outputs=torch.ones_like(du2dy), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        gradp = autograd.grad(outputs=p, inputs=data[0], grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)
        dpdx = gradp[0][:, 0][:, None]
        dpdy = gradp[0][:, 1][:, None]

        f_1 = data[1]
        f_2 = data[2]
        w_1 = data[3]
        w_2 = data[4]

        res1 = du1dt + w_1 * du1dx + w_2 * du1dy - v_p * (d2u1dx2 + d2u1dy2) + dpdx
        res2 = du2dt + w_1 * du2dx + w_2 * du2dy - v_p * (d2u2dx2 + d2u2dy2) + dpdy

        loss1 = nn.MSELoss()(res1, f_1)
        loss2 = nn.MSELoss()(res2, f_2)
        return loss1, loss2

    def loss_omega_n(data):
        output = model(data[0])
        u1 = output[:, 0][:, None]
        u2 = output[:, 1][:, None]
        p = output[:, 2][:, None]
        gradu1 = autograd.grad(outputs=u1, inputs=data[0], grad_outputs=torch.ones_like(u1), create_graph=True, retain_graph=True)
        du1dx = gradu1[0][:, 0][:, None]
        du1dy = gradu1[0][:, 1][:, None]
        du1dt = gradu1[0][:, 2][:, None]
        d2u1dx2 = autograd.grad(outputs=du1dx, inputs=data[0], grad_outputs=torch.ones_like(du1dx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2u1dy2 = autograd.grad(outputs=du1dy, inputs=data[0], grad_outputs=torch.ones_like(du1dy), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        gradu2 = autograd.grad(outputs=u2, inputs=data[0], grad_outputs=torch.ones_like(u2), create_graph=True, retain_graph=True)
        du2dx = gradu2[0][:, 0][:, None]
        du2dy = gradu2[0][:, 1][:, None]
        du2dt = gradu2[0][:, 2][:, None]
        d2u2dx2 = autograd.grad(outputs=du2dx, inputs=data[0], grad_outputs=torch.ones_like(du2dx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2u2dy2 = autograd.grad(outputs=du2dy, inputs=data[0], grad_outputs=torch.ones_like(du2dy), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        gradp = autograd.grad(outputs=p, inputs=data[0], grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)
        dpdx = gradp[0][:, 0][:, None]
        dpdy = gradp[0][:, 1][:, None]

        f_1 = data[1]
        f_2 = data[2]
        w_1 = data[3]
        w_2 = data[4]

        res1 = du1dt + w_1 * du1dx + w_2 * du1dy - v_n * (d2u1dx2 + d2u1dy2) + dpdx
        res2 = du2dt + w_1 * du2dx + w_2 * du2dy - v_n * (d2u2dx2 + d2u2dy2) + dpdy

        loss1 = nn.MSELoss()(res1, f_1)
        loss2 = nn.MSELoss()(res2, f_2)
        return loss1, loss2

    def loss_div(data_p, data_n):
        u_p = model(data_p[0])
        u_p_1 = u_p[:, 0][:, None]
        u_p_2 = u_p[:, 1][:, None]
        du_p_1dx = autograd.grad(outputs=u_p_1, inputs=data_p[0], grad_outputs=torch.ones_like(u_p_1), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        du_p_2dy = autograd.grad(outputs=u_p_2, inputs=data_p[0], grad_outputs=torch.ones_like(u_p_2), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        u_n = model(data_n[0])
        u_n_1 = u_n[:, 0][:, None]
        u_n_2 = u_n[:, 1][:, None]
        du_n_1dx = autograd.grad(outputs=u_n_1, inputs=data_n[0], grad_outputs=torch.ones_like(u_n_1), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        du_n_2dy = autograd.grad(outputs=u_n_2, inputs=data_n[0], grad_outputs=torch.ones_like(u_n_2), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        res1 = du_p_1dx + du_p_2dy
        res2 = du_n_1dx + du_n_2dy

        loss1 = nn.MSELoss()(res1, torch.zeros_like(res1))
        loss2 = nn.MSELoss()(res2, torch.zeros_like(res2))
        return loss1, loss2

    def loss_boundary(data):
        output = model(data[0])
        u1 = output[:, 0][:, None]
        u2 = output[:, 1][:, None]
        g1 = data[1]
        g2 = data[2]
        loss1 = nn.MSELoss()(u1, g1)
        loss2 = nn.MSELoss()(u2, g2)
        return loss1, loss2

    def loss_initial(data):
        output = model(data[0])
        u1 = output[:, 0][:, None]
        u2 = output[:, 1][:, None]
        u01 = data[1]
        u02 = data[2]
        loss1 = nn.MSELoss()(u1, u01)
        loss2 = nn.MSELoss()(u2, u02)
        return loss1, loss2

    def loss_interface_phi(data):
        x_p = data[0]
        x_n = data[1]
        phi_1 = data[2]
        phi_2 = data[3]
        u_p = model(x_p)
        u_n = model(x_n)
        u_p_1 = u_p[:, 0][:, None]
        u_p_2 = u_p[:, 1][:, None]
        u_n_1 = u_n[:, 0][:, None]
        u_n_2 = u_n[:, 1][:, None]
        loss1 = nn.MSELoss()(u_p_1 - u_n_1, phi_1)
        loss2 = nn.MSELoss()(u_p_2 - u_n_2, phi_2)
        return loss1, loss2

    def loss_interface_psi(data):
        x_p = data[0]
        x_n = data[1]
        psi_1 = data[2]
        psi_2 = data[3]
        u_p = model(x_p)
        u_n = model(x_n)
        u_p_1 = u_p[:, 0][:, None]
        u_p_2 = u_p[:, 1][:, None]
        p_p = u_p[:, 2][:, None]
        u_n_1 = u_n[:, 0][:, None]
        u_n_2 = u_n[:, 1][:, None]
        p_n = u_n[:, 2][:, None]
        gradu_p_1 = autograd.grad(outputs=u_p_1, inputs=x_p, grad_outputs=torch.ones_like(u_p_1), create_graph=True, retain_graph=True)
        du_p_1dx = gradu_p_1[0][:, 0][:, None]
        du_p_1dy = gradu_p_1[0][:, 1][:, None]
        gradu_p_2 = autograd.grad(outputs=u_p_2, inputs=x_p, grad_outputs=torch.ones_like(u_p_2), create_graph=True, retain_graph=True)
        du_p_2dx = gradu_p_2[0][:, 0][:, None]
        du_p_2dy = gradu_p_2[0][:, 1][:, None]
        gradu_n_1 = autograd.grad(outputs=u_n_1, inputs=x_n, grad_outputs=torch.ones_like(u_n_1), create_graph=True, retain_graph=True)
        du_n_1dx = gradu_n_1[0][:, 0][:, None]
        du_n_1dy = gradu_n_1[0][:, 1][:, None]
        gradu_n_2 = autograd.grad(outputs=u_n_2, inputs=x_n, grad_outputs=torch.ones_like(u_n_2), create_graph=True, retain_graph=True)
        du_n_2dx = gradu_n_2[0][:, 0][:, None]
        du_n_2dy = gradu_n_2[0][:, 1][:, None]
        res1 = v_p * (du_p_1dx + du_p_1dy) - p_p - v_n * (du_n_1dx + du_n_1dy) + p_n
        res2 = v_p * (du_p_2dx + du_p_2dy) - p_p - v_n * (du_n_2dx + du_n_2dy) + p_n
        loss1 = nn.MSELoss()(res1, psi_1)
        loss2 = nn.MSELoss()(res2, psi_2)
        return loss1, loss2

    optimizer = SOAP(params=model.parameters(), lr=lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    best_loss = 1000

    for epoch in range(epochs + 1):
        loss1, loss2 = loss_omega_p(data_op)
        loss3, loss4 = loss_omega_n(data_on)
        loss5, loss6 = loss_div(data_op, data_on)
        loss7, loss8 = loss_boundary(data_b)
        loss9, loss10 = loss_initial(data_ini)
        loss11, loss12 = loss_interface_phi(data_if_phi)
        loss13, loss14 = loss_interface_psi(data_if_psi)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14
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
            print('loss5:', loss5.item(), 'loss6:', loss6.item(), 'loss7:', loss7.item(), 'loss8:', loss8.item(), 'loss9:', loss9.item())
            print('loss10:', loss10.item(), 'loss11:', loss11.item(), 'loss12:', loss12.item(), 'loss13:', loss13.item(), 'loss14:', loss14.item())
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
        data_length = 3 * Xop_len + 3 * Xon_len + 2 * Xini_len + 2 * Xb_len + 2 * Xif_phi_len + 2 * Xif_psi_len# 输入数据长度和

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

    def u_1_NN(data, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)[0]

    def u_2_NN(data, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)[1]

    def p_NN(data, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)[2]

    def res_omega_p_1(func_params, data, f, w_1, w_2):
        gradu = torch.func.jacrev(u_1_NN, argnums=0)(data, func_params)
        gradp = torch.func.jacrev(p_NN, argnums=0)(data, func_params)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_1_NN, argnums=0), argnums=0)(data, func_params)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        dpdx = gradp[0]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        res = dudt + w_1 * dudx + w_2 * dudy - v_p * (d2udx2 + d2udy2) + dpdx - f
        return res

    def res_omega_p_2(func_params, data, f, w_1, w_2):
        gradu = torch.func.jacrev(u_2_NN, argnums=0)(data, func_params)
        gradp = torch.func.jacrev(p_NN, argnums=0)(data, func_params)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_2_NN, argnums=0), argnums=0)(data, func_params)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        dpdy = gradp[1]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        res = dudt + w_1 * dudx + w_2 * dudy - v_p * (d2udx2 + d2udy2) + dpdy - f
        return res

    def res_omega_n_1(func_params, data, f, w_1, w_2):
        gradu = torch.func.jacrev(u_1_NN, argnums=0)(data, func_params)
        gradp = torch.func.jacrev(p_NN, argnums=0)(data, func_params)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_1_NN, argnums=0), argnums=0)(data, func_params)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        dpdx = gradp[0]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        res = dudt + w_1 * dudx + w_2 * dudy - v_n * (d2udx2 + d2udy2) + dpdx - f
        return res

    def res_omega_n_2(func_params, data, f, w_1, w_2):
        gradu = torch.func.jacrev(u_2_NN, argnums=0)(data, func_params)
        gradp = torch.func.jacrev(p_NN, argnums=0)(data, func_params)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_2_NN, argnums=0), argnums=0)(data, func_params)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        dpdy = gradp[1]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        res = dudt + w_1 * dudx + w_2 * dudy - v_n * (d2udx2 + d2udy2) + dpdy - f
        return res

    def res_div_p(func_params, data):
        gradu1 = torch.func.jacrev(u_1_NN, argnums=0)(data, func_params)
        gradu2 = torch.func.jacrev(u_2_NN, argnums=0)(data, func_params)
        du1dx = gradu1[0]
        du2dy = gradu2[1]
        res = du1dx + du2dy
        return res

    def res_div_n(func_params, data):
        gradu1 = torch.func.jacrev(u_1_NN, argnums=0)(data, func_params)
        gradu2 = torch.func.jacrev(u_2_NN, argnums=0)(data, func_params)
        du1dx = gradu1[0]
        du2dy = gradu2[1]
        res = du1dx + du2dy
        return res

    def res_boundary_1(func_params, data, g1):
        u = u_1_NN(data, func_params)
        g1 = g1[0]
        res = u - g1
        return res

    def res_boundary_2(func_params, data, g2):
        u = u_2_NN(data, func_params)
        g2 = g2[0]
        res = u - g2
        return res

    def res_initial_1(func_params, data, u0):
        u = u_1_NN(data, func_params)
        u0 = u0[0]
        res = u - u0
        return res

    def res_initial_2(func_params, data, u0):
        u = u_2_NN(data, func_params)
        u0 = u0[0]
        res = u - u0
        return res

    def res_interface_phi_1(func_params, data_p, data_n, phi):
        up = u_1_NN(data_p, func_params)
        un = u_1_NN(data_n, func_params)
        phi = phi[0]
        res = up - un - phi
        return res

    def res_interface_phi_2(func_params, data_p, data_n, phi):
        up = u_2_NN(data_p, func_params)
        un = u_2_NN(data_n, func_params)
        phi = phi[0]
        res = up - un - phi
        return res

    def res_interface_psi_1(func_params, data_p, data_n, psi):
        p_p = p_NN(data_p, func_params)
        p_n = p_NN(data_n, func_params)
        gradu_p = torch.func.jacrev(u_1_NN, argnums=0)(data_p, func_params)
        gradu_n = torch.func.jacrev(u_1_NN, argnums=0)(data_n, func_params)
        du_pdx = gradu_p[0]
        du_pdy = gradu_p[1]
        du_ndx = gradu_n[0]
        du_ndy = gradu_n[1]
        psi = psi[0]
        res = v_p * (du_pdx + du_pdy) - p_p - v_n * (du_ndx + du_ndy) + p_n - psi
        return res

    def res_interface_psi_2(func_params, data_p, data_n, psi):
        p_p = p_NN(data_p, func_params)
        p_n = p_NN(data_n, func_params)
        gradu_p = torch.func.jacrev(u_2_NN, argnums=0)(data_p, func_params)
        gradu_n = torch.func.jacrev(u_2_NN, argnums=0)(data_n, func_params)
        du_pdx = gradu_p[0]
        du_pdy = gradu_p[1]
        du_ndx = gradu_n[0]
        du_ndy = gradu_n[1]
        psi = psi[0]
        res = v_p * (du_pdx + du_pdy) - p_p - v_n * (du_ndx + du_ndy) + p_n - psi
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
    NL = [3 * len(data_op[0]) + 3 * len(data_on[0]) + 2 * len(data_b[0]) + 2 * len(data_ini[0]) + 2 * len(data_if_phi[0]) + 2 * len(data_if_psi[0]),
          len(data_op[0]), len(data_op[0]), len(data_on[0]), len(data_on[0]), len(data_op[0]), len(data_on[0]), len(data_b[0]), len(data_b[0]),
          len(data_ini[0]), len(data_ini[0]), len(data_if_phi[0]), len(data_if_phi[0]), len(data_if_psi[0]), len(data_if_psi[0])]
    NL_sqrt = np.sqrt(NL)
    func_params = dict(model.named_parameters())
    p_vec_o, dp_o, L_o, J_o = generate_initial_LM(func_params, NL[1], NL[3], NL[7], NL[9], NL[11], NL[13])
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
                Lop_1 = torch.vmap((res_omega_p_1), (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[1], data_op[3], data_op[4]).flatten().detach()
                Lop_2 = torch.vmap((res_omega_p_2), (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[2], data_op[3], data_op[4]).flatten().detach()
                Lon_1 = torch.vmap((res_omega_n_1), (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[1], data_on[3], data_on[4]).flatten().detach()
                Lon_2 = torch.vmap((res_omega_n_2), (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[2], data_on[3], data_on[4]).flatten().detach()
                Ldiv_p = torch.vmap((res_div_p), (None, 0))(func_params, data_op[0]).flatten().detach()
                Ldiv_n = torch.vmap((res_div_n), (None, 0))(func_params, data_on[0]).flatten().detach()
                Lb_1 = torch.vmap((res_boundary_1), (None, 0, 0))(func_params, data_b[0], data_b[1]).flatten().detach()
                Lb_2 = torch.vmap((res_boundary_2), (None, 0, 0))(func_params, data_b[0], data_b[2]).flatten().detach()
                Lini_1 = torch.vmap((res_initial_1), (None, 0, 0))(func_params, data_ini[0], data_ini[1]).flatten().detach()
                Lini_2 = torch.vmap((res_initial_2), (None, 0, 0))(func_params, data_ini[0], data_ini[2]).flatten().detach()
                Lif_phi_1 = torch.vmap((res_interface_phi_1), (None, 0, 0, 0))(func_params, data_if_phi[0], data_if_phi[1], data_if_phi[2]).flatten().detach()
                Lif_phi_2 = torch.vmap((res_interface_phi_2), (None, 0, 0, 0))(func_params, data_if_phi[0], data_if_phi[1], data_if_phi[3]).flatten().detach()
                Lif_psi_1 = torch.vmap((res_interface_psi_1), (None, 0, 0, 0))(func_params, data_if_psi[0], data_if_psi[1], data_if_psi[2]).flatten().detach()
                Lif_psi_2 = torch.vmap((res_interface_psi_2), (None, 0, 0, 0))(func_params, data_if_psi[0], data_if_psi[1], data_if_psi[3]).flatten().detach()
                L = torch.cat((Lop_1 / NL_sqrt[1], Lop_2 / NL_sqrt[2], Lon_1 / NL_sqrt[3], Lon_2 / NL_sqrt[4], Ldiv_p / NL_sqrt[5],
                               Ldiv_n / NL_sqrt[6], Lb_1 / NL_sqrt[7], Lb_2 / NL_sqrt[8], Lini_1 / NL_sqrt[9], Lini_2 / NL_sqrt[10],
                               Lif_phi_1 / NL_sqrt[11], Lif_phi_2 / NL_sqrt[12], Lif_psi_1 / NL_sqrt[13], Lif_psi_2 / NL_sqrt[14]))
                L = L.reshape(NL[0], 1).detach()
                lsop_1_sum = torch.sum(Lop_1 * Lop_1) / NL[1]
                lsop_2_sum = torch.sum(Lop_2 * Lop_2) / NL[2]
                lson_1_sum = torch.sum(Lon_1 * Lon_1) / NL[3]
                lson_2_sum = torch.sum(Lon_2 * Lon_2) / NL[4]
                lsdiv_p_sum = torch.sum(Ldiv_p * Ldiv_p) / NL[5]
                lsdiv_n_sum = torch.sum(Ldiv_n * Ldiv_n) / NL[6]
                lsb_1_sum = torch.sum(Lb_1 * Lb_1) / NL[7]
                lsb_2_sum = torch.sum(Lb_2 * Lb_2) / NL[8]
                lsini_1_sum = torch.sum(Lini_1 * Lini_1) / NL[9]
                lsini_2_sum = torch.sum(Lini_2 * Lini_2) / NL[10]
                lsif_phi_1_sum = torch.sum(Lif_phi_1 * Lif_phi_1) / NL[11]
                lsif_phi_2_sum = torch.sum(Lif_phi_2 * Lif_phi_2) / NL[12]
                lsif_psi_1_sum = torch.sum(Lif_psi_1 * Lif_psi_1) / NL[13]
                lsif_psi_2_sum = torch.sum(Lif_psi_2 * Lif_psi_2) / NL[14]
                loss_dbg_old = [lsop_1_sum.item(), lsop_2_sum.item(), lson_1_sum.item(), lson_2_sum.item(), lsdiv_p_sum.item(),
                                lsdiv_n_sum.item(), lsb_1_sum.item(), lsb_2_sum.item(), lsini_1_sum.item(), lsini_2_sum.item(),
                                lsif_phi_1_sum.item(), lsif_phi_2_sum.item(), lsif_psi_1_sum.item(), lsif_psi_2_sum.item()]
            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]
            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = get_p_vec(func_params).detach()  # get p_vec for p_vec_old if neccessary

            if criterion:
                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_p_1), (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[1], data_op[3], data_op[4])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_op_1 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_op_1, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_p_2), (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[2], data_op[3], data_op[4])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_op_2 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_op_2, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_n_1), (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[1], data_on[3], data_on[4])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_on_1 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_on_1, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_n_2), (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[2], data_on[3], data_on[4])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_on_2 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_on_2, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_div_p), (None, 0))(func_params, data_op[0])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_div_p = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_div_p, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_div_n), (None, 0))(func_params, data_on[0])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_div_n = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_div_n, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_boundary_1), (None, 0, 0))(func_params, data_b[0], data_b[1])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_b_1 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_b_1, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_boundary_2), (None, 0, 0))(func_params, data_b[0], data_b[2])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_b_2 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_b_2, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_initial_1), (None, 0, 0))(func_params, data_ini[0], data_ini[1])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_ini_1 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_ini_1, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_initial_2), (None, 0, 0))(func_params, data_ini[0], data_ini[2])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_ini_2 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_ini_2, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_phi_1), (None, 0, 0, 0))(func_params, data_if_phi[0], data_if_phi[1], data_if_phi[2])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_if_phi_1 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if_phi_1, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_phi_2), (None, 0, 0, 0))(func_params, data_if_phi[0], data_if_phi[1], data_if_phi[3])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_if_phi_2 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if_phi_2, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_psi_1), (None, 0, 0, 0))(func_params, data_if_psi[0], data_if_psi[1], data_if_psi[2])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_if_psi_1 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if_psi_1, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_psi_2), (None, 0, 0, 0))(func_params, data_if_psi[0], data_if_psi[1], data_if_psi[3])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    J_if_psi_2 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if_psi_2, g.reshape(len(g), -1)])
                    cnt = 1

                J = torch.cat((J_op_1 / NL_sqrt[1], J_op_2 / NL_sqrt[2], J_on_1 / NL_sqrt[3], J_on_2 / NL_sqrt[4], J_div_p / NL_sqrt[5],
                               J_div_n / NL_sqrt[6], J_b_1 / NL_sqrt[7], J_b_2 / NL_sqrt[8], J_ini_1 / NL_sqrt[9], J_ini_2 / NL_sqrt[10],
                               J_if_phi_1 / NL_sqrt[11], J_if_phi_2 / NL_sqrt[12], J_if_psi_1 / NL_sqrt[13], J_if_psi_2 / NL_sqrt[14])).detach()
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
            Lop_1 = torch.vmap((res_omega_p_1), (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[1], data_op[3], data_op[4]).flatten().detach()
            Lop_2 = torch.vmap((res_omega_p_2), (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[2], data_op[3], data_op[4]).flatten().detach()
            Lon_1 = torch.vmap((res_omega_n_1), (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[1], data_on[3], data_on[4]).flatten().detach()
            Lon_2 = torch.vmap((res_omega_n_2), (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[2], data_on[3], data_on[4]).flatten().detach()
            Ldiv_p = torch.vmap((res_div_p), (None, 0))(func_params, data_op[0]).flatten().detach()
            Ldiv_n = torch.vmap((res_div_n), (None, 0))(func_params, data_on[0]).flatten().detach()
            Lb_1 = torch.vmap((res_boundary_1), (None, 0, 0))(func_params, data_b[0], data_b[1]).flatten().detach()
            Lb_2 = torch.vmap((res_boundary_2), (None, 0, 0))(func_params, data_b[0], data_b[2]).flatten().detach()
            Lini_1 = torch.vmap((res_initial_1), (None, 0, 0))(func_params, data_ini[0], data_ini[1]).flatten().detach()
            Lini_2 = torch.vmap((res_initial_2), (None, 0, 0))(func_params, data_ini[0], data_ini[2]).flatten().detach()
            Lif_phi_1 = torch.vmap((res_interface_phi_1), (None, 0, 0, 0))(func_params, data_if_phi[0], data_if_phi[1], data_if_phi[2]).flatten().detach()
            Lif_phi_2 = torch.vmap((res_interface_phi_2), (None, 0, 0, 0))(func_params, data_if_phi[0], data_if_phi[1], data_if_phi[3]).flatten().detach()
            Lif_psi_1 = torch.vmap((res_interface_psi_1), (None, 0, 0, 0))(func_params, data_if_psi[0], data_if_psi[1], data_if_psi[2]).flatten().detach()
            Lif_psi_2 = torch.vmap((res_interface_psi_2), (None, 0, 0, 0))(func_params, data_if_psi[0], data_if_psi[1], data_if_psi[3]).flatten().detach()
            L = torch.cat((Lop_1 / NL_sqrt[1], Lop_2 / NL_sqrt[2], Lon_1 / NL_sqrt[3], Lon_2 / NL_sqrt[4], Ldiv_p / NL_sqrt[5],
                           Ldiv_n / NL_sqrt[6], Lb_1 / NL_sqrt[7], Lb_2 / NL_sqrt[8], Lini_1 / NL_sqrt[9], Lini_2 / NL_sqrt[10],
                           Lif_phi_1 / NL_sqrt[11], Lif_phi_2 / NL_sqrt[12], Lif_psi_1 / NL_sqrt[13], Lif_psi_2 / NL_sqrt[14]))
            L = L.reshape(NL[0], 1).detach()
            loss_new = torch.sum(L * L).item()
            lsop_1_sum = torch.sum(Lop_1 * Lop_1) / NL[1]
            lsop_2_sum = torch.sum(Lop_2 * Lop_2) / NL[2]
            lson_1_sum = torch.sum(Lon_1 * Lon_1) / NL[3]
            lson_2_sum = torch.sum(Lon_2 * Lon_2) / NL[4]
            lsdiv_p_sum = torch.sum(Ldiv_p * Ldiv_p) / NL[5]
            lsdiv_n_sum = torch.sum(Ldiv_n * Ldiv_n) / NL[6]
            lsb_1_sum = torch.sum(Lb_1 * Lb_1) / NL[7]
            lsb_2_sum = torch.sum(Lb_2 * Lb_2) / NL[8]
            lsini_1_sum = torch.sum(Lini_1 * Lini_1) / NL[9]
            lsini_2_sum = torch.sum(Lini_2 * Lini_2) / NL[10]
            lsif_phi_1_sum = torch.sum(Lif_phi_1 * Lif_phi_1) / NL[11]
            lsif_phi_2_sum = torch.sum(Lif_phi_2 * Lif_phi_2) / NL[12]
            lsif_psi_1_sum = torch.sum(Lif_psi_1 * Lif_psi_1) / NL[13]
            lsif_psi_2_sum = torch.sum(Lif_psi_2 * Lif_psi_2) / NL[14]
            loss_dbg_new = [lsop_1_sum.item(), lsop_2_sum.item(), lson_1_sum.item(), lson_2_sum.item(), lsdiv_p_sum.item(),
                            lsdiv_n_sum.item(), lsb_1_sum.item(), lsb_2_sum.item(), lsini_1_sum.item(), lsini_2_sum.item(),
                            lsif_phi_1_sum.item(), lsif_phi_2_sum.item(), lsif_psi_1_sum.item(), lsif_psi_2_sum.item()]

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