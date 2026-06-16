# -*- coding: utf-8 -*-
# @Time    : 2026/5/26 下午8:55
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


def error_compute(model, func_params, data):
    def u_NN(model, params, x):
        return torch.func.functional_call(model, params, x)

    pred = u_NN(model, func_params, data[0])
    u_pred = pred[:, 0]
    v_pred = pred[:, 1]
    u_true = data[1].reshape(-1)
    v_true = data[2].reshape(-1)

    err_u = u_pred - u_true
    err_v = v_pred - v_true

    err_norm = torch.sqrt(torch.sum(err_u**2 + err_v**2))
    true_norm = torch.sqrt(torch.sum(u_true**2 + v_true**2))
    relative_error = err_norm / true_norm
    return relative_error


def optimize_parameters_SOAP(data_op, data_on, data_b, data_ini, data_if, nu_p, nu_n, model, epochs, lr, lossval, lossval_dbg, relative_error, device):
    def loss_omega_p(data):
        output = model(data[0])
        u = output[:, 0][:, None]
        v = output[:, 1][:, None]
        p = output[:, 2][:, None]

        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudt = gradu[0][:, 2][:, None]
        d2udx2 = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(dudx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2udy2 = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(dudy), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        gradv = autograd.grad(outputs=v, inputs=data[0], grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)
        dvdx = gradv[0][:, 0][:, None]
        dvdy = gradv[0][:, 1][:, None]
        dvdt = gradv[0][:, 2][:, None]
        d2vdx2 = autograd.grad(outputs=dvdx, inputs=data[0], grad_outputs=torch.ones_like(dvdx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2vdy2 = autograd.grad(outputs=dvdy, inputs=data[0], grad_outputs=torch.ones_like(dvdy), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        gradp = autograd.grad(outputs=p, inputs=data[0], grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)
        dpdx = gradp[0][:, 0][:, None]
        dpdy = gradp[0][:, 1][:, None]

        f_1 = data[1]
        f_2 = data[2]
        w_1 = data[3]
        w_2 = data[4]

        res1 = dudt + w_1 * dudx + w_2 * dudy - nu_p * (d2udx2 + d2udy2) + dpdx
        res2 = dvdt + w_1 * dvdx + w_2 * dvdy - nu_p * (d2vdx2 + d2vdy2) + dpdy

        loss1 = nn.MSELoss()(res1, f_1)
        loss2 = nn.MSELoss()(res2, f_2)
        return loss1, loss2

    def loss_omega_n(data):
        output = model(data[0])
        u = output[:, 0][:, None]
        v = output[:, 1][:, None]
        p = output[:, 2][:, None]

        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudt = gradu[0][:, 2][:, None]
        d2udx2 = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(dudx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2udy2 = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(dudy), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        gradv = autograd.grad(outputs=v, inputs=data[0], grad_outputs=torch.ones_like(v), create_graph=True,
                              retain_graph=True)
        dvdx = gradv[0][:, 0][:, None]
        dvdy = gradv[0][:, 1][:, None]
        dvdt = gradv[0][:, 2][:, None]
        d2vdx2 = autograd.grad(outputs=dvdx, inputs=data[0], grad_outputs=torch.ones_like(dvdx), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        d2vdy2 = autograd.grad(outputs=dvdy, inputs=data[0], grad_outputs=torch.ones_like(dvdy), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        gradp = autograd.grad(outputs=p, inputs=data[0], grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)
        dpdx = gradp[0][:, 0][:, None]
        dpdy = gradp[0][:, 1][:, None]

        f_1 = data[1]
        f_2 = data[2]
        w_1 = data[3]
        w_2 = data[4]
        res1 = dudt + w_1 * dudx + w_2 * dudy - nu_n * (d2udx2 + d2udy2) + dpdx
        res2 = dvdt + w_1 * dvdx + w_2 * dvdy - nu_n * (d2vdx2 + d2vdy2) + dpdy

        loss1 = nn.MSELoss()(res1, f_1)
        loss2 = nn.MSELoss()(res2, f_2)
        return loss1, loss2


    def loss_div(data_p, data_n):
        V_p_output = model(data_p[0])
        u_p = V_p_output[:, 0][:, None]
        v_p = V_p_output[:, 1][:, None]
        du_pdx = autograd.grad(outputs=u_p, inputs=data_p[0], grad_outputs=torch.ones_like(u_p), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        dv_pdy = autograd.grad(outputs=v_p, inputs=data_p[0], grad_outputs=torch.ones_like(v_p), create_graph=True, retain_graph=True)[0][:, 1][:, None]


        V_n_output = model(data_n[0])
        u_n = V_n_output[:, 0][:, None]
        v_n = V_n_output[:, 1][:, None]
        du_ndx = autograd.grad(outputs=u_n, inputs=data_n[0], grad_outputs=torch.ones_like(u_n), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        dv_ndy = autograd.grad(outputs=v_n, inputs=data_n[0], grad_outputs=torch.ones_like(v_n), create_graph=True, retain_graph=True)[0][:, 1][:, None]

        res1 = du_pdx + dv_pdy
        res2 = du_ndx + dv_ndy

        loss1 = nn.MSELoss()(res1, torch.zeros_like(res1))
        loss2 = nn.MSELoss()(res2, torch.zeros_like(res2))
        return loss1, loss2

    def loss_boundary(data):
        output = model(data[0])
        u = output[:, 0][:, None]
        v = output[:, 1][:, None]
        g1 = data[1]
        g2 = data[2]
        loss1 = nn.MSELoss()(u, g1)
        loss2 = nn.MSELoss()(v, g2)
        return loss1, loss2

    def loss_initial(data):
        output = model(data[0])
        u = output[:, 0][:, None]
        v = output[:, 1][:, None]
        u01 = data[1]
        u02 = data[2]
        loss1 = nn.MSELoss()(u, u01)
        loss2 = nn.MSELoss()(v, u02)
        return loss1, loss2

    def loss_interface_D(data):
        x_p = data[0]
        x_n = data[1]

        V_p_output = model(x_p)
        u_p = V_p_output[:, 0][:, None]
        v_p = V_p_output[:, 1][:, None]

        V_n_output = model(x_n)
        u_n = V_n_output[:, 0][:, None]
        v_n = V_n_output[:, 1][:, None]

        phi1 = data[3]
        phi2 = data[4]
        loss1 = nn.MSELoss()(u_p - u_n, phi1)
        loss2 = nn.MSELoss()(v_p - v_n, phi2)
        return loss1, loss2

    def loss_interface_N(data):
        x_p = data[0]
        x_n = data[1]

        V_p_output = model(x_p)
        u_p = V_p_output[:, 0][:, None]
        v_p = V_p_output[:, 1][:, None]
        p_p = V_p_output[:, 2][:, None]

        V_n_output = model(x_n)
        u_n = V_n_output[:, 0][:, None]
        v_n = V_n_output[:, 1][:, None]
        p_n = V_n_output[:, 2][:, None]

        gradup = autograd.grad(outputs=u_p, inputs=x_p, grad_outputs=torch.ones_like(u_p), create_graph=True, retain_graph=True)
        dupdx = gradup[0][:, 0][:, None]
        dupdy = gradup[0][:, 1][:, None]

        gradun = autograd.grad(outputs=u_n, inputs=x_n, grad_outputs=torch.ones_like(u_n), create_graph=True, retain_graph=True)
        dundx = gradun[0][:, 0][:, None]
        dundy = gradun[0][:, 1][:, None]

        gradvp = autograd.grad(outputs=v_p, inputs=x_p, grad_outputs=torch.ones_like(v_p), create_graph=True, retain_graph=True)
        dvpdx = gradvp[0][:, 0][:, None]
        dvpdy = gradvp[0][:, 1][:, None]

        gradvn = autograd.grad(outputs=v_n, inputs=x_n, grad_outputs=torch.ones_like(v_n), create_graph=True, retain_graph=True)
        dvndx = gradvn[0][:, 0][:, None]
        dvndy = gradvn[0][:, 1][:, None]

        nor = data[2]
        n1 = nor[:, 0][:, None]
        n2 = nor[:, 1][:, None]

        sigma_p1 = -p_p * n1 + nu_p * (2 * dupdx * n1 + (dupdy + dvpdx) * n2)
        sigma_n1 = -p_n * n1 + nu_n * (2 * dundx * n1 + (dundy + dvndx) * n2)
        res1 = sigma_p1 - sigma_n1

        sigma_p2 = -p_p * n2 + nu_p * ((dupdy + dvpdx) * n1 + 2 * dvpdy * n2)
        sigma_n2 = -p_n * n2 + nu_n * ((dundy + dvndx) * n1 + 2 * dvndy * n2)
        res2 = sigma_p2 - sigma_n2

        loss1 = nn.MSELoss()(res1, data[5])
        loss2 = nn.MSELoss()(res2, data[6])
        return loss1, loss2

    params = model.parameters()
    optimizer = SOAP(params, lr=lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)
    best_loss = float('inf')
    num_test = 10000
    data_test = generate_test_data(num_test, device)

    for epoch in range(epochs + 1):
        loss1, loss2 = loss_omega_p(data_op)
        loss3, loss4 = loss_omega_n(data_on)
        loss5, loss6 = loss_div(data_op, data_on)
        loss7, loss8 = loss_boundary(data_b)
        loss9, loss10 = loss_initial(data_ini)
        loss11, loss12 = loss_interface_D(data_if)
        loss13, loss14 = loss_interface_N(data_if)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + 100*loss7 + 100*loss8 + 100*loss9 + 100*loss10 + 100*loss11 + 100*loss12 + loss13 + loss14
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            print('epoch:', epoch)
            print('loss:', loss.item(), 'loss1:', loss1.item(), 'loss2:', loss2.item(), 'loss3:', loss3.item(), 'loss4:', loss4.item())
            print('loss5:', loss5.item(), 'loss6:', loss6.item(), 'loss7:', loss7.item(), 'loss8:', loss8.item(), 'loss9:', loss9.item())
            print('loss10:', loss10.item(), 'loss11:', loss11.item(), 'loss12:', loss12.item(), 'loss13:', loss13.item(), 'loss14:', loss14.item())

            func_params = dict(model.named_parameters())
            error = error_compute(model, func_params, data_test)
            relative_error.append(error.item())
            print(f"relative error: {relative_error[-1]:.4e}")
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
            lossval.append(loss.item())
            lossval_dbg.append([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()])

    print('best epoch:', best_epoch, 'best loss:', best_loss)
    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), 'best_model_XIPINN_SOAP.mdl')
    return model, lossval, lossval_dbg, relative_error



def optimize_parameters_LM(data_op, data_on, data_b, data_ini, data_if, nu_p, nu_n, model, tr_iter_max, mu, lossval, lossval_dbg, relative_error, mu_div, mu_mul, device):
    def get_p_vec(func_params):
        p_vec = []
        cnt = 0
        for p in func_params:
            p_vec = func_params[p].contiguous().view(-1) if cnt == 0 else torch.cat(
                [p_vec, func_params[p].contiguous().view(-1)])
            cnt = 1
        return p_vec

    def count_parameters(func_params):
        return sum(x.numel() for x in func_params.values())

    def u_NN(data, func_params, model):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)[0]

    def v_NN(data, func_params, model):
        v = torch.func.functional_call(model, func_params, data)
        return v.squeeze(0).squeeze(0)[1]

    def p_NN(data, func_params, model):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)[2]

    def res_omega_p_u(func_params, data, f, w_1, w_2):
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params, model)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data, func_params, model)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        dpdx = torch.func.jacrev(p_NN, argnums=0)(data, func_params, model)[0]
        res = dudt + w_1 * dudx + w_2 * dudy - nu_p * (d2udx2 + d2udy2) + dpdx - f
        return res

    def res_omega_n_u(func_params, data, f, w_1, w_2):
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params, model)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data, func_params, model)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        dpdx = torch.func.jacrev(p_NN, argnums=0)(data, func_params, model)[0]
        res = dudt + w_1 * dudx + w_2 * dudy - nu_n * (d2udx2 + d2udy2) + dpdx - f
        return res


    def res_omega_p_v(func_params, data, f, w_1, w_2):
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        gradv = torch.func.jacrev(v_NN, argnums=0)(data, func_params, model)
        grad2v = torch.func.jacrev(torch.func.jacrev(v_NN, argnums=0), argnums=0)(data, func_params, model)
        dvdx = gradv[0]
        dvdy = gradv[1]
        dvdt = gradv[2]
        d2vdx2 = grad2v[0][0]
        d2vdy2 = grad2v[1][1]
        dpdy = torch.func.jacrev(p_NN, argnums=0)(data, func_params, model)[1]
        res = dvdt + w_1 * dvdx + w_2 * dvdy - nu_p * (d2vdx2 + d2vdy2) + dpdy - f
        return res

    def res_omega_n_v(func_params, data, f, w_1, w_2):
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        gradv = torch.func.jacrev(v_NN, argnums=0)(data, func_params, model)
        grad2v = torch.func.jacrev(torch.func.jacrev(v_NN, argnums=0), argnums=0)(data, func_params, model)
        dvdx = gradv[0]
        dvdy = gradv[1]
        dvdt = gradv[2]
        d2vdx2 = grad2v[0][0]
        d2vdy2 = grad2v[1][1]
        dpdy = torch.func.jacrev(p_NN, argnums=0)(data, func_params, model)[1]
        res = dvdt + w_1 * dvdx + w_2 * dvdy - nu_n * (d2vdx2 + d2vdy2) + dpdy - f
        return res


    def res_div_p(func_params, data):
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params, model)
        gradv = torch.func.jacrev(v_NN, argnums=0)(data, func_params, model)
        dudx = gradu[0]
        dvdy = gradv[1]
        res = dudx + dvdy
        return res

    def res_div_n(func_params, data):
        gradu = torch.func.jacrev(u_NN, argnums=0)(data, func_params, model)
        gradv = torch.func.jacrev(v_NN, argnums=0)(data, func_params, model)
        dudx = gradu[0]
        dvdy = gradv[1]
        res = dudx + dvdy
        return res


    def res_boundary_u(func_params, data, gu):
        g = gu[0]
        u = u_NN(data, func_params, model)
        res = u - g
        return res

    def res_boundary_v(func_params, data, gv):
        g = gv[0]
        v = v_NN(data, func_params, model)
        res = v - g
        return res


    def res_initial_u(func_params, data, u0):
        u = u_NN(data, func_params, model)
        u0 = u0[0]
        res = u - u0
        return res

    def res_initial_v(func_params, data, v0):
        v = v_NN(data, func_params, model)
        v0 = v0[0]
        res = v - v0
        return res

    def res_interface_D_u(func_params, data_p, data_n, phi):
        up = u_NN(data_p, func_params, model)
        un = u_NN(data_n, func_params, model)
        phi = phi[0]
        res = up - un - phi
        return res

    def res_interface_D_v(func_params, data_p, data_n, phi):
        vp = v_NN(data_p, func_params, model)
        vn = v_NN(data_n, func_params, model)
        phi = phi[0]
        res = vp - vn - phi
        return res

    def res_interface_N_1(func_params, data_p, data_n, nor, psi):
        gradup = torch.func.jacrev(u_NN, argnums=0)(data_p, func_params, model)
        dupdx = gradup[0]
        dupdy = gradup[1]

        gradun = torch.func.jacrev(u_NN, argnums=0)(data_n, func_params, model)
        dundx = gradun[0]
        dundy = gradun[1]

        gradvp = torch.func.jacrev(v_NN, argnums=0)(data_p, func_params, model)
        dvpdx = gradvp[0]

        gradvn = torch.func.jacrev(v_NN, argnums=0)(data_n, func_params, model)
        dvndx = gradvn[0]

        p_p = p_NN(data_p, func_params, model)
        p_n = p_NN(data_n, func_params, model)

        n1 = nor[0]
        n2 = nor[1]
        psi = psi[0]

        sigma_p1 = -p_p * n1 + nu_p * (2 * dupdx * n1 + (dupdy + dvpdx) * n2)
        sigma_n1 = -p_n * n1 + nu_n * (2 * dundx * n1 + (dundy + dvndx) * n2)
        res = sigma_p1 - sigma_n1 - psi
        return res


    def res_interface_N_2(func_params, data_p, data_n, nor, psi):
        gradup = torch.func.jacrev(u_NN, argnums=0)(data_p, func_params, model)
        dupdy = gradup[1]

        gradun = torch.func.jacrev(u_NN, argnums=0)(data_n, func_params, model)
        dundy = gradun[1]

        gradvp = torch.func.jacrev(v_NN, argnums=0)(data_p, func_params, model)
        dvpdx = gradvp[0]
        dvpdy = gradvp[1]

        gradvn = torch.func.jacrev(v_NN, argnums=0)(data_n, func_params, model)
        dvndx = gradvn[0]
        dvndy = gradvn[1]

        p_p = p_NN(data_p, func_params, model)
        p_n = p_NN(data_n, func_params, model)

        n1 = nor[0]
        n2 = nor[1]
        psi = psi[0]

        sigma_p2 = -p_p * n2 + nu_p * ((dupdy + dvpdx) * n1 + 2 * dvpdy * n2)
        sigma_n2 = -p_n * n2 + nu_n * ((dundy + dvndx) * n1 + 2 * dvndy * n2)
        res = sigma_p2 - sigma_n2 - psi
        return res

    # tolerence for LM
    tol_main = 10 ** (-13)
    tol_machine = 10 ** (-15)
    mu_max = 10 ** 8
    # iteration check
    ls_check = 10
    ls_check0 = ls_check - 1
    # 残差数量统计
    Nup = len(data_op[0])
    Nun = len(data_on[0])
    Nvp = len(data_op[0])
    Nvn = len(data_on[0])
    Ndivp = len(data_op[0])
    Ndivn = len(data_on[0])
    Nub = len(data_b[0])
    Nvb = len(data_b[0])
    Nuini = len(data_ini[0])
    Nvini = len(data_ini[0])
    NifD1 = len(data_if[0])
    NifD2 = len(data_if[0])
    NifN1 = len(data_if[0])
    NifN2 = len(data_if[0])
    total_res = Nup + Nun + Nvp + Nvn + Ndivp + Ndivn + Nub + Nvb + Nuini + Nvini + NifD1 + NifD2 + NifN1 + NifN2
    NL = [total_res, Nup, Nun, Nvp, Nvn, Ndivp, Ndivn, Nub, Nvb, Nuini, Nvini, NifD1, NifD2, NifN1, NifN2]
    NL_sqrt = [np.sqrt(n) if n > 0 else 1.0 for n in NL]

    func_params = dict(model.named_parameters())
    p_vec_old = get_p_vec(func_params).double().to(device)
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
    data_test = generate_test_data(num_test, device)
    try:
        while (lossval[-1] > tol_main) and (step <= tr_iter_max):
            torch.cuda.empty_cache()
            if Comput_old:  # need to compute loss_old and J_olds
                ### computation of loss
                Lup = torch.vmap(res_omega_p_u, (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[1], data_op[3], data_op[4]).flatten().detach()
                Lun = torch.vmap(res_omega_n_u, (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[1], data_on[3], data_on[4]).flatten().detach()
                Lvp = torch.vmap(res_omega_p_v, (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[2], data_op[3], data_op[4]).flatten().detach()
                Lvn = torch.vmap(res_omega_n_v, (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[2], data_on[3], data_on[4]).flatten().detach()
                Ldivp = torch.vmap(res_div_p, (None, 0))(func_params, data_op[0]).flatten().detach()
                Ldivn = torch.vmap(res_div_n, (None, 0))(func_params, data_on[0]).flatten().detach()
                Lub = torch.vmap(res_boundary_u, (None, 0, 0))(func_params, data_b[0], data_b[1]).flatten().detach()
                Lvb = torch.vmap(res_boundary_v, (None, 0, 0))(func_params, data_b[0], data_b[2]).flatten().detach()
                Luini = torch.vmap(res_initial_u, (None, 0, 0))(func_params, data_ini[0], data_ini[1]).flatten().detach()
                Lvini = torch.vmap(res_initial_v, (None, 0, 0))(func_params, data_ini[0], data_ini[2]).flatten().detach()
                LifDu = torch.vmap(res_interface_D_u, (None, 0, 0, 0))(func_params, data_if[0], data_if[1], data_if[3]).flatten().detach()
                LifDv = torch.vmap(res_interface_D_v, (None, 0, 0, 0))(func_params, data_if[0], data_if[1], data_if[4]).flatten().detach()
                LifN1 = torch.vmap(res_interface_N_1, (None, 0, 0, 0, 0))(func_params, data_if[0], data_if[1], data_if[2], data_if[5]).flatten().detach()
                LifN2 = torch.vmap(res_interface_N_2, (None, 0, 0, 0, 0))(func_params, data_if[0], data_if[1], data_if[2], data_if[6]).flatten().detach()

                L = torch.cat([
                    Lup / NL_sqrt[1],
                    Lun / NL_sqrt[2],
                    Lvp / NL_sqrt[3],
                    Lvn / NL_sqrt[4],
                    Ldivp / NL_sqrt[5],
                    Ldivn / NL_sqrt[6],
                    Lub / NL_sqrt[7],
                    Lvb / NL_sqrt[8],
                    Luini / NL_sqrt[9],
                    Lvini / NL_sqrt[10],
                    LifDu / NL_sqrt[11],
                    LifDv / NL_sqrt[12],
                    LifN1 / NL_sqrt[13],
                    LifN2 / NL_sqrt[14]
                ]).reshape(-1, 1).detach()

                loss_dbg_old = [
                    (Lup ** 2).mean().item(),
                    (Lun ** 2).mean().item(),
                    (Lvp ** 2).mean().item(),
                    (Lvn ** 2).mean().item(),
                    (Lub ** 2).mean().item(),
                    (Lvb ** 2).mean().item(),
                    (Luini ** 2).mean().item(),
                    (Lvini ** 2).mean().item(),
                    (LifDu ** 2).mean().item(),
                    (LifDv ** 2).mean().item(),
                    (LifN1 ** 2).mean().item(),
                    (LifN2 ** 2).mean().item()
                ]

            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]
            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = get_p_vec(func_params).detach()  # get p_vec for p_vec_old if neccessary

            if criterion:
                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_p_u), (None, 0, 0, 0, 0))(func_params,
                                                                                                    data_op[0],
                                                                                                    data_op[1],
                                                                                                    data_op[3],
                                                                                                    data_op[4])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Jup = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Jup, g.reshape(len(g), -1)])
                    cnt = 1
                Jup = Jup / NL_sqrt[1]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_n_u), (None, 0, 0, 0, 0))(func_params,
                                                                                                    data_on[0],
                                                                                                    data_on[1],
                                                                                                    data_on[3],
                                                                                                    data_on[4])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Jun = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Jun, g.reshape(len(g), -1)])
                    cnt = 1
                Jun = Jun / NL_sqrt[2]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_p_v), (None, 0, 0, 0, 0))(func_params,
                                                                                                    data_op[0],
                                                                                                    data_op[2],
                                                                                                    data_op[3],
                                                                                                    data_op[4])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Jvp = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Jvp, g.reshape(len(g), -1)])
                    cnt = 1
                Jvp = Jvp / NL_sqrt[3]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_omega_n_v), (None, 0, 0, 0, 0))(func_params,
                                                                                                    data_on[0],
                                                                                                    data_on[2],
                                                                                                    data_on[3],
                                                                                                    data_on[4])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Jvn = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Jvn, g.reshape(len(g), -1)])
                    cnt = 1
                Jvn = Jvn / NL_sqrt[4]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_div_p), (None, 0))(func_params, data_op[0])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Jdivp = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Jdivp, g.reshape(len(g), -1)])
                    cnt = 1
                Jdivp = Jdivp / NL_sqrt[5]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_div_n), (None, 0))(func_params, data_on[0])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Jdivn = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Jdivn, g.reshape(len(g), -1)])
                    cnt = 1
                Jdivn = Jdivn / NL_sqrt[6]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_boundary_u), (None, 0, 0))(func_params, data_b[0],
                                                                                               data_b[1])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Jub = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Jub, g.reshape(len(g), -1)])
                    cnt = 1
                Jub = Jub / NL_sqrt[7]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_boundary_v), (None, 0, 0))(func_params, data_b[0],
                                                                                               data_b[2])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Jvb = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Jvb, g.reshape(len(g), -1)])
                    cnt = 1
                Jvb = Jvb / NL_sqrt[8]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_initial_u), (None, 0, 0))(func_params, data_ini[0],
                                                                                              data_ini[1])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Juini = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Juini, g.reshape(len(g), -1)])
                    cnt = 1
                Juini = Juini / NL_sqrt[9]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_initial_v), (None, 0, 0))(func_params, data_ini[0],
                                                                                              data_ini[2])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    Jvini = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([Jvini, g.reshape(len(g), -1)])
                    cnt = 1
                Jvini = Jvini / NL_sqrt[10]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_D_u), (None, 0, 0, 0))(func_params,
                                                                                                     data_if[0],
                                                                                                     data_if[1],
                                                                                                     data_if[3])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    JifDu = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([JifDu, g.reshape(len(g), -1)])
                    cnt = 1
                JifDu = JifDu / NL_sqrt[11]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_D_v), (None, 0, 0, 0))(func_params,
                                                                                                     data_if[0],
                                                                                                     data_if[1],
                                                                                                     data_if[4])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    JifDv = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([JifDv, g.reshape(len(g), -1)])
                    cnt = 1
                JifDv = JifDv / NL_sqrt[12]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_N_1), (None, 0, 0, 0, 0))(func_params,
                                                                                                        data_if[0],
                                                                                                        data_if[1],
                                                                                                        data_if[2],
                                                                                                        data_if[5])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    JifN1 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([JifN1, g.reshape(len(g), -1)])
                    cnt = 1
                JifN1 = JifN1 / NL_sqrt[13]

                per_sample_grads = torch.vmap(torch.func.jacrev(res_interface_N_2), (None, 0, 0, 0, 0))(func_params,
                                                                                                        data_if[0],
                                                                                                        data_if[1],
                                                                                                        data_if[2],
                                                                                                        data_if[6])
                cnt = 0
                for g in per_sample_grads:
                    g = per_sample_grads[g].detach()
                    JifN2 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([JifN2, g.reshape(len(g), -1)])
                    cnt = 1
                JifN2 = JifN2 / NL_sqrt[14]

                J = torch.cat([Jup, Jun, Jvp, Jvn, Jdivp, Jdivn, Jub, Jvb, Juini, Jvini, JifDu, JifDv, JifN1, JifN2], dim=0).detach()
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
            Lup = torch.vmap(res_omega_p_u, (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[1], data_op[3],
                                                                data_op[4]).flatten().detach()
            Lun = torch.vmap(res_omega_n_u, (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[1], data_on[3],
                                                                data_on[4]).flatten().detach()
            Lvp = torch.vmap(res_omega_p_v, (None, 0, 0, 0, 0))(func_params, data_op[0], data_op[2], data_op[3],
                                                                data_op[4]).flatten().detach()
            Lvn = torch.vmap(res_omega_n_v, (None, 0, 0, 0, 0))(func_params, data_on[0], data_on[2], data_on[3],
                                                                data_on[4]).flatten().detach()
            Ldivp = torch.vmap(res_div_p, (None, 0))(func_params, data_op[0]).flatten().detach()
            Ldivn = torch.vmap(res_div_n, (None, 0))(func_params, data_on[0]).flatten().detach()
            Lub = torch.vmap(res_boundary_u, (None, 0, 0))(func_params, data_b[0], data_b[1]).flatten().detach()
            Lvb = torch.vmap(res_boundary_v, (None, 0, 0))(func_params, data_b[0], data_b[2]).flatten().detach()
            Luini = torch.vmap(res_initial_u, (None, 0, 0))(func_params, data_ini[0], data_ini[1]).flatten().detach()
            Lvini = torch.vmap(res_initial_v, (None, 0, 0))(func_params, data_ini[0], data_ini[2]).flatten().detach()
            LifDu = torch.vmap(res_interface_D_u, (None, 0, 0, 0))(func_params, data_if[0], data_if[1],
                                                                   data_if[3]).flatten().detach()
            LifDv = torch.vmap(res_interface_D_v, (None, 0, 0, 0))(func_params, data_if[0], data_if[1],
                                                                   data_if[4]).flatten().detach()
            LifN1 = torch.vmap(res_interface_N_1, (None, 0, 0, 0, 0))(func_params, data_if[0], data_if[1], data_if[2],
                                                                      data_if[5]).flatten().detach()
            LifN2 = torch.vmap(res_interface_N_2, (None, 0, 0, 0, 0))(func_params, data_if[0], data_if[1], data_if[2],
                                                                      data_if[6]).flatten().detach()

            L = torch.cat([
                Lup / NL_sqrt[1],
                Lun / NL_sqrt[2],
                Lvp / NL_sqrt[3],
                Lvn / NL_sqrt[4],
                Ldivp / NL_sqrt[5],
                Ldivn / NL_sqrt[6],
                Lub / NL_sqrt[7],
                Lvb / NL_sqrt[8],
                Luini / NL_sqrt[9],
                Lvini / NL_sqrt[10],
                LifDu / NL_sqrt[11],
                LifDv / NL_sqrt[12],
                LifN1 / NL_sqrt[13],
                LifN2 / NL_sqrt[14]
            ]).reshape(-1, 1).detach()

            loss_new = torch.sum(L * L).item()
            loss_dbg_new = [
                    (Lup ** 2).mean().item(),
                    (Lun ** 2).mean().item(),
                    (Lvp ** 2).mean().item(),
                    (Lvn ** 2).mean().item(),
                    (Lub ** 2).mean().item(),
                    (Lvb ** 2).mean().item(),
                    (Luini ** 2).mean().item(),
                    (Lvini ** 2).mean().item(),
                    (LifDu ** 2).mean().item(),
                    (LifDv ** 2).mean().item(),
                    (LifN1 ** 2).mean().item(),
                    (LifN2 ** 2).mean().item()
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
                print(f" relative error: {relative_error[-1]:.4e}")
            step += 1
            error = error_compute(model, func_params, data_test)
            relative_error.append(error.item())

        print("Step %s: " % (step - 1))
        print(f" training loss: {lossval[-1]:.4e}")
        print('finished')
        torch.save(model.state_dict(), 'best_model_XIPINN_LM.mdl')
        return model, lossval, lossval_dbg, relative_error


    except KeyboardInterrupt:
        print('Interrupt')
        print('steps = ', step)
        torch.save(model.state_dict(), 'best_model_XIPINN_LM.mdl')
        return model, lossval, lossval_dbg, relative_error

