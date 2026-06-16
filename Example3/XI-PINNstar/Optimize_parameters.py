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
    def u_NN(data, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)
    u_numerical = u_NN(data[0], func_params)
    relative_l2 = torch.norm(u_numerical[:, 0][:, None] - data[1]) / torch.norm(data[1]) + torch.norm(u_numerical[:, 1][:, None] - data[2]) / torch.norm(data[2])
    return relative_l2


def optimize_parameters_SOAP(data_op, data_on, data_b, data_ini, data_if, nu_p, nu_n, V_model, P_model, epochs, lr, lossval, lossval_dbg, relative_error, device):
    def loss_omega_p(data):
        V_output = V_model(data[0])
        P_output = P_model(data[1])
        u = V_output[:, 0][:, None]
        v = V_output[:, 1][:, None]
        p = P_output
        phix = data[6]
        phiy = data[7]
        phit = data[8]
        phix2 = data[9]
        phiy2 = data[10]

        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudt = gradu[0][:, 2][:, None]
        dudz = gradu[0][:, 3][:, None]
        gradux = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(dudx), create_graph=True, retain_graph=True)
        d2udx2 = gradux[0][:, 0][:, None]
        d2udxdz = gradux[0][:, 3][:, None]
        graduy = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(dudy), create_graph=True, retain_graph=True)
        d2udy2 = graduy[0][:, 1][:, None]
        d2udydz = graduy[0][:, 3][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(dudz), create_graph=True, retain_graph=True)[0][:, 3][:, None]

        gradv = autograd.grad(outputs=v, inputs=data[0], grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)
        dvdx = gradv[0][:, 0][:, None]
        dvdy = gradv[0][:, 1][:, None]
        dvdt = gradv[0][:, 2][:, None]
        dvdz = gradv[0][:, 3][:, None]
        gradvx = autograd.grad(outputs=dvdx, inputs=data[0], grad_outputs=torch.ones_like(dvdx), create_graph=True, retain_graph=True)
        d2vdx2 = gradvx[0][:, 0][:, None]
        d2vdxdz = gradvx[0][:, 3][:, None]
        gradvy = autograd.grad(outputs=dvdy, inputs=data[0], grad_outputs=torch.ones_like(dvdy), create_graph=True, retain_graph=True)
        d2vdy2 = gradvy[0][:, 1][:, None]
        d2vdydz = gradvy[0][:, 3][:, None]
        d2vdz2 = autograd.grad(outputs=dvdz, inputs=data[0], grad_outputs=torch.ones_like(dvdz), create_graph=True, retain_graph=True)[0][:, 3][:, None]

        gradp = autograd.grad(outputs=p, inputs=data[1], grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)
        dpdx = gradp[0][:, 0][:, None]
        dpdy = gradp[0][:, 1][:, None]

        f_1 = data[2]
        f_2 = data[3]
        w_1 = data[4]
        w_2 = data[5]

        Ut = dudt + dudz * phit
        Ux = dudx + dudz * phix
        Uy = dudy + dudz * phiy
        Uxx = d2udx2 + 2 * phix * d2udxdz + phix * phix * d2udz2 + phix2 * dudz
        Uyy = d2udy2 + 2 * phiy * d2udydz + phiy * phiy * d2udz2 + phiy2 * dudz

        Vt = dvdt + dvdz * phit
        Vx = dvdx + dvdz * phix
        Vy = dvdy + dvdz * phiy
        Vxx = d2vdx2 + 2 * phix * d2vdxdz + phix * phix * d2vdz2 + phix2 * dvdz
        Vyy = d2vdy2 + 2 * phiy * d2vdydz + phiy * phiy * d2vdz2 + phiy2 * dvdz

        res1 = Ut + w_1 * Ux + w_2 * Uy - nu_p * (Uxx + Uyy) + dpdx
        res2 = Vt + w_1 * Vx + w_2 * Vy - nu_p * (Vxx + Vyy) + dpdy

        loss1 = nn.MSELoss()(res1, f_1)
        loss2 = nn.MSELoss()(res2, f_2)
        return loss1, loss2

    def loss_omega_n(data):
        V_output = V_model(data[0])
        P_output = P_model(data[1])
        u = V_output[:, 0][:, None]
        v = V_output[:, 1][:, None]
        p = P_output
        phix = -data[6]
        phiy = -data[7]
        phit = -data[8]
        phix2 = -data[9]
        phiy2 = -data[10]

        gradu = autograd.grad(outputs=u, inputs=data[0], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudt = gradu[0][:, 2][:, None]
        dudz = gradu[0][:, 3][:, None]
        gradux = autograd.grad(outputs=dudx, inputs=data[0], grad_outputs=torch.ones_like(dudx), create_graph=True, retain_graph=True)
        d2udx2 = gradux[0][:, 0][:, None]
        d2udxdz = gradux[0][:, 3][:, None]
        graduy = autograd.grad(outputs=dudy, inputs=data[0], grad_outputs=torch.ones_like(dudy), create_graph=True, retain_graph=True)
        d2udy2 = graduy[0][:, 1][:, None]
        d2udydz = graduy[0][:, 3][:, None]
        d2udz2 = autograd.grad(outputs=dudz, inputs=data[0], grad_outputs=torch.ones_like(dudz), create_graph=True, retain_graph=True)[0][:, 3][:, None]

        gradv = autograd.grad(outputs=v, inputs=data[0], grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)
        dvdx = gradv[0][:, 0][:, None]
        dvdy = gradv[0][:, 1][:, None]
        dvdt = gradv[0][:, 2][:, None]
        dvdz = gradv[0][:, 3][:, None]
        gradvx = autograd.grad(outputs=dvdx, inputs=data[0], grad_outputs=torch.ones_like(dvdx), create_graph=True, retain_graph=True)
        d2vdx2 = gradvx[0][:, 0][:, None]
        d2vdxdz = gradvx[0][:, 3][:, None]
        gradvy = autograd.grad(outputs=dvdy, inputs=data[0], grad_outputs=torch.ones_like(dvdy), create_graph=True, retain_graph=True)
        d2vdy2 = gradvy[0][:, 1][:, None]
        d2vdydz = gradvy[0][:, 3][:, None]
        d2vdz2 = autograd.grad(outputs=dvdz, inputs=data[0], grad_outputs=torch.ones_like(dvdz), create_graph=True, retain_graph=True)[0][:, 3][:, None]

        gradp = autograd.grad(outputs=p, inputs=data[1], grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)
        dpdx = gradp[0][:, 0][:, None]
        dpdy = gradp[0][:, 1][:, None]

        f_1 = data[2]
        f_2 = data[3]
        w_1 = data[4]
        w_2 = data[5]

        Ut = dudt + dudz * phit
        Ux = dudx + dudz * phix
        Uy = dudy + dudz * phiy
        Uxx = d2udx2 + 2 * phix * d2udxdz + phix * phix * d2udz2 + phix2 * dudz
        Uyy = d2udy2 + 2 * phiy * d2udydz + phiy * phiy * d2udz2 + phiy2 * dudz

        Vt = dvdt + dvdz * phit
        Vx = dvdx + dvdz * phix
        Vy = dvdy + dvdz * phiy
        Vxx = d2vdx2 + 2 * phix * d2vdxdz + phix * phix * d2vdz2 + phix2 * dvdz
        Vyy = d2vdy2 + 2 * phiy * d2vdydz + phiy * phiy * d2vdz2 + phiy2 * dvdz

        res1 = Ut + w_1 * Ux + w_2 * Uy - nu_n * (Uxx + Uyy) + dpdx
        res2 = Vt + w_1 * Vx + w_2 * Vy - nu_n * (Vxx + Vyy) + dpdy

        loss1 = nn.MSELoss()(res1, f_1)
        loss2 = nn.MSELoss()(res2, f_2)
        return loss1, loss2


    def loss_div(data_p, data_n):
        V_p_output = V_model(data_p[0])
        phi_px = data_p[6]
        phi_py = data_p[7]
        phi_nx = -data_n[6]
        phi_ny = -data_n[7]
        u_p = V_p_output[:, 0][:, None]
        v_p = V_p_output[:, 1][:, None]
        du_pdx = autograd.grad(outputs=u_p, inputs=data_p[0], grad_outputs=torch.ones_like(u_p), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        du_pdz = autograd.grad(outputs=u_p, inputs=data_p[0], grad_outputs=torch.ones_like(u_p), create_graph=True, retain_graph=True)[0][:, 3][:, None]
        dv_pdy = autograd.grad(outputs=v_p, inputs=data_p[0], grad_outputs=torch.ones_like(v_p), create_graph=True, retain_graph=True)[0][:, 1][:, None]
        dv_pdz = autograd.grad(outputs=v_p, inputs=data_p[0], grad_outputs=torch.ones_like(v_p), create_graph=True, retain_graph=True)[0][:, 3][:, None]
        U_px = du_pdx + du_pdz * phi_px
        V_py = dv_pdy + dv_pdz * phi_py

        V_n_output = V_model(data_n[0])
        u_n = V_n_output[:, 0][:, None]
        v_n = V_n_output[:, 1][:, None]
        du_ndx = autograd.grad(outputs=u_n, inputs=data_n[0], grad_outputs=torch.ones_like(u_n), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        du_ndz = autograd.grad(outputs=u_n, inputs=data_n[0], grad_outputs=torch.ones_like(u_n), create_graph=True, retain_graph=True)[0][:, 3][:, None]
        dv_ndy = autograd.grad(outputs=v_n, inputs=data_n[0], grad_outputs=torch.ones_like(v_n), create_graph=True, retain_graph=True)[0][:, 1][:, None]
        dv_ndz = autograd.grad(outputs=v_n, inputs=data_n[0], grad_outputs=torch.ones_like(v_n), create_graph=True, retain_graph=True)[0][:, 3][:, None]
        U_nx = du_ndx + du_ndz * phi_nx
        V_ny = dv_ndy + dv_ndz * phi_ny

        res1 = U_px + V_py
        res2 = U_nx + V_ny

        loss1 = nn.MSELoss()(res1, torch.zeros_like(res1))
        loss2 = nn.MSELoss()(res2, torch.zeros_like(res2))
        return loss1, loss2

    def loss_boundary(data):
        V_output = V_model(data[0])
        u = V_output[:, 0][:, None]
        v = V_output[:, 1][:, None]
        g1 = data[1]
        g2 = data[2]
        loss1 = nn.MSELoss()(u, g1)
        loss2 = nn.MSELoss()(v, g2)
        return loss1, loss2

    def loss_initial(data):
        V_output = V_model(data[0])
        u = V_output[:, 0][:, None]
        v = V_output[:, 1][:, None]
        u01 = data[1]
        u02 = data[2]
        loss1 = nn.MSELoss()(u, u01)
        loss2 = nn.MSELoss()(v, u02)
        return loss1, loss2

    def loss_interface(data):
        x_lf = data[0]
        x_sign_p = data[1]
        x_sign_n = data[2]
        nor = data[3]
        psi1 = data[4]
        psi2 = data[5]
        phi_px = data[6]
        phi_py = data[7]
        phi_nx = -data[6]
        phi_ny = -data[7]

        V_output = V_model(x_lf)
        p_p = P_model(x_sign_p)
        p_n = P_model(x_sign_n)
        u = V_output[:, 0][:, None]
        v = V_output[:, 1][:, None]

        gradu = autograd.grad(outputs=u, inputs=x_lf, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
        dudx = gradu[0][:, 0][:, None]
        dudy = gradu[0][:, 1][:, None]
        dudz = gradu[0][:, 3][:, None]

        gradv = autograd.grad(outputs=v, inputs=x_lf, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)
        dvdx = gradv[0][:, 0][:, None]
        dvdy = gradv[0][:, 1][:, None]
        dvdz = gradv[0][:, 3][:, None]

        U_px = dudx + dudz * phi_px
        U_py = dudy + dudz * phi_py
        V_px = dvdx + dvdz * phi_px
        V_py = dvdy + dvdz * phi_py

        U_nx = dudx + dudz * phi_nx
        U_ny = dudy + dudz * phi_ny
        V_nx = dvdx + dvdz * phi_nx
        V_ny = dvdy + dvdz * phi_ny

        n1 = nor[:, 0][:, None]
        n2 = nor[:, 1][:, None]

        sigma_p1 = -p_p * n1 + nu_p * (2 * U_px * n1 + (U_py + V_px) * n2)
        sigma_n1 = -p_n * n1 + nu_n * (2 * U_nx * n1 + (U_ny + V_nx) * n2)
        res1 = sigma_p1 - sigma_n1

        sigma_p2 = -p_p * n2 + nu_p * ((U_py + V_px) * n1 + 2 * V_py * n2)
        sigma_n2 = -p_n * n2 + nu_n * ((U_ny + V_nx) * n1 + 2 * V_ny * n2)
        res2 = sigma_p2 - sigma_n2

        loss1 = nn.MSELoss()(res1, psi1)
        loss2 = nn.MSELoss()(res2, psi2)
        return loss1, loss2

    params = list(V_model.parameters()) + list(P_model.parameters())
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
        loss11, loss12 = loss_interface(data_if)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            print('epoch:', epoch)
            print('loss:', loss.item(), 'loss1:', loss1.item(), 'loss2:', loss2.item(), 'loss3:', loss3.item(), 'loss4:', loss4.item())
            print('loss5:', loss5.item(), 'loss6:', loss6.item(), 'loss7:', loss7.item(), 'loss8:', loss8.item(), 'loss9:', loss9.item())
            print('loss10:', loss10.item(), 'loss11:', loss11.item(), 'loss12:', loss12.item())

            func_params = dict(V_model.named_parameters())
            error = error_compute(V_model, func_params, data_test)
            relative_error.append(error.item())
            print(f"relative error: {relative_error[-1]:.4e}")
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_epoch = epoch
                best_state_dict_V = copy.deepcopy(V_model.state_dict())
                best_state_dict_P = copy.deepcopy(P_model.state_dict())
            lossval.append(loss.item())
            lossval_dbg.append([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()])

    print('best epoch:', best_epoch, 'best loss:', best_loss)
    V_model.load_state_dict(best_state_dict_V)
    P_model.load_state_dict(best_state_dict_P)
    torch.save(V_model.state_dict(), 'best_V_model_XIPINN_SOAP.mdl')
    torch.save(P_model.state_dict(), 'best_P_model_XIPINN_SOAP.mdl')
    return V_model, P_model, lossval, lossval_dbg, relative_error



def optimize_parameters_LM(data_op, data_on, data_b, data_ini, data_if, nu_p, nu_n, V_model, P_model, tr_iter_max, mu, lossval, lossval_dbg, relative_error, mu_div, mu_mul, device):
    def get_all_params_vec(V_model, P_model):
        V_params = torch.cat([p.contiguous().view(-1) for p in V_model.parameters()])
        P_params = torch.cat([p.contiguous().view(-1) for p in P_model.parameters()])
        return torch.cat([V_params, P_params]), V_params.numel(), P_params.numel()

    all_params, num_V, num_P = get_all_params_vec(V_model, P_model)

    unflatten_V = []
    offset = 0
    for name, p in V_model.named_parameters():
        n = p.numel()
        unflatten_V.append((name, p.shape, n, offset))
        offset += n

    unflatten_P = []
    offset = 0
    for name, p in P_model.named_parameters():
        n = p.numel()
        unflatten_P.append((name, p.shape, n, offset))
        offset += n

    def vec_to_params(vec):
        vec_p = vec[:num_V]
        Vd = {}
        for name, shape, n, st in unflatten_V:
            end = st + n
            Vd[name] = vec_p[st:end].reshape(shape)
        vec_n = vec[num_V:]
        Pd = {}
        for name, shape, n, st in unflatten_P:
            end = st + n
            Pd[name] = vec_n[st:end].reshape(shape)
        return Vd, Pd

    def u_NN(data, func_params, V_model):
        u = torch.func.functional_call(V_model, func_params, data)
        return u.squeeze(0).squeeze(0)[0]

    def v_NN(data, func_params, V_model):
        v = torch.func.functional_call(V_model, func_params, data)
        return v.squeeze(0).squeeze(0)[1]

    def p_NN(data, func_params, P_model):
        u = torch.func.functional_call(P_model, func_params, data)
        return u.squeeze(0).squeeze(0)

    def res_omega_p_u(vec, data_lf, data_sign, f, w_1, w_2, phix, phiy, phit, phix2, phiy2):
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        phix = phix[0]
        phiy = phiy[0]
        phit = phit[0]
        phix2 = phix2[0]
        phiy2 = phiy2[0]

        V_params, P_params = vec_to_params(vec)
        gradu = torch.func.jacrev(u_NN, argnums=0)(data_lf, V_params, V_model)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data_lf, V_params, V_model)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        dudz = gradu[3]
        d2udx2 = grad2u[0][0]
        d2udxdz = grad2u[0][3]
        d2udy2 = grad2u[1][1]
        d2udydz = grad2u[1][3]
        d2udz2 = grad2u[3][3]

        dpdx = torch.func.jacrev(p_NN, argnums=0)(data_sign, P_params, P_model)[0]

        Ut = dudt + dudz * phit
        Ux = dudx + phix * dudz
        Uy = dudy + phiy * dudz
        Uxx = d2udx2 + 2 * phix * d2udxdz + phix * phix * d2udz2 + phix2 * dudz
        Uyy = d2udy2 + 2 * phiy * d2udydz + phiy * phiy * d2udz2 + phiy2 * dudz
        res = Ut + w_1 * Ux + w_2 * Uy - nu_p * (Uxx + Uyy) + dpdx - f
        return res

    def res_omega_n_u(vec, data_lf, data_sign, f, w_1, w_2, phix, phiy, phit, phix2, phiy2):
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        phix = -phix[0]
        phiy = -phiy[0]
        phit = -phit[0]
        phix2 = -phix2[0]
        phiy2 = -phiy2[0]

        V_params, P_params = vec_to_params(vec)
        gradu = torch.func.jacrev(u_NN, argnums=0)(data_lf, V_params, V_model)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_NN, argnums=0), argnums=0)(data_lf, V_params, V_model)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        dudz = gradu[3]
        d2udx2 = grad2u[0][0]
        d2udxdz = grad2u[0][3]
        d2udy2 = grad2u[1][1]
        d2udydz = grad2u[1][3]
        d2udz2 = grad2u[3][3]

        dpdx = torch.func.jacrev(p_NN, argnums=0)(data_sign, P_params, P_model)[0]

        Ut = dudt + dudz * phit
        Ux = dudx + phix * dudz
        Uy = dudy + phiy * dudz
        Uxx = d2udx2 + 2 * phix * d2udxdz + phix * phix * d2udz2 + phix2 * dudz
        Uyy = d2udy2 + 2 * phiy * d2udydz + phiy * phiy * d2udz2 + phiy2 * dudz
        res = Ut + w_1 * Ux + w_2 * Uy - nu_n * (Uxx + Uyy) + dpdx - f
        return res


    def res_omega_p_v(vec, data_lf, data_sign, f, w_1, w_2, phix, phiy, phit, phix2, phiy2):
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        phix = phix[0]
        phiy = phiy[0]
        phit = phit[0]
        phix2 = phix2[0]
        phiy2 = phiy2[0]

        V_params, P_params = vec_to_params(vec)
        gradv = torch.func.jacrev(v_NN, argnums=0)(data_lf, V_params, V_model)
        grad2v = torch.func.jacrev(torch.func.jacrev(v_NN, argnums=0), argnums=0)(data_lf, V_params, V_model)
        dvdx = gradv[0]
        dvdy = gradv[1]
        dvdt = gradv[2]
        dvdz = gradv[3]
        d2vdx2 = grad2v[0][0]
        d2vdxdz = grad2v[0][3]
        d2vdy2 = grad2v[1][1]
        d2vdydz = grad2v[1][3]
        d2vdz2 = grad2v[3][3]

        dpdy = torch.func.jacrev(p_NN, argnums=0)(data_sign, P_params, P_model)[1]

        Vt = dvdt + dvdz * phit
        Vx = dvdx + phix * dvdz
        Vy = dvdy + phiy * dvdz
        Vxx = d2vdx2 + 2 * phix * d2vdxdz + phix * phix * d2vdz2 + phix2 * dvdz
        Vyy = d2vdy2 + 2 * phiy * d2vdydz + phiy * phiy * d2vdz2 + phiy2 * dvdz
        res = Vt + w_1 * Vx + w_2 * Vy - nu_p * (Vxx + Vyy) + dpdy - f
        return res


    def res_omega_n_v(vec, data_lf, data_sign, f, w_1, w_2, phix, phiy, phit, phix2, phiy2):
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        phix = -phix[0]
        phiy = -phiy[0]
        phit = -phit[0]
        phix2 = -phix2[0]
        phiy2 = -phiy2[0]

        V_params, P_params = vec_to_params(vec)
        gradv = torch.func.jacrev(v_NN, argnums=0)(data_lf, V_params, V_model)
        grad2v = torch.func.jacrev(torch.func.jacrev(v_NN, argnums=0), argnums=0)(data_lf, V_params, V_model)
        dvdx = gradv[0]
        dvdy = gradv[1]
        dvdt = gradv[2]
        dvdz = gradv[3]
        d2vdx2 = grad2v[0][0]
        d2vdxdz = grad2v[0][3]
        d2vdy2 = grad2v[1][1]
        d2vdydz = grad2v[1][3]
        d2vdz2 = grad2v[3][3]

        dpdy = torch.func.jacrev(p_NN, argnums=0)(data_sign, P_params, P_model)[1]

        Vt = dvdt + dvdz * phit
        Vx = dvdx + phix * dvdz
        Vy = dvdy + phiy * dvdz
        Vxx = d2vdx2 + 2 * phix * d2vdxdz + phix * phix * d2vdz2 + phix2 * dvdz
        Vyy = d2vdy2 + 2 * phiy * d2vdydz + phiy * phiy * d2vdz2 + phiy2 * dvdz
        res = Vt + w_1 * Vx + w_2 * Vy - nu_n * (Vxx + Vyy) + dpdy - f
        return res


    def res_div_p(vec, data_lf, phix, phiy):
        phix = phix[0]
        phiy = phiy[0]
        V_params, _ = vec_to_params(vec)
        gradu = torch.func.jacrev(u_NN, argnums=0)(data_lf, V_params, V_model)
        gradv = torch.func.jacrev(v_NN, argnums=0)(data_lf, V_params, V_model)
        dudx = gradu[0]
        dudz = gradu[3]
        dvdy = gradv[1]
        dvdz = gradv[3]
        Ux = dudx + phix * dudz
        Vy = dvdy + phiy * dvdz
        res = Ux + Vy
        return res

    def res_div_n(vec, data_lf, phix, phiy):
        phix = -phix[0]
        phiy = -phiy[0]
        V_params, _ = vec_to_params(vec)
        gradu = torch.func.jacrev(u_NN, argnums=0)(data_lf, V_params, V_model)
        gradv = torch.func.jacrev(v_NN, argnums=0)(data_lf, V_params, V_model)
        dudx = gradu[0]
        dudz = gradu[3]
        dvdy = gradv[1]
        dvdz = gradv[3]
        Ux = dudx + phix * dudz
        Vy = dvdy + phiy * dvdz
        res = Ux + Vy
        return res


    def res_boundary_u(vec, data_lf, gu):
        g = gu[0]
        V_params, _ = vec_to_params(vec)
        u = u_NN(data_lf, V_params, V_model)
        res = u - g
        return res

    def res_boundary_v(vec, data_lf, gv):
        g = gv[0]
        V_params, _ = vec_to_params(vec)
        v = v_NN(data_lf, V_params, V_model)
        res = v - g
        return res


    def res_initial_u(vec, data_lf, u0):
        V_params, _ = vec_to_params(vec)
        u = u_NN(data_lf, V_params, V_model)
        u0 = u0[0]
        res = u - u0
        return res

    def res_initial_v(vec, data_lf, v0):
        V_params, _ = vec_to_params(vec)
        v = v_NN(data_lf, V_params, V_model)
        v0 = v0[0]
        res = v - v0
        return res

    def res_interface_psi_1(vec, data_lf, data_sign_p, data_sign_n, nor, psi, phix, phiy):
        V_params, P_params = vec_to_params(vec)
        gradu = torch.func.jacrev(u_NN, argnums=0)(data_lf, V_params, V_model)
        dudx = gradu[0]
        dudy = gradu[1]
        dudz = gradu[3]

        gradv = torch.func.jacrev(v_NN, argnums=0)(data_lf, V_params, V_model)
        dvdx = gradv[0]
        dvdz = gradv[3]

        p_p = p_NN(data_sign_p, P_params, P_model)
        p_n = p_NN(data_sign_n, P_params, P_model)

        phi_px = phix[0]
        phi_py = phiy[0]
        phi_nx = -phix[0]
        phi_ny = -phiy[0]

        U_px = dudx + dudz * phi_px
        U_py = dudy + dudz * phi_py
        V_px = dvdx + dvdz * phi_px

        U_nx = dudx + dudz * phi_nx
        U_ny = dudy + dudz * phi_ny
        V_nx = dvdx + dvdz * phi_nx

        n1 = nor[0]
        n2 = nor[1]
        psi = psi[0]

        sigma_p1 = -p_p * n1 + nu_p * (2 * U_px * n1 + (U_py + V_px) * n2)
        sigma_n1 = -p_n * n1 + nu_n * (2 * U_nx * n1 + (U_ny + V_nx) * n2)
        res = sigma_p1 - sigma_n1 - psi
        return res


    def res_interface_psi_2(vec, data_lf, data_sign_p, data_sign_n, nor, psi, phix, phiy):
        V_params, P_params = vec_to_params(vec)
        gradu = torch.func.jacrev(u_NN, argnums=0)(data_lf, V_params, V_model)
        dudy = gradu[1]
        dudz = gradu[3]

        gradv = torch.func.jacrev(v_NN, argnums=0)(data_lf, V_params, V_model)
        dvdx = gradv[0]
        dvdy = gradv[1]
        dvdz = gradv[3]

        p_p = p_NN(data_sign_p, P_params, P_model)
        p_n = p_NN(data_sign_n, P_params, P_model)

        phi_px = phix[0]
        phi_py = phiy[0]
        phi_nx = -phix[0]
        phi_ny = -phiy[0]

        U_py = dudy + dudz * phi_py
        V_px = dvdx + dvdz * phi_px
        V_py = dvdy + dvdz * phi_py

        U_ny = dudy + dudz * phi_ny
        V_nx = dvdx + dvdz * phi_nx
        V_ny = dvdy + dvdz * phi_ny

        n1 = nor[0]
        n2 = nor[1]
        psi = psi[0]

        sigma_p2 = -p_p * n2 + nu_p * ((U_py + V_px) * n1 + 2 * V_py * n2)
        sigma_n2 = -p_n * n2 + nu_n * ((U_ny + V_nx) * n1 + 2 * V_ny * n2)
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
    Nif1 = len(data_if[0])
    Nif2 = len(data_if[0])
    total_res = Nup + Nun + Nvp + Nvn + Ndivp + Ndivn + Nub + Nvb + Nuini + Nvini + Nif1 + Nif2
    NL = [total_res, Nup, Nun, Nvp, Nvn, Ndivp, Ndivn, Nub, Nvb, Nuini, Nvini, Nif1, Nif2]
    NL_sqrt = [np.sqrt(n) if n > 0 else 1.0 for n in NL]

    all_params, _, _ = get_all_params_vec(V_model, P_model)
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
    data_test = generate_test_data(num_test, device)
    try:
        while (lossval[-1] > tol_main) and (step <= tr_iter_max):
            torch.cuda.empty_cache()
            if Comput_old:  # need to compute loss_old and J_olds
                ### computation of loss
                Lup = torch.vmap(res_omega_p_u, (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(all_params, data_op[0], data_op[1], data_op[2], data_op[4], data_op[5], data_op[6], data_op[7], data_op[8], data_op[9], data_op[10]).flatten().detach()
                Lun = torch.vmap(res_omega_n_u, (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(all_params, data_on[0], data_on[1], data_on[2], data_on[4], data_on[5], data_on[6], data_on[7], data_on[8], data_on[9], data_on[10]).flatten().detach()
                Lvp = torch.vmap(res_omega_p_v, (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(all_params, data_op[0], data_op[1], data_op[3], data_op[4], data_op[5], data_op[6], data_op[7], data_op[8], data_op[9], data_op[10]).flatten().detach()
                Lvn = torch.vmap(res_omega_n_v, (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(all_params, data_on[0], data_on[1], data_on[3], data_on[4], data_on[5], data_on[6], data_on[7], data_on[8], data_on[9], data_on[10]).flatten().detach()
                Ldivp = torch.vmap(res_div_p, (None, 0, 0, 0))(all_params, data_op[0], data_op[6], data_op[7]).flatten().detach()
                Ldivn = torch.vmap(res_div_n, (None, 0, 0, 0))(all_params, data_on[0], data_on[6], data_on[7]).flatten().detach()
                Lub = torch.vmap(res_boundary_u, (None, 0, 0))(all_params, data_b[0], data_b[1]).flatten().detach()
                Lvb = torch.vmap(res_boundary_v, (None, 0, 0))(all_params, data_b[0], data_b[2]).flatten().detach()
                Luini = torch.vmap(res_initial_u, (None, 0, 0))(all_params, data_ini[0], data_ini[1]).flatten().detach()
                Lvini = torch.vmap(res_initial_v, (None, 0, 0))(all_params, data_ini[0], data_ini[2]).flatten().detach()
                Lif1 = torch.vmap(res_interface_psi_1, (None, 0, 0, 0, 0, 0, 0, 0))(all_params, data_if[0], data_if[1], data_if[2], data_if[3], data_if[4], data_if[6], data_if[7]).flatten().detach()
                Lif2 = torch.vmap(res_interface_psi_2, (None, 0, 0, 0, 0, 0, 0, 0))(all_params, data_if[0], data_if[1], data_if[2], data_if[3], data_if[5], data_if[6], data_if[7]).flatten().detach()

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
                    Lif1 / NL_sqrt[11],
                    Lif2 / NL_sqrt[12]
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
                    (Lif1 ** 2).mean().item(),
                    (Lif2 ** 2).mean().item(),
                ]

            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]
            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = all_params.detach().clone()  # get p_vec for p_vec_old if neccessary

            if criterion:
                Jup = torch.vmap(torch.func.jacrev(res_omega_p_u, argnums=0), (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
                    all_params, data_op[0], data_op[1], data_op[2], data_op[4], data_op[5],
                    data_op[6], data_op[7], data_op[8], data_op[9], data_op[10]
                )
                Jup = Jup / NL_sqrt[1]

                Jun = torch.vmap(torch.func.jacrev(res_omega_n_u, argnums=0), (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
                    all_params, data_on[0], data_on[1], data_on[2], data_on[4], data_on[5],
                    data_on[6], data_on[7], data_on[8], data_on[9], data_on[10]
                )
                Jun = Jun / NL_sqrt[2]

                Jvp = torch.vmap(torch.func.jacrev(res_omega_p_v, argnums=0), (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
                    all_params, data_op[0], data_op[1], data_op[3], data_op[4], data_op[5],
                    data_op[6], data_op[7], data_op[8], data_op[9], data_op[10]
                )
                Jvp = Jvp / NL_sqrt[3]

                Jvn = torch.vmap(torch.func.jacrev(res_omega_n_v, argnums=0), (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
                    all_params, data_on[0], data_on[1], data_on[3], data_on[4], data_on[5],
                    data_on[6], data_on[7], data_on[8], data_on[9], data_on[10]
                )
                Jvn = Jvn / NL_sqrt[4]

                Jdivp = torch.vmap(torch.func.jacrev(res_div_p, argnums=0), (None, 0, 0, 0))(
                    all_params, data_op[0], data_op[6], data_op[7]
                )
                Jdivp = Jdivp / NL_sqrt[5]

                Jdivn = torch.vmap(torch.func.jacrev(res_div_n, argnums=0), (None, 0, 0, 0))(
                    all_params, data_on[0], data_on[6], data_on[7]
                )
                Jdivn = Jdivn / NL_sqrt[6]

                Jub = torch.vmap(torch.func.jacrev(res_boundary_u, argnums=0), (None, 0, 0))(
                    all_params, data_b[0], data_b[1]
                )
                Jub = Jub / NL_sqrt[7]

                Jvb = torch.vmap(torch.func.jacrev(res_boundary_v, argnums=0), (None, 0, 0))(
                    all_params, data_b[0], data_b[2]
                )
                Jvb = Jvb / NL_sqrt[8]

                Juini = torch.vmap(torch.func.jacrev(res_initial_u, argnums=0), (None, 0, 0))(
                    all_params, data_ini[0], data_ini[1]
                )
                Juini = Juini / NL_sqrt[9]

                Jvini = torch.vmap(torch.func.jacrev(res_initial_v, argnums=0), (None, 0, 0))(
                    all_params, data_ini[0], data_ini[2]
                )
                Jvini = Jvini / NL_sqrt[10]

                Jif1 = torch.vmap(torch.func.jacrev(res_interface_psi_1, argnums=0), (None, 0, 0, 0, 0, 0, 0, 0))(
                    all_params, data_if[0], data_if[1], data_if[2], data_if[3], data_if[4],
                    data_if[6], data_if[7]
                )
                Jif1 = Jif1 / NL_sqrt[11]

                Jif2 = torch.vmap(torch.func.jacrev(res_interface_psi_2, argnums=0), (None, 0, 0, 0, 0, 0, 0, 0))(
                    all_params, data_if[0], data_if[1], data_if[2], data_if[3], data_if[5],
                    data_if[6], data_if[7]
                )
                Jif2 = Jif2 / NL_sqrt[12]

                J = torch.cat([Jup, Jun, Jvp, Jvn, Jdivp, Jdivn, Jub, Jvb, Juini, Jvini, Jif1, Jif2], dim=0).detach()
                J_product = J.t() @ J
                rhs = - J.t() @ L

            with torch.no_grad():
                ### solve the linear system
                dp = torch.linalg.solve(J_product + mu * I_pvec, rhs)
                all_params += dp.view(-1)
                V_params_dict, P_params_dict = vec_to_params(all_params)
                V_model.load_state_dict(V_params_dict)
                P_model.load_state_dict(P_params_dict)

            ### Compute loss_new
            Lup = torch.vmap(res_omega_p_u, (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(all_params, data_op[0], data_op[1],
                                                                                  data_op[2], data_op[4], data_op[5],
                                                                                  data_op[6], data_op[7], data_op[8],
                                                                                  data_op[9], data_op[10]).flatten().detach()
            Lun = torch.vmap(res_omega_n_u, (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(all_params, data_on[0], data_on[1],
                                                                                  data_on[2], data_on[4], data_on[5],
                                                                                  data_on[6], data_on[7], data_on[8],
                                                                                  data_on[9], data_on[10]).flatten().detach()
            Lvp = torch.vmap(res_omega_p_v, (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(all_params, data_op[0], data_op[1],
                                                                                  data_op[3], data_op[4], data_op[5],
                                                                                  data_op[6], data_op[7], data_op[8],
                                                                                  data_op[9], data_op[10]).flatten().detach()
            Lvn = torch.vmap(res_omega_n_v, (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(all_params, data_on[0], data_on[1],
                                                                                  data_on[3], data_on[4], data_on[5],
                                                                                  data_on[6], data_on[7], data_on[8],
                                                                                  data_on[9], data_on[10]).flatten().detach()
            Ldivp = torch.vmap(res_div_p, (None, 0, 0, 0))(all_params, data_op[0], data_op[6], data_op[7]).flatten().detach()
            Ldivn = torch.vmap(res_div_n, (None, 0, 0, 0))(all_params, data_on[0], data_on[6], data_on[7]).flatten().detach()
            Lub = torch.vmap(res_boundary_u, (None, 0, 0))(all_params, data_b[0], data_b[1]).flatten().detach()
            Lvb = torch.vmap(res_boundary_v, (None, 0, 0))(all_params, data_b[0], data_b[2]).flatten().detach()
            Luini = torch.vmap(res_initial_u, (None, 0, 0))(all_params, data_ini[0], data_ini[1]).flatten().detach()
            Lvini = torch.vmap(res_initial_v, (None, 0, 0))(all_params, data_ini[0], data_ini[2]).flatten().detach()
            Lif1 = torch.vmap(res_interface_psi_1, (None, 0, 0, 0, 0, 0, 0, 0))(all_params, data_if[0], data_if[1], data_if[2],
                                                                                data_if[3], data_if[4], data_if[6],
                                                                                data_if[7]).flatten().detach()
            Lif2 = torch.vmap(res_interface_psi_2, (None, 0, 0, 0, 0, 0, 0, 0))(all_params, data_if[0], data_if[1], data_if[2],
                                                                                data_if[3], data_if[5], data_if[6],
                                                                                data_if[7]).flatten().detach()

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
                Lif1 / NL_sqrt[11],
                Lif2 / NL_sqrt[12]
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
                (Lif1 ** 2).mean().item(),
                (Lif2 ** 2).mean().item(),
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

            func_params, _ = vec_to_params(all_params)
            error = error_compute(V_model, func_params, data_test)
            relative_error.append(error.item())

        print(f"Step {step - 1}: training loss = {lossval[-1]:.4e}")
        print("finished")

        with torch.no_grad():
            V_params_dict, P_params_dict = vec_to_params(all_params)
            V_model.load_state_dict(V_params_dict)
            P_model.load_state_dict(P_params_dict)
        torch.save(V_model.state_dict(), 'best_V_model_XIPINN_LM.mdl')
        torch.save(P_model.state_dict(), 'best_P_model_XIPINN_LM.mdl')
        return P_model, V_model, lossval, lossval_dbg, relative_error

    except KeyboardInterrupt:
        print("Interrupted")
        print("steps =", step)
        with torch.no_grad():
            V_params_dict, P_params_dict = vec_to_params(all_params)
            V_model.load_state_dict(V_params_dict)
            P_model.load_state_dict(P_params_dict)
        torch.save(V_model.state_dict(), 'best_V_model_XIPINN_LM.mdl')
        torch.save(P_model.state_dict(), 'best_P_model_XIPINN_LM.mdl')
        return P_model, V_model, lossval, lossval_dbg, relative_error

