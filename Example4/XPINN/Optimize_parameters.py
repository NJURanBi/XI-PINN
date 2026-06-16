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


def error_compute(model_p, model_n, func_params_p, func_params_n, data_p, data_n):
    # 前向传播两个子网络
    def u_NN(data, model, func_params):
        u = torch.func.functional_call(model, func_params, data)
        return u.squeeze(0).squeeze(0)

    pred_p = u_NN(data_p[0], model_p, func_params_p)
    pred_n = u_NN(data_n[0], model_n, func_params_n)

    pred = torch.cat([pred_p[:, :2], pred_n[:, :2]], dim=0)  

    true_p = torch.cat([data_p[1].reshape(-1,1), data_p[2].reshape(-1,1)], dim=1)
    true_n = torch.cat([data_n[1].reshape(-1,1), data_n[2].reshape(-1,1)], dim=1)
    true = torch.cat([true_p, true_n], dim=0)

    err = pred - true
    err_norm = torch.sqrt(torch.sum(err.pow(2)))
    true_norm = torch.sqrt(torch.sum(true.pow(2)))
    relative_l2 = err_norm / true_norm

    return relative_l2


def optimize_parameters_SOAP(data_op, data_on, data_b, data_inip, data_inin, data_if, nu_p, nu_n, model_p, model_n, epochs, lr, lossval, lossval_dbg, relative_error, device):
    def loss_omega_p(data):
        output = model_p(data[0])
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
        output = model_n(data[0])
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
        V_p_output = model_p(data_p[0])
        u_p = V_p_output[:, 0][:, None]
        v_p = V_p_output[:, 1][:, None]
        du_pdx = autograd.grad(outputs=u_p, inputs=data_p[0], grad_outputs=torch.ones_like(u_p), create_graph=True, retain_graph=True)[0][:, 0][:, None]
        dv_pdy = autograd.grad(outputs=v_p, inputs=data_p[0], grad_outputs=torch.ones_like(v_p), create_graph=True, retain_graph=True)[0][:, 1][:, None]


        V_n_output = model_n(data_n[0])
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
        V_p_output =  model_p(data[0])
        u = V_p_output[:, 0][:, None]
        v = V_p_output[:, 1][:, None]
        g1 = data[1]
        g2 = data[2]
        loss1 = nn.MSELoss()(u, g1)
        loss2 = nn.MSELoss()(v, g2)
        return loss1, loss2

    def loss_initial_p(data):
        V_p_output = model_p(data[0])
        u = V_p_output[:, 0][:, None]
        v = V_p_output[:, 1][:, None]
        u01 = data[1]
        u02 = data[2]
        loss1 = nn.MSELoss()(u, u01)
        loss2 = nn.MSELoss()(v, u02)
        return loss1, loss2

    def loss_initial_n(data):
        V_n_output = model_n(data[0])
        u = V_n_output[:, 0][:, None]
        v = V_n_output[:, 1][:, None]
        u01 = data[1]
        u02 = data[2]
        loss1 = nn.MSELoss()(u, u01)
        loss2 = nn.MSELoss()(v, u02)
        return loss1, loss2

    def loss_interface_D(data):
        V_p_output = model_p(data[0])
        u_p = V_p_output[:, 0][:, None]
        v_p = V_p_output[:, 1][:, None]

        V_n_output = model_n(data[0])
        u_n = V_n_output[:, 0][:, None]
        v_n = V_n_output[:, 1][:, None]

        phi1 = data[2]
        phi2 = data[3]
        loss1 = nn.MSELoss()(u_p - u_n, phi1)
        loss2 = nn.MSELoss()(v_p - v_n, phi2)
        return loss1, loss2

    def loss_interface_N(data):
        V_p_output = model_p(data[0])
        u_p = V_p_output[:, 0][:, None]
        v_p = V_p_output[:, 1][:, None]
        p_p = V_p_output[:, 2][:, None]

        V_n_output = model_n(data[0])
        u_n = V_n_output[:, 0][:, None]
        v_n = V_n_output[:, 1][:, None]
        p_n = V_n_output[:, 2][:, None]

        gradup = autograd.grad(outputs=u_p, inputs=data[0], grad_outputs=torch.ones_like(u_p), create_graph=True, retain_graph=True)
        dupdx = gradup[0][:, 0][:, None]
        dupdy = gradup[0][:, 1][:, None]

        gradun = autograd.grad(outputs=u_n, inputs=data[0], grad_outputs=torch.ones_like(u_n), create_graph=True, retain_graph=True)
        dundx = gradun[0][:, 0][:, None]
        dundy = gradun[0][:, 1][:, None]

        gradvp = autograd.grad(outputs=v_p, inputs=data[0], grad_outputs=torch.ones_like(v_p), create_graph=True, retain_graph=True)
        dvpdx = gradvp[0][:, 0][:, None]
        dvpdy = gradvp[0][:, 1][:, None]

        gradvn = autograd.grad(outputs=v_n, inputs=data[0], grad_outputs=torch.ones_like(v_n), create_graph=True, retain_graph=True)
        dvndx = gradvn[0][:, 0][:, None]
        dvndy = gradvn[0][:, 1][:, None]

        nor = data[1]
        n1 = nor[:, 0][:, None]
        n2 = nor[:, 1][:, None]

        sigma_p1 = -p_p * n1 + nu_p * (2 * dupdx * n1 + (dupdy + dvpdx) * n2)
        sigma_n1 = -p_n * n1 + nu_n * (2 * dundx * n1 + (dundy + dvndx) * n2)
        res1 = sigma_p1 - sigma_n1

        sigma_p2 = -p_p * n2 + nu_p * ((dupdy + dvpdx) * n1 + 2 * dvpdy * n2)
        sigma_n2 = -p_n * n2 + nu_n * ((dundy + dvndx) * n1 + 2 * dvndy * n2)
        res2 = sigma_p2 - sigma_n2

        loss1 = nn.MSELoss()(res1, data[4])
        loss2 = nn.MSELoss()(res2, data[5])
        return loss1, loss2

    params = list(model_p.parameters()) + list(model_n.parameters())
    optimizer = SOAP(params, lr=lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)
    best_loss = float('inf')
    num_test = 10000
    data_test_p, data_test_n = generate_test_data(num_test, device)

    for epoch in range(epochs + 1):
        loss1, loss2 = loss_omega_p(data_op)
        loss3, loss4 = loss_omega_n(data_on)
        loss5, loss6 = loss_div(data_op, data_on)
        loss7, loss8 = loss_boundary(data_b)
        loss_p1, loss_p2 = loss_initial_p(data_inip)
        loss_n1, loss_n2 = loss_initial_n(data_inin)
        loss9 = loss_p1 + loss_n1
        loss10 = loss_p2 + loss_n2
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
            lossval_dbg.append([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()])

    print('best epoch:', best_epoch, 'best loss:', best_loss)
    model_p.load_state_dict(best_state_dict_p)
    model_n.load_state_dict(best_state_dict_n)
    torch.save(model_p.state_dict(), 'best_model_p_XIPINN_SOAP.mdl')
    torch.save(model_n.state_dict(), 'best_model_n_XIPINN_SOAP.mdl')
    return model_p, model_n, lossval, lossval_dbg, relative_error



def optimize_parameters_LM(data_op, data_on, data_b, data_inip, data_inin, data_if, nu_p, nu_n, model_p, model_n, tr_iter_max, mu, lossval, lossval_dbg, relative_error, mu_div, mu_mul, device):
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

    def u_p_NN(data, func_params, model_p):
        u = torch.func.functional_call(model_p, func_params, data)
        return u.squeeze(0).squeeze(0)[0]

    def u_n_NN(data, func_params, model_n):
        u = torch.func.functional_call(model_n, func_params, data)
        return u.squeeze(0).squeeze(0)[0]

    def v_p_NN(data, func_params, model_p):
        v = torch.func.functional_call(model_p, func_params, data)
        return v.squeeze(0).squeeze(0)[1]

    def v_n_NN(data, func_params, model_n):
        v = torch.func.functional_call(model_n, func_params, data)
        return v.squeeze(0).squeeze(0)[1]

    def p_p_NN(data, func_params, model_p):
        u = torch.func.functional_call(model_p, func_params, data)
        return u.squeeze(0).squeeze(0)[2]

    def p_n_NN(data, func_params, model_n):
        u = torch.func.functional_call(model_n, func_params, data)
        return u.squeeze(0).squeeze(0)[2]

    def res_omega_p_u(vec, data, f, w_1, w_2):
        params_p, _ = vec_to_params(vec)
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        gradu = torch.func.jacrev(u_p_NN, argnums=0)(data, params_p, model_p)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_p_NN, argnums=0), argnums=0)(data, params_p, model_p)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        dpdx = torch.func.jacrev(p_p_NN, argnums=0)(data, params_p, model_p)[0]
        res = dudt + w_1 * dudx + w_2 * dudy - nu_p * (d2udx2 + d2udy2) + dpdx - f
        return res

    def res_omega_n_u(vec, data, f, w_1, w_2):
        _, params_n = vec_to_params(vec)
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        gradu = torch.func.jacrev(u_n_NN, argnums=0)(data, params_n, model_n)
        grad2u = torch.func.jacrev(torch.func.jacrev(u_n_NN, argnums=0), argnums=0)(data, params_n, model_n)
        dudx = gradu[0]
        dudy = gradu[1]
        dudt = gradu[2]
        d2udx2 = grad2u[0][0]
        d2udy2 = grad2u[1][1]
        dpdx = torch.func.jacrev(p_n_NN, argnums=0)(data, params_n, model_n)[0]
        res = dudt + w_1 * dudx + w_2 * dudy - nu_n * (d2udx2 + d2udy2) + dpdx - f
        return res


    def res_omega_p_v(vec, data, f, w_1, w_2):
        params_p, _ = vec_to_params(vec)
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        gradv = torch.func.jacrev(v_p_NN, argnums=0)(data, params_p, model_p)
        grad2v = torch.func.jacrev(torch.func.jacrev(v_p_NN, argnums=0), argnums=0)(data, params_p, model_p)
        dvdx = gradv[0]
        dvdy = gradv[1]
        dvdt = gradv[2]
        d2vdx2 = grad2v[0][0]
        d2vdy2 = grad2v[1][1]
        dpdy = torch.func.jacrev(p_p_NN, argnums=0)(data, params_p, model_p)[1]
        res = dvdt + w_1 * dvdx + w_2 * dvdy - nu_p * (d2vdx2 + d2vdy2) + dpdy - f
        return res

    def res_omega_n_v(vec, data, f, w_1, w_2):
        _, params_n = vec_to_params(vec)
        f = f[0]
        w_1 = w_1[0]
        w_2 = w_2[0]
        gradv = torch.func.jacrev(v_n_NN, argnums=0)(data, params_n, model_n)
        grad2v = torch.func.jacrev(torch.func.jacrev(v_n_NN, argnums=0), argnums=0)(data, params_n, model_n)
        dvdx = gradv[0]
        dvdy = gradv[1]
        dvdt = gradv[2]
        d2vdx2 = grad2v[0][0]
        d2vdy2 = grad2v[1][1]
        dpdy = torch.func.jacrev(p_n_NN, argnums=0)(data, params_n, model_n)[1]
        res = dvdt + w_1 * dvdx + w_2 * dvdy - nu_n * (d2vdx2 + d2vdy2) + dpdy - f
        return res


    def res_div_p(vec, data):
        params_p, _ = vec_to_params(vec)
        gradu = torch.func.jacrev(u_p_NN, argnums=0)(data, params_p, model_p)
        gradv = torch.func.jacrev(v_p_NN, argnums=0)(data, params_p, model_p)
        dudx = gradu[0]
        dvdy = gradv[1]
        res = dudx + dvdy
        return res

    def res_div_n(vec, data):
        _, params_n = vec_to_params(vec)
        gradu = torch.func.jacrev(u_n_NN, argnums=0)(data, params_n, model_n)
        gradv = torch.func.jacrev(v_n_NN, argnums=0)(data, params_n, model_n)
        dudx = gradu[0]
        dvdy = gradv[1]
        res = dudx + dvdy
        return res


    def res_boundary_u(vec, data, gu):
        params_p, _ = vec_to_params(vec)
        g = gu[0]
        u = u_p_NN(data, params_p, model_p)
        res = u - g
        return res

    def res_boundary_v(vec, data, gv):
        params_p, _ = vec_to_params(vec)
        g = gv[0]
        v = v_p_NN(data, params_p, model_p)
        res = v - g
        return res


    def res_initial_u_p(vec, data, u0):
        params_p, _ = vec_to_params(vec)
        u = u_p_NN(data, params_p, model_p)
        u0 = u0[0]
        res = u - u0
        return res

    def res_initial_v_p(vec, data, v0):
        params_p, _ = vec_to_params(vec)
        v = v_p_NN(data, params_p, model_p)
        v0 = v0[0]
        res = v - v0
        return res

    def res_initial_u_n(vec, data, u0):
        _, params_n = vec_to_params(vec)
        u = u_n_NN(data, params_n, model_n)
        u0 = u0[0]
        res = u - u0
        return res

    def res_initial_v_n(vec, data, v0):
        _, params_n = vec_to_params(vec)
        v = v_n_NN(data, params_n, model_n)
        v0 = v0[0]
        res = v - v0
        return res

    def res_interface_D_u(vec, data, phi):
        params_p, params_n = vec_to_params(vec)
        up = u_p_NN(data, params_p, model_p)
        un = u_n_NN(data, params_n, model_n)
        phi = phi[0]
        res = up - un - phi
        return res

    def res_interface_D_v(vec, data, phi):
        params_p, params_n = vec_to_params(vec)
        vp = v_p_NN(data, params_p, model_p)
        vn = v_n_NN(data, params_n, model_n)
        phi = phi[0]
        res = vp - vn - phi
        return res

    def res_interface_N_1(vec, data, nor, psi):
        params_p, params_n = vec_to_params(vec)
        gradup = torch.func.jacrev(u_p_NN, argnums=0)(data, params_p, model_p)
        dupdx = gradup[0]
        dupdy = gradup[1]

        gradun = torch.func.jacrev(u_n_NN, argnums=0)(data, params_n, model_n)
        dundx = gradun[0]
        dundy = gradun[1]

        gradvp = torch.func.jacrev(v_p_NN, argnums=0)(data, params_p, model_p)
        dvpdx = gradvp[0]

        gradvn = torch.func.jacrev(v_n_NN, argnums=0)(data, params_n, model_n)
        dvndx = gradvn[0]

        p_p = p_p_NN(data, params_p, model_p)
        p_n = p_n_NN(data, params_n, model_n)

        n1 = nor[0]
        n2 = nor[1]
        psi = psi[0]

        sigma_p1 = -p_p * n1 + nu_p * (2 * dupdx * n1 + (dupdy + dvpdx) * n2)
        sigma_n1 = -p_n * n1 + nu_n * (2 * dundx * n1 + (dundy + dvndx) * n2)
        res = sigma_p1 - sigma_n1 - psi
        return res


    def res_interface_N_2(vec, data, nor, psi):
        params_p, params_n = vec_to_params(vec)
        gradup = torch.func.jacrev(u_p_NN, argnums=0)(data, params_p, model_p)
        dupdy = gradup[1]

        gradun = torch.func.jacrev(u_n_NN, argnums=0)(data, params_n, model_n)
        dundy = gradun[1]

        gradvp = torch.func.jacrev(v_p_NN, argnums=0)(data, params_p, model_p)
        dvpdx = gradvp[0]
        dvpdy = gradvp[1]

        gradvn = torch.func.jacrev(v_n_NN, argnums=0)(data, params_n, model_n)
        dvndx = gradvn[0]
        dvndy = gradvn[1]

        p_p = p_p_NN(data, params_p, model_p)
        p_n = p_n_NN(data, params_n, model_n)

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
    Nuinip = len(data_inip[0])
    Nvinip = len(data_inip[0])
    Nuinin = len(data_inin[0])
    Nvinin = len(data_inin[0])
    NifD1 = len(data_if[0])
    NifD2 = len(data_if[0])
    NifN1 = len(data_if[0])
    NifN2 = len(data_if[0])
    total_res = Nup + Nun + Nvp + Nvn + Ndivp + Ndivn + Nub + Nvb + Nuinip + Nuinin + Nvinip + Nvinin + NifD1 + NifD2 + NifN1 + NifN2
    NL = [total_res, Nup, Nun, Nvp, Nvn, Ndivp, Ndivn, Nub, Nvb, Nuinip, Nuinin, Nvinip, Nvinin, NifD1, NifD2, NifN1, NifN2]
    NL_sqrt = [np.sqrt(n) if n > 0 else 1.0 for n in NL]

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
    data_test_p, data_test_n = generate_test_data(num_test, device)
    try:
        while (lossval[-1] > tol_main) and (step <= tr_iter_max):
            torch.cuda.empty_cache()
            if Comput_old:  # need to compute loss_old and J_olds
                ### computation of loss
                Lup = torch.vmap(res_omega_p_u, (None, 0, 0, 0, 0))(all_params, data_op[0], data_op[1], data_op[3], data_op[4]).flatten().detach()
                Lun = torch.vmap(res_omega_n_u, (None, 0, 0, 0, 0))(all_params, data_on[0], data_on[1], data_on[3], data_on[4]).flatten().detach()
                Lvp = torch.vmap(res_omega_p_v, (None, 0, 0, 0, 0))(all_params, data_op[0], data_op[2], data_op[3], data_op[4]).flatten().detach()
                Lvn = torch.vmap(res_omega_n_v, (None, 0, 0, 0, 0))(all_params, data_on[0], data_on[2], data_on[3], data_on[4]).flatten().detach()
                Ldivp = torch.vmap(res_div_p, (None, 0))(all_params, data_op[0]).flatten().detach()
                Ldivn = torch.vmap(res_div_n, (None, 0))(all_params, data_on[0]).flatten().detach()
                Lub = torch.vmap(res_boundary_u, (None, 0, 0))(all_params, data_b[0], data_b[1]).flatten().detach()
                Lvb = torch.vmap(res_boundary_v, (None, 0, 0))(all_params, data_b[0], data_b[2]).flatten().detach()
                Luinip = torch.vmap(res_initial_u_p, (None, 0, 0))(all_params, data_inip[0], data_inip[1]).flatten().detach()
                Luinin = torch.vmap(res_initial_u_n, (None, 0, 0))(all_params, data_inin[0], data_inin[1]).flatten().detach()
                Lvinip = torch.vmap(res_initial_v_p, (None, 0, 0))(all_params, data_inip[0], data_inip[2]).flatten().detach()
                Lvinin = torch.vmap(res_initial_v_n, (None, 0, 0))(all_params, data_inin[0], data_inin[2]).flatten().detach()
                LuifD = torch.vmap(res_interface_D_u, (None, 0, 0))(all_params, data_if[0], data_if[2]).flatten().detach()
                LvifD = torch.vmap(res_interface_D_v, (None, 0, 0))(all_params, data_if[0], data_if[3]).flatten().detach()
                LifN1 = torch.vmap(res_interface_N_1, (None, 0, 0, 0))(all_params, data_if[0], data_if[1], data_if[4]).flatten().detach()
                LifN2 = torch.vmap(res_interface_N_2, (None, 0, 0, 0))(all_params, data_if[0], data_if[1], data_if[5]).flatten().detach()

                L = torch.cat([
                    Lup / NL_sqrt[1],
                    Lun / NL_sqrt[2],
                    Lvp / NL_sqrt[3],
                    Lvn / NL_sqrt[4],
                    Ldivp / NL_sqrt[5],
                    Ldivn / NL_sqrt[6],
                    Lub / NL_sqrt[7],
                    Lvb / NL_sqrt[8],
                    Luinip / NL_sqrt[9],
                    Luinin / NL_sqrt[10],
                    Lvinip / NL_sqrt[11],
                    Lvinin / NL_sqrt[12],
                    LuifD / NL_sqrt[13],
                    LvifD / NL_sqrt[14],
                    LifN1 / NL_sqrt[15],
                    LifN2 / NL_sqrt[16]
                ]).reshape(-1, 1).detach()

                loss_dbg_old = [
                    (Lup ** 2).mean().item(),
                    (Lun ** 2).mean().item(),
                    (Lvp ** 2).mean().item(),
                    (Lvn ** 2).mean().item(),
                    (Ldivp ** 2).mean().item(),
                    (Ldivn ** 2).mean().item(),
                    (Lub ** 2).mean().item(),
                    (Lvb ** 2).mean().item(),
                    (Luinip ** 2).mean().item(),
                    (Luinin ** 2).mean().item(),
                    (Lvinip ** 2).mean().item(),
                    (Lvinin ** 2).mean().item(),
                    (LuifD ** 2).mean().item(),
                    (LvifD ** 2).mean().item(),
                    (LifN1 ** 2).mean().item(),
                    (LifN2 ** 2).mean().item()
                ]

            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]
            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = all_params.detach().clone()  # get p_vec for p_vec_old if neccessary

            if criterion:
                # 正域动量 u 的 Jacobian
                Jup = torch.vmap(torch.func.jacrev(res_omega_p_u, argnums=0), (None, 0, 0, 0, 0))(
                    all_params, data_op[0], data_op[1], data_op[3], data_op[4]
                )
                Jup = Jup / NL_sqrt[1]

                # 负域动量 u 的 Jacobian
                Jun = torch.vmap(torch.func.jacrev(res_omega_n_u, argnums=0), (None, 0, 0, 0, 0))(
                    all_params, data_on[0], data_on[1], data_on[3], data_on[4]
                )
                Jun = Jun / NL_sqrt[2]

                # 正域动量 v 的 Jacobian
                Jvp = torch.vmap(torch.func.jacrev(res_omega_p_v, argnums=0), (None, 0, 0, 0, 0))(
                    all_params, data_op[0], data_op[2], data_op[3], data_op[4]
                )
                Jvp = Jvp / NL_sqrt[3]

                # 负域动量 v 的 Jacobian
                Jvn = torch.vmap(torch.func.jacrev(res_omega_n_v, argnums=0), (None, 0, 0, 0, 0))(
                    all_params, data_on[0], data_on[2], data_on[3], data_on[4]
                )
                Jvn = Jvn / NL_sqrt[4]

                # 正域散度 Jacobian
                Jdivp = torch.vmap(torch.func.jacrev(res_div_p, argnums=0), (None, 0))(
                    all_params, data_op[0]
                )
                Jdivp = Jdivp / NL_sqrt[5]

                # 负域散度 Jacobian
                Jdivn = torch.vmap(torch.func.jacrev(res_div_n, argnums=0), (None, 0))(
                    all_params, data_on[0]
                )
                Jdivn = Jdivn / NL_sqrt[6]

                # 边界 u Jacobian
                Jub = torch.vmap(torch.func.jacrev(res_boundary_u, argnums=0), (None, 0, 0))(
                    all_params, data_b[0], data_b[1]
                )
                Jub = Jub / NL_sqrt[7]

                # 边界 v Jacobian
                Jvb = torch.vmap(torch.func.jacrev(res_boundary_v, argnums=0), (None, 0, 0))(
                    all_params, data_b[0], data_b[2]
                )
                Jvb = Jvb / NL_sqrt[8]

                # 正域初始 u Jacobian
                Juinip = torch.vmap(torch.func.jacrev(res_initial_u_p, argnums=0), (None, 0, 0))(
                    all_params, data_inip[0], data_inip[1]
                )
                Juinip = Juinip / NL_sqrt[9]

                # 负域初始 u Jacobian
                Juinin = torch.vmap(torch.func.jacrev(res_initial_u_n, argnums=0), (None, 0, 0))(
                    all_params, data_inin[0], data_inin[1]
                )
                Juinin = Juinin / NL_sqrt[10]

                # 正域初始 v Jacobian
                Jvinip = torch.vmap(torch.func.jacrev(res_initial_v_p, argnums=0), (None, 0, 0))(
                    all_params, data_inip[0], data_inip[2]
                )
                Jvinip = Jvinip / NL_sqrt[11]

                # 负域初始 v Jacobian
                Jvinin = torch.vmap(torch.func.jacrev(res_initial_v_n, argnums=0), (None, 0, 0))(
                    all_params, data_inin[0], data_inin[2]
                )
                Jvinin = Jvinin / NL_sqrt[12]

                # 界面速度 u Jacobian
                JuifD = torch.vmap(torch.func.jacrev(res_interface_D_u, argnums=0), (None, 0, 0))(
                    all_params, data_if[0], data_if[2]
                )
                JuifD = JuifD / NL_sqrt[13]

                # 界面速度 v Jacobian
                JvifD = torch.vmap(torch.func.jacrev(res_interface_D_v, argnums=0), (None, 0, 0))(
                    all_params, data_if[0], data_if[3]
                )
                JvifD = JvifD / NL_sqrt[14]

                # 界面应力跳跃分量 1 Jacobian
                JifN1 = torch.vmap(torch.func.jacrev(res_interface_N_1, argnums=0), (None, 0, 0, 0))(
                    all_params, data_if[0], data_if[1], data_if[4]
                )
                JifN1 = JifN1 / NL_sqrt[15]

                # 界面应力跳跃分量 2 Jacobian
                JifN2 = torch.vmap(torch.func.jacrev(res_interface_N_2, argnums=0), (None, 0, 0, 0))(
                    all_params, data_if[0], data_if[1], data_if[5]
                )
                JifN2 = JifN2 / NL_sqrt[16]

                # 拼接总 Jacobian 矩阵，形状 (N_res, n_params)
                J = torch.cat([
                    Jup, Jun, Jvp, Jvn, Jdivp, Jdivn, Jub, Jvb,
                    Juinip, Juinin, Jvinip, Jvinin,
                    JuifD, JvifD, JifN1, JifN2
                ], dim=0).detach()

                # 计算 Hessian 近似与右端项
                J_product = J.t() @ J
                rhs = - J.t() @ L

            with torch.no_grad():
                dp = torch.linalg.solve(J_product + mu * I_pvec, rhs)
                all_params += dp.view(-1)


            ### Compute loss_new
            Lup = torch.vmap(res_omega_p_u, (None, 0, 0, 0, 0))(all_params, data_op[0], data_op[1], data_op[3],
                                                                data_op[4]).flatten().detach()
            Lun = torch.vmap(res_omega_n_u, (None, 0, 0, 0, 0))(all_params, data_on[0], data_on[1], data_on[3],
                                                                data_on[4]).flatten().detach()
            Lvp = torch.vmap(res_omega_p_v, (None, 0, 0, 0, 0))(all_params, data_op[0], data_op[2], data_op[3],
                                                                data_op[4]).flatten().detach()
            Lvn = torch.vmap(res_omega_n_v, (None, 0, 0, 0, 0))(all_params, data_on[0], data_on[2], data_on[3],
                                                                data_on[4]).flatten().detach()
            Ldivp = torch.vmap(res_div_p, (None, 0))(all_params, data_op[0]).flatten().detach()
            Ldivn = torch.vmap(res_div_n, (None, 0))(all_params, data_on[0]).flatten().detach()
            Lub = torch.vmap(res_boundary_u, (None, 0, 0))(all_params, data_b[0], data_b[1]).flatten().detach()
            Lvb = torch.vmap(res_boundary_v, (None, 0, 0))(all_params, data_b[0], data_b[2]).flatten().detach()
            Luinip = torch.vmap(res_initial_u_p, (None, 0, 0))(all_params, data_inip[0],
                                                               data_inip[1]).flatten().detach()
            Luinin = torch.vmap(res_initial_u_n, (None, 0, 0))(all_params, data_inin[0],
                                                               data_inin[1]).flatten().detach()
            Lvinip = torch.vmap(res_initial_v_p, (None, 0, 0))(all_params, data_inip[0],
                                                               data_inip[2]).flatten().detach()
            Lvinin = torch.vmap(res_initial_v_n, (None, 0, 0))(all_params, data_inin[0],
                                                               data_inin[2]).flatten().detach()
            LuifD = torch.vmap(res_interface_D_u, (None, 0, 0))(all_params, data_if[0], data_if[2]).flatten().detach()
            LvifD = torch.vmap(res_interface_D_v, (None, 0, 0))(all_params, data_if[0], data_if[3]).flatten().detach()
            LifN1 = torch.vmap(res_interface_N_1, (None, 0, 0, 0))(all_params, data_if[0], data_if[1],
                                                                   data_if[4]).flatten().detach()
            LifN2 = torch.vmap(res_interface_N_2, (None, 0, 0, 0))(all_params, data_if[0], data_if[1],
                                                                   data_if[5]).flatten().detach()

            L = torch.cat([
                Lup / NL_sqrt[1],
                Lun / NL_sqrt[2],
                Lvp / NL_sqrt[3],
                Lvn / NL_sqrt[4],
                Ldivp / NL_sqrt[5],
                Ldivn / NL_sqrt[6],
                Lub / NL_sqrt[7],
                Lvb / NL_sqrt[8],
                Luinip / NL_sqrt[9],
                Luinin / NL_sqrt[10],
                Lvinip / NL_sqrt[11],
                Lvinin / NL_sqrt[12],
                LuifD / NL_sqrt[13],
                LvifD / NL_sqrt[14],
                LifN1 / NL_sqrt[15],
                LifN2 / NL_sqrt[16]
            ]).reshape(-1, 1).detach()

            loss_new = torch.sum(L * L).item()
            loss_dbg_new = [
                    (Lup ** 2).mean().item(),
                    (Lun ** 2).mean().item(),
                    (Lvp ** 2).mean().item(),
                    (Lvn ** 2).mean().item(),
                    (Ldivp ** 2).mean().item(),
                    (Ldivn ** 2).mean().item(),
                    (Lub ** 2).mean().item(),
                    (Lvb ** 2).mean().item(),
                    (Luinip ** 2).mean().item(),
                    (Luinin ** 2).mean().item(),
                    (Lvinip ** 2).mean().item(),
                    (Lvinin ** 2).mean().item(),
                    (LuifD ** 2).mean().item(),
                    (LvifD ** 2).mean().item(),
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
                            all_params -= dp.view(-1)
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

