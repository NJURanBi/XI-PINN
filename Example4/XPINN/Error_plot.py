import numpy as np
import matplotlib.pyplot as plt
import torch
from Network import Vanilla_Net
torch.set_default_dtype(torch.float64)

# ------------------ 随机种子 ------------------
np.random.seed(42)
torch.manual_seed(42)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ================== 加载回流模型 ==================
flow_model_paths = [
    'inverse_mapping_0.32.mdl',
    'inverse_mapping_0.52.mdl',
    'inverse_mapping_0.73.mdl',
    'inverse_mapping_1.mdl',
]

flow_models = []
for path in flow_model_paths:
    model = Vanilla_Net(3, 32, 2, 3).double()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    flow_models.append(model)

# ================== 水平集函数 ==================
def level_set_function(data, models):
    t_nodes = [0.0, 0.32, 0.52, 0.73]
    t = data[:, 2]
    N = data.shape[0]
    device = data.device
    X_final = torch.empty(N, 1, device=device)
    Y_final = torch.empty(N, 1, device=device)

    # 区间1
    mask = t < t_nodes[1]
    if mask.any():
        sub_data = data[mask]
        out = models[0](sub_data)
        dt = sub_data[:, 2:3] - t_nodes[0]
        x = sub_data[:, 0:1] + dt * out[:, 0:1]
        y = sub_data[:, 1:2] + dt * out[:, 1:2]
        X_final[mask] = x
        Y_final[mask] = y

    # 区间2
    mask = (t >= t_nodes[1]) & (t < t_nodes[2])
    if mask.any():
        sub_data = data[mask]
        out = models[1](sub_data)
        dt = sub_data[:, 2:3] - t_nodes[1]
        x = sub_data[:, 0:1] + dt * out[:, 0:1]
        y = sub_data[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[1] * torch.ones_like(x)], dim=1)
        out = models[0](cur)
        dt = t_nodes[1] - t_nodes[0]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        X_final[mask] = x
        Y_final[mask] = y

    # 区间3
    mask = (t >= t_nodes[2]) & (t < t_nodes[3])
    if mask.any():
        sub_data = data[mask]
        out = models[2](sub_data)
        dt = sub_data[:, 2:3] - t_nodes[2]
        x = sub_data[:, 0:1] + dt * out[:, 0:1]
        y = sub_data[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[2] * torch.ones_like(x)], dim=1)
        out = models[1](cur)
        dt = t_nodes[2] - t_nodes[1]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[1] * torch.ones_like(x)], dim=1)
        out = models[0](cur)
        dt = t_nodes[1] - t_nodes[0]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        X_final[mask] = x
        Y_final[mask] = y

    # 区间4
    mask = t >= t_nodes[3]
    if mask.any():
        sub_data = data[mask]
        out = models[3](sub_data)
        dt = sub_data[:, 2:3] - t_nodes[3]
        x = sub_data[:, 0:1] + dt * out[:, 0:1]
        y = sub_data[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[3] * torch.ones_like(x)], dim=1)
        out = models[2](cur)
        dt = t_nodes[3] - t_nodes[2]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[2] * torch.ones_like(x)], dim=1)
        out = models[1](cur)
        dt = t_nodes[2] - t_nodes[1]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        cur = torch.cat([x, y, t_nodes[1] * torch.ones_like(x)], dim=1)
        out = models[0](cur)
        dt = t_nodes[1] - t_nodes[0]
        x = cur[:, 0:1] + dt * out[:, 0:1]
        y = cur[:, 1:2] + dt * out[:, 1:2]
        X_final[mask] = x
        Y_final[mask] = y

    lf = (X_final - 0.5) ** 2 + (Y_final - 0.75) ** 2 - 0.15 ** 2
    return lf

# ================== 真解 ==================
def get_u_true_np(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    data_tensor = torch.tensor(data, dtype=torch.float64)
    phi = level_set_function(data_tensor, flow_models).detach().cpu().numpy()[:, 0]
    mask = phi > 0
    u_p = np.exp(x) * np.sin(np.pi * y + np.pi * t)
    v_p = (1.0 / np.pi) * np.exp(x) * np.cos(np.pi * y + np.pi * t)
    u_n = np.cos(t) * np.cos(np.pi * x) * np.sin(np.pi * y)
    v_n = -np.cos(t) * np.sin(np.pi * x) * np.cos(np.pi * y)
    u = np.where(mask, u_p, u_n)
    v = np.where(mask, v_p, v_n)
    return u, v

def get_u_true_and_grad(data):
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    data_tensor = torch.tensor(data, dtype=torch.float64)
    phi = level_set_function(data_tensor, flow_models).detach().cpu().numpy()[:, 0]
    mask = phi > 0

    u_p = np.exp(x) * np.sin(np.pi * y + np.pi * t)
    v_p = (1.0 / np.pi) * np.exp(x) * np.cos(np.pi * y + np.pi * t)
    du_p_dx = u_p
    du_p_dy = np.pi * np.exp(x) * np.cos(np.pi * y + np.pi * t)
    dv_p_dx = v_p
    dv_p_dy = -np.exp(x) * np.sin(np.pi * y + np.pi * t)

    u_n = np.cos(t) * np.cos(np.pi * x) * np.sin(np.pi * y)
    v_n = -np.cos(t) * np.sin(np.pi * x) * np.cos(np.pi * y)
    du_n_dx = -np.pi * np.cos(t) * np.sin(np.pi * x) * np.sin(np.pi * y)
    du_n_dy = np.pi * np.cos(t) * np.cos(np.pi * x) * np.cos(np.pi * y)
    dv_n_dx = -np.pi * np.cos(t) * np.cos(np.pi * x) * np.cos(np.pi * y)
    dv_n_dy = np.pi * np.cos(t) * np.sin(np.pi * x) * np.sin(np.pi * y)

    u = np.where(mask, u_p, u_n)
    v = np.where(mask, v_p, v_n)
    grad_u_x = np.where(mask, du_p_dx, du_n_dx)
    grad_u_y = np.where(mask, du_p_dy, du_n_dy)
    grad_v_x = np.where(mask, dv_p_dx, dv_n_dx)
    grad_v_y = np.where(mask, dv_p_dy, dv_n_dy)

    grad_u = np.stack([grad_u_x, grad_u_y], axis=1)
    grad_v = np.stack([grad_v_x, grad_v_y], axis=1)
    return u, v, grad_u, grad_v

# ================== X‑PINN 预测函数 ==================
def xpinn_output(data_tensor, model_p, model_n, flow_models):
    lf = level_set_function(data_tensor, flow_models)
    mask = (lf >= 0).double()
    out_p = model_p(data_tensor)
    out_n = model_n(data_tensor)
    return mask * out_p + (1 - mask) * out_n

# ================== 加载 X‑PINN 模型 ==================
model_p_path = 'best_model_p_LM_medium.mdl'
model_n_path = 'best_model_n_LM_medium.mdl'

model_p = Vanilla_Net(3, 45, 3, 2).double()
model_n = Vanilla_Net(3, 45, 3, 2).double()
model_p.load_state_dict(torch.load(model_p_path, map_location='cpu'))
model_n.load_state_dict(torch.load(model_n_path, map_location='cpu'))
model_p.eval()
model_n.eval()

# ================== 网格设置 ==================
times = [0.25, 0.5, 0.75, 1.0]
grid_size = 201
x_vec = np.linspace(0, 1, grid_size)
y_vec = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x_vec, y_vec)

grid_datas = []
for t_val in times:
    data_tmp = np.hstack([X.ravel()[:, None],
                          Y.ravel()[:, None],
                          t_val * np.ones((grid_size**2, 1))])
    grid_datas.append(data_tmp)

# 真解与预测
true_fields_u, true_fields_v = [], []
pred_fields_u, pred_fields_v = [], []
lf_fields = []

for data_tmp in grid_datas:
    u_t, v_t = get_u_true_np(data_tmp)
    true_fields_u.append(u_t.reshape(grid_size, grid_size))
    true_fields_v.append(v_t.reshape(grid_size, grid_size))

    inp = torch.tensor(data_tmp, dtype=torch.float64)
    with torch.no_grad():
        pred = xpinn_output(inp, model_p, model_n, flow_models).cpu().numpy()
    pred_u = pred[:, 0]
    pred_v = pred[:, 1]
    pred_fields_u.append(pred_u.reshape(grid_size, grid_size))
    pred_fields_v.append(pred_v.reshape(grid_size, grid_size))

    lf = level_set_function(inp, flow_models).detach().cpu().numpy()
    lf_fields.append(lf.reshape(grid_size, grid_size))

# 转换为模长
mod_true_fields = []
mod_pred_fields = []
mod_error_fields = []
for i in range(len(times)):
    mod_true = np.sqrt(true_fields_u[i]**2 + true_fields_v[i]**2)
    mod_pred = np.sqrt(pred_fields_u[i]**2 + pred_fields_v[i]**2)
    mod_error = np.abs(mod_true - mod_pred)
    mod_true_fields.append(mod_true)
    mod_pred_fields.append(mod_pred)
    mod_error_fields.append(mod_error)

# ================== 蒙特卡洛误差 ==================
N_mc = 100000
mc_points = np.random.uniform(0, 1, (N_mc, 2))
t_mc = np.random.uniform(0, 1, N_mc)
mc_data = np.hstack([mc_points, t_mc[:, None]])

u_true, v_true, grad_u_true, grad_v_true = get_u_true_and_grad(mc_data)

mc_input = torch.tensor(mc_data, requires_grad=True, dtype=torch.float64)
pred_mc = xpinn_output(mc_input, model_p, model_n, flow_models)

grad_u_pred = torch.autograd.grad(
    pred_mc[:, 0].sum(), mc_input, create_graph=False, retain_graph=True
)[0][:, :2].detach().cpu().numpy()
grad_v_pred = torch.autograd.grad(
    pred_mc[:, 1].sum(), mc_input, create_graph=False, retain_graph=False
)[0][:, :2].detach().cpu().numpy()

pred_u_np = pred_mc[:, 0].detach().cpu().numpy()
pred_v_np = pred_mc[:, 1].detach().cpu().numpy()

def rel_error_components(u_pred, u_true, grad_pred, grad_true):
    err_u = u_pred - u_true
    err_g = grad_pred - grad_true
    norm_u_sq = np.mean(u_true**2)
    norm_e_sq = np.mean(err_u**2)
    norm_h1_sq = norm_u_sq + np.mean(np.sum(grad_true**2, axis=1))
    norm_h1_e_sq = norm_e_sq + np.mean(np.sum(err_g**2, axis=1))
    rel_L2 = np.sqrt(norm_e_sq / norm_u_sq) if norm_u_sq > 0 else np.inf
    rel_H1 = np.sqrt(norm_h1_e_sq / norm_h1_sq) if norm_h1_sq > 0 else np.inf
    return rel_L2, rel_H1

rel_L2_u, rel_H1_u = rel_error_components(pred_u_np, u_true, grad_u_pred, grad_u_true)
rel_L2_v, rel_H1_v = rel_error_components(pred_v_np, v_true, grad_v_pred, grad_v_true)

# 模长误差
mod_true_mc = np.sqrt(u_true**2 + v_true**2)
mod_pred_mc = np.sqrt(pred_u_np**2 + pred_v_np**2)
rel_L2_mod = np.sqrt(np.mean((mod_pred_mc - mod_true_mc)**2) / np.mean(mod_true_mc**2))

print(f"蒙特卡洛点数: {N_mc}")
print(f"u 分量 - L2: {rel_L2_u:.6e}, H1: {rel_H1_u:.6e}")
print(f"v 分量 - L2: {rel_L2_v:.6e}, H1: {rel_H1_v:.6e}")
print(f"模长 |(u,v)^T| - 相对 L2 误差: {rel_L2_mod:.6e}")

# ================== 绘图：仅绘制模长 ==================
def plot_modulus(pred_fields, error_fields, lf_fields):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.90,
                        hspace=0.1, wspace=0.30)

    im0 = None
    im1 = None
    for col, t_val in enumerate(times):
        ax0 = axes[0, col]
        pcm0 = ax0.pcolormesh(X, Y, pred_fields[col],
                              shading='auto', cmap='viridis')
        ax0.contour(X, Y, lf_fields[col], levels=[0],
                    colors='k', linewidths=1.2, linestyles='--')
        ax0.set_title(f't = {t_val}', fontsize=12)
        ax0.set_aspect('equal')
        if col == 0:
            im0 = pcm0
            ax0.set_ylabel('$y$', fontsize=10)

        ax1 = axes[1, col]
        pcm1 = ax1.pcolormesh(X, Y, error_fields[col],
                              shading='auto', cmap='inferno')
        ax1.contour(X, Y, lf_fields[col], levels=[0],
                    colors='white', linewidths=1.0, linestyles='--')
        ax1.set_title(f't = {t_val}', fontsize=12)
        ax1.set_aspect('equal')
        if col == 0:
            im1 = pcm1
            ax1.set_ylabel('$y$', fontsize=10)
        ax1.set_xlabel('$x$', fontsize=10)

    cbar0 = fig.colorbar(im0, ax=axes[0, :], location='right',
                         aspect=16, shrink=0.95)
    cbar1 = fig.colorbar(im1, ax=axes[1, :], location='right',
                         aspect=16, shrink=0.95)
    cbar0.ax.tick_params(labelsize=14)
    cbar1.ax.tick_params(labelsize=14)
    cbar0.ax.yaxis.get_offset_text().set_fontsize(14)
    cbar1.ax.yaxis.get_offset_text().set_fontsize(14)

    fig.text(0.03, 0.71, r'Numerical $|(u,v)^T|$',
             va='center', ha='center', rotation='vertical', fontsize=14)
    fig.text(0.03, 0.30, 'Absolute Error',
             va='center', ha='center', rotation='vertical', fontsize=14)

    fig.suptitle(r'X-PINN prediction and error of $|(u,v)^T|$', fontsize=18,
                 x=0.4, y=0.95)
    return fig

fig_mod = plot_modulus(mod_pred_fields, mod_error_fields, lf_fields)
fig_mod.savefig('pred_error_modulus_XPINN.jpg', dpi=300)
plt.show()