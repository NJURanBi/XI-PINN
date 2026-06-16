import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch
from Network import Vanilla_Net
import matplotlib.patches as mpatches

# ------------------ 随机种子 ------------------
np.random.seed(42)
torch.manual_seed(42)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ======================== 问题参数 ========================
t_val = 0.5
grid_size = 401
x1 = np.linspace(-1, 1, grid_size)
x2 = np.linspace(-1, 1, grid_size)
X1, X2 = np.meshgrid(x1, x2)

# 单位圆掩码
mask_circle = X1**2 + X2**2 <= 1.0
X1_in = X1[mask_circle].reshape(-1, 1)
X2_in = X2[mask_circle].reshape(-1, 1)
data_grid = np.hstack((X1_in, X2_in, t_val * np.ones((X1_in.shape[0], 1))))

# ------------------ 水平集函数及导数 ------------------
def level_set_function(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t  = data[:, 2]
    xi = x1 * np.cos(t) + x2 * np.sin(t)
    eta = -x1 * np.sin(t) + x2 * np.cos(t)
    phi = (xi**2 + eta**2) - 0.4 - (1.0 / np.pi) * np.sin(np.pi * xi) * np.cos(np.pi * eta)
    return phi[:, None]

# 一阶导数 (phi_x1, phi_x2)
def get_phi_grad(data):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t  = data[:, 2]
    xi = x1 * np.cos(t) + x2 * np.sin(t)
    eta = -x1 * np.sin(t) + x2 * np.cos(t)
    phi_xi = 2*xi - np.cos(np.pi*xi)*np.cos(np.pi*eta)
    phi_eta = 2*eta + np.sin(np.pi*xi)*np.sin(np.pi*eta)
    dxi_dx1 = np.cos(t); deta_dx1 = -np.sin(t)
    dxi_dx2 = np.sin(t); deta_dx2 = np.cos(t)
    phi_x1 = phi_xi*dxi_dx1 + phi_eta*deta_dx1
    phi_x2 = phi_xi*dxi_dx2 + phi_eta*deta_dx2
    return phi_x1[:, None], phi_x2[:, None]

# 真解 (u, v)
def get_u(data):
    x1 = data[:,0]; x2 = data[:,1]; t = data[:,2]
    u_base1 = np.exp(x1)*np.sin(x2+t)
    u_base2 = np.exp(x1)*np.cos(x2+t)
    phi = level_set_function(data)[:,0]
    phi_x1, phi_x2 = get_phi_grad(data)
    phi_x1 = phi_x1.flatten(); phi_x2 = phi_x2.flatten()
    u_p1 = u_base1 + phi * phi_x2
    u_p2 = u_base2 - phi * phi_x1
    u_n1 = u_base1; u_n2 = u_base2
    mask = phi > 0
    u1 = np.where(mask, u_p1, u_n1)
    u2 = np.where(mask, u_p2, u_n2)
    return u1[:, None], u2[:, None]

# 真解及其梯度（解析）
def get_u_true_and_grad(data):
    x1 = data[:,0]; x2 = data[:,1]; t = data[:,2]
    u_base1 = np.exp(x1)*np.sin(x2+t)
    u_base2 = np.exp(x1)*np.cos(x2+t)
    du_base1_x1 = u_base1
    du_base1_x2 = np.exp(x1)*np.cos(x2+t)
    du_base2_x1 = u_base2
    du_base2_x2 = -np.exp(x1)*np.sin(x2+t)

    phi = level_set_function(data)[:,0]
    phi_x1, phi_x2 = get_phi_grad(data)
    phi_x1 = phi_x1.flatten(); phi_x2 = phi_x2.flatten()
    # 二阶导数
    def get_phi_hessian(data):
        x1_d = data[:,0]; x2_d = data[:,1]; t_d = data[:,2]
        xi = x1_d*np.cos(t_d) + x2_d*np.sin(t_d)
        eta = -x1_d*np.sin(t_d) + x2_d*np.cos(t_d)
        phi_xixi = 2 + np.pi * np.sin(np.pi*xi) * np.cos(np.pi*eta)
        phi_xieta = np.pi * np.cos(np.pi*xi) * np.sin(np.pi*eta)
        phi_etaeta = 2 + np.pi * np.sin(np.pi*xi) * np.cos(np.pi*eta)
        c = np.cos(t_d); s = np.sin(t_d)
        phi_x1x1 = c*c * phi_xixi - 2*c*s * phi_xieta + s*s * phi_etaeta
        phi_x1x2 = c*s * (phi_xixi - phi_etaeta) + (c*c - s*s) * phi_xieta
        phi_x2x2 = s*s * phi_xixi + 2*s*c * phi_xieta + c*c * phi_etaeta
        return phi_x1x1[:, None], phi_x1x2[:, None], phi_x2x2[:, None]

    phi_x1x1, phi_x1x2, phi_x2x2 = get_phi_hessian(data)
    phi_x1x1 = phi_x1x1.flatten(); phi_x1x2 = phi_x1x2.flatten(); phi_x2x2 = phi_x2x2.flatten()

    du_p1_x1 = du_base1_x1 + phi_x1*phi_x2 + phi*phi_x1x2
    du_p1_x2 = du_base1_x2 + phi_x2*phi_x2 + phi*phi_x2x2
    du_p2_x1 = du_base2_x1 - phi_x1*phi_x1 - phi*phi_x1x1
    du_p2_x2 = du_base2_x2 - phi_x2*phi_x1 - phi*phi_x1x2

    du_n1_x1 = du_base1_x1; du_n1_x2 = du_base1_x2
    du_n2_x1 = du_base2_x1; du_n2_x2 = du_base2_x2

    mask = phi > 0
    u1 = np.where(mask, u_base1 + phi*phi_x2, u_base1)
    u2 = np.where(mask, u_base2 - phi*phi_x1, u_base2)
    grad1_x1 = np.where(mask, du_p1_x1, du_n1_x1)
    grad1_x2 = np.where(mask, du_p1_x2, du_n1_x2)
    grad2_x1 = np.where(mask, du_p2_x1, du_n2_x1)
    grad2_x2 = np.where(mask, du_p2_x2, du_n2_x2)

    grad1 = np.stack([grad1_x1, grad1_x2], axis=1)
    grad2 = np.stack([grad2_x1, grad2_x2], axis=1)
    return u1, u2, grad1, grad2

# ------------------ 网格上的真解（仅圆内） ------------------
u_true1_in, u_true2_in = get_u(data_grid)
lf_in = level_set_function(data_grid)

# ------------------ X‑PINN 模型（两个子网络，输出2维） ------------------
model_p = Vanilla_Net(3, 45, 2, 3).double()
model_n = Vanilla_Net(3, 45, 2, 3).double()
model_p.load_state_dict(torch.load('best_V_p_model_XIPINN_LM_large.mdl'))
model_n.load_state_dict(torch.load('best_V_n_model_XIPINN_LM_large.mdl'))
model_p.eval()
model_n.eval()

def XPINN_output(data_tensor, model_p, model_n):
    """data_tensor: (N,3) 包含 x1,x2,t"""
    x1 = data_tensor[:, 0]
    x2 = data_tensor[:, 1]
    t = data_tensor[:, 2]
    xi = x1 * torch.cos(t) + x2 * torch.sin(t)
    eta = -x1 * torch.sin(t) + x2 * torch.cos(t)
    lf = (xi ** 2 + eta ** 2) - 0.4 - (1.0 / np.pi) * torch.sin(torch.pi * xi) * torch.cos(torch.pi * eta)
    output_p = model_p(data_tensor)   # (N,2)
    output_n = model_n(data_tensor)
    mask = (lf >= 0).float().unsqueeze(1)  # (N,1)
    output = mask * output_p + (1 - mask) * output_n
    return output

# 在网格上预测
data_tensor = torch.tensor(data_grid).double()
with torch.no_grad():
    pred_in = XPINN_output(data_tensor, model_p, model_n).cpu().numpy()

u_pred1_in = pred_in[:, 0:1]
u_pred2_in = pred_in[:, 1:2]

# 转向模长
mod_true_in = np.sqrt(u_true1_in**2 + u_true2_in**2)
mod_pred_in = np.sqrt(u_pred1_in**2 + u_pred2_in**2)
error_mod_in = np.abs(mod_true_in - mod_pred_in)

# 映射回完整网格，圆外置 NaN
def to_full_grid(values_1d):
    out = np.full((grid_size, grid_size), np.nan)
    out[mask_circle] = values_1d.flatten()
    return out

mod_pred_2d = to_full_grid(mod_pred_in)
error_mod_2d = to_full_grid(error_mod_in)
mod_true_2d = to_full_grid(mod_true_in)
lf_2d = to_full_grid(lf_in)

# ------------------ 蒙特卡洛积分（圆内） ------------------
N_mc = 100000
mc_all = np.random.uniform(-1, 1, (N_mc*2, 2))
r2 = mc_all[:,0]**2 + mc_all[:,1]**2
mc_valid = mc_all[r2 <= 1.0]
if mc_valid.shape[0] < N_mc:
    while mc_valid.shape[0] < N_mc:
        extra = np.random.uniform(-1, 1, (N_mc, 2))
        r2_extra = extra[:,0]**2 + extra[:,1]**2
        mc_valid = np.vstack([mc_valid, extra[r2_extra <= 1.0]])
mc_points = mc_valid[:N_mc, :]
mc_data = np.hstack([mc_points, np.random.uniform(0, 1, (N_mc, 1))])

# 真解与解析梯度
u1_true, u2_true, grad1_true, grad2_true = get_u_true_and_grad(mc_data)

# 预测及梯度
mc_input = torch.tensor(mc_data, requires_grad=True, dtype=torch.float64)
pred_mc = XPINN_output(mc_input, model_p, model_n)

grad1_all = torch.autograd.grad(pred_mc[:,0].sum(), mc_input,
                                create_graph=False, retain_graph=True)[0].detach().cpu().numpy()
grad2_all = torch.autograd.grad(pred_mc[:,1].sum(), mc_input,
                                create_graph=False, retain_graph=False)[0].detach().cpu().numpy()
grad1_pred_corr = grad1_all[:, :2]
grad2_pred_corr = grad2_all[:, :2]

u1_pred = pred_mc[:,0].detach().cpu().numpy()
u2_pred = pred_mc[:,1].detach().cpu().numpy()

# 相对误差计算（传统两个分量）
def rel_errors(u_pred, u_true, grad_pred, grad_true):
    err_u = u_pred - u_true
    err_g = grad_pred - grad_true
    norm_u_sq = np.mean(u_true**2)
    norm_e_sq = np.mean(err_u**2)
    norm_h1_sq = norm_u_sq + np.mean(np.sum(grad_true**2, axis=1))
    norm_h1_e_sq = norm_e_sq + np.mean(np.sum(err_g**2, axis=1))
    return np.sqrt(norm_e_sq / norm_u_sq), np.sqrt(norm_h1_e_sq / norm_h1_sq)

rel_L2_1, rel_H1_1 = rel_errors(u1_pred, u1_true, grad1_pred_corr, grad1_true)
rel_L2_2, rel_H1_2 = rel_errors(u2_pred, u2_true, grad2_pred_corr, grad2_true)

# 模长相对 L2 误差
mod_true_mc = np.sqrt(u1_true**2 + u2_true**2)
mod_pred_mc = np.sqrt(u1_pred**2 + u2_pred**2)
rel_L2_mod = np.sqrt(np.mean((mod_pred_mc - mod_true_mc)**2) / np.mean(mod_true_mc**2))

print(f"蒙特卡洛点数(圆内): {N_mc}")
print(f"u 分量 - 相对 L2 误差 : {rel_L2_1:.6e}，相对 H1 误差 : {rel_H1_1:.6e}")
print(f"v 分量 - 相对 L2 误差 : {rel_L2_2:.6e}，相对 H1 误差 : {rel_H1_2:.6e}")
print(f"模长 |(u,v)^T| - 相对 L2 误差 : {rel_L2_mod:.6e}")

# ------------------ 绘图函数（仅绘制模长） ------------------
step = 4
X1_low = X1[::step, ::step]
X2_low = X2[::step, ::step]
mask_low = mask_circle[::step, ::step]

def plot_modulus(mod_pred_2d, error_2d, mod_true_2d, cross_direction='y'):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), constrained_layout=True)
    # ---------- 数值解 (左图) ----------
    im0 = axes[0].pcolormesh(X1_low, X2_low,
                             np.where(mask_low, mod_pred_2d[::step, ::step], np.nan),
                             shading='auto', cmap='viridis')
    axes[0].contour(X1_low, X2_low,
                    np.where(mask_low, lf_2d[::step, ::step], np.nan),
                    levels=[0], colors='k', linewidths=1.2, linestyles='--')
    axes[0].set_title(r'Numerical $|(u,v)^T|$', fontsize=12)
    axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
    axes[0].set_aspect('equal')
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    cbar0.ax.tick_params(labelsize=10)
    cbar0.ax.yaxis.get_offset_text().set_fontsize(10)
    # ---------- 绝对误差 (中图) ----------
    im1 = axes[1].pcolormesh(X1_low, X2_low,
                             np.where(mask_low, error_2d[::step, ::step], np.nan),
                             shading='auto', cmap='inferno')
    axes[1].contour(X1_low, X2_low,
                    np.where(mask_low, lf_2d[::step, ::step], np.nan),
                    levels=[0], colors='white', linewidths=1.0, linestyles='--')
    axes[1].set_title('Absolute Error', fontsize=12)
    axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')
    axes[1].set_aspect('equal')
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    cbar1.ax.tick_params(labelsize=10)
    cbar1.ax.yaxis.get_offset_text().set_fontsize(10)
    # ---------- 截面图（固定为 y=0） ----------
    idx = np.argmin(np.abs(x2 - 0.0))
    line_x = np.where(mask_circle[idx, :], x1, np.nan)
    mod_true_line = mod_true_2d[idx, :]
    mod_pred_line = mod_pred_2d[idx, :]
    lf_line = lf_2d[idx, :]
    error_line = error_2d[idx, :]
    x_label = '$x$'
    fixed_str = '$y = 0$'
    mask_pos = (x1 > 0) & mask_circle[idx, :]
    if np.any(mask_pos):
        lf_pos = lf_line[mask_pos]
        interface_idx = np.arange(len(lf_line))[mask_pos][np.nanargmin(np.abs(lf_pos))]
    else:
        interface_idx = np.nanargmin(np.abs(lf_line))
    interface_pos = x1[interface_idx]

    axes[2].plot(line_x, mod_true_line, 'b-', lw=1.5, label='True')
    axes[2].plot(line_x, mod_pred_line, 'r--', lw=1.5, label='Predicted')
    axes[2].axvline(interface_pos, color='grey', ls=':', lw=1.2, label='Interface')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(r'$|(u,v)^T|$')
    axes[2].set_title(f'Cross‑section at {fixed_str}, $t={t_val}$', fontsize=12)
    axes[2].set_xlim([-1, 1])
    all_y = np.concatenate([mod_true_line[~np.isnan(mod_true_line)],
                            mod_pred_line[~np.isnan(mod_pred_line)]])
    if len(all_y) > 0:
        ymax = np.nanmax(all_y)
        axes[2].set_ylim(top=ymax * 1.3)
    axes[2].legend(loc='upper right')
    # 浅色高亮区域
    zoom_w = 0.15
    axes[2].axvspan(interface_pos - zoom_w, interface_pos + zoom_w, alpha=0.2, color='yellow')
    # ----- 内嵌放大图 -----
    inset_ax = inset_axes(axes[2], width="50%", height="50%",
                           bbox_to_anchor=(0.2, 0.4, 0.6, 0.5),
                           bbox_transform=axes[2].transAxes, loc='lower left')
    inset_ax.plot(line_x, error_line, 'm-', lw=1.5)
    inset_ax.axvline(interface_pos, color='grey', ls=':', lw=1)
    inset_ax.set_xlim(interface_pos - zoom_w, interface_pos + zoom_w)
    inset_ax.set_xlabel(x_label, fontsize=6)
    inset_ax.set_ylabel('$|error|$', fontsize=7)
    inset_ax.set_title('Zoom: Abs. Error', fontsize=8)
    inset_ax.tick_params(labelsize=6)
    zoom_mask = (line_x >= interface_pos - zoom_w) & (line_x <= interface_pos + zoom_w)
    error_zoom = error_line[zoom_mask]
    if len(error_zoom) > 0:
        ymin_inset = max(0, np.nanmin(error_zoom) * 0.9)
        inset_ax.set_ylim(bottom=ymin_inset)
    # ----- 手动绘制连接线 -----
    ylim = axes[2].get_ylim()
    y_mid = np.mean(ylim)
    x_left = interface_pos - zoom_w
    x_right = interface_pos + zoom_w
    con_left = mpatches.ConnectionPatch(
        xyA=(0, 1), xyB=(x_left, y_mid),
        coordsA="axes fraction", coordsB="data",
        axesA=inset_ax, axesB=axes[2],
        color="0.4", linewidth=1.2, linestyle='-'
    )
    fig.add_artist(con_left)
    con_right = mpatches.ConnectionPatch(
        xyA=(1, 1), xyB=(x_right, y_mid),
        coordsA="axes fraction", coordsB="data",
        axesA=inset_ax, axesB=axes[2],
        color="0.4", linewidth=1.2, linestyle='-'
    )
    fig.add_artist(con_right)

    inset_ax.patch.set_edgecolor('0.5')
    inset_ax.patch.set_linewidth(1.0)
    fig.suptitle(r'X-PINN prediction of $|(u,v)^T|$ at $t=%.1f$' % t_val, fontsize=15)
    fig.savefig('pred_error_modulus_XPINN.jpg', dpi=300)

# 绘制模长图，截面为 y=0
plot_modulus(mod_pred_2d, error_mod_2d, mod_true_2d, cross_direction='y')

plt.show()