import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import torch
from Network import Vanilla_Net

# ------------------ 设置随机种子以保证可重复性 ------------------
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

# ------------------ 参数与网格 ------------------
beta_p, beta_n = 10.0, 0.1
grid_size = 401
x1 = np.linspace(-1, 1, grid_size)
x2 = np.linspace(-1, 1, grid_size)
X1, X2 = np.meshgrid(x1, x2)
t_val = .5
data = np.hstack((X1.ravel()[:, None], X2.ravel()[:, None],
                  t_val * np.ones((grid_size**2, 1))))
data = torch.tensor(data, dtype=torch.float64)
# ------------------ 辅助函数 ------------------
def level_set_function(data, model):
    t = data[:, 2]
    output = model(data)
    X = data[:, 0] + t * output[:, 0]
    Y = data[:, 1] + t * output[:, 1]
    lf = (X - 0.3) ** 2 + Y ** 2 - (torch.pi / 6) ** 2
    return lf[:, None]


def compute_phi_derivative(data, model):
    x = data.clone().detach().requires_grad_(True)
    phi = level_set_function(x, model)
    grad = torch.autograd.grad(phi.sum(), x, retain_graph=True, create_graph=True)[0]
    dphi_dx = grad[:, 0][:, None]
    dphi_dy = grad[:, 1][:, None]
    dphi_dt = grad[:, 2][:, None]
    d2phi_dx2 = torch.autograd.grad(dphi_dx.sum(), x, retain_graph=True, create_graph=False)[0][:, 0][:, None]
    d2phi_dy2 = torch.autograd.grad(dphi_dy.sum(), x, retain_graph=False, create_graph=False)[0][:, 1][:, None]
    return dphi_dx, dphi_dy, dphi_dt, d2phi_dx2, d2phi_dy2

def add_dimension(data, model):
    lf = level_set_function(data, model)
    data_add = torch.hstack((data, torch.abs(lf)))
    return data_add

def get_u(data, beta_p, beta_n, model):
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    lf = level_set_function(data, model)[:, 0]
    F = torch.sqrt((x1 - 0.3 * torch.cos(np.pi * t))**2 + (x2 - 0.3 * torch.sin(np.pi * t))**2)
    u_p = 6 / (beta_p * np.pi) * F**5 + (np.pi / 6)**4 * (1 / beta_n - 1 / beta_p)
    u_n = 6 / (beta_n * np.pi) * F**5
    u_x = torch.where(lf >= 0, u_p, u_n)
    return u_x[:, None]

def get_u_true_and_grad(data, beta_p, beta_n, model):
    """计算真解及其梯度（对 x1, x2）的解析式，返回 numpy 数组"""
    x1 = data[:, 0]
    x2 = data[:, 1]
    t = data[:, 2]
    lf = level_set_function(data, model)[:, 0]
    center_x1 = 0.3 * torch.cos(np.pi * t)
    center_x2 = 0.3 * torch.sin(np.pi * t)
    dx1 = x1 - center_x1
    dx2 = x2 - center_x2
    F = torch.sqrt(dx1**2 + dx2**2)

    # 分段值
    u_p = 6 / (beta_p * np.pi) * F**5 + (np.pi / 6)**4 * (1 / beta_n - 1 / beta_p)
    u_n = 6 / (beta_n * np.pi) * F**5
    u = torch.where(lf >= 0, u_p, u_n)

    # 梯度: dF/dx1 = (x1 - cx)/F, dF/dx2 = (x2 - cy)/F
    # d(u_p)/dxi = 30/(beta_p*pi) * F^4 * (xi - ci)/F = 30/(beta_p*pi) * F^3 * (xi - ci)
    # d(u_n)/dxi = 30/(beta_n*pi) * F^4 * (xi - ci)/F = 30/(beta_n*pi) * F^3 * (xi - ci)
    grad_factor_p = 30 / (beta_p * np.pi) * F**3
    grad_factor_n = 30 / (beta_n * np.pi) * F**3
    grad_u_x1 = torch.where(lf >= 0, grad_factor_p * dx1, grad_factor_n * dx1)
    grad_u_x2 = torch.where(lf >= 0, grad_factor_p * dx2, grad_factor_n * dx2)

    return u, torch.vstack([grad_u_x1, grad_u_x2]).T  # (N, 2)

# ------------------ 加载模型并预测网格解 ------------------
inverse_mapping = Vanilla_Net(3, 64, 2, 1).double()
inverse_mapping.load_state_dict(torch.load('inverse_mapping.mdl'))
inverse_mapping.eval()

model = Vanilla_Net(4, 32, 1, 2).double()
model.load_state_dict(torch.load('best_model_XIPINN_LM_XI-PINNstar_(3,32).mdl'))
model.eval()

# ------------------ 计算网格上的解（用于绘图） ------------------
u_true_grid = get_u(data, beta_p, beta_n, inverse_mapping).cpu().detach().numpy()
lf_grid = level_set_function(data, inverse_mapping).cpu().detach().numpy()
data_add = add_dimension(data, inverse_mapping)

with torch.no_grad():
    u_pred_grid = model(data_add).cpu().numpy()

error_grid = np.abs(u_true_grid - u_pred_grid)

# 转为二维
u_true_grid = u_true_grid.reshape(grid_size, grid_size)
u_pred_grid = u_pred_grid.reshape(grid_size, grid_size)
error_grid = error_grid.reshape(grid_size, grid_size)
lf_grid = lf_grid.reshape(grid_size, grid_size)


# ---------- 蒙特卡洛估计 L2 和 H1 相对误差 ----------
N_mc = 100000
# 生成随机点（固定时间 t = 0.5）
mc_points = np.random.uniform(-1, 1, (N_mc, 2))
mc_data_tmp = np.hstack([mc_points, np.random.uniform(0, 1, (N_mc, 1))])
mc_data_tmp = torch.tensor(mc_data_tmp).double()
# 真解与梯度
u_true_mc, grad_u_true_mc = get_u_true_and_grad(mc_data_tmp, beta_p, beta_n, inverse_mapping)
u_true_mc = u_true_mc.cpu().detach().numpy()
grad_u_true_mc = grad_u_true_mc.cpu().detach().numpy()
# 预测解与梯度（需用自动微分）
mc_data_add = add_dimension(mc_data_tmp, inverse_mapping)  # (N, 4)
u_pred_mc_tensor = model(mc_data_add)
# 计算梯度：输出对输入的梯度（只取前两列，即 x1, x2）
grad_pred = torch.autograd.grad(
    outputs=u_pred_mc_tensor, inputs=mc_data_add,
    grad_outputs=torch.ones_like(u_pred_mc_tensor),
    create_graph=False, retain_graph=False
)[0]
grad_pred = grad_pred.detach().cpu().numpy()
dphidx, dphidy, _, _, _ = compute_phi_derivative(mc_data_tmp, inverse_mapping)
dphidx = dphidx.detach().cpu().numpy()
dphidy = dphidy.detach().cpu().numpy()
phi = level_set_function(mc_data_tmp, inverse_mapping).cpu().detach().numpy()
dUdx = np.where(phi >= 0, grad_pred[:, 0][:, None] + dphidx * grad_pred[:, 3][:, None], grad_pred[:, 0][:, None] - dphidx * grad_pred[:, 3][:, None])
dUdy = np.where(phi >= 0, grad_pred[:, 1][:, None] + dphidy * grad_pred[:, 3][:, None], grad_pred[:, 1][:, None] - dphidy * grad_pred[:, 3][:, None])
u_pred_mc = u_pred_mc_tensor.detach().cpu().numpy().flatten()
grad_pred_mc = np.hstack((dUdx, dUdy))

# 相对误差计算
err_u = u_pred_mc - u_true_mc
err_grad = grad_pred_mc - grad_u_true_mc
# L2 norms (squared, mean)
norm_u_true_sq = np.mean(u_true_mc**2)
norm_u_err_sq = np.mean(err_u**2)
# H1 norms
norm_h1_true_sq = norm_u_true_sq + np.mean(np.sum(grad_u_true_mc**2, axis=1))
norm_h1_err_sq = norm_u_err_sq + np.mean(np.sum(err_grad**2, axis=1))

rel_L2 = np.sqrt(norm_u_err_sq / norm_u_true_sq)
rel_H1 = np.sqrt(norm_h1_err_sq / norm_h1_true_sq)

print(f"蒙特卡洛点数: {N_mc}")
print(f"相对 L2 误差 : {rel_L2:.6e}")
print(f"相对 H1 误差 : {rel_H1:.6e}")

# ---------- 绘图（降采样二维场） ----------
step = 4
X1_low = X1[::step, ::step]
X2_low = X2[::step, ::step]
u_pred_low = u_pred_grid[::step, ::step]
error_low = error_grid[::step, ::step]
lf_low = lf_grid[::step, ::step]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), constrained_layout=True)

im0 = axes[0].pcolormesh(X1_low, X2_low, u_pred_low, shading='auto', cmap='viridis')
axes[0].contour(X1_low, X2_low, lf_low, levels=[0], colors='k', linewidths=1.2, linestyles='--')
axes[0].set_title('Numerical Solution', fontsize=12)
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$y$')
axes[0].set_aspect('equal')
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.ax.tick_params(labelsize=10)
cbar0.ax.yaxis.get_offset_text().set_fontsize(10)


im1 = axes[1].pcolormesh(X1_low, X2_low, error_low, shading='auto', cmap='inferno')
axes[1].contour(X1_low, X2_low, lf_low, levels=[0], colors='white', linewidths=1.0, linestyles='--')
axes[1].set_title('Absolute Error Distribution', fontsize=12)
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('$y$')
axes[1].set_aspect('equal')
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.ax.tick_params(labelsize=10)
cbar1.ax.yaxis.get_offset_text().set_fontsize(10)

# 截面图
x1_fixed = 0.0
idx_x1 = np.argmin(np.abs(x1 - x1_fixed))
u_true_line = u_true_grid[:, idx_x1]
u_pred_line = u_pred_grid[:, idx_x1]
lf_line = lf_grid[:, idx_x1]
error_line = error_grid[:, idx_x1]
x2_line = x2

interface_idx = np.argmin(np.abs(lf_line))
x2_interface = x2_line[interface_idx]

axes[2].plot(x2_line, u_true_line, 'b-', linewidth=1.5, label='True')
axes[2].plot(x2_line, u_pred_line, 'r--', linewidth=1.5, label='Predicted')
axes[2].axvline(x2_interface, color='grey', linestyle=':', linewidth=1.2, label='Interface')
axes[2].set_xlabel('$y$')
axes[2].set_ylabel('$u$')
axes[2].set_title(f'Cross‑section at $x = {x1_fixed}$, $t={t_val}$', fontsize=12)
axes[2].legend(loc='upper right')

zoom_width = 0.1
axes[2].axvspan(x2_interface - zoom_width, x2_interface + zoom_width,
                alpha=0.2, color='yellow')

inset_ax = inset_axes(axes[2], width="50%", height="50%",
                       bbox_to_anchor=(0.45, 0.25, 0.6, 0.5),
                       bbox_transform=axes[2].transAxes, loc='center')
inset_ax.plot(x2_line, error_line, 'm-', linewidth=1.5)
inset_ax.axvline(x2_interface, color='grey', linestyle=':', linewidth=1)
inset_ax.set_xlim(x2_interface - zoom_width, x2_interface + zoom_width)
inset_ax.set_xlabel('$y$', fontsize=6)
inset_ax.set_ylabel('$|error|$', fontsize=7)
inset_ax.set_title('Zoom: Abs. Error', fontsize=8)
inset_ax.tick_params(labelsize=6)

mark_inset(axes[2], inset_ax, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=1.2)

plt.suptitle(f'XI-PINN$^*$ prediction and error at $t={t_val}$', fontsize=15)
plt.savefig('pred_error_crossection.jpg', dpi=300)
plt.show()