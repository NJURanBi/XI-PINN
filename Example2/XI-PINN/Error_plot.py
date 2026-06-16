import numpy as np
import matplotlib.pyplot as plt
import torch
from Network import Vanilla_Net          # 假设网络结构与之前相同，仅输入维度变为 5

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

# ================= 问题参数 =================
a, b, c = 0.9, 0.7, 0.5
t_val = 1

# ================= 辅助函数 =================
def level_set_function(data):
    """3D 椭球界面水平集函数"""
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]
    F1 = (x1 * np.cos(np.pi * t / 2) + x2 * np.sin(np.pi * t / 2)) ** 2
    F2 = (-x1 * np.sin(np.pi * t / 2) + x2 * np.cos(np.pi * t / 2)) ** 2
    F3 = (x3 - 0.5 * t + 0.25) ** 2
    lf = F1 / a**2 + F2 / b**2 + F3 / c**2 - 1
    return lf[:, None]

def add_dimension(data):
    lf = level_set_function(data)
    add_x = np.ones_like(lf)
    index_n = np.where(lf < 0)[0]
    add_x[index_n] = -1
    data = np.hstack((data, add_x))
    return data

def get_u(data):
    """真解"""
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t = data[:, 3]
    lf = level_set_function(data)[:, 0]
    u_p = np.exp(-(x1**2 + x2**2 + x3**2)) * np.cos(t)
    u_n = (np.exp(-t) * np.sin(np.pi * x1) * np.sin(np.pi * x2) * np.exp(x3) + 1) / 2
    return np.where(lf >= 0, u_p, u_n)[:, None]

def get_u_true_and_grad(data):
    """
    真解及其空间梯度（对 x1, x2, x3）的解析式
    返回: u (N,), grad (N, 3)
    """
    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]
    t  = data[:, 3]
    lf = level_set_function(data)[:, 0]

    # ----- 区域 p -----
    u_p = np.exp(-(x1**2 + x2**2 + x3**2)) * np.cos(t)
    # 梯度
    grad_p_x1 = -2 * x1 * u_p
    grad_p_x2 = -2 * x2 * u_p
    grad_p_x3 = -2 * x3 * u_p

    # ----- 区域 n -----
    factor = np.exp(-t)
    sin1 = np.sin(np.pi * x1)
    sin2 = np.sin(np.pi * x2)
    cos1 = np.cos(np.pi * x1)
    cos2 = np.cos(np.pi * x2)
    exp_x3 = np.exp(x3)

    u_n = (factor * sin1 * sin2 * exp_x3 + 1) / 2
    grad_n_x1 = factor * np.pi * cos1 * sin2 * exp_x3 / 2
    grad_n_x2 = factor * sin1 * np.pi * cos2 * exp_x3 / 2
    grad_n_x3 = factor * sin1 * sin2 * exp_x3 / 2

    u = np.where(lf >= 0, u_p, u_n)
    grad_x1 = np.where(lf >= 0, grad_p_x1, grad_n_x1)
    grad_x2 = np.where(lf >= 0, grad_p_x2, grad_n_x2)
    grad_x3 = np.where(lf >= 0, grad_p_x3, grad_n_x3)

    return u, np.stack([grad_x1, grad_x2, grad_x3], axis=1)

# ---- 计算水平集函数对空间变量的导数（用于 XI‑PINN 梯度修正） ----

# ================= 1. 网格生成（z=0, t=0.5 平面） =================
grid_size = 401
x1 = np.linspace(-1, 1, grid_size)
x2 = np.linspace(-1, 1, grid_size)
X1, X2 = np.meshgrid(x1, x2)

data_grid = np.hstack((
    X1.ravel()[:, None],
    X2.ravel()[:, None],
    np.zeros((grid_size**2, 1)),          # z = 0
    t_val * np.ones((grid_size**2, 1))   # t = 0.5
))

# 真解
u_true_grid = get_u(data_grid)
lf_grid = level_set_function(data_grid)

# ================= 2. 加载 XI‑PINN 模型 =================
model = Vanilla_Net(5, 64, 1, 3).double()
# 请将文件名替换为你的训练好的模型路径
model.load_state_dict(torch.load('best_model_XIPINN_LM_4_64.mdl', map_location='cpu'))
model.eval()

# 预测
data_add_grid = add_dimension(data_grid)
data_add_tensor = torch.tensor(data_add_grid).double()
with torch.no_grad():
    u_pred_grid = model(data_add_tensor).cpu().numpy()

error_grid = np.abs(u_true_grid - u_pred_grid)

# 转为二维图像
u_pred_2d = u_pred_grid.reshape(grid_size, grid_size)
error_2d  = error_grid.reshape(grid_size, grid_size)
lf_2d     = lf_grid.reshape(grid_size, grid_size)

# ================= 3. 蒙特卡洛误差 (L2 & H1) =================
N_mc = 100000
mc_points = np.random.uniform(-1, 1, (N_mc, 3))   # (x1, x2, x3)
mc_data = np.hstack([mc_points, np.random.uniform(0, 1, (N_mc, 1))])

# 真解与梯度
u_true_mc, grad_true_mc = get_u_true_and_grad(mc_data)  # (N,), (N,3)

# 预测与梯度修正
mc_data_add = add_dimension(mc_data)  # 现在包含 |lf|
mc_input = torch.tensor(mc_data_add, requires_grad=True, dtype=torch.float64)
u_pred_tensor = model(mc_input)

# 自动微分求 du/d(inputs)，输入顺序: x1, x2, x3, t, |lf|
grad_pred_mc = torch.autograd.grad(
    outputs=u_pred_tensor, inputs=mc_input,
    grad_outputs=torch.ones_like(u_pred_tensor),
    create_graph=False, retain_graph=False
)[0][:,:3].detach().cpu().numpy()

u_pred_mc = u_pred_tensor.detach().cpu().numpy().flatten()

# 链式法则修正空间梯度
phi = level_set_function(mc_data)[:, 0]
sign_phi = np.where(phi >= 0, 1.0, -1.0)

# 相对误差
err_u = u_pred_mc - u_true_mc
err_grad = grad_pred_mc - grad_true_mc

norm_u_true_sq = np.mean(u_true_mc**2)
norm_u_err_sq = np.mean(err_u**2)

norm_h1_true_sq = norm_u_true_sq + np.mean(np.sum(grad_true_mc**2, axis=1))
norm_h1_err_sq  = norm_u_err_sq + np.mean(np.sum(err_grad**2, axis=1))

rel_L2 = np.sqrt(norm_u_err_sq / norm_u_true_sq)
rel_H1 = np.sqrt(norm_h1_err_sq / norm_h1_true_sq)

print(f"蒙特卡洛点数: {N_mc}")
print(f"相对 L2 误差 : {rel_L2:.6e}")
print(f"相对 H1 误差 : {rel_H1:.6e}")

# ================= 4. 绘图（降采样） =================
step = 4
X1_low = X1[::step, ::step]
X2_low = X2[::step, ::step]
u_pred_low = u_pred_2d[::step, ::step]
error_low  = error_2d[::step, ::step]
lf_low     = lf_2d[::step, ::step]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)

# ---------- 左：预测解 ----------
im0 = axes[0].pcolormesh(X1_low, X2_low, u_pred_low, shading='auto', cmap='viridis')
axes[0].contour(X1_low, X2_low, lf_low, levels=[0], colors='k', linewidths=1.2, linestyles='--')
axes[0].set_title('Numerical Solution', fontsize=12)
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$y$')
axes[0].set_aspect('equal')
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.ax.tick_params(labelsize=10)
cbar0.ax.yaxis.get_offset_text().set_fontsize(10)

# ---------- 右：绝对误差 ----------
im1 = axes[1].pcolormesh(X1_low, X2_low, error_low, shading='auto', cmap='inferno')
axes[1].contour(X1_low, X2_low, lf_low, levels=[0], colors='white', linewidths=1.0, linestyles='--')
axes[1].set_title('Absolute Error Distribution', fontsize=12)
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('$y$')
axes[1].set_aspect('equal')
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.ax.tick_params(labelsize=10)
cbar1.ax.yaxis.get_offset_text().set_fontsize(10)

plt.suptitle(f'XI‑PINN prediction on slice $z=0$ at $t={t_val}$', fontsize=15)
plt.savefig('pred_error_3D_slice.jpg', dpi=300)
plt.show()