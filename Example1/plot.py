import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================
# 1. 数据导入——逐一列出所有情况
# ============================================

# 格式: (方法名, 规模标记, L, W, loss文件路径, l2相对误差文件路径)
# 规模标记: 'small' 或 'large'，用于控制颜色深浅和线型

cases = [
    ('PINN',     'small', '3', '32', 'loss_data_PINN_(3,32).npz',    'relative_errors_data_PINN_(3,32).npz'),
    ('PINN',     'large', '3', '64', 'loss_data_PINN_(3,64).npz',    'relative_errors_data_PINN_(3,64).npz'),
    ('X-PINN',    'small', '3', '23', 'loss_data_X-PINN_(3,23).npz',   'relative_errors_data_X-PINN_(3,23).npz'),
    ('X-PINN',    'large', '3', '45', 'loss_data_X-PINN_(3,45).npz',   'relative_errors_data_X-PINN_(3,45).npz'),
    ('XI-PINN',  'small', '3', '32', 'loss_data_XI-PINN_(3,32).npz', 'relative_errors_data_XI-PINN_(3,32).npz'),
    ('XI-PINN',  'large', '3', '64', 'loss_data_XI-PINN_(3,64).npz', 'relative_errors_data_XI-PINN_(3,64).npz'),
    ('XI-PINN*', 'small', '3', '32', 'loss_data_XI-PINNstar_(3,32).npz','relative_errors_data_XI-PINNstar_(3,32).npz'),
    ('XI-PINN*', 'large', '3', '64', 'loss_data_XI-PINNstar_(3,64).npz','relative_errors_data_XI-PINNstar_(3,64).npz'),
]

num_epochs = 2001              # 原始数据点数
epochs_all = np.arange(num_epochs)   # 0,1,...,2000
epochs = epochs_all[::10]      # 每隔10个取一个点，共201个点
data = {'epoch': epochs}
def load_npz_first_array(filepath):
    """加载指定npz文件中的第一个数组，并展平为一维数组"""
    with np.load(filepath) as npz:
        arr = npz[list(npz.keys())[0]]
    return arr.flatten()
for method, size, L, W, loss_file, l2_file in cases:
    try:
        loss_full = np.abs(load_npz_first_array(loss_file))
        l2_full   = np.abs(load_npz_first_array(l2_file))
    except Exception as e:
        print(f"读取文件出错: {loss_file} 或 {l2_file}，错误: {e}")
        # 出错时用NaN填充，绘图时会自动跳过
        loss_full = np.full(num_epochs, np.nan)
        l2_full   = np.full(num_epochs, np.nan)
    # 每隔10个点采样
    loss = loss_full[::10]
    l2   = l2_full[::10]
    # 避免对数坐标出现0或负数，设置极小值下限
    eps = 1e-16
    loss[loss < eps] = eps
    l2[l2 < eps] = eps
    col_loss = f'{method}_{size}_loss'
    col_l2   = f'{method}_{size}_L2'
    data[col_loss] = loss
    data[col_l2]   = l2
# ============================================
# 2. 绘图设置
# ============================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 15,
    'axes.labelsize': 12,
    'axes.titlesize': 18,
    'legend.fontsize': 9,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 3.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
method_colors = {
    'PINN':     '#E64B35',
    'X-PINN':    '#4DBBD5',
    'XI-PINN':  '#00A087',
    'XI-PINN*': '#F39B2C'
}
def lighten_color(hex_color, factor=0.6):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f'#{r:02x}{g:02x}{b:02x}'
color_map = {}
for method, base in method_colors.items():
    color_map[(method, 'small')] = lighten_color(base, factor=0.1)
    color_map[(method, 'large')] = base
linestyle_map = {'small': '--', 'large': '-'}
# ============================================
# 3. 创建画布与双栏子图
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
handles = []
# ---------- 左图：Loss ----------
for method, size, L, W, _, _ in cases:
    col_loss = f'{method}_{size}_loss'
    label = f'{method} ({L},{W})'
    line, = ax1.loglog(
        data['epoch'], data[col_loss],
        color=color_map[(method, size)],
        linestyle=linestyle_map[size],
        label=label,
        marker='o' if size == 'large' else None,
        markersize=3,
        markevery=(0.1, 0.2)      # 标记在10%和20%位置
    )
    if len(handles) < len(cases):
        handles.append(line)
ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
ax1.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
ax1.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
ax1.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax1.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
# ---------- 右图：L2 相对误差 ----------
for method, size, L, W, _, _ in cases:
    col_l2 = f'{method}_{size}_L2'
    label = f'{method} ({L},{W})'
    ax2.loglog(
        data['epoch'], data[col_l2],
        color=color_map[(method, size)],
        linestyle=linestyle_map[size],
        label=label,
        marker='o' if size == 'large' else None,
        markersize=3,
        markevery=(0.1, 0.2)
    )
ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Relative L2 Error')
ax2.set_title('Relative $L^2$ Error')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
ax2.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
ax2.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax2.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
# ---------- 共享图例 ----------
fig.legend(
    handles,
    [h.get_label() for h in handles],
    loc='center right',
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,
    frameon=True,
    fancybox=True,
    shadow=False,
    borderpad=0.5,
    labelspacing=0.3,
    title='Method & Size(L, W)',
    title_fontsize=10
)
plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig('loss_l2_dualplot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('loss_l2_dualplot.eps', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()