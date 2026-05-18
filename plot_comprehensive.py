"""
综合可视化：多调制方式 × 多信道 × 多维度对比
产出论文级配图（PDF矢量格式）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfc

# ==================== 全局样式 ====================
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# ==================== 读取全部数据 ====================
df_bpsk = pd.read_csv('data/bpsk_ber_results.csv')
df_qpsk = pd.read_csv('data/qpsk_ber_results.csv')
df_16qam = pd.read_csv('data/16qam_ber_results.csv')
df_rayleigh = pd.read_csv('data/rayleigh_ber_results.csv')

# ==================== 图A：多调制方式BER对比（AWGN） ====================
figA, axA = plt.subplots(figsize=(9, 6))

# BPSK
axA.semilogy(df_bpsk['Eb_N0_dB'], df_bpsk['Simulated_BER'], 'o-',
             color='#E74C3C', markerfacecolor='white', markeredgewidth=1.5,
             label='BPSK (Sim.)')

# QPSK
axA.semilogy(df_qpsk['Eb_N0_dB'], df_qpsk['Simulated_BER'], 's-',
             color='#3498DB', markerfacecolor='white', markeredgewidth=1.5,
             label='QPSK (Sim.)')

# 16QAM
axA.semilogy(df_16qam['Eb_N0_dB'], df_16qam['Simulated_BER'], 'D-',
             color='#2ECC71', markerfacecolor='white', markeredgewidth=1.5,
             label='16QAM (Sim.)')

# 理论曲线（虚线）
Eb_N0_dense = np.linspace(-3, 17, 100)
Eb_N0_lin = 10**(Eb_N0_dense/10)
axA.semilogy(Eb_N0_dense, 0.5*erfc(np.sqrt(Eb_N0_lin)), '--',
             color='#E74C3C', alpha=0.5, label='BPSK/QPSK (Theory)')
axA.semilogy(Eb_N0_dense, (3/8)*erfc(np.sqrt(0.4*Eb_N0_lin)), '--',
             color='#2ECC71', alpha=0.5, label='16QAM (Theory)')

axA.set_xlabel('$E_b/N_0$ (dB)', fontsize=13)
axA.set_ylabel('Bit Error Rate (BER)', fontsize=13)
axA.set_title('BER Performance Comparison: BPSK vs QPSK vs 16QAM (AWGN)',
              fontweight='bold', fontsize=14)
axA.set_ylim(1e-5, 0.5)
axA.grid(True, which='major', alpha=0.4)
axA.grid(True, which='minor', linestyle='--', alpha=0.2)
axA.legend(loc='lower left', frameon=True, fancybox=True)
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)

# 添加关键标注
axA.annotate('BPSK & QPSK:\nsame BER,\nQPSK has 2×\nspectral efficiency',
             xy=(6.8, 1e-3), xytext=(2, 1e-4),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=8, color='gray', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

figA.savefig('figures/figA_multimod_BER_AWGN.pdf')
figA.savefig('figures/figA_multimod_BER_AWGN.png')
print("图A已保存")

# ==================== 图B：AWGN vs 瑞利衰落对比 ====================
figB, axB = plt.subplots(figsize=(9, 6))

# AWGN仿真（BPSK）
axB.semilogy(df_bpsk['Eb_N0_dB'], df_bpsk['Simulated_BER'], 'o-',
             color='#3498DB', markerfacecolor='white', markeredgewidth=1.5,
             label='BPSK over AWGN (Sim.)')

# 瑞利衰落仿真
axB.semilogy(df_rayleigh['Eb_N0_dB'], df_rayleigh['Simulated_BER_Rayleigh'], 's-',
             color='#E74C3C', markerfacecolor='white', markeredgewidth=1.5,
             label='BPSK over Rayleigh Fading (Sim.)')

# 理论曲线
axB.semilogy(df_rayleigh['Eb_N0_dB'], df_rayleigh['Theoretical_BER_AWGN'], '--',
             color='#3498DB', alpha=0.5, label='AWGN Theory')
axB.semilogy(df_rayleigh['Eb_N0_dB'], df_rayleigh['Theoretical_BER_Rayleigh'], '--',
             color='#E74C3C', alpha=0.5, label='Rayleigh Theory')

axB.set_xlabel('$E_b/N_0$ (dB)', fontsize=13)
axB.set_ylabel('Bit Error Rate (BER)', fontsize=13)
axB.set_title('Channel Comparison: AWGN vs Rayleigh Fading (BPSK)',
              fontweight='bold', fontsize=14)
axB.set_ylim(1e-5, 0.5)
axB.grid(True, which='major', alpha=0.4)
axB.grid(True, which='minor', linestyle='--', alpha=0.2)
axB.legend(loc='lower left', frameon=True, fancybox=True)
axB.spines['top'].set_visible(False)
axB.spines['right'].set_visible(False)

# 添加关键标注
axB.annotate('Rayleigh fading:\nBER ∝ 1/SNR\n(slow decay)',
             xy=(20, 1e-2), fontsize=9, color='#E74C3C',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axB.annotate('AWGN:\nBER ∝ e^(-SNR)\n(fast decay)',
             xy=(10, 1e-4), fontsize=9, color='#3498DB',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

figB.savefig('figures/figB_channel_comparison.pdf')
figB.savefig('figures/figB_channel_comparison.png')
print("图B已保存")

# ==================== 图C：频谱效率对比 ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfc

# 读取之前保存的数据
df_bpsk = pd.read_csv('data/bpsk_ber_results.csv')
df_qpsk = pd.read_csv('data/qpsk_ber_results.csv')
df_16qam = pd.read_csv('data/16qam_ber_results.csv')

# 定义查找函数（加入保护）
def find_required_snr(df, ber_col, target=1e-3):
    """找到达到目标BER所需的最小Eb/N0，找不到则返回NaN"""
    # 确保BER列是数值
    ber_values = pd.to_numeric(df[ber_col], errors='coerce')
    # 找到第一个小于等于target的位置
    idx = (ber_values <= target)
    if idx.any():
        i = idx.idxmax()  # 第一个True的位置
        if i == 0:
            return df['Eb_N0_dB'].iloc[i]
        # 线性插值（在对数域）
        x1, x2 = df['Eb_N0_dB'].iloc[i-1], df['Eb_N0_dB'].iloc[i]
        y1, y2 = ber_values.iloc[i-1], ber_values.iloc[i]
        # 防止零值取对数出错
        if y1 <= 0 or y2 <= 0:
            return df['Eb_N0_dB'].iloc[i]
        log_target = np.log10(target)
        log_y1, log_y2 = np.log10(y1), np.log10(y2)
        t = (log_target - log_y1) / (log_y2 - log_y1)
        return x1 + t * (x2 - x1)
    else:
        print(f"警告：在 {ber_col} 中未达到 BER <= {target}，请扩展 SNR 范围")
        return float('nan')  # 返回NaN

snr_bpsk = find_required_snr(df_bpsk, 'Simulated_BER')
snr_qpsk = find_required_snr(df_qpsk, 'Simulated_BER')
snr_16qam = find_required_snr(df_16qam, 'Simulated_BER')

print(f"Required SNR for BER=1e-3: BPSK={snr_bpsk:.2f}, QPSK={snr_qpsk:.2f}, 16QAM={snr_16qam:.2f}")

# 绘图
figC, axC = plt.subplots(figsize=(8, 5.5))
modulations = ['BPSK', 'QPSK', '16QAM']
spectral_eff = [1, 2, 4]
required_snr = [snr_bpsk, snr_qpsk, snr_16qam]
colors = ['#E74C3C', '#3498DB', '#2ECC71']

for i in range(3):
    if np.isnan(required_snr[i]):
        print(f"跳过 {modulations[i]}，因为缺少数据")
        continue
    axC.scatter(required_snr[i], spectral_eff[i], s=200, 
                c=colors[i], edgecolors='black', linewidth=1.2, zorder=5)
    axC.annotate(modulations[i], 
                 (required_snr[i], spectral_eff[i]),
                 textcoords="offset points", xytext=(10, 10),
                 fontsize=11, fontweight='bold', color=colors[i])

# 只对有效点画连线
valid_idx = [i for i, v in enumerate(required_snr) if not np.isnan(v)]
if len(valid_idx) >= 2:
    valid_snr = [required_snr[i] for i in valid_idx]
    valid_se = [spectral_eff[i] for i in valid_idx]
    axC.plot(valid_snr, valid_se, '--', color='gray', alpha=0.5, zorder=1)

axC.set_xlabel('Required $E_b/N_0$ for BER = $10^{-3}$ (dB)', fontsize=13)
axC.set_ylabel('Spectral Efficiency (bps/Hz)', fontsize=13)
axC.set_title('Spectral Efficiency vs Power Efficiency Trade-off',
              fontweight='bold', fontsize=14)
axC.grid(True, alpha=0.3)
axC.spines['top'].set_visible(False)
axC.spines['right'].set_visible(False)

figC.savefig('figures/figC_spectral_efficiency.pdf')
figC.savefig('figures/figC_spectral_efficiency.png')
print("图C已保存")

# ==================== 图D：综合4面板图（论文级多子图） ====================
figD, axes = plt.subplots(2, 2, figsize=(14, 11))

# 面板(a)：多调制方式BER对比
ax = axes[0, 0]
ax.semilogy(df_bpsk['Eb_N0_dB'], df_bpsk['Simulated_BER'], 'o-', color='#E74C3C',
            markerfacecolor='white', markersize=5, label='BPSK')
ax.semilogy(df_qpsk['Eb_N0_dB'], df_qpsk['Simulated_BER'], 's-', color='#3498DB',
            markerfacecolor='white', markersize=5, label='QPSK')
ax.semilogy(df_16qam['Eb_N0_dB'], df_16qam['Simulated_BER'], 'D-', color='#2ECC71',
            markerfacecolor='white', markersize=5, label='16QAM')
ax.set_xlabel('$E_b/N_0$ (dB)')
ax.set_ylabel('BER')
ax.set_title('(a) Multi-Modulation BER (AWGN)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 面板(b)：AWGN vs 瑞利衰落
ax = axes[0, 1]
ax.semilogy(df_rayleigh['Eb_N0_dB'], df_rayleigh['Theoretical_BER_AWGN'], '-',
            color='#3498DB', linewidth=2, label='AWGN Theory')
ax.semilogy(df_rayleigh['Eb_N0_dB'], df_rayleigh['Simulated_BER_Rayleigh'], 'o-',
            color='#E74C3C', markerfacecolor='white', markersize=5, label='Rayleigh Sim.')
ax.semilogy(df_rayleigh['Eb_N0_dB'], df_rayleigh['Theoretical_BER_Rayleigh'], '--',
            color='#E74C3C', alpha=0.5, label='Rayleigh Theory')
ax.set_xlabel('$E_b/N_0$ (dB)')
ax.set_ylabel('BER')
ax.set_title('(b) AWGN vs Rayleigh Fading')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 面板(c)：QPSK星座图（带噪声样本）
ax = axes[1, 0]
# 生成QPSK星座点
constellation_qpsk = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
# 为每个星座点生成一些噪声样本用于展示
np.random.seed(123)
for c in constellation_qpsk:
    noise_samples = c + 0.15 * (np.random.randn(200) + 1j * np.random.randn(200))
    ax.scatter(np.real(noise_samples), np.imag(noise_samples), s=3, alpha=0.3, color='#3498DB')
# 绘制理想星座点
ax.scatter(np.real(constellation_qpsk), np.imag(constellation_qpsk), 
           s=120, c='#E74C3C', marker='o', edgecolors='black', linewidth=1.5, zorder=5)
# 标注
labels = ['00', '01', '11', '10']
for i, (c, label) in enumerate(zip(constellation_qpsk, labels)):
    ax.annotate(label, (np.real(c), np.imag(c)), textcoords="offset points", 
                xytext=(8, 8), fontsize=9, fontweight='bold')
# 绘制判决边界
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_xlabel('In-Phase (I)')
ax.set_ylabel('Quadrature (Q)')
ax.set_title('(c) QPSK Constellation (with noise samples)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 面板(d)：仿真参数摘要表格
ax = axes[1, 1]
ax.axis('off')
table_data = [
    ['Parameter', 'BPSK', 'QPSK', '16QAM'],
    ['Bits/Symbol', '1', '2', '4'],
    ['Spectral Eff.', '1 bps/Hz', '2 bps/Hz', '4 bps/Hz'],
    [f'BER=10⁻³ at', f'{snr_bpsk:.1f} dB', f'{snr_qpsk:.1f} dB', f'{snr_16qam:.1f} dB'],
    ['Channel', 'AWGN / Rayleigh', 'AWGN', 'AWGN'],
    ['Min Errors', '100', '100', '100'],
    ['Total Bits Processed', '>10⁶', '>10⁶', '>10⁷'],
]
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.22, 0.18, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2.0)
# 表头样式
for j in range(4):
    table[0, j].set_facecolor('#2C3E50')
    table[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('(d) Simulation Parameters Summary', fontweight='bold', y=1.02)

# 整体标题
figD.suptitle('Communication System Simulation: Comprehensive Performance Analysis',
              fontsize=16, fontweight='bold', y=1.01)

plt.tight_layout()
figD.savefig('figures/figD_comprehensive_analysis.pdf')
figD.savefig('figures/figD_comprehensive_analysis.png')
print("图D已保存（综合4面板图）")

print("\n🎉 全部图表生成完成！")