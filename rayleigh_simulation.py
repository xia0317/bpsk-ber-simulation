"""
扩展三：BPSK在瑞利衰落信道下的BER仿真

瑞利衰落信道模型：
- 信道增益 h = (randn + j×randn) / √2，其中实部和虚部都是N(0, 1/2)
- |h| 服从瑞利分布
- 接收信号 y = h × s + n
- 接收端假设已知h（理想信道估计），通过均衡恢复：y_eq = y / h = s + n/h
"""

import numpy as np
from scipy.special import erfc
import pandas as pd
np.random.seed(42)
Eb_N0_dB = np.arange(0, 31)  # 瑞利衰落需要更大SNR范围
N_BITS = int(1e5)
MIN_ERRORS = 100

simulated_ber_rayleigh = []
theoretical_ber_rayleigh = []
theoretical_ber_awgn = []  # 对比：AWGN理论值

print("=" * 60)
print("BPSK BER仿真开始（瑞利衰落信道）")
print("=" * 60)

for Eb_N0_dB_val in Eb_N0_dB:
    Eb_N0_linear = 10 ** (Eb_N0_dB_val / 10)
    noise_std = np.sqrt(1 / (2 * Eb_N0_linear))
    
    total_errors = 0
    total_bits = 0
    iteration = 0
    
    while total_errors < MIN_ERRORS and iteration < 500:
        iteration += 1
        
        # --- BPSK调制 ---
        bits = np.random.randint(0, 2, N_BITS)
        signal = 2 * bits - 1  # BPSK: 0→-1, 1→+1
        
        # --- ★★★ 生成瑞利衰落信道系数 ★★★ ---
        # 实部和虚部各从N(0,1/2)采样，使得|h|²的期望为1
        # 即信道平均增益为1（不放大也不衰减，只引入随机性）
        h = (np.random.randn(N_BITS) + 1j * np.random.randn(N_BITS)) / np.sqrt(2)
        
        # --- AWGN噪声 ---
        noise = noise_std * (np.random.randn(N_BITS) + 1j * np.random.randn(N_BITS))
        
        # --- 通过瑞利衰落信道 ---
        # 接收信号 = 信道增益 × 发送信号 + 噪声
        received = h * signal + noise
        
        # --- ★★★ 均衡（理想信道估计，接收端已知h） ★★★ ---
        # 除以信道系数恢复发送信号
        # 注意：这放大了噪声！当|h|很小时（深衰落），噪声被严重放大
        equalized = received / h
        # 均衡后的信号 = signal + noise/h
        # 当|h|很小时，noise/h可以非常大
        
        # --- 解调判决（取实部，与BPSK相同） ---
        decoded = (np.real(equalized) >= 0).astype(int)
        
        # --- 统计误码 ---
        errors = np.sum(bits != decoded)
        total_errors += errors
        total_bits += N_BITS
    
    ber_sim = total_errors / total_bits
    simulated_ber_rayleigh.append(ber_sim)
    
    # 瑞利衰落理论BER
    snr = Eb_N0_linear
    ber_theory_rayleigh = 0.5 * (1 - np.sqrt(snr / (snr + 2)))
    theoretical_ber_rayleigh.append(ber_theory_rayleigh)
    
    # AWGN理论BER（对比用）
    ber_theory_awgn_val = 0.5 * erfc(np.sqrt(Eb_N0_linear))
    theoretical_ber_awgn.append(ber_theory_awgn_val)
    
    print(f"Eb/N0 = {Eb_N0_dB_val:2d} dB | 仿真BER = {ber_sim:.2e} | 理论BER(瑞利) = {ber_theory_rayleigh:.2e} | 理论BER(AWGN) = {ber_theory_awgn_val:.2e}")

print("\n瑞利衰落信道仿真完成！")
print("注意：瑞利衰落下的BER下降速度远慢于AWGN——这就是多径效应的代价。")

# 保存结果
df_rayleigh = pd.DataFrame({
    'Eb_N0_dB': Eb_N0_dB,
    'Simulated_BER_Rayleigh': simulated_ber_rayleigh,
    'Theoretical_BER_Rayleigh': theoretical_ber_rayleigh,
    'Theoretical_BER_AWGN': theoretical_ber_awgn
})
df_rayleigh.to_csv('data/rayleigh_ber_results.csv', index=False)