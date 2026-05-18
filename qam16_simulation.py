"""
扩展二：16QAM在AWGN信道下的BER仿真
"""

import numpy as np
from scipy.special import erfc
import pandas as pd
np.random.seed(42)
Eb_N0_dB = np.arange(0, 18)  # 16QAM需要更高SNR范围
N_BITS = int(1e5)
MIN_ERRORS = 100

simulated_ber_16qam = []
theoretical_ber_16qam = []

# 16QAM星座点映射（Gray编码，4×4网格）
# 16QAM每符号4比特，每个维度2比特（4个电平）
# 电平映射：00→-3, 01→-1, 11→+1, 10→+3（Gray编码）
level_map = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1): +1,
    (1, 0): +3
}

print("=" * 60)
print("16QAM BER仿真开始（AWGN信道）")
print("=" * 60)

for Eb_N0_dB_val in Eb_N0_dB:
    Eb_N0_linear = 10 ** (Eb_N0_dB_val / 10)
    
    # 16QAM：每符号4比特，符号能量Es = 4×Eb（归一化前）
    # 16QAM星座点归一化因子：平均能量 = (1/16)×4×((-3)²+(-1)²+(+1)²+(+3)²) = 10
    # 归一化后 Es = 1，因此需要乘以 normalize_factor
    normalize_factor = 1.0 / np.sqrt(10)
    Es_N0_linear = 4 * Eb_N0_linear
    noise_std = np.sqrt(1 / (2 * Es_N0_linear))  # 复噪声每维标准差
    
    total_errors = 0
    total_bits = 0
    iteration = 0
    
    while total_errors < MIN_ERRORS and iteration < 200:
        iteration += 1
        
        # --- 生成比特流（需为4的倍数） ---
        n_bits = (N_BITS // 4) * 4
        bits = np.random.randint(0, 2, n_bits)
        
        # --- 16QAM调制 ---
        bits_reshaped = bits.reshape(-1, 4)  # 每行4比特
        
        # 每4比特映射为一个复数符号
        symbols = np.zeros(len(bits_reshaped), dtype=complex)
        for i in range(len(bits_reshaped)):
            b0, b1, b2, b3 = bits_reshaped[i]
            # 前两比特映射到I路，后两比特映射到Q路
            I_level = level_map[(b0, b1)]
            Q_level = level_map[(b2, b3)]
            symbols[i] = (I_level + 1j * Q_level) * normalize_factor
        
        # --- 生成复AWGN噪声 ---
        noise = (noise_std * np.random.randn(len(symbols)) + 
                 1j * noise_std * np.random.randn(len(symbols)))
        
        # --- 通过信道 ---
        received = symbols + noise
        
        # --- 16QAM解调 ---
        # 反归一化
        received_denorm = received / normalize_factor
        
        # 分别对I路和Q路进行4电平判决
        decoded_I_level = np.zeros(len(received_denorm), dtype=int)
        decoded_Q_level = np.zeros(len(received_denorm), dtype=int)
        
        # I路判决（基于接收值的实部）
        I_real = np.real(received_denorm)
        decoded_I_level = np.where(I_real < -2, -3,
                          np.where(I_real < 0, -1,
                          np.where(I_real < 2, +1, +3)))
        
        # Q路判决（基于接收值的虚部）
        Q_imag = np.imag(received_denorm)
        decoded_Q_level = np.where(Q_imag < -2, -3,
                          np.where(Q_imag < 0, -1,
                          np.where(Q_imag < 2, +1, +3)))
        
        # 反向查找Gray编码
        level_to_bits = {-3: (0, 0), -1: (0, 1), +1: (1, 1), +3: (1, 0)}
        decoded_bits = np.zeros(n_bits, dtype=int)
        for i in range(len(symbols)):
            I_bits = level_to_bits[decoded_I_level[i]]
            Q_bits = level_to_bits[decoded_Q_level[i]]
            decoded_bits[4*i:4*i+4] = [I_bits[0], I_bits[1], Q_bits[0], Q_bits[1]]
        
        # --- 统计误码 ---
        errors = np.sum(bits != decoded_bits)
        total_errors += errors
        total_bits += n_bits
    
    ber_sim = total_errors / total_bits
    simulated_ber_16qam.append(ber_sim)
    
    # 16QAM理论BER近似公式
    ber_theory = (3/8) * erfc(np.sqrt(0.4 * Eb_N0_linear))
    theoretical_ber_16qam.append(ber_theory)
    
    print(f"Eb/N0 = {Eb_N0_dB_val:2d} dB | 仿真BER = {ber_sim:.2e} | 理论BER(近似) = {ber_theory:.2e}")

print("\n16QAM仿真完成！")

# 保存结果
df_16qam = pd.DataFrame({
    'Eb_N0_dB': Eb_N0_dB,
    'Simulated_BER': simulated_ber_16qam,
    'Theoretical_BER': theoretical_ber_16qam
})
df_16qam.to_csv('data/16qam_ber_results.csv', index=False)