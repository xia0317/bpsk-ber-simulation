"""
【最终修正版】QPSK 在 AWGN 信道下的 BER 仿真
映射规则：bit 1 → +1, bit 0 → -1 (与 BPSK 一致)
"""
import numpy as np
from scipy.special import erfc
import pandas as pd

np.random.seed(42)
Eb_N0_dB = np.arange(-3, 11)
N_BITS = int(1e5)
MIN_ERRORS = 100

simulated_ber_qpsk = []
theoretical_ber_qpsk = []

print("=" * 60)
print("QPSK BER 仿真开始（AWGN）")
print("=" * 60)

for Eb_N0_dB_val in Eb_N0_dB:
    Eb_N0_linear = 10 ** (Eb_N0_dB_val / 10)
    Es_N0_linear = 2 * Eb_N0_linear          # QPSK 符号能量 = 2*Eb
    noise_std = np.sqrt(1 / (2 * Es_N0_linear))  # 复噪声每维标准差
    
    total_errors = 0
    total_bits = 0
    iteration = 0
    
    while total_errors < MIN_ERRORS and iteration < 100:
        iteration += 1
        
        n_bits = (N_BITS // 2) * 2
        bits = np.random.randint(0, 2, n_bits)
        
        # QPSK 调制（Gray 编码）
        bits_reshaped = bits.reshape(-1, 2)
        # ★ 修改这里：bit 1 → +1, bit 0 → -1
        I = 2 * bits_reshaped[:, 0] - 1
        Q = 2 * bits_reshaped[:, 1] - 1
        symbols = (I + 1j * Q) / np.sqrt(2)   # 归一化
        
        # 复噪声
        noise = (noise_std * np.random.randn(len(symbols)) +
                 1j * noise_std * np.random.randn(len(symbols)))
        
        # 信道
        received = symbols + noise
        
        # ★ 解调判决：>=0 判为 1
        decoded_I = (np.real(received) >= 0).astype(int)
        decoded_Q = (np.imag(received) >= 0).astype(int)
        
        decoded_bits = np.column_stack([decoded_I, decoded_Q]).flatten()
        
        # 统计误码
        errors = np.sum(bits != decoded_bits)
        total_errors += errors
        total_bits += n_bits
    
    ber_sim = total_errors / total_bits
    simulated_ber_qpsk.append(ber_sim)
    
    # QPSK 理论 BER 与 BPSK 相同
    ber_theory = 0.5 * erfc(np.sqrt(Eb_N0_linear))
    theoretical_ber_qpsk.append(ber_theory)
    
    print(f"Eb/N0 = {Eb_N0_dB_val:2d} dB | 仿真 BER = {ber_sim:.2e} | 理论 BER = {ber_theory:.2e}")

print("\nQPSK 仿真完成！")

# 保存正确结果
df_qpsk = pd.DataFrame({
    'Eb_N0_dB': Eb_N0_dB,
    'Simulated_BER': simulated_ber_qpsk,
    'Theoretical_BER': theoretical_ber_qpsk
})
df_qpsk.to_csv('data/qpsk_ber_results.csv', index=False)
print("数据已保存至 qpsk_ber_results.csv")