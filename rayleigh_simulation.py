import numpy as np
from scipy.special import erfc
import pandas as pd
np.random.seed(42)
Eb_N0_dB = np.arange(0, 31)  
N_BITS = int(1e5)
MIN_ERRORS = 100

simulated_ber_rayleigh = []
theoretical_ber_rayleigh = []
theoretical_ber_awgn = [] 

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

        bits = np.random.randint(0, 2, N_BITS)
        signal = 2 * bits - 1  # BPSK: 0→-1, 1→+1

        h = (np.random.randn(N_BITS) + 1j * np.random.randn(N_BITS)) / np.sqrt(2)

        noise = noise_std * (np.random.randn(N_BITS) + 1j * np.random.randn(N_BITS))

        received = h * signal + noise

        decoded = (np.real(equalized) >= 0).astype(int)

        errors = np.sum(bits != decoded)
        total_errors += errors
        total_bits += N_BITS
    
    ber_sim = total_errors / total_bits
    simulated_ber_rayleigh.append(ber_sim)

    snr = Eb_N0_linear
    ber_theory_rayleigh = 0.5 * (1 - np.sqrt(snr / (snr + 2)))
    theoretical_ber_rayleigh.append(ber_theory_rayleigh)

    ber_theory_awgn_val = 0.5 * erfc(np.sqrt(Eb_N0_linear))
    theoretical_ber_awgn.append(ber_theory_awgn_val)
    
    print(f"Eb/N0 = {Eb_N0_dB_val:2d} dB | 仿真BER = {ber_sim:.2e} | 理论BER(瑞利) = {ber_theory_rayleigh:.2e} | 理论BER(AWGN) = {ber_theory_awgn_val:.2e}")

print("\n瑞利衰落信道仿真完成！")
print("注意：瑞利衰落下的BER下降速度远慢于AWGN——这就是多径效应的代价。")

df_rayleigh = pd.DataFrame({
    'Eb_N0_dB': Eb_N0_dB,
    'Simulated_BER_Rayleigh': simulated_ber_rayleigh,
    'Theoretical_BER_Rayleigh': theoretical_ber_rayleigh,
    'Theoretical_BER_AWGN': theoretical_ber_awgn
})
df_rayleigh.to_csv('data/rayleigh_ber_results.csv', index=False)