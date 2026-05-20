import numpy as np
from scipy.special import erfc
import pandas as pd
np.random.seed(42)
Eb_N0_dB = np.arange(0, 18)  
N_BITS = int(1e5)
MIN_ERRORS = 100

simulated_ber_16qam = []
theoretical_ber_16qam = []

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

    normalize_factor = 1.0 / np.sqrt(10)
    Es_N0_linear = 4 * Eb_N0_linear
    noise_std = np.sqrt(1 / (2 * Es_N0_linear))  
    
    total_errors = 0
    total_bits = 0
    iteration = 0
    
    while total_errors < MIN_ERRORS and iteration < 200:
        iteration += 1

        n_bits = (N_BITS // 4) * 4
        bits = np.random.randint(0, 2, n_bits)

        bits_reshaped = bits.reshape(-1, 4)  

        symbols = np.zeros(len(bits_reshaped), dtype=complex)
        for i in range(len(bits_reshaped)):
            b0, b1, b2, b3 = bits_reshaped[i]

            I_level = level_map[(b0, b1)]
            Q_level = level_map[(b2, b3)]
            symbols[i] = (I_level + 1j * Q_level) * normalize_factor

        noise = (noise_std * np.random.randn(len(symbols)) + 
                 1j * noise_std * np.random.randn(len(symbols)))

        received = symbols + noise

        received_denorm = received / normalize_factor

        decoded_I_level = np.zeros(len(received_denorm), dtype=int)
        decoded_Q_level = np.zeros(len(received_denorm), dtype=int)

        I_real = np.real(received_denorm)
        decoded_I_level = np.where(I_real < -2, -3,
                          np.where(I_real < 0, -1,
                          np.where(I_real < 2, +1, +3)))

        Q_imag = np.imag(received_denorm)
        decoded_Q_level = np.where(Q_imag < -2, -3,
                          np.where(Q_imag < 0, -1,
                          np.where(Q_imag < 2, +1, +3)))

        level_to_bits = {-3: (0, 0), -1: (0, 1), +1: (1, 1), +3: (1, 0)}
        decoded_bits = np.zeros(n_bits, dtype=int)
        for i in range(len(symbols)):
            I_bits = level_to_bits[decoded_I_level[i]]
            Q_bits = level_to_bits[decoded_Q_level[i]]
            decoded_bits[4*i:4*i+4] = [I_bits[0], I_bits[1], Q_bits[0], Q_bits[1]]

        errors = np.sum(bits != decoded_bits)
        total_errors += errors
        total_bits += n_bits
    
    ber_sim = total_errors / total_bits
    simulated_ber_16qam.append(ber_sim)

    ber_theory = (3/8) * erfc(np.sqrt(0.4 * Eb_N0_linear))
    theoretical_ber_16qam.append(ber_theory)
    
    print(f"Eb/N0 = {Eb_N0_dB_val:2d} dB | 仿真BER = {ber_sim:.2e} | 理论BER(近似) = {ber_theory:.2e}")

print("\n16QAM仿真完成！")

df_16qam = pd.DataFrame({
    'Eb_N0_dB': Eb_N0_dB,
    'Simulated_BER': simulated_ber_16qam,
    'Theoretical_BER': theoretical_ber_16qam
})
df_16qam.to_csv('data/16qam_ber_results.csv', index=False)