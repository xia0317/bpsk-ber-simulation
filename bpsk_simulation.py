import numpy as np
from scipy.special import erfc

np.random.seed(42)
Eb_N0_dB = np.arange(-3, 11)
num_bits = int(1e5)
min_errors = 100

simulated_ber = []
theoretical_ber = []

for snr_db in Eb_N0_dB:
    print(f"Simulating Eb/N0 = {snr_db} dB...")
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    total_errors, total_bits = 0, 0

    while total_errors < min_errors:
        bits = np.random.randint(0, 2, num_bits)
        signal = 2 * bits - 1
        noise = noise_std * np.random.randn(num_bits)
        received_signal = signal + noise
        decoded_bits = (received_signal >= 0).astype(int)
        total_errors += np.sum(bits != decoded_bits)
        total_bits += num_bits

    ber_sim = total_errors / total_bits
    simulated_ber.append(ber_sim)
    ber_theory = 0.5 * erfc(np.sqrt(snr_linear))
    theoretical_ber.append(ber_theory)
    print(f"  Simulated BER = {ber_sim:.2e}, Theoretical BER = {ber_theory:.2e}")

print("Simulation finished!")
import pandas as pd

df = pd.DataFrame({
    'Eb_N0_dB': Eb_N0_dB,
    'Simulated_BER': simulated_ber,
    'Theoretical_BER': theoretical_ber
})
df['Error_Ratio'] = df['Simulated_BER'] / df['Theoretical_BER']

print(df.head(10))
print(df.describe())
df.to_csv('data/bpsk_ber_results.csv', index=False)
print("Data saved to bpsk_ber_results.csv")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.format': 'pdf'})

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.semilogy(df['Eb_N0_dB'], df['Simulated_BER'], 'o-', color='#E74C3C', markerfacecolor='white', label='Simulation (Monte Carlo)')
ax.semilogy(df['Eb_N0_dB'], df['Theoretical_BER'], '-', color='#2C3E50', linewidth=2, label='Theory')
ax.set_xlabel('$E_b/N_0$ (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('BER Performance of BPSK over AWGN Channel', fontweight='bold')
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend()
fig.savefig('figures/ber_waterfall.pdf')
fig.savefig('figures/ber_waterfall.png')
plt.show()
print("Figures saved!")