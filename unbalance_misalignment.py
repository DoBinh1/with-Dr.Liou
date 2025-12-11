import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys


def analyze_rotating_fault(file_path):
    # 1. LOAD DATA
    data = sio.loadmat(file_path)
    x = data['Signal'][0,0]['y_values'][0,0]['values']
    x = x[:,0]
    fs = 25600

    # 2. FFT PROCESS
    N = len(x)
    X_fft = np.abs(np.fft.fft(x)) * 2 / N
    freqs = np.fft.fftfreq(N, d=1/fs)
    
    mask = freqs >= 0
    freqs = freqs[mask]
    X_fft = X_fft[mask]

    # 3. FIND 1X (Fundamental Frequency)
    range_1X = (freqs >= 30) & (freqs <= 70)

    idx_max = np.argmax(X_fft[range_1X])
    f_1X = freqs[range_1X][idx_max]
    A_1X = X_fft[range_1X][idx_max]

    harmonics = [(1, f_1X, A_1X)]

    # 4. FIND HARMONICS (2X - 5X)
    tol = 0.05 * f_1X # Tolerance 5%
    for k in range(2, 6):
        f_target = k * f_1X
        range_k = (freqs >= f_target - tol) & (freqs <= f_target + tol)
        if np.any(range_k):
            idx_k = np.argmax(X_fft[range_k])
            harmonics.append((k, freqs[range_k][idx_k], X_fft[range_k][idx_k]))

    # 5. PRINT RESULTS
    print(f"\n=== ANALYSIS RESULT (File: {file_path.split('/')[-1]}) ===")
    print(f"Estimated Speed: {f_1X * 60:.0f} RPM")
    print(f"{'Order':<10} {'Freq (Hz)':<12} {'Amp (g)':<12}")
    print("-" * 36)
    for h in harmonics:
        print(f"{h[0]:<10} {h[1]:<12.2f} {h[2]:<12.4f}")

    # 6. PLOT
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, X_fft, 'b', linewidth=0.8)
    
    # Plot harmonic peak
    max_peak_val = 0
    for h in harmonics:
        plt.plot(h[1], h[2], 'rv', markersize=8)
        plt.text(h[1], h[2], f'{h[0]}X', ha='center', va='bottom', 
                 color='red', fontweight='bold', fontsize=10)
        if h[2] > max_peak_val: max_peak_val = h[2]

    plt.xlim(0, 500) 
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (g)')
    if max_peak_val > 0:
        plt.ylim(0, max_peak_val * 1.3)
    plt.title(f'FFT SPECTRUM - Unbalance/Misalignment (f1X = {f_1X:.2f} Hz)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- RUN ---
# Change directory
file_path = "E:/dr.liou/KAIST/1st/vibration/0Nm_Normal.mat"
analyze_rotating_fault(file_path)