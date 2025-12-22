import numpy as np
import scipy.io as sio
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from vmdpy import VMD

# --- Load Data ---
mat_data = sio.loadmat(r'E:\dr.liou\KAIST\1st\vibration\0Nm_BPFI_03.mat')
signal_full = mat_data['Signal'][0, 0]['y_values'][0, 0]['values']
x = signal_full[0:25600, 0]

fs = 25600
N = len(x)
t = np.arange(N) / fs

# --- Add Noise ---
SNR_dB = -5
# Calculate signal power and noise power
sig_power = np.mean(x**2)
snr_linear = 10**(SNR_dB / 10)
noise_power = sig_power / snr_linear
# Generate white Gaussian noise
noise = np.random.normal(0, np.sqrt(noise_power), N)
x_noisy = x + noise

# --- VMD Decomposition ---
alpha = 1000  # Penalty factor   
tau = 0.01    # Time-step of the dual ascent        
K = 4         # Number of modes       
DC = 0        # 0 (no DC part imposed)     
init = 1      # 1 (initializes omegas uniformly)     
tol = 0.005   # Tolerance      

# Run VMD
u, u_hat, omega = VMD(x_noisy, alpha, tau, K, DC, init, tol)

# --- Analysis and Plotting ---
num_imfs = u.shape[0]
L = u.shape[1]

plt.figure(figsize=(10, 12))

for i in range(num_imfs):
    imf = u[i, :]
    
    # Envelope Analysis
    analytic_signal = hilbert(imf) 
    envelope = np.abs(analytic_signal)
    
    # Remove DC component from envelope
    envelope_detrended = envelope - np.mean(envelope)
    
    # Fast Fourier Transform
    Y = np.fft.fft(envelope_detrended)
    
    # Amplitude Spectrum calculation
    P2 = np.abs(Y / L)
    P1 = P2[0 : L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    
    # Frequency axis
    f = fs * np.arange(L // 2 + 1) / L
    
    # Plotting
    plt.subplot(num_imfs, 1, i + 1)
    plt.plot(f, P1, 'b', linewidth=1.2)
    plt.title(f'Envelope Spectrum of IMF {i+1}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 500])
    plt.grid(True)

plt.tight_layout()
plt.show()