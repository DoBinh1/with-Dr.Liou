import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    Data loading function. 
    """
    data = sio.loadmat(filepath)
    x = data['Signal'][0,0]['y_values'][0,0]['values']
    x = x[:,0]
    return x

# --- 1. LOAD DATA ---
# Change directory
file_path = "E:/dr.liou/KAIST/1st/vibration/0Nm_BPFO_03.mat" 
x, fs = load_data(file_path)
N = len(x)
t = np.arange(N) / fs

# --- 2. FIND RESONANCE REGION (SPECTRAL KURTOSIS) ---
win_sizes = [64, 128, 256, 512, 1024]
best_SK_max = 0
best_fc = 0
best_freqs = []
best_SK_curve = []

print("Calculating Spectral Kurtosis...")

for win in win_sizes:
    noverlap = int(win * 0.75)
    # STFT (Spectrogram)
    f, t_spec, Zxx = signal.stft(x, fs, window='hamming', nperseg=win, noverlap=noverlap)
    
    S_pow = np.abs(Zxx)**2
    
    # Calculate Kurtosis
    # SK = Mean(x^4) / Mean(x^2)^2 - 2 
    mean_pow = np.mean(S_pow, axis=1)
    mean_pow_sq = np.mean(S_pow**2, axis=1)
    
    valid_idx = mean_pow > 1e-10
    sk_curve = np.zeros_like(mean_pow)
    sk_curve[valid_idx] = (mean_pow_sq[valid_idx] / (mean_pow[valid_idx]**2)) - 2
    
    curr_max_sk = np.max(sk_curve)
    
    if curr_max_sk > best_SK_max:
        best_SK_max = curr_max_sk
        best_idx = np.argmax(sk_curve)
        best_fc = f[best_idx]
        best_freqs = f
        best_SK_curve = sk_curve

# Identify resonance region (SK > 30% max)
threshold = best_SK_max * 0.3
res_indices = np.where(best_SK_curve > threshold)[0]

if len(res_indices) > 0:
    f_low = max(best_freqs[res_indices[0]], 50)
    f_high = min(best_freqs[res_indices[-1]], fs/2 - 100)
else:
    f_low, f_high = 1000, 2000 # Fallback

if (f_high - f_low) < 500:
    f_high = f_low + 500

print(f"RESONANCE REGION: [{f_low:.0f} - {f_high:.0f}] Hz (Center: {best_fc:.0f} Hz)")

# --- 3. FILTER BANDPASS + ENVELOPE ---
# Butterworth Bandpass
nyq = 0.5 * fs
b, a = signal.butter(4, [f_low/nyq, f_high/nyq], btype='bandpass')
x_filt = signal.filtfilt(b, a, x)

# Hilbert transform
analytic_signal = signal.hilbert(x_filt)
envelope = np.abs(analytic_signal)
envelope = envelope - np.mean(envelope) # Loại bỏ thành phần DC

# Envelope Spectrum
env_fft = np.abs(np.fft.fft(envelope)) * 2 / N
f_env = np.fft.fftfreq(N, d=1/fs)
half_N = N // 2
f_env_plot = f_env[:half_N]
env_fft_plot = env_fft[:half_N]

# --- 4. PLOT ---
plt.figure(figsize=(12, 8))

# Subplot 1: Spectral Kurtosis
plt.subplot(2, 2, 1)
plt.plot(best_freqs, best_SK_curve, 'b', linewidth=1.2)
plt.axvspan(f_low, f_high, color='r', alpha=0.2, label='Resonance Region')
plt.title('SPECTRAL KURTOSIS')
plt.xlabel('Frequency (Hz)')
plt.ylabel('SK')
plt.grid(True)
plt.legend()

# Subplot 2: Signal after filtering
plt.subplot(2, 2, 2)
plt.plot(t, x_filt, 'r')
plt.title(f'SIGNAL AFTER FILTERING [{f_low:.0f}-{f_high:.0f}] Hz')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim([0, 0.3])
plt.grid(True)

# Subplot 3: Envelope
plt.subplot(2, 2, 3)
plt.plot(t, envelope, 'b')
plt.title('ENVELOPE SIGNAL')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim([0, 0.3])
plt.grid(True)

# Subplot 4: Envelope Spectrum
plt.subplot(2, 2, 4)
plt.plot(f_env_plot, env_fft_plot, 'b', linewidth=1.2)
plt.title('ENVELOPE SPECTRUM')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim([0, 600])
plt.grid(True)

# Peak Picking
peaks, _ = signal.find_peaks(env_fft_plot, height=np.max(env_fft_plot)*0.15, distance=int(20/(fs/N)))

# Plot peak
plt.plot(f_env_plot[peaks], env_fft_plot[peaks], 'rv', markersize=8)
for p in peaks[:7]:
    plt.text(f_env_plot[p], env_fft_plot[p]*1.05, f"{f_env_plot[p]:.1f}", 
             ha='center', fontsize=9)

plt.tight_layout()
plt.show()