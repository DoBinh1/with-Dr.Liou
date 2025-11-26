clear; clc;

%% 1. LOAD DATA
cd('E:\dr.liou\KAIST\1st\vibration');
load("4Nm_BPFI_10.mat");
x = Signal.y_values.values(1:25600, 1);
fs = 25600;
N = length(x);
t = (0:N-1)/fs;

%% 2. FIND THE RESONANCE REGION (SPECTRAL KURTOSIS)
win_sizes = [64, 128, 256, 512, 1024];
best_SK = 0; best_fc = 0; best_F = []; best_SK_curve = [];

for win = win_sizes
    [S, F, ~] = spectrogram(x, hamming(win), round(win*0.75), win, fs);
    S_pow = abs(S).^2;
    SK = zeros(size(F));
    for k = 1:length(F)
        s = S_pow(k,:);
        mu2 = mean(s);
        if mu2 > 1e-10
            SK(k) = mean(s.^2)/(mu2^2) - 2;
        end
    end
    [max_sk, idx] = max(SK);
    if max_sk > best_SK
        best_SK = max_sk;
        best_fc = F(idx);
        best_F = F;
        best_SK_curve = SK;
    end
end

% RESONANCE REGION (SK > 30% max)
res_idx = find(best_SK_curve > best_SK*0.3);
f_low = max(best_F(res_idx(1)), 50);
f_high = min(best_F(res_idx(end)), fs/2-100);
if (f_high - f_low) < 500, f_high = f_low + 500; end

fprintf('RESONANCE REGION: [%.0f - %.0f] Hz (fc = %.0f Hz)\n', f_low, f_high, best_fc);

%% 3. BANDPASS + ENVELOPE
[b, a] = butter(4, [f_low f_high]/(fs/2), 'bandpass');
x_filt = filtfilt(b, a, x);
envelope = abs(hilbert(x_filt)) - mean(abs(hilbert(x_filt)));

% ENVELOPE
env_fft = abs(fft(envelope))*2/N;
f_env = (0:N-1)*fs/N;

%% 4. PLOT
figure('Position', [100 100 1200 600]);

subplot(2,2,1);
plot(best_F, best_SK_curve, 'b', 'LineWidth', 1.2);
hold on; xregion(f_low, f_high, 'FaceColor', 'r', 'FaceAlpha', 0.2);
xlabel('Frequency (Hz)'); ylabel('SK'); title('SPECTRAL KURTOSIS'); grid on;

subplot(2,2,2);
plot(t, x_filt, 'r');
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('SIGNAL AFTER FILTERING [%.0f-%.0f] Hz', f_low, f_high));
xlim([0 0.3]); grid on;

subplot(2,2,3);
plot(t, abs(hilbert(x_filt)), 'b');
xlabel('Time (s)'); ylabel('Amplitude'); title('ENVELOPE');
xlim([0 0.3]); grid on;

subplot(2,2,4);
plot(f_env(1:N/2), env_fft(1:N/2), 'b', 'LineWidth', 1.2);
xlabel('Frequency (Hz)'); ylabel('Amplitude'); title('ENVELOPE SPECTRUM');
xlim([0 600]); grid on;

% Frequency peak numbering
[pks, locs] = findpeaks(env_fft(1:N/2), f_env(1:N/2), ...
    'MinPeakHeight', max(env_fft)*0.15, 'MinPeakDistance', 20);
hold on; plot(locs, pks, 'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
for i = 1:min(7, length(pks))
    text(locs(i), pks(i)*1.1, sprintf('%.1f', locs(i)), 'FontSize', 9, 'HorizontalAlignment', 'center');
end
