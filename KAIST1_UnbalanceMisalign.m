function [peaks_info] = analyze_rotating_fault(x, fs)
    N = length(x);
    X_fft = abs(fft(x))*2/N;
    f = (0:N-1)*fs/N;
    
    %% FIND 1X
    range_1X = (f >= 30) & (f <= 70);
    [A_1X, idx] = max(X_fft(range_1X));
    f_temp = f(range_1X);
    f_1X = f_temp(idx);
    
    %% FIND 2X, 3X, 4X
    tol = 0.05 * f_1X;  % tolerance 5% of f_1X
    
    harmonics = struct();
    harmonics.f = [f_1X];
    harmonics.A = [A_1X];
    harmonics.order = [1];
    
    for k = 2:5
        f_expected = k * f_1X;
        range_k = (f >= f_expected-tol) & (f <= f_expected+tol);
        if any(range_k)
            [A_k, idx_k] = max(X_fft(range_k));
            f_temp_k = f(range_k);
            f_k = f_temp_k(idx_k);
            
            harmonics.f = [harmonics.f, f_k];
            harmonics.A = [harmonics.A, A_k];
            harmonics.order = [harmonics.order, k];
        end
    end
    
    %% PRINT RESULT
    peaks_info = harmonics;
    
    fprintf('=== Analysis Result ===\n');
    fprintf('Toc do uoc tinh: %.0f RPM\n', f_1X * 60);
    fprintf('\n%-10s %-12s %-12s\n', 'Order', 'Frequency (Hz)', 'Amplitude (g)');
    fprintf('------------------------------------\n');
    for i = 1:length(harmonics.order)
        fprintf('%-10d %-12.2f %-12.4f\n', ...
            harmonics.order(i), harmonics.f(i), harmonics.A(i));
    end
    
    %% PLOT
    figure;
    plot(f(1:N/2), X_fft(1:N/2), 'b'); hold on;
    for i = 1:length(harmonics.f)
        xline(harmonics.f(i), 'r--', sprintf('%dX', harmonics.order(i)));
        plot(harmonics.f(i), harmonics.A(i), 'rv', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    end
    xlim([0 min(500, fs/2)]); grid on;
    xlabel('Frequency (Hz)'); ylabel('Amplitude (g)');
    title(sprintf('FFT SPECTRUM - %s (f_{1X} = %.2f Hz)', f_1X));
    
end


%% Start analyzing
clear all; clc;
cd("KAIST\1st\vibration\")
load("0Nm_Misalign_03.mat");
x = Signal.y_values.values(:,1);
fs = 25600;

[peaks] = analyze_rotating_fault(x, fs);
