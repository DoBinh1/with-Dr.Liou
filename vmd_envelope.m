cd('E:\dr.liou\KAIST\1st\vibration');
load("0Nm_BPFO_03.mat");
x = Signal.y_values.values(1:25600, 1);
fs = 25600;
N = length(x);
t = (0:N-1)/fs;

SNR_dB = -20;
x_noisy = awgn(x, SNR_dB, 'measured');

[imfs,residual,info] = vmd(x_noisy, ...
    AbsoluteTolerance=5e-06, ...
    RelativeTolerance=0.005, ...
    MaxIterations=500, ...
    NumIMF=4, ...
    InitialIMFs=zeros(25600,4), ...
    PenaltyFactor=1000, ...
    InitialLM=complex(zeros(25601,1)), ...
    LMUpdateRate=0.01, ...
    InitializeMethod='peaks');

[L,num_imfs] = size(imfs);

for i=1:num_imfs
    IMF = imfs(:,i);
    
    % Envelope
    analytic_signal = hilbert(x); 
    envelope = abs(analytic_signal);

    envelope = envelope - mean(envelope);

    Y = fft(envelope);

    P2 = abs(Y/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);

    f = fs * (0:(L/2)) / L;

    subplot(num_imfs, 1, i);
    plot(f, P1, 'b', 'LineWidth', 1.2);
    title(['Phổ đường bao của IMF ', num2str(i)]);
    xlabel('Tần số (Hz)');
    xlim([0,500]);
    ylabel('Biên độ');
    grid on;
end    