% Simple spectrogram script
clear; close all;

% Parameters
filename = 'raw\gnss_rx.bin';
fs = 24e6;              % Sample rate (Hz)
n_fft = 8192*2;           % FFT size
overlap = 6144*2;         % Overlap (75%)

% Read data
file = fopen(filename, 'r');
data = fread(file, 24000000*2, 'float32');
fclose(file);

% Convert to complex
data_complex = data(1:2:end) + 1i * data(2:2:end);

% Compute spectrogram with window
win = hann(n_fft, 'periodic');
[S, F, T] = spectrogram(data_complex, win, overlap, n_fft, fs, 'centered');

% Plot
S_db = 10*log10(abs(S).^2);
medfilt2(S_db, [5 5]);
imagesc(T, F/1e6, S_db);
axis xy;
caxis([-50 -10]);
colormap(jet);
xlabel('Time (s)');
ylabel('Frequency (MHz)');
title('Spectrogram');
colorbar;