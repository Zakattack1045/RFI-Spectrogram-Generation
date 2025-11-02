% Parameters
Fs = 24e6;                 % sampling rate (Hz)
duration_s = 2;          % seconds to read
N = round(Fs * duration_s);  % number of complex samples to read

% Open file and read only the first 10 s worth of IQ (float32, little-endian)
fname = 'raw\gnss_rx_24.bin';
f = fopen(fname, 'rb', 'ieee-le');
if f < 0
    error('Failed to open file: %s', fname);
end

% Read interleaved I,Q as a 2-by-N matrix for efficiency
% Note: fread reads column-major; each column will be [I; Q]
[iq, count] = fread(f, [2, N], 'single=>single');
fclose(f);

% Handle case where file is shorter than requested
if count < 2*N
    warning('File had only %.3f seconds of data. Proceeding with available samples.', count/2/Fs);
end

% Split into I and Q and form complex vector
I = iq(1, :);
Q = iq(2, :);
X = complex(I, Q);

% Example: compute and plot a 2D spectrogram (two-sided, centered)
[Power, Freq, Time] = pspectrum(X, Fs, 'spectrogram', ...
    'TwoSided', true, 'FrequencyLimits', [-Fs/2, Fs/2]);
PowdB = 10*log10(Power + eps);

figure;
imagesc(Time, Freq, PowdB);
axis xy; colormap turbo; colorbar;
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Spectrogram (first 10 s, dB)');