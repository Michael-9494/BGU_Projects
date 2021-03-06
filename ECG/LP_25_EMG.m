function Hd = LP_25_EMG
%LP_50_EMG Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 9.11 and Signal Processing Toolbox 8.7.
% Generated on: 29-Dec-2021 09:26:10

% IIR maximally flat Lowpass filter designed using the MAXFLAT function.

% All frequency values are in Hz.
Fs = 300;  % Sampling Frequency

Nb   = 16;  % Numerator Order
Na   = 16;  % Denominator Order
F3dB = 25;  % 3-dB Frequency

h  = fdesign.lowpass('Nb,Na,F3dB', Nb, Na, F3dB, Fs);
Hd = design(h, 'butter');

% [EOF]
