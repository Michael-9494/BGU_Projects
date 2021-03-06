function Hd = LP_100_EMG
%LP_100_EMG Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 9.11 and DSP System Toolbox 9.13.
% Generated on: 31-Dec-2021 15:23:55

% IIR maximally flat Lowpass filter designed using the MAXFLAT function.

% All frequency values are in Hz.
Fs = 300;  % Sampling Frequency

Nb   = 16;   % Numerator Order
Na   = 16;   % Denominator Order
F3dB = 100;  % 3-dB Frequency

h  = fdesign.lowpass('Nb,Na,F3dB', Nb, Na, F3dB, Fs);
Hd = design(h, 'butter');

% [EOF]
