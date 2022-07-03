function Hd = butterworth100
%BUTTERWORTH100 Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 9.10 and Signal Processing Toolbox 8.6.
% Generated on: 08-Dec-2021 20:42:09

% Butterworth Bandpass filter designed using FDESIGN.BANDPASS.

% All frequency values are in Hz.
Fs = 200;  % Sampling Frequency

N   = 100;  % Order
Fc1 = 8;    % First Cutoff Frequency
Fc2 = 13;   % Second Cutoff Frequency

% Construct an FDESIGN object and call its BUTTER method.
h  = fdesign.bandpass('N,F3dB1,F3dB2', N, Fc1, Fc2, Fs);
Hd = design(h, 'butter');

% [EOF]
