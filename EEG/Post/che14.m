function Hd = che120
%CHE120 Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 9.10 and Signal Processing Toolbox 8.6.
% Generated on: 08-Dec-2021 20:50:03

% Chebyshev Type I Bandpass filter designed using FDESIGN.BANDPASS.

% All frequency values are in Hz.
Fs = 200;  % Sampling Frequency

N      = 4;  % Order
Fpass1 = 8;   % First Passband Frequency
Fpass2 = 13;  % Second Passband Frequency
Apass  = 1;   % Passband Ripple (dB)

% Construct an FDESIGN object and call its CHEBY1 method.
h  = fdesign.bandpass('N,Fp1,Fp2,Ap', N, Fpass1, Fpass2, Apass, Fs);
Hd = design(h, 'cheby1');

% [EOF]
