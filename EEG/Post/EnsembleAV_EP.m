function [AV_sig,t_AV_sig]=EnsembleAV_EP(signal,Fs,f_pulse,M)
%This function returns the Averege of 1 segment from signal by using
% ENSAMBLE AVERAGING,to do so first we choose the segment size acoording
% to the visual stemulation frequncy.
%   signal - the original signal
%   Fs - sample reate of the signal
%   F_pulse - the visual stemulation frequncy
%   M - number of segment to avrege
T = 1/f_pulse; %the time for one segment
N = T*Fs; %the sample number of each segment
AV_mat = reshape(signal(1 : N*M ), N, M); %each row is the next segment
AV_sig = sum(AV_mat,2)/M; %Avreging all segment
t_AV_sig = (0: 1/Fs : (N-1)/Fs)'; %the time of the first segment
end