function [S_signal,t_AV_sig]=ExpoAV_EP(signal,Fs,f_pulse,alpha,M)
%This function returns the Averege of 1 segment from signal by 
% using EXPONENTIAL AVERAGING, to do so first we will build a weight
% vector ftom the recursuve formula and then after multiplying it with
%  its corrollated segment the sum of it will give us the average signal.
%
%   signal - the original signal
%   Fs - sample reate of the signal
%   F_pulse - the visual stemulation frequncy
%   alpha - the weight
%   M - number of segment to avrege

n = 0:M-1; %this will be the power fo the wiegt vec
exp_vec = alpha*(1-alpha).^n'; %a weight vector to mulyply with the segmentation
weights = flip(exp_vec); %the correct order of elements for the rucursive formula
T = 1/f_pulse; %the time for one segment
N = T*Fs; %the sample number of each segment
AV_mat = reshape(signal(1 : N*M ), N, M); %each row is the next segment
S_mat = AV_mat*weights; %The recursive formula
S_signal = sum(S_mat,2); %Avreging all segment
t_AV_sig = (0:1/Fs:(N-1)/Fs)';%the time of the first segment
end