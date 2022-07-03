function [N_IMG]=Negative(IMG)
% Inputs:
% IMG - Gray levels image
%
% Outputs:
% N_IMG - Negative image of IMG

N_IMG = 256*ones(size(IMG),'uint8')-IMG;
