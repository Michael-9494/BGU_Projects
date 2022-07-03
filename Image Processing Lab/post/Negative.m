function [N_IMG]=Negative(IMG)
% Inputs
% IMG - Gray levels image
%
% Outputs:
% N_IMG - Negative image of IMG
%to make it a negative we need to make all the Dark as Light and the
%opposit. first we will multiply the data by minus one so the higher number
%becomes the lower now. and then we will add 1 to each pixel so we will
%stay in between 0-1

N_IMG = IMG.*(-1);
N_IMG = N_IMG+ones(size(N_IMG));

end 