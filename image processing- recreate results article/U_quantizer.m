function [Qsignal,Q] = U_quantizer(input_img,bits)
% Inputs: 
% Signal - Original signal [vector]
% bits - Amount of bits for quantization
%
% Outputs:
% Qsignal - New quantized signal [vector]
% Q - Contains vector with decision & quantization
% levels [ (quantization levels) X 2 ]


%in this function we will take a signal with its range and quantizisd it
Sig = input_img;
Sig = im2double(Sig);
number_pixels = size(Sig,1)*size(Sig,2);
Sig = reshape(Sig,number_pixels,1); % shifting the image into vector

N = bits;
M = 2^N; %number of steps
% Xmin = min(Range);
% Xmax = max(Range);
quant_interval = max(max(Sig)) - min(min(Sig)); %the range
delta = quant_interval/M;
%now we make the quantization vec, with their value we will produce
%quantizisd vector
Q = 1:M;
Q = (Q + ((min(min(Sig))/delta) -(1/2))*ones(1,M))*delta;


% map the signal in the range to [0,M-1] discrete steps of the range

quant_amp = (Sig-min(min(Sig)))*(M-1);
semi_quant = quant_amp/quant_interval;
Qsignal = round(semi_quant);
% replace the Qsignal with the level as associated with Q (the decision)
% the +1 is for level zero in Q to be one
Qsignal = Q(Qsignal+1);

%reshaping the image to the original image size
Qsignal = reshape(Qsignal,size(input_img,1),size(input_img,2)); 
% Qsignal = uint8(Qsignal*255);
end