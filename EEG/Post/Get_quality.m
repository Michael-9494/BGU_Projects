function [SNR,PRD,RMS]=Get_quality(Signal,QSignal)
% input:
% Signal is the raw signal
% QSignal is the quantized same signal.
% output: Quantization error parameters accordng to course sylabus
Ss=var(Signal);
Sq=mean((Signal(:)-QSignal(:)).^2);% or (var - mean^2)
SNR=10*log10(Ss/Sq);
RMS=sqrt(Sq);% equal to 'std(Signal)'
PRD=sqrt(Sq*length(Signal)/(Signal(:)'*Signal(:)))*100;
% if mean=0 then also 'PRD=sqrt(Sq/Ss)*100'