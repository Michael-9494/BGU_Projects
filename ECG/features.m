function [meanRR,BPM,Var_RR] = features(QRS_detect,Fs)

% features function gives the outputs we neet for learning
t_Qrs=find(QRS_detect==1)*(1/Fs);%find wehere the output is equal to 1. after multiply by Ts
% % we will get coresponding time vector
RR_interval=diff(t_Qrs);% find diff between two elements
%
meanRR=mean(RR_interval);
% BPM = 60/meanRR;
BPM = sum(QRS_detect)/(length(QRS_detect)/(Fs*60));

NN_interval=RR_interval(RR_interval>0.8*meanRR & RR_interval<1.2*meanRR);
%

% pNN50=sum(abs(diff(NN_interval))>((50*10^-3)*Fs))*((100)/length(NN_interval));
Var_RR = var(RR_interval);
end

