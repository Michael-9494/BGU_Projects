function [FeatsVector,Feat_title]=FeatExt(Phoneme,Fs,framedPhoneme)
%FeatExt the phoneme and give clculates all the features that is requaierd
% Phoneme – one phoneme (after pre-processing)
% Fs – sampling frequency
%  framedPhoneme – the phoneme after framing
% OUTPUT:
% FeatsVector – 1X24 vector of features of the analyzed phoneme
%  Feat_title – 1X24 cell array of the names of the calculated features
P = Fs/1000 +2;
FeatsVector = [];
Feat_title = [];

% 1 - mean energy
Mean_NRG = mean(calcNRG(framedPhoneme));
FeatsVector = [FeatsVector; Mean_NRG];
Feat_title = [Feat_title; {'mean energy'}];
% 2 - mean ZCR
Zero_mean_Crossing_Signal = mean(calcZCR(framedPhoneme));
FeatsVector = [FeatsVector; Zero_mean_Crossing_Signal];

Feat_title = [Feat_title; {'mean ZCR'}];
% 3- Pitch
[Pitch,~] = sift(framedPhoneme,Fs);

FeatsVector = [FeatsVector; Pitch];
Feat_title = [Feat_title; {'pitch'}];
% 4 - LPC coefficients
Lags = 2;
[Formamt,outAR] = formants(Phoneme,P,[],Fs,2*Fs/Lags);
FeatsVector = [FeatsVector; outAR(2:end)'];
for i=2:19
    Feat_title = [Feat_title; {sprintf('LPC coefficient #%d',i-1)}];
end
% 5 - first 3 formants
FeatsVector = [FeatsVector; Formamt(1); Formamt(2); Formamt(3)];
for i=1:3
    Feat_title = [Feat_title; {sprintf('Formant #%d',i)}];
end
end





