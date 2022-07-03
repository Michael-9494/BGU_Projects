%% Exp4 - Group 8
close all;clear all



% 1.2
[speech,Fs] = audioread('shalom_example.wav');

alpha =0.999; % Degree of pre-emphasis
figure
freqz([1 -alpha],1,[],Fs); % Calculate and display frequency response
 title 'Pre-Emphasis fitler';
figure
[z,p,~] = zplane([1 -alpha],1);
findzeros = findobj(z,'Type','line'); findpoles = findobj(p,'Type','line');
title('Zero - Pole Map');
set(findzeros,'Color','m','linewidth',1.2,'markersize',9);
set(findpoles,'Color','b','linewidth',1.2,'markersize',9);

% show Raw speech in time:
t = [0:length(speech)-1]*1/Fs; % time vector
figure,subplot(2,1,1), plot(t,speech);xlim([0 t(end)]);
xlabel 'Time[S]'; ylabel 'Amp'; title 'speech';
% show Raw speech in Frequency:
Y = fft(speech); L=length(speech);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
subplot(2,1,2), plot(f,P1) ; xlim([0 8000]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'speech in Frequency';



WindowLength=30*10^-3; % 1 seconds window
Overlap=50; % 50% overlap
[ProcessedSig,~]=PreProcess(speech,Fs,alpha,WindowLength,Overlap);

% show ProcessedSig speech in time:
figure,subplot(2,1,1), plot(t,ProcessedSig)
xlabel 'Time[S]'; ylabel 'Amp'; title 'speech after PreProcess';xlim([0 t(end)]);
% show ProcessedSig speech in Frequency:
Y = fft(ProcessedSig); L=length(ProcessedSig);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
subplot(2,1,2), plot(f,P1) ; xlim([0 8000]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'speech in Frequency';


% Window
N = 64;
rect = rectwin(N);
wvtool(rect)


%% segmantation:

close all; clear all; clc
[speech,Fs] = audioread('shalom_example.wav');
alpha=0.98;
WindowLength=30*10^-3;  % 30 [mS] window
Overlap=50;             % 50% overlap
[ProcessedSig,FramedSig]=PreProcess(speech,Fs,alpha,WindowLength,Overlap);

% 2.2

Idx=FindWordIdx(FramedSig,Fs,WindowLength,Overlap);

t=[0:length(speech)-1]*1/Fs; % time vector
figure,subplot(3,1,1)
plot(t,ProcessedSig);xlim([0 t(end)]);
ylim([min(ProcessedSig) max(ProcessedSig)]);
hold on
line([t(Idx(1)) t(Idx(1))],ylim,'color','m','linewidth',1.2)
line([t(Idx(2)) t(Idx(2))],ylim,'color','m','linewidth',1.2)
title 'signal - marker the speech '; xlabel 'Time [S]' ; ylabel 'Amplitude'
legend('speech','start ','end');grid on


% 2.4
   %playerObj = audioplayer(speech,Fs);
   %play(playerObj);
%
%    playerObj = audioplayer(ProcessedSig,Fs);  play(playerObj);
dt = 35*10^-3; % minimum time above threshold 'eta' [mS]
eta =50;
winlen = 30*10^-3; % 30 [mS]

[seg_ind,delta]=segmentation(ProcessedSig,winlen,eta,dt,Fs,Idx);

%  ProcessedSig = ProcessedSig(Idx(1):Idx(2));
t=[0:length(ProcessedSig)-1]*1/Fs;
delta=delta(1:length(ProcessedSig));
% show results:
subplot(3,1,2), plot(t,ProcessedSig); hold on;grid on
% show segmentation lines:
for i=1:length(seg_ind)
    line([t(seg_ind(i)) t(seg_ind(i))],ylim,'color','m','linewidth',1.2)
end
xlim([0 t(end)])
title 'Segmentation - Shalom speech signal' ;xlabel 'Time [S]' ;ylabel 'Amplitude'
legend('speech','segments');  axis tight;
hold on

subplot(3,1,3),plot(t,delta); hold on
yline(eta,'r','linewidth',1.2)
for i=1:length(seg_ind)

    line([t(seg_ind(i)) t(seg_ind(i))],ylim,'color','m','linewidth',1.2)
end
xlim([0 t(end)]);grid on
title 'Spectral error measure' ;xlabel 'Time [S]' ;ylabel 'Amplitude'
legend('\Delta_1',['\eta=' num2str(eta)],'segments');


%% 3

PhonemeSig = ProcessedSig(seg_ind(2):seg_ind(3));

% 3.1 Periodogram
figure
[Peri,omeg]=periodogram(PhonemeSig,[],'onesided');

 plot((Fs/(2*pi))*omeg,10*log10(abs(Peri))); grid on
title 'Periodogram - nonparametric Power Spectral Density ' ;
xlabel 'Frequency [Hz]' ;ylabel 'Magnitude [dB]'

rng default
% 3.3
% LPC analysis
P = Fs/1000 +2; 
% g-variance of the prediction error
[a,g]=lpc(PhonemeSig,P);
std_g = sqrt(g);
M = length(PhonemeSig);
%use the Random Noise Generator on samples
add_noise_to_filt=filter(1,a,std_g*randn(M,1));
% Autoregressive all-pole model parameters — Yule-Walker method
[a1,g1]=aryule(add_noise_to_filt,P);
std_g1 = sqrt(g1);
%add noise var to frequency response
[h,omeg1]=freqz(std_g1,a1);

 
% find formants:
rts = roots(a1);
%Because the LPC coefficients are real-valued, the roots occur in complex
% conjugate pairs. Retain only the roots with one sign for the imaginary
% part and determine the angles corresponding to the roots.
rts = rts(imag(rts)>=0);
[~,Poles_idx]=maxk(abs(rts),3);
angz = atan2(imag(rts),real(rts));
Formants = angz(Poles_idx).*(Fs/(2*pi));
Formants = sort(Formants);
h1 = figure;
 plot((Fs/(2*pi))*omeg1,20*log10(abs(h)),'LineWidth',1.4,'Color','r');
  xlabel 'Frequency (Hz)' ;ylabel 'Gain [dB]'; hold on
 plot((Fs/(2*pi))*omeg,10*log10(Peri),'Color','b'); grid on

hold on
 title 'Spectral Estimation of the Phoneme /a/ ' ;
 xlabel 'Frequency [Hz]' ; ylabel 'Gain [dB]'
 legend('LPC Spectral Estimation','Periodogram')
annotation(h1,'textbox',[ 0.15 0.09 0.09 0.1],'String',...
 {sprintf('Estimated Formants F1=%d, F2=%d, F3=%d',Formants(1),Formants(2),...
 Formants(3))},'FitBoxToText','on');grid on


h2 = figure;
[z,p,k]  = zplane(1,a1); hold on; grid on
findzeros = findobj(z,'Type','line'); findpoles = findobj(p,'Type','line');
findk = findobj(k,'Type','line');
set(findzeros,'Color','b','linewidth',1.1,'markersize',9);
set(findpoles,'Color','b','linewidth',1.1,'markersize',9);
set(findk,'Color','b','linewidth',1.1,'markersize',9);
hold on
[z1,~,~]=zplane(rts(Poles_idx(1:3)));
findzeros1 = findobj(z1,'Type','line');
set(findzeros1,'Color','r','linewidth',1.2,'markersize',9);
xlim([-1.1 1.1]);ylim([-1.1 1.1]);grid on
title('AR Model Poles- using LPC- Phoneme \a\ ')
legend('LPC Zeros','LPC Poles','Z-plane Axes','Chosen Poles','Location','southeast')


 
%       [h1,h2]=estimatePhonemeFormants(PhonemeSig,Fs,'\a\');


%% 4.5
N=(30*10^-3)*Fs; % number of samples in each frame
%  win=rectwin(N);
 Overlap=100;
 Overlap_sam= ((Overlap)*N)/100;
 FramedSig=enframe(ProcessedSig,Overlap_sam); 
% Energy and ZCR
NRG=calcNRG(FramedSig);
[M,~]=size(FramedSig);
NRG=reshape((repmat(NRG,1,N))',[N*M,1]);
ZCR=calcZCR(FramedSig);
ZCR=reshape((repmat(ZCR,1,N))',[N*M,1]);
ZCR = ZCR-min(ZCR);ZCR = ZCR/max(ZCR)*max(NRG);

figure
subplot(2,1,1), plot(t(1:length(ProcessedSig)),ProcessedSig); hold on;grid on
% show segmentation lines:
for i=1:length(seg_ind)
    line([t(seg_ind(i)) t(seg_ind(i))],ylim,'color','m','linewidth',1.2)
end
xlim([0 t(end)])
title 'Segmentation - Shalom speech signal' ;xlabel 'Time [S]' ;ylabel 'Amplitude'
legend('speech','segments');  axis tight;
hold on

subplot(2,1,2)
plot(t(1:length(NRG)),NRG,'color','r','LineWidth',1.2);hold on
plot(t(1:length(ZCR)),ZCR,'color','k','LineWidth',1.2);
xlim([0 t(end)]);grid on;hold on
for i=1:length(seg_ind)
    line([t(seg_ind(i)) t(seg_ind(i))],ylim,'color','m','linewidth',1.2)
end
title (['Energy and ZCR of the signal in ' num2str((N/Fs)*10^3) '[mS] window']) ;xlabel 'Time [S]' ;
ylabel 'Arbitrary amplitude'
legend('NRG','ZCR')

%% 4.6 FeatExt :
% extract features from each frame:
% \sh\ 
% take the first phoneme from segmentation
Phoneme_sh=ProcessedSig(seg_ind(1):seg_ind(2));
framed_Phoneme=enframe(Phoneme_sh,rectwin(N),Overlap_sam);
[FeatsVector_sh,~]=FeatExt(Phoneme_sh,Fs,framed_Phoneme);
 [h11,h21]=estimatePhonemeFormants(Phoneme_sh,Fs,'\Sh\');
% \a\ 
Phoneme_a=ProcessedSig(seg_ind(2):seg_ind(3));
framed_Phoneme=enframe(Phoneme_a,rectwin(N),Overlap_sam);
[FeatsVector_a,~]=FeatExt(Phoneme_a,Fs,framed_Phoneme); 

% \l\ 
Phoneme_L=ProcessedSig(seg_ind(3):seg_ind(4));
framed_Phoneme=enframe(Phoneme_L,rectwin(N),Overlap_sam);
[FeatsVector_L,~]=FeatExt(Phoneme_L,Fs,framed_Phoneme);
%  [h12,h22]=estimatePhonemeFormants(Phoneme_L,Fs,'\L\');
% \o\  
Phoneme_o=ProcessedSig(seg_ind(4):seg_ind(5));
framed_Phoneme=enframe(Phoneme_o,rectwin(N),Overlap_sam);
[FeatsVector_o,~]=FeatExt(Phoneme_o,Fs,framed_Phoneme);
% [h13,h23]=estimatePhonemeFormants(Phoneme_sh,Fs,'\o\');
% \m\ - 5 
Phoneme_m=ProcessedSig(seg_ind(5):seg_ind(end));
framed_Phoneme=enframe(Phoneme_m,rectwin(N),Overlap_sam);
[FeatsVector_m,~]=FeatExt(Phoneme_m,Fs,framed_Phoneme);
% [h14,h24]=estimatePhonemeFormants(Phoneme_m,Fs,'\M\');


%% 6
% 6.2
% do stft with 50 % overlap
N=[  (20*10^-3) (30*10^-3) (60*10^-3)]*Fs;
figure
for i=1:length(N)
    window=hamming(N(i));
    hold on
    subplot(1,3,i)
    noverlap = ((50)*N(i))/100;
    nfft = N(i);
    spectrogram(ProcessedSig,window,noverlap,nfft,Fs,'yaxis')
    title(['\Shalom\-Spectrogram. window ' num2str((N(i)/Fs)*10^3) '[mS]->N=' num2str(N(i)) '. 50% overlap'])
    colormap jet
end

% do stft with 90 % overlap
N=[  (20*10^-3) (30*10^-3) (60*10^-3)]*Fs;
figure
for i=1:length(N)
    window=hamming(N(i));
    hold on
    subplot(1,3,i)
    noverlap = ((90)*N(i))/100;
    nfft = N(i);
    spectrogram(ProcessedSig,window,noverlap,nfft,Fs,'yaxis')
    title(['\Shalom\-Spectrogram. window ' num2str((N(i)/Fs)*10^3) '[mS]->N=' num2str(N(i)) '. 90% overlap'])
    colormap jet
end
