%% Exp3 postG8
clear all; clc;
%% EXP 1
T1 = readmatrix("Michael_Mor_T1.xlsx"); 
EEG = T1(:,1);
alpha = T1(:,2); beta = T1(:,3); delta_sig = T1(:,4); theta = T1(:,5);
Fs = 200; 
t = 0: 1/Fs :(length(EEG)-1)/Fs;

figure
subplot(5,1,1)
plot(t,alpha)
title '\alpha'
subplot(5,1,2)
plot(t,beta)
title '\beta'
subplot(5,1,3)
plot(t,delta_sig)
title '\delta_\sigma'
subplot(5,1,4)
plot(t,theta)
title '\theta'
subplot(5,1,5)
plot(t,EEG)
title 'EEG'
%% 1.1
alpha_nomove = alpha(1:14778);
EEG_nomove = EEG(1:14778);
t_nomove = t(1:14778);
butterw4 = filter(butterworth4,EEG_nomove);
che14 = filter(che14,EEG_nomove);
butterw100 = filter(butterworth100,EEG_nomove);
che1100 = filter(che1100,EEG_nomove);

figure;
subplot(6,1,1)
plot(t_nomove,butterw4)
ylim([-20 20])
title 'filtered with butterworth with order 5'
subplot(6,1,2)
plot(t_nomove,che14)
ylim([-20 20])
title 'filtered with chebyshev1 with order 5'
subplot(6,1,3)
plot(t_nomove,butterw100)
ylim([-20 20])
title 'filtered with butterworth with order 100'
subplot(6,1,4)
plot(t_nomove,che1100)
ylim([-20 20])
title 'filtered with chebyshev1 with order 100'
subplot(6,1,5)
plot(t_nomove,alpha_nomove')
ylim([-20 20])
title 'alpha'
subplot(6,1,6)
plot(t_nomove,EEG_nomove)
title 'EEG'


% 1.1.1
error_butter4 = (butterw4 - alpha_nomove); 
error_che14 = (che14 - alpha_nomove); 
error_butter100 = (butterw100 - alpha_nomove); 
error_che1100 = (che1100 - alpha_nomove);

% butterworth filter order 4
MeanDiff = mean(error_butter4) - 0;
Var_a = var(error_butter4);
Var_b = 0;
SEComb = sqrt(Var_a/length(error_butter4) + Var_b/length(error_butter4));
df = length(error_butter4) - 2;
tstat = MeanDiff/SEComb;
P_butter4 = 1 - tcdf(tstat , df)

% butterworth filter order 100
MeanDiff = mean(error_butter100) - 0;
Var_a = var(error_butter100);
Var_b = 0;
SEComb = sqrt(Var_a/length(error_butter100) + Var_b/length(error_butter100));
df = length(error_butter100) - 2;
tstat = MeanDiff/SEComb;
P_butter100 = 1 - tcdf(tstat , df)

% chebyshev 1 filter order 100
MeanDiff = mean(error_che1100) - 0;
Var_a = var(error_che1100);
Var_b = 0;
SEComb = sqrt(Var_a/length(error_che1100) + Var_b/length(error_che1100));
df = length(error_che1100) - 2;
tstat = MeanDiff/SEComb;
P_che1100 = 1 - tcdf(tstat , df)

% chebyshev 1 filter order 4
MeanDiff = mean(error_che14) - 0;
Var_a = var(error_che14);
Var_b = 0;
SEComb = sqrt(Var_a/length(error_che14) + Var_b/length(error_che14));
df = length(error_che14) - 2;
tstat = MeanDiff/SEComb;
P_che14 = 1 - tcdf(tstat , df)

figure;
subplot(4,1,1)
plot(error_butter4)
title(['error for butterworth order 4 with P=',num2str(P_butter4)])
subplot(4,1,2)
plot(error_butter100)
title(['error for butterworth order 100 with P=',num2str(P_butter100)])
subplot(4,1,3)
plot(error_che14)
title(['error for chebyshev order 4 with P=',num2str(P_che14)])
subplot(4,1,4)
plot(error_che1100)
title(['error for chebyshev order 100 with P=',num2str(P_che1100)])


% 1.1.2
corr_butter20 = xcorr(butterw4,alpha_nomove,'normalized'); 
corr_che120 = xcorr(che14,alpha_nomove,'normalized'); 
corr_butter100 = xcorr(butterw100,alpha_nomove,'normalized'); 
corr_che1100 = xcorr(che1100,alpha_nomove,'normalized');

corrb20 = max(corr_butter20)
corrb100 = max(corr_butter100)
corrc20 = max(corr_che120)
corrc100 = max(corr_che1100)

% 1.1.3

% 1.1.4
labels = readtable('Labels.xlsx');
alpha_closed = [alpha(1:Fs*9.85) ; alpha(Fs*19.54:Fs*29.61) ; alpha(Fs*39.59:Fs*49.65) ; alpha(Fs*59.62:Fs*69.84) ; alpha(Fs*79.64:Fs*89.63) ; alpha(Fs*99.72:Fs*109.72)];
alpha_open = [alpha(Fs*9.85:Fs*19.54) ; alpha(Fs*29.61:Fs*39.59) ; alpha(Fs*49.65:Fs*59.62) ; alpha(Fs*69.84:Fs*79.64) ; alpha(Fs*89.63:Fs*99.72) ; alpha(Fs*109.72:end)];

Vara_c = var(alpha_closed);
Vara_o = var(alpha_open);

err_a = (Vara_c-Vara_o)/Vara_c*100

% 1.1.5
EEG_closed = [EEG(1:Fs*9.85) ; EEG(Fs*19.54:Fs*29.61) ; EEG(Fs*39.59:Fs*49.65) ; EEG(Fs*59.62:Fs*69.84) ; EEG(Fs*79.64:Fs*89.63) ; EEG(Fs*99.72:Fs*109.72)];
EEG_open = [EEG(Fs*9.85:Fs*19.54) ; EEG(Fs*29.61:Fs*39.59) ; EEG(Fs*49.65:Fs*59.62) ; EEG(Fs*69.84:Fs*79.64) ; EEG(Fs*89.63:Fs*99.72) ; EEG(Fs*109.72:end)];

VarEEG_c = var(EEG_closed);
VarEEG_o = var(EEG_open);

err_EEG = (VarEEG_c-VarEEG_o)/VarEEG_c*100


%% 1.2
EEG_closed_nomove = [EEG(1:Fs*9.85) ; EEG(Fs*19.54:Fs*29.61) ; EEG(Fs*39.59:Fs*49.65) ; EEG(Fs*59.62:Fs*69.84)];
EEG_closed_move = [EEG(Fs*79.64:Fs*89.63) ; EEG(Fs*99.72:Fs*109.72)];

figure
subplot(2,1,1)
Fs = 200;
Y1 = abs(fft(EEG_closed_nomove))
L=length(EEG_closed_nomove)
f = Fs*(1:(L))/L
nChannel = width(EEG_closed_nomove)-1
plot(f,Y1);
title('EEG with closed eyes and no movement')
xlabel('frequency(Hz)')
ylabel('|FFT|')
ylim([0 2*10^4])
subplot(2,1,2)
Y2 = abs(fft(EEG_closed_move))
L=length(EEG_closed_move)
f = Fs*(1:(L))/L
nChannel = width(EEG_closed_move)-1
plot(f,Y2);
ylim([0 2*10^4])
title('EEG with closed eyes with movement')
xlabel('frequency(Hz)')
ylabel('|FFT|')

%% 1.4
beta_closed = [beta(1:Fs*9.85) ; beta(Fs*19.54:Fs*29.61) ; beta(Fs*39.59:Fs*49.65) ; beta(Fs*59.62:Fs*69.84) ; beta(Fs*79.64:Fs*89.63) ; beta(Fs*99.72:Fs*109.72)];
beta_open = [beta(Fs*9.85:Fs*19.54) ; beta(Fs*29.61:Fs*39.59) ; beta(Fs*49.65:Fs*59.62) ; beta(Fs*69.84:Fs*79.64) ; beta(Fs*89.63:Fs*99.72) ; beta(Fs*109.72:end)];

figure; subplot(2,1,1);
t = 0: 1/Fs :(length(beta_closed)-1)/Fs;
plot(t,beta_closed)
title('beta with closed eyes')
xlim([0 60]); ylim([-40 40]);
subplot(2,1,2);
t = 0: 1/Fs :(length(beta_open)-1)/Fs;
plot(t,beta_open)
title('beta with open eyes')
xlim([0 60]); ylim([-40 40]);

Varbeta_c = var(beta_closed);
Varbeta_o = var(beta_open);

err_var_beta = (Varbeta_c-Varbeta_o)/Varbeta_c*100



%% E 2
% 2.1.1:
clear all
clc
% create VEPs:
fs = 200;
t = 0: 1/fs :0.2-1/fs;
N75 = (1/20)*normpdf(t,0.075,0.01);
P100 = -(1/10)*normpdf(t,0.1,0.01);
Vep_1_cycle= N75+P100;


t_full = 0: 1/fs :60-1/fs;
Vep_full = repmat(Vep_1_cycle,1,300);

% show signal before noise addition 
figure, plot(t_full,Vep_full); xlim([0 1]);
title 'Simulated VEPs signal'
ylabel('Volts [\muV]');xlabel('Time [Sec]');grid on;

% add noise to signal
rng default
rng(2)
SNR = -4.7;
noise_sig =awgn(Vep_full,SNR) ;
figure, plot(t_full,noise_sig); xlim([0 1]);
title 'Simulated VEPs signal with noise'
ylabel('Volts [\muV]');xlabel('Time [Sec]');grid on;
% check initial SNR
noise = noise_sig-Vep_full;

SNR_ = snr(Vep_full,noise) %SNR=-3.0721


% 2.1.1:
% filtering using EnsembleAV_EP
%try to get SNR = 6[dB]-> 1.6 sec->
%1.6/0.2=8 cycles

F_pulse = 5; % [Hz] the visual stemulation frequncy
[AV_sig,t_AV_sig]=EnsembleAV_EP(noise_sig,fs,F_pulse,8);
figure, plot(t_AV_sig,AV_sig,'LineWidth',1);
title(['Ensembl Averaging M =' num2str(8) ]);
xlabel('Time [s]'), ylabel('Volts [\muV]') ,grid on;

 noise_1_6 = mean(reshape(noise(1:40*8),40,8),2);
%check if we reached the required  SNR:
SNR_1_6_sec = snr(AV_sig,noise_1_6)

%try to get SNR = 12[dB]-> 6.4 sec->
%6.4/0.2=32 cycles
[AV_sig,t_AV_sig]=EnsembleAV_EP(noise_sig,fs,F_pulse,32);
figure, plot(t_AV_sig,AV_sig,'LineWidth',1);
title(['Ensembl Averaging M =' num2str(32) ]);
xlabel('Time [s]'), ylabel('Volts [\muV]') ,grid on;
noise_6_4 = mean(reshape(noise(1:40*32),40,32),2);
%check if we reached the required  SNR:
SNR_6_4_sec = snr(AV_sig,noise_6_4)



%% 2.2
% create 1 min of 2 different VEPs, 30 sec each:
clear all
clc
% create VEPs:
fs = 200;
t = 0: 1/fs :0.2-1/fs;
N75 = (1/20)*normpdf(t,0.075,0.01);
P100 = -(1/10)*normpdf(t,0.1,0.01);
Vep_1_cycle= N75+P100;

% other 30 sec
N75_2 = (1/6)*normpdf(t,0.075,0.02);
P100_2 = -(1/4)*normpdf(t,0.1,0.02);
Vep_2_cycle= N75_2+P100_2;

t_full = 0: 1/fs :60-1/fs;
% add two VEPs for 1 minute sample
Vep_Full =cat(2, repmat(Vep_1_cycle,1,150), repmat(Vep_2_cycle,1,150));

% show signal before noise addition 
figure, plot(t_full,Vep_Full); xlim([29 31]);
title 'Simulated VEPs signal'
ylabel('Volts [\muV]');xlabel('Time [Sec]');grid on;

% add noise to signal
rng default
SNR = -6;
noise_sig =awgn(Vep_Full,SNR) ;
% check initial SNR
noise = noise_sig-Vep_Full;
SNR_2 = snr(Vep_Full,noise) %SNR= -3.0295

% do Expo averaging
% window 1: first VEP
alpha = [ 3/4, 1/2, 1/10 , 1/50 , 1/80, 1/11000, ]; %we choos alpha to be alpha = 1/M
F_pulse = 5; % [Hz] the visual stemulation frequncy
figure

for i=1:length(alpha)
    [S_signal,t_AV_sig]=ExpoAV_EP(noise_sig(1:40*50),fs,F_pulse,alpha(i),50);
    noisee = mean(reshape(noise(1:40*50),40,50),2);
    SNRr = snr(S_signal,noisee);
    subplot(2,3,i), plot(t_AV_sig,S_signal,'LineWidth',1);
    title(['W_{1- 1st VEP}: Exponential Averaging \alpha =' num2str(alpha(i)) ]);
    xlabel('Time [s]'), ylabel('Volts [\muV]') ,grid on;
end

% window 2: mix of VEPs
figure
for i=1:length(alpha)
    [S_signal,t_AV_sig]=ExpoAV_EP(noise_sig(40*125:40*175-1),fs,F_pulse,alpha(i),50);
    subplot(2,3,i), plot(t_AV_sig,S_signal,'LineWidth',1);
    title(['W_{2- mix of VEPs}: Exponential Averaging \alpha =' num2str(alpha(i)) ]);
    xlabel('Time [s]'), ylabel('Volts [\muV]') ,grid on;
end

% window 3:Second VEP
figure
for i=1:length(alpha)
    [S_signal,t_AV_sig]=ExpoAV_EP(noise_sig(40*200:40*250-1),fs,F_pulse,alpha(i),50);
    subplot(2,3,i), plot(t_AV_sig,S_signal,'LineWidth',1);
    title(['W_{3- 2nd VEP}: Exponential Averaging \alpha =' num2str(alpha(i)) ]);
    xlabel('Time [s]'), ylabel('Volts [\muV]') ,grid on;
end
%% 2.3.1
clear all
X = load("Ben_Orit_ex2.mat");

EEG = X.data(:,1)-mean(X.data(:,1));
flesh =  X.data(:,2);
Fs = 1000;
F_pulse = 1/(1000e-3);
tt = 0: 1/Fs : (length(EEG)-1)/Fs;
% 
 figure, 
subplot(3,1,1), plot(tt,EEG); hold on
ylabel 'Volts [\muV]'; title 'EEG signal';grid on
subplot(3,1,2),plot(tt,flesh);hold on
xlabel 'Time [s]'; ylabel 'Volts [V]'; title 'fliker output';grid on
hold on
Y = fft(EEG); L=length(EEG);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
subplot(3,1,3), plot(f,P1) ;xlim([0 500]);
 xlim([0 100]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'EEG in Frequency';

 % filtering
f_remove = 50; f_sample = Fs; T = 1/f_sample;
w_0 = (2*pi*f_remove)/f_sample; p = exp(1i*w_0);
b = poly([p conj(p)]); a=poly([0.9*p 0.9*conj(p)]);
syms G
sys=tf(b,a);
G_iir=solve(G*((2-2*cos(w_0))/(1-2*0.9*cos(w_0)+(0.9)^2))==1 ,G);
G_iir=double(G_iir);
V_o_after_fil=filter((G_iir)*b,a,EEG);
V_o_after_fil=V_o_after_fil((length(b))/2:end);

% shift the real VEP so that first sample is the fitst in Vep cycle 
V_o_after_fil = V_o_after_fil(541:end);
 
 %after powerline filtering time 
t = tt(1:length(V_o_after_fil));
 Y = fft(V_o_after_fil);
 L=length(V_o_after_fil);
 P2 = abs(Y/L);
 P1 = P2(1:L/2+1);
 P1(2:end-1) = 2*P1(2:end-1);
 f = Fs*(0:(L/2))/L;


M = [ 5 15 30 59];
figure
for i=1:length(M)
    subplot(2,2,i)
    [AV_sig,t_AV_sig]=EnsembleAV_EP(V_o_after_fil',Fs,F_pulse,M(i));
    plot(t_AV_sig,-AV_sig,'LineWidth',1);
    title(['Ensembl Averaging M =' num2str(M(i)) ]);
    xlabel('Time [s]'), ylabel('Volts [\muV]') ,grid on;xlim([0 1])
end




%% 2.3.2
clear all; clc;
EEG_2 = load("EEG_2.mat");
EEG_2 = EEG_2.veps-mean(EEG_2.veps);%zero mean
Fs = 500;
F_pulse = 2.5; % [Hz] the visual stemulation frequncy
t = linspace(0,length(EEG_2)/Fs,length(EEG_2))';

First = EEG_2(1:length(EEG_2)/2);
Second = EEG_2(length(EEG_2)/2+1:end);


% try averaging homogenuous and exp on looking_at_light VEP
M = [5 20 200];
% Ensemble
figure
for i=1:length(M)
    [AV_sig,t_AV_sig]=EnsembleAV_EP(First,Fs,F_pulse,M(i));
    subplot(3,1,i), plot(t_AV_sig,AV_sig,'LineWidth',1);
    title(['EnsembleAV EP M =' num2str(M(i)) ' - First Half'  ]);
    xlabel('Time [s]'), ylabel('Volts [\muV]') ,grid on;

end

% try averaging homogenuous and exp on not_looking_at_light VEP
M = [5 20 200];
% Ensemble
figure
for i=1:length(M)
    [AV_sig,t_AV_sig]=EnsembleAV_EP(Second,Fs,F_pulse,M(i));
    subplot(3,1,i), plot(t_AV_sig,AV_sig,'LineWidth',1);
    title(['EnsembleAV EP M =' num2str(M(i)) ' - Second Half '  ]);
    xlabel('Time [s]'), ylabel('Volts [\muV]') ,grid on;

end



%% EXP 3

clear all; clc;
data = load("Data.mat");
EEG = (data.DATA(1,:)-mean(data.DATA(1,:))); %\muV
ECG_ref = (data.DATA(2,:)-mean(data.DATA(2,:))); % mV
EOG_R_ref = (data.DATA(3,:)-mean(data.DATA(3,:))); %\muV
EOG_L_ref = (data.DATA(4,:)-mean(data.DATA(4,:))); %\muV

fs_1 = 125; fs_2 = 250; fs_3 = 50;

t_1 = 0 : 1/fs_1 : (length(EEG)-1)/fs_1;
t_2 = 0 : 1/fs_2 : (length(ECG_ref)-1)/fs_2;
t_3 = 0 : 1/fs_3 : (length(EOG_R_ref)-1)/fs_3;


% 3.1 
% interpolate signals to 250 [Hz]

EEG_250 = interp(EEG,2);EEG_250 = EEG_250(1:length(ECG_ref));
EOG_R_ref_250 = interp(EOG_R_ref,5);
EOG_R_ref_250 = EOG_R_ref_250(1:length(ECG_ref));
EOG_L_ref_250 = interp(EOG_L_ref,5);
EOG_L_ref_250 = EOG_L_ref_250(1:length(ECG_ref));


% 3.2
% add noise to EEG
t_250 = 0 : 1/250 : (length(EEG_250)-1)/250;
noise = 0.005*sin(2*pi*(50)*t_250); %\muV
EEG_250_noise = EEG_250+noise;

% show all sig in spectrum:
figure, subplot(3,1,1),
Y = fft(EEG_250_noise); L=length(EEG_250_noise);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = 250*(0:(L/2))/L;
plot(f,P1) ; xlim([0 60]);
ylabel '|Amplitude|'; title 'EEG sig in spectrum';
grid on; hold on
 subplot(3,1,2),
Y = fft(ECG_ref); L=length(ECG_ref);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = 250*(0:(L/2))/L;
plot(f,P1) ; xlim([0 60]);
 ylabel '|Amplitude|'; title 'ECG ref ';
grid on; hold on
 subplot(3,1,3),
Y = fft(EOG_R_ref_250); L=length(EOG_R_ref_250);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = 250*(0:(L/2))/L;
plot(f,P1) ; xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'EOG_{R} ref';
grid on; hold on





% 3.3

figure, plot(t_250,EEG_250,'b',t_250,EEG_250_noise,'m --',t_250,ECG_ref,'r',...
            t_250,EOG_R_ref_250,t_250,EOG_L_ref_250);
title('EEG_{250}+noise ');xlabel('Time [Sec]'), ylabel('Volts [\muV]') ,grid on;
 legend('EEG','EEG +noise','ECG [mV]','EOG_R','EOG_L');
xlim([330 340])





%% 3.4 +3.5
% first, filer the powerline noise
% get a reference signals

Raw_Signal = EEG_250;
n = 0:(length(EEG_250)-1);
noise1 = sin((50/(1/250))*2*pi*n);
noise2 = sin((50/(1/250))*2*pi*n+pi/2);
 DC_noise=[(ones(1,length(EEG_250)))];

Noise_ref = [noise1;noise2;DC_noise];

[r,c] = size(Noise_ref);
N = length(Raw_Signal);
R=Noise_ref*Noise_ref'/N; %cor mat 
options.mu = 0.001; %1/(trace(R));
options.W0 = zeros(r,1);

figure,subplot(4,1,1),plot(t_250,EEG_250)
 xlim([330 340]);
title('EEG ');ylabel('Volts [\muV]');

%ploting the filterd signal lms
[Clean_powerline_Signal,~] = LMS(Raw_Signal,Noise_ref,options);
subplot(4,1,2)
plot(t_250,Clean_powerline_Signal)
title('EEG - removed (PL) noise');ylabel('Volts [\muV]');grid on
 xlim([330 340])
hold on

% now, filer the ECG noise
% get a reference signals
Noise_ref = [ECG_ref];
[r,~] = size(Noise_ref);
Raw_Signal = Clean_powerline_Signal;
N = length(Raw_Signal);
R=Noise_ref*Noise_ref'/N; %cor mat 
options.mu =0.001; %1/(trace(R));
options.W0 = zeros(r,1);
hold on

%ploting the filterd signal lms
[Clean_powerline_ECG_Signal,~] = LMS(Raw_Signal,Noise_ref,options);
subplot(4,1,3)
plot(t_250,Clean_powerline_ECG_Signal)
title('EEG - removed (PL&ECG) noise ');ylabel('Volts [\muV]');xlabel('Time');grid on
 xlim([330 340]);

hold on
% now, filer the EOG noise
% get a reference signals
Noise_ref = [EOG_R_ref_250];
[r,c] = size(Noise_ref);
Raw_Signal = Clean_powerline_ECG_Signal;
N = length(Raw_Signal);
R=Noise_ref*Noise_ref'/N; %cor mat 
options.mu =  0.001;%1/(trace(R));
options.W0 = zeros(r,1);

%ploting the filterd signal lms
[Clean_powerline_ECG_EOG_Signal,WLMS] = LMS(Raw_Signal,Noise_ref,options);
subplot(4,1,4)
plot(t_250,Clean_powerline_ECG_EOG_Signal)
title('EEG - removed (P & ECG & EOG) noise ');ylabel('Volts [\muV]');xlabel('Time');grid on
 xlim([330 340])


% now in spectrum:
figure, subplot(4,1,1),
Y = fft(EEG_250_noise); L=length(EEG_250_noise);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = 250*(0:(L/2))/L;
plot(f,P1) ; xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'EEG in Frequency';
grid on; hold on;ylim([0 6]*(10^-3))

Y = fft(Clean_powerline_Signal); L=length(Clean_powerline_Signal);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = 250*(0:(L/2))/L;
subplot(4,1,2), plot(f,P1);xlim([0 60]); grid on
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'EEG - removed (PL) noise';
hold on;ylim([0 6]*(10^-3))

Y = fft(Clean_powerline_ECG_Signal); L=length(Clean_powerline_ECG_Signal);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = 250*(0:(L/2))/L;
subplot(4,1,3), plot(f,P1); xlim([0 60]); grid on
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; 
title 'EEG - removed (PL&ECG) noise';
hold on;ylim([0 6]*(10^-3))

Y = fft(Clean_powerline_ECG_EOG_Signal); L=length(Clean_powerline_ECG_EOG_Signal);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = 250*(0:(L/2))/L;
subplot(4,1,4), plot(f,P1); xlim([0 60]); grid on;ylim([0 6]*(10^-3))
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; 
title 'EEG - removed (P & ECG & EOG) noise';

[SNR,PRD,RMS]=Get_quality(EEG_250,Clean_powerline_ECG_EOG_Signal)

