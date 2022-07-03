%% Exp1
clear all; clc;
% 2
ECG = table2array(readtable ('2_5_b_g8.csv'));
ECG = ECG(1003:2000,:);
ECG1 = interp((ECG(:,2)-mean(ECG(:,2))),3);
t_ECG1 = interp(ECG(:,1),3);

%plotting the signal from lab1
Fs = 300; %Hz
figure,plot(t_ECG1,ECG1);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal 60 bpm'
grid on;

figure(1),subplot(2,2,1),plot(t_ECG1,ECG1);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal 60 bpm'
grid on;
figure(1),subplot(2,2,2),
Y = fft(ECG1); L=length(ECG1);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) ; %xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'ECG signal ';
grid on; hold on

% 3
%network noise
% t = (0:1/Fs:5)';
rng default
SNR = 30;
NetNoise =  sin(2*pi*50*t_ECG1);
P_L_noise_Amp=sqrt(var(ECG1)/((10^(SNR/10)*var(NetNoise)))); %the noise amp
NetNoise=P_L_noise_Amp*NetNoise; %powerline noise.

NetNoisyECG = ECG1+NetNoise;

figure(1),subplot(2,2,3), plot(t_ECG1,NetNoisyECG);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal + P.L noise'
grid on;

figure(1),subplot(2,2,4),
Y = fft(NetNoisyECG); L=length(NetNoisyECG);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) ; %xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'ECG signal + P.L noise';
grid on; hold on

SNR = snr(NetNoisyECG,NetNoise) %SNR = 30.0039

% add EMG noise:
figure(2),subplot(2,2,1),plot(t_ECG1,ECG1);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal 60 bpm'
grid on;
figure(2),subplot(2,2,2),
Y = fft(ECG1); L=length(ECG1);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) ; %xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'ECG signal ';
grid on; hold on

EMG_noise_sig =awgn(ECG1,SNR,'measured') ;
figure(2),subplot(2,2,3), plot(t_ECG1,EMG_noise_sig); xlim([0 5]);
title 'ECG signal + EMG noise'
ylabel('Volts [V]');xlabel('Time [Sec]');grid on;
% check initial SNR
noise = EMG_noise_sig-ECG1;

SNR_ = snr(EMG_noise_sig,noise) %SNR=30.0296

figure(2),subplot(2,2,4),
Y = fft(EMG_noise_sig); L=length(EMG_noise_sig);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) ; %xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'ECG signal + EMG noise';
grid on; hold on

% 1.4
% filter Power Line noise:
tic
clean_p_L = filter(NOtch_50,NetNoisyECG);
Time_FIR = toc

figure(3),subplot(2,2,1),plot(t_ECG1,ECG1);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal 60 bpm'
grid on;
figure(3),subplot(2,2,2),
Y = fft(ECG1); L=length(ECG1);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) ; %xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'ECG signal ';
grid on; hold on

figure(3),subplot(2,2,3),plot(t_ECG1,clean_p_L);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal after PL removal'
grid on;
figure(3),subplot(2,2,4),
Y = fft(clean_p_L); L=length(clean_p_L);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) ; %xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'ECG  after PL removal';
grid on; hold on

LP_50_EMG = LP_50_EMG;
% filter EMG noise:
tic
clean_EMG = filter(LP_100_EMG,EMG_noise_sig);
Time_FIR_EMG = toc
delay=mean(grpdelay(LP_100_EMG)); %delay cause by filter.
clean_EMG(1:delay)=[]; %fix the delay.
t_clean_EMG=t_ECG1(1:end-delay+1); %time update.


figure(4),subplot(2,2,1),plot(t_ECG1,ECG1);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal 60 bpm'
grid on;
figure(4),subplot(2,2,2),
Y = fft(ECG1); L=length(ECG1);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) ; %xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'ECG signal ';
grid on; hold on

figure(4),subplot(2,2,3),plot(t_clean_EMG,clean_EMG);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal after EMG removal'
grid on;
figure(4),subplot(2,2,4),
Y = fft(clean_EMG); L=length(clean_EMG);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) ; %xlim([0 60]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'ECG  after EMG removal';
grid on; hold on


% 1.5

BLWNoise = 0.05* sin(2*pi*0.2*t_ECG1); % 0.2 [Hz] 12 Breaths per minute
BLWNoisyECG = ECG1+BLWNoise;

figure(5),subplot(2,1,1),plot(t_ECG1,BLWNoisyECG);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal + BLW'
grid on;
figure(5),subplot(2,1,2),
Y = fft(BLWNoisyECG); L=length(BLWNoisyECG);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) ; xlim([-5 Fs/2]);
xlabel 'Frequency[Hz]'; ylabel '|Amplitude|'; title 'ECG signal + BLW ';
grid on; hold on


% FIR:
tic
clean_BLW_FIR = filter(FIR_BLW_05_Hz,BLWNoisyECG);
Time_clean_BLW_FIR = toc
%IIR
tic
clean_BLW_IIR = filter(IIRBLW05Hz,BLWNoisyECG);
Time_clean_BLW_IIR = toc

IIRBLW05Hz = IIRBLW05Hz;

% filtfilt
clean_BLW_filtfilt = filtfilt(IIRBLW05Hz.sosMatrix,IIRBLW05Hz.ScaleValues,BLWNoisyECG);
Time_clean_BLW_filtfilt = toc

figure(6),subplot(4,1,1),plot(t_ECG1,BLWNoisyECG);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]'; title 'ECG signal + BLW'
grid on;
figure(6),subplot(4,1,2),plot(t_ECG1,clean_BLW_FIR);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]';
title (['clean BLW FIR Run_{t}=' num2str(10^3*Time_clean_BLW_FIR) ' [ms]' ])
grid on;
figure(6),subplot(4,1,3),plot(t_ECG1,clean_BLW_IIR);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]';
title (['clean BLW IIR Run_{t}=' num2str(10^3*Time_clean_BLW_IIR) ' [ms]' ])
grid on;
figure(6),subplot(4,1,4),plot(t_ECG1,clean_BLW_filtfilt);xlim([0 5]);
xlabel 'Time[sec]'; ylabel 'Voltage[V]';
title (['clean BLW filtfilt Run_{t}=' num2str(10^3*Time_clean_BLW_filtfilt) ' [ms]' ])
grid on;

% threshold=0.2;window_length=8;minimal_sample=6;
% [inxN,QRS,ECG_newN] = Balda_QRS(clean_BLW_filtfilt,threshold,window_length,minimal_sample);

threshold=0.6*(max(clean_BLW_filtfilt));
win_length=8;
% get the QRS complex
QRS_detect=PanThomp_QRSs(clean_BLW_filtfilt',win_length,Fs,threshold,1);
[meanRR,BPM,pNN50] = features(QRS_detect,Fs)

figure
plot(t_ECG1,QRS_detect,'LineWidth',1.5)
hold on
plot(t_ECG1,clean_BLW_filtfilt)
title 'QRS detection'
xlabel 'time [sec]'

%% Exp2
clear all
clc;
LP_50_EMG = LP_50_EMG;
IIRBLW05Hz = IIRBLW05Hz;
% 3
Fs = 300; %Hz
Testlabels = readtable('Test_labels.xlsx');
Trainlabels = readtable('Train_labels.xlsx');

% reading train data
Path = 'Train/';
File = dir( fullfile(Path,'*.mat'));
FileNames = {File.name}';
Length_Names = size(FileNames,1);
tic
for i=1:Length_Names
    filename=strcat(Path, Trainlabels(i,1).Name{1});
    if strcmp(table2cell(Trainlabels(i,2)),{'A'})==1
        feature_lable_train(i)= 1;
    end
    if strcmp(table2cell(Trainlabels(i,2)),{'N'})==1
        feature_lable_train(i)= 0;
    end
    Train{i}.data=load(filename)';

     %remove DC
    x_f = fft(Train{1,i}.data.val);
    x_f(1) = 0;
    Train{i}.data = ifft(x_f);
    
    % if the signal is upside down
     if (mean(Train{i}.data)<-0.4)
         Train{i}.data = (-1)*Train{i}.data;
     end
%     Max_val_Train(i) = max(Train{i}.data);
%     function [P1,max_freq,mean_freq] = FFT( data ,Lable,flag,Fs)
    [~,max_freq_Train(i),mean_P_wave_freq_Train(i)] = FFT( Train{i}.data ,feature_lable_train(i),0,Fs);
    Train{i}.data = (Train{i}.data-mean(Train{i}.data))/std(Train{i}.data);%normalized
    Train{i}.data=filtfilt(LP_50_EMG.sosMatrix,LP_50_EMG.ScaleValues,Train{1,i}.data); % EMG noise
    Train{i}.data=filtfilt(IIRBLW05Hz.sosMatrix,IIRBLW05Hz.ScaleValues,Train{1,i}.data); %BLW
    Train{i}.data=filter(NOtch_50,Train{1,i}.data); % net noise


    threshold=0.1*(max(Train{1,i}.data));
    win_length=10;
    QRS_detect=PanThomp_QRSs(Train{1,i}.data,win_length,Fs,threshold,0);
    [meanRR(i),BPM(i),Var_RR(i)] = features(QRS_detect,Fs);
    FramedSig=enframe(Train{1,i}.data,128);
    spectral_centroid_Train(i) = spec_centroid(FramedSig,Fs);
end
Preprocess_and_Train = toc

Idx_A_Train = find(strcmpi(Trainlabels{:,2},'A'));
Labels = zeros(1,length(Train));
Labels(Idx_A_Train)=1;

AF_sum = sum(Labels)
Feature_Mat_Train = [max_freq_Train' mean_P_wave_freq_Train' meanRR' Var_RR'  spectral_centroid_Train' Labels'];
Feature_Table_Train = array2table(Feature_Mat_Train,'VariableNames',...
    {'max_freq','mean_P_wave_freq','meanRR','Var_RR','spectral_centroid','Labels'});


% reading test data
clear Test Max_val_Test meanRR BPM SDNN rMSSD Var_RR
Path = 'Test/';
File = dir( fullfile(Path,'*.mat'));
FileNames = {File.name}';
Length_Names = size(FileNames,1);
tic
for i=1:Length_Names
    filename=strcat(Path, Testlabels(i,1).Name{1});
    if strcmp(table2cell(Testlabels(i,2)),{'A'})==1
        feature_lable_Test(i)= 1;
    end
    if strcmp(table2cell(Testlabels(i,2)),{'N'})==1
        feature_lable_Test(i)= 0;
    end
    Test{i}.data=load(filename)';

     %remove DC
    x_f = fft(Test{1,i}.data.val);
    x_f(1) = 0;
    Test{i}.data = ifft(x_f);
    
    % if the signal is upside down
    if (mean(Test{i}.data)<-0.4)
        Test{i}.data = (-1)*Test{i}.data;
    end
 
%     function [P1,max_freq,mean_freq] = FFT( data ,Lable,flag,Fs)
    [~,max_freq_Test(i),mean_P_wave_freq_Test(i)] = FFT( Test{i}.data ,feature_lable_Test(i),0,Fs);
    Test{i}.data = (Test{i}.data-mean(Test{i}.data))/std(Test{i}.data);%normalized
    Test{i}.data=filtfilt(LP_50_EMG.sosMatrix,LP_50_EMG.ScaleValues,Test{1,i}.data); % EMG noise
    Test{i}.data=filter(NOtch_50,Test{1,i}.data); % net noise
    Test{i}.data=filtfilt(IIRBLW05Hz.sosMatrix,IIRBLW05Hz.ScaleValues,Test{1,i}.data); %BLW
  

    threshold=0.1*(max(Test{1,i}.data));
    win_length=10;
    QRS_detect=PanThomp_QRSs(Test{1,i}.data,win_length,Fs,threshold,0);
    [meanRR(i),BPM(i),Var_RR(i)] = features(QRS_detect,Fs);
    FramedSig=enframe(Test{1,i}.data,128);
    spectral_centroid_Test(i) = spec_centroid(FramedSig,Fs);
end
Preprocess_and_Test = toc

Idx_A_Test = find(strcmpi(Testlabels{:,2},'A'));
Labels_Test = zeros(1,length(Test));
Labels_Test(Idx_A_Test)=1;

AF_sum = sum(Labels_Test)
Feature_Mat_Test = [max_freq_Test' mean_P_wave_freq_Test' meanRR'  Var_RR' spectral_centroid_Test' Labels_Test'];
Feature_Table_Test = array2table(Feature_Mat_Test,'VariableNames',...
    {'max_freq','mean_P_wave_freq','meanRR','Var_RR','spectral_centroid','Labels'});

%% 3
clc
clear Train_py  ECGarrayTrain
Path = 'Train/';
File = dir( fullfile(Path,'*.mat'));
FileNames = {File.name}';
Length_Names = size(FileNames,1);
LP_25_EMG =LP_25_EMG;
tic
for i=1:Length_Names
    filename=strcat(Path, Trainlabels(i,1).Name{1});
    Train_py{i}.data=load(filename)';

    %remove DC
    x_f = fft(Train_py{1,i}.data.val);
    x_f(1) = 0;
    Train_py{i}.data = ifft(x_f);
    % if the signal is upside down
    if (mean(Train_py{i}.data)<-0.4)
        Train_py{i}.data = (-1)*Train_py{i}.data;
    end

    Train_py{i}.data=filtfilt(LP_25_EMG.sosMatrix,LP_25_EMG.ScaleValues,Train_py{1,i}.data); % EMG noise
    %     Train_py{i}.data(1:delay) = []; % delay
    Train_py{i}.data=filtfilt(IIRBLW05Hz.sosMatrix,IIRBLW05Hz.ScaleValues,Train_py{1,i}.data); %BLW
    Train_py{i}.data=filter(NOtch_50,Train_py{1,i}.data); % net noise

    if (length(Train_py{i}.data)>3600)
        n = length(Train_py{i}.data);
        midpoint = round(n/2);
        Train_py{i}.data = Train_py{i}.data(midpoint-1800+1:midpoint+1800);
% new Fs=50
        Train_py{i}.data = decimate(Train_py{i}.data,6);
    else
        n = length(Train_py{i}.data);
        FL = floor(n/Fs)*Fs;
        Train_py{i}.data = Train_py{i}.data(1:FL);
        Train_py{i}.data = resample(Train_py{i}.data,lcm(FL,3600)/FL,round(lcm(FL,3600)*(1/600)));
        % new Fs=50
    end
    Train_py{i}.data = (Train_py{i}.data-mean(Train_py{i}.data))/std(Train_py{i}.data);
    ECGarrayTrain(i,:) = Train_py{i}.data;
end
PY_T = toc
save('ECGarrayTrain.mat', 'ECGarrayTrain')

Idx_A_Train_py = find(strcmpi(Trainlabels{:,2},'A'));% find A
LabelsTrain = zeros(1,length(Train_py));% N
LabelsTrain(Idx_A_Train_py)=1;
LabelsTrain = LabelsTrain';
save('LabelsTrain.mat', 'LabelsTrain')



clear Test_py  ECGarrayTest
Path = 'Test/';
File = dir( fullfile(Path,'*.mat'));
FileNames = {File.name}';
Length_Names = size(FileNames,1);
tic
for i=1:Length_Names
    filename=strcat(Path, Testlabels(i,1).Name{1});
    Test_py{i}.data=load(filename)';
     %remove DC
    x_f = fft(Test_py{1,i}.data.val);
    x_f(1) = 0;
    Test_py{i}.data = ifft(x_f);
    % if the signal is upside down
    if (mean(Test_py{i}.data)<-0.4)
        Test_py{i}.data = (-1)*Test_py{i}.data;
    end

    Test_py{i}.data=filtfilt(LP_25_EMG.sosMatrix,LP_25_EMG.ScaleValues,Test_py{1,i}.data); % EMG noise
    %     Train_py{i}.data(1:delay) = []; % delay
    Test_py{i}.data=filtfilt(IIRBLW05Hz.sosMatrix,IIRBLW05Hz.ScaleValues,Test_py{1,i}.data); %BLW
    Test_py{i}.data=filter(NOtch_50,Test_py{1,i}.data); % net noise

    if (length(Test_py{i}.data)>3600)
        n = length(Test_py{i}.data);
        midpoint = round(n/2);
        Test_py{i}.data = Test_py{i}.data(midpoint-1800+1:midpoint+1800);
% new Fs=50
        Test_py{i}.data = decimate(Test_py{i}.data,6);
    else
        n = length(Test_py{i}.data);
        FL = floor(n/Fs)*Fs;
        Test_py{i}.data = Test_py{i}.data(1:FL);
        Test_py{i}.data = resample(Test_py{i}.data,lcm(FL,3600)/FL,round(lcm(FL,3600)*(6/3600)));
        % new Fs=50
    end
    Test_py{i}.data = (Test_py{i}.data-mean(Test_py{i}.data))/std(Test_py{i}.data);
    ECGarrayTest(i,:) = Test_py{i}.data;
end
save('ECGarrayTest.mat', 'ECGarrayTest')


% Test
Idx_A_Test_py = find(strcmpi(Testlabels{:,2},'A'));% find A
LabelsTest = zeros(1,length(Test_py))';% N
LabelsTest(Idx_A_Test_py)=1;
save('LabelsTest.mat', 'LabelsTest')



%% 4


liveECG = [];
for i=1:length(Train)
    curr = Train(i);
    liveECG = [liveECG,curr{1}.data];
    clear curr;
end
t_liveECG = [0:1/Fs:(length(liveECG)-1)/Fs];




