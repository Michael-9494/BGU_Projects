function exportGraphs(folder_name,Signal,Fs,phon,seg_ind,STFTwinLength,...
    STFToverlap,STFTnfft,STFTcmin,STFTcmax,NRG,ZCR,Flag)
% • folder_name – full address of the folder to save the files in
% • Signal – the recorded signal
% • Fs – sampling frequency
% • phon – an array of chars, representing the phonemes
% • seg_ind – result of the segmentation process
% • STFTwinLength – window length for the STFT (in samples)
% • STFToverlap – overlap length for the STFT (in samples)
% • STFTnfft – nfft length (in samples)
% • STFTcmin – minimum value for color axis scaling
% • STFTcmax – maximum value for color axis scaling
% • NRG – energy signal
% • ZCR – ZCR signal
% • Flag – indicates which graphs to generate.

alpha=0.999;
WindowLength=30*10^-3;  % 30 [mS] window
Overlap=50;             % 50% overlap
[ProcessedSig,FramedSig]=PreProcess(Signal,Fs,alpha,WindowLength,Overlap);


Idx=FindWordIdx(FramedSig,Fs,WindowLength,Overlap);

t=[0:length(Signal)-1]*1/Fs; % time vector

N=(30*10^-3)*Fs; % number of samples in each frame
Overlap_sam= ((Overlap)*N)/100;

% \sh\
Phoneme_sh=ProcessedSig(seg_ind(1):seg_ind(2));
framed_Phoneme_sh=enframe(Phoneme_sh,hamming(N),Overlap_sam);
[FeatsVector_sh,~]=FeatExt(Phoneme_sh,Fs,framed_Phoneme_sh);

% \a\
Phoneme_a=ProcessedSig(seg_ind(2):seg_ind(3));
framed_Phoneme_a=enframe(Phoneme_a,hamming(N),Overlap_sam);
[FeatsVector_a,~]=FeatExt(Phoneme_a,Fs,framed_Phoneme_a);

% \l\
Phoneme_L=ProcessedSig(seg_ind(3):seg_ind(4));
framed_Phoneme_L=enframe(Phoneme_L,hamming(N),Overlap_sam);
[FeatsVector_L,~]=FeatExt(Phoneme_L,Fs,framed_Phoneme_L);

% \o\
Phoneme_o=ProcessedSig(seg_ind(4):seg_ind(5));
framed_Phoneme_o=enframe(Phoneme_o,hamming(N),Overlap_sam);
[FeatsVector_L,~]=FeatExt(Phoneme_L,Fs,framed_Phoneme_L);

% \m\ - 5
Phoneme_m=ProcessedSig(seg_ind(5):seg_ind(end));
framed_Phoneme_m=enframe(Phoneme_m,hamming(N),Overlap_sam);
[FeatsVector_L,~]=FeatExt(Phoneme_L,Fs,framed_Phoneme_L);


h = figure;
XT=[];
for i=1:length(seg_ind)-1
    plot(t(seg_ind(i):seg_ind(i+1)),Signal(seg_ind(i):seg_ind(i+1)));hold on
    XT=[XT t(round((seg_ind(i)+seg_ind(i+1))/2))];
end
hold on
for i=2:length(seg_ind)-1
    line([t(seg_ind(i)) t(seg_ind(i))],[-1 1],'color','m','linewidth',1.4);hold on
end
hold on
ylim(1.2*max(abs(Signal))*[-1 1]);
xticks(XT(2:end-1));
xticklabels({'SH','A','L','O','M'});
xlabel 'Time [S]'; xlim([0 t(end)])
title 'Segmented pronounciation of the word 'Shalom' '; ylabel 'Amplitude'
% legend('speech','start_{speech} ','end_{speech}');
grid on
saveas(h,fullfile(folder_name,'Processed_Signal'),'fig');
saveas(h,fullfile(folder_name,'Processed_Signal'),'jpg');




h1 = figure;
[~,F,T,P] = spectrogram(Signal,STFTwinLength,STFToverlap,STFTnfft,Fs,'yaxis');
surf(T,F,10*log10(P),'edgecolor','none');
view([0,90]);
title '"shalom" Spectrogram';xlabel 'Time [S]';ylabel 'Frequency [Hz]';
caxis([STFTcmin STFTcmax]); colormap jet
xlim([t(1),t(end)]);
ylim([0,Fs/2]);
for i=2:length(seg_ind)-1
    line([t(seg_ind(i)) t(seg_ind(i))],ylim,[-1 1],'color','k','linewidth',1.4)
end

saveas(h1,fullfile(folder_name,'Spectrogram'),'fig');
saveas(h1,fullfile(folder_name,'Spectrogram'),'jpg');


if Flag~=0

    h2 =figure;
    plot(NRG,'color','r','LineWidth',1.2);hold on
    plot(ZCR,'color','k','LineWidth',1.2);hold on
    for i=2:length(seg_ind)-1
    line([(seg_ind(i))/(Overlap_sam) (seg_ind(i))/(Overlap_sam)],ylim,'color','m','linewidth',1.4)
    end
    grid on
    title 'Energy and ZCR  of the signal' ;%xlabel 'Time [S]' ;
    ylabel 'Arbitrary amplitude'
    legend('NRG','ZCR')
    saveas(h2,fullfile(folder_name,'Energy_and_ZCR'),'fig');
    saveas(h2,fullfile(folder_name,'Energy_and_ZCR'),'jpg');

    [h11,h21]=estimatePhonemeFormants(Phoneme_sh,Fs,phon(1,:));
    saveas(h11,fullfile(folder_name,'estimate_Phoneme_sh_Formants'),'fig');
    saveas(h11,fullfile(folder_name,'estimate_Phoneme_sh_Formants'),'jpg');
    saveas(h21,fullfile(folder_name,'AR_Model_Poles_Phoneme_sh'),'fig');
    saveas(h21,fullfile(folder_name,'AR_Model_Poles_Phoneme_sh'),'jpg');

    [h12,h22]=estimatePhonemeFormants(Phoneme_a,Fs,phon(2,:));
    saveas(h12,fullfile(folder_name,'estimate_Phoneme_a_Formants'),'fig');
    saveas(h12,fullfile(folder_name,'estimate_Phoneme_a_Formants'),'jpg');
    saveas(h22,fullfile(folder_name,'AR_Model_Poles_Phoneme_a'),'fig');
    saveas(h22,fullfile(folder_name,'AR_Model_Poles_Phoneme_a'),'jpg');

    [h13,h23]=estimatePhonemeFormants(Phoneme_L,Fs,phon(3,:));
    saveas(h13,fullfile(folder_name,'estimate_Phoneme_L_Formants'),'fig');
    saveas(h13,fullfile(folder_name,'estimate_Phoneme_L_Formants'),'jpg');
    saveas(h23,fullfile(folder_name,'AR_Model_Poles_Phoneme_L'),'fig');
    saveas(h23,fullfile(folder_name,'AR_Model_Poles_Phoneme_L'),'jpg');

    [h14,h24]=estimatePhonemeFormants(Phoneme_o,Fs,phon(4,:));
    saveas(h14,fullfile(folder_name,'estimate_Phoneme_o_Formants'),'fig');
    saveas(h14,fullfile(folder_name,'estimate_Phoneme_o_Formants'),'jpg');
    saveas(h24,fullfile(folder_name,'AR_Model_Poles_Phoneme_o'),'fig');
    saveas(h24,fullfile(folder_name,'AR_Model_Poles_Phoneme_o'),'jpg');

    [h15,h25]=estimatePhonemeFormants(Phoneme_m,Fs,phon(5,:));
    saveas(h15,fullfile(folder_name,'estimate_Phoneme_m_Formants'),'fig');
    saveas(h15,fullfile(folder_name,'estimate_Phoneme_m_Formants'),'jpg');
    saveas(h25,fullfile(folder_name,'AR_Model_Poles_Phoneme_m'),'fig');
    saveas(h25,fullfile(folder_name,'AR_Model_Poles_Phoneme_m'),'jpg');

end