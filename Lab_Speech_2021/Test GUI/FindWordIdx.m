function Idx = FindWordIdx(FramedSig,Fs,WindowLength,Overlap)
% FindWordIdx finds the start and end of speech using energy 
% calculation. based of certain threshold
% FramedSig – the framed speech signal after preprocessing.
% Fs – sampling frequency
% WindowLength – length of test and reference windows [seconds]
% Overlap – percentage of overlap between adjacent frames [0-100]

% OUTPUT:
% Idx – 2 integer vector: start and end indices of detected word.

Signal_Energy=calcNRG(FramedSig);%calculate  of each frame
thresh=max(Signal_Energy)*0.001;
Frames_with_Speech=find(Signal_Energy>thresh);
WindowLength_samples=WindowLength*Fs;   % [sec]*[sample/sec]=[sample]
overlap_in_samples=((Overlap)*WindowLength_samples)/100; % overlap in samples

Idx=zeros(1,2);
% find relevant index acording to the overlaping.
idx_start=(Frames_with_Speech(1))*overlap_in_samples;
Idx(1)=idx_start;
idx_end=(Frames_with_Speech(end))*overlap_in_samples;
Idx(2)=idx_end+WindowLength_samples-1;

end


