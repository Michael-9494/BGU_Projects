function [inx,QRS_finding,ECG_new] = Balda_QRS(ECG,threshold,window_length,minimal_sample)
for i = 6:length(ECG)
    ECG_1tag(i-5) = abs(ECG(i-1) - ECG(i-3));
    ECG_2tag(i-5) = abs(ECG(i-1) - 2*ECG(i-3) + ECG(i-5));
end
ECG_new = 1.3*ECG_1tag + 1.1*ECG_2tag;

counter = 0;
QRS_finding = zeros(1,length(ECG));

for i = 1:length(ECG_new)-window_length;
    for k = 1:window_length;
        if ECG_new(i-1+k)>=threshold;
            counter = counter+1;
        end
    end
    QRS_finding(i:i+window_length) = counter>=minimal_sample;
    counter = 0;
end
inx = find(QRS_finding);
end








