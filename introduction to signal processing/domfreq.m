function [domfreq_1 ,domfreq_2] = domfreq(fs,x_t)
%this function find the 2 dominant frequencys by taking the sample and
%computing its fft, and from that we can see the highes amplitutde sor each
%signal

y=fft(x_t); %calculating the fast foiurier transform of the signal
p2 = abs(y/fs); %Compute the minus and plus spectrum 
p1 = p2(1:fs/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero


[pks,locs] = findpeaks(p1,'MinPeakHeight',0.6);  %to see dominnant values on frequncy plane
check = isempty(locs);
if check == 1
    domfreq_1 = 0  ;
    domfreq_2 = 0;
    
    return
end

domfreq_1 = locs(1)-1;  
domfreq_2 = locs(2)-1;



