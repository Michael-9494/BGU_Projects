function [P1,max_freq,mean_freq] = FFT_nolable( data ,flag,Fs)

Y = fft(data);
L=length(data);
P2 = abs(Y/L); P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
max_freq = max(P1);
f = Fs*(0:(L/2))/L;
ff = find(f>8 & f<50);
PP1 = P1(ff);
mean_freq = mean(PP1);
end 