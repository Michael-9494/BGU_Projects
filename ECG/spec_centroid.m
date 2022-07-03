function spectral_centroid=spec_centroid(Signal,Fs)
%this function get a matrix of framed signal we will first compute the FFT
%of each row then we will calculate the central spectral mass by
%multiplying a matching frequncy vector 
%spectral_centroid a vector where each row is a different spectral mass
%each index is for a new frame
%Signal - the signal in frames
%Fs - the sampling frequncy
[R,C] = size(Signal);

%finding the FFT of each row 
n = 2^nextpow2(C);
FFT_mat = fft(Signal,n,2);
P2 = abs(FFT_mat/C); %the duble sided fft
P1 = P2(:,1:n/2+1); %making it 1 side cabes on P2
P1(:,2:end-1) = 2*P1(:,2:end-1);

freq_array = Fs*(0:(n/2))/C;
spectral_centroid = mean((P1*freq_array')./sum(P1,2)); 
end