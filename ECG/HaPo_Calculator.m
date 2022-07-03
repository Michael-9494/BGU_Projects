function HaPo=HaPo_Calculator(Signal,Fs)
%this function get a matrix of framed signal we will first compute the FFT
%of each row then we will calculate the Half Point of each segment
%HaPo = Half Point is the location where you can divide the area of a plot
%by 2
%we will use the buikd in function cumtrapz which caclculate the cumulative
%area of each index
%Signal - the signal in frames
%Fs - the sampling frequncy

%finding the FFT of each row 

[R,C] = size(Signal);%the signal is a matrix of segment
n = 2^nextpow2(C);
FFT_mat = fft(Signal,n,2);
P2 = abs(FFT_mat/C); %the duble sided fft
P1 = P2(:,1:n/2+1); %making it 1 side cabes on P2

freq_array = Fs*(0:(n/2))/C;

Q = cumtrapz(freq_array,P1')';
for i = 1:(length(R))
   HaPo_index(:,i) =  find(Q(i,:)>=Q(i,end)/2,1);
end

HaPo = mean(freq_array(HaPo_index-ones(1,length(HaPo_index))));
end