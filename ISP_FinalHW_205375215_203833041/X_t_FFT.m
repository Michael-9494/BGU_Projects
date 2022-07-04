function [x_t] = X_t_FFT(n,t)
%X_T_FFT calculate the fourier reconstruction of the signal that have neen given 
%   to us. thr DC componant was analytically calculated.
% n represent the nomber of harmonics we want to reconstruct with
% t is the time vector we are  using

T0 = 10;
w0 = 2*pi/T0;

D0=0.3;% analytically calculated

k = 1:n;
Dk = (6.*sin(2.*k.*w0)-6.*sin(3.*k.*w0)+1i.*(2.*cos(k.*w0)-2.*cos(4.*k.*w0)+3.*(exp(1i.*k.*w0)-exp(4.*1i.*k.*w0))))./(k.*pi.*2);

 Xf_n=zeros(1,length(t)); % preallocation

for i_1=1:length(t) % a loop for each point in time
 Xf_n(i_1) = 2.*sum(Dk.*exp(1i.*k.*w0.*t(i_1)));
end


Xf_n = D0 + Xf_n; % add dc componant



x_t = Xf_n ;
end

