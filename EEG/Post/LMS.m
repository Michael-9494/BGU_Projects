function [Clean_Signal,W]=LMS(Raw_Signal,Noise_ref,Options)
%this function finds will genarate the lms algorithem in a loop to find the
%proper weight
N = length(Raw_Signal);
R = (Noise_ref*Noise_ref')/N; %each value of R vector is calculated by 
%                               1/N*sum(XiVj) for v matrix

    if isempty(Options.mu)==1
      mu = (1/trace(R))/2;
    else
        mu = Options.mu;
    end
    if isempty(Options.W0)==1
        W0 = 0;
    else
        W0 = Options.W0;
    end

weighted_noise = zeros(1,N);
W(:,1) = W0;

for i=1:N-1
    e = Raw_Signal(i)-Noise_ref(:,i)'* W(:,i);
    W(:,i+1) = W(:,i)+mu*e*(Noise_ref(i));
    weighted_noise(i) = W(:,i)'*Noise_ref(:,i);
end
Clean_Signal = Raw_Signal - weighted_noise;
