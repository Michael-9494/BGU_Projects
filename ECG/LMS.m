function [Clean_Signal,W] = LMS(Raw_Signal,Noise_ref,Options)
if nargin<3 %in case the funcion gets only 2 arguments
    Options.mu = 1/trace((1/length(Raw_Signal))*Noise_ref*Noise_ref')
    Options.W0 = zeros(1,size(Noise_ref,1))
end
if ~isfield(Options,'mu') %in case mu (length step) field is empty
    Options.mu = 1/trace((1/length(Raw_Signal))*Noise_ref*Noise_ref')
end
if ~isfield(Options,'W0') %in case W0 (starting weighs field is empty
    Options.W0 = zeros(1,size(Noise_ref,1))
end
Total_Noise = zeros(1,length(Raw_Signal));
W(:,1) = Options.W0;
for n = 1:(length(Raw_Signal)-1)
    e = Raw_Signal(n)- Noise_ref(:,n)'*W(:,n); % error calculation
    W(:,n+1) = W(:,n) + Options.mu*e*Noise_ref(:,n); %next weights calculation
    Total_Noise(n) = W(:,n)'*Noise_ref(:,n); %the weighted noise
end
Clean_Signal = Raw_Signal-Total_Noise;

end

