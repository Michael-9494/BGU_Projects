function EnergySignal=calcNRG(framedSignal)
% calcNRG calculate the the energy of each frame
% framedSignal – a matrix of the framed signal, after preprocessing
% OUTPUT:
% EnergySignal – a column vector of the energy values of the signal

% take the length of each frame
[M,N]=size(framedSignal); 
window=rectwin(N);
% sum( _ ,2) in order to sum each frame
EnergySignal=(sum((framedSignal).^2,2))/sum(window.^2);
% make the range [0 1]
%  min_E = min(EnergySignal); max_E = max(EnergySignal);
%  EnergySignal=(EnergySignal-min_E)/(max_E-min_E); 
%  EnergySignal=reshape((repmat(EnergySignal,1,N))',[N*M,1]);
end
