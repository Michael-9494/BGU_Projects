function [Pv_final] = histogram_model(v,M,sigma,mu,Pn)
% This function finds the optimal values of: sigma,mu,Pn using 
% otimization tool
Pv=[];
for i=1:M
    for j=1:length(v)
        gn(i,j)=(1/sqrt(2*pi*(sigma(i)).^2)*exp(-(v(j)-mu(i)).^2/(2*sigma(i).^2)));
    end
end
for i=1:M
    Pv(i,:)=Pn(i)*gn(i,:);
end
Pv_final=sum(Pv,1);
end