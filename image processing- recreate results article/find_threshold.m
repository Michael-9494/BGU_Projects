function [final_T] = find_threshold(x,M)
% The variabels
s=x(1:M); %sigma
mu=x(M+1:2*M);
p=x(2*M+1:3*M);
s=sort(s);
p=sort(p);
mu = sort(mu);
% finding T by solving quadratic equation
sS1=s(1)^2;
sS2=s(2)^2;
sM1=mu(1)^2;
sM2=mu(2)^2;
A=sS1-sS2;
B=2*(mu(1)*sS2-mu(2)*sS1);
C=(sS1*sM2-sS2*sM1)+2*(sS1*sS2)*(log((s(2)*p(1))/s(1)*p(2)));
proot=[A,B,C];
T=roots(proot);
% chosing T between mu(1) and mu(2)
if(T(1)>mu(1) && T(1)<mu(2))
    final_T=abs(T(1));
else
    final_T =abs( T(2));
end
 
end