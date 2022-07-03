function Hd = FIR_Notch
%50HZBS Returns a discrete-time filter object.


% All frequency values are in Hz.
Fs = 300;  % Sampling Frequency
f_remove=50;
w=(2*pi*f_remove)/Fs;
p=exp(1i*w);
b=poly([p conj(p)]);
a=1;
syms G
sys=tf(b,a)
G_fir=solve(G*((2-2*cos(w))) == 1,G);
G_fir=double(G_fir);

Hd = dfilt.dffir(G_fir);
end
% [EOF]
