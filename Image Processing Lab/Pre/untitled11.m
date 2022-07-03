
clear all, clc
x = -4:4;y = -4:4;
sigma = 1;

z1 = (1/(( sqrt(2*pi)*sigma^2)))*exp( (1/(sigma^2)* ((x.^2)) ));
z2 = (1/(( sqrt(2*pi)*sigma^2)))*exp( (1/(sigma^2)* ((y.^2)) ));
z = fftshift(z1'*z2);
figure, mesh(z)

w = fspecial('gaussian',5,1)
figure,mesh(w)