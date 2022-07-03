% 1.To perform Basic Image Processing Operations                                                       
%   a. Arithmetic and Logical Operations
clc;
clear all;
close all;
A=imread('cameraman.tif');
subplot(5,3,1)
imshow(A)
title('Image A');
B=imread('rice.png');
subplot(5,3,2)
imshow(B)
title ('Image B');
C=zeros(size(A));
for (x= 100: 200)
    for (y=100: 200)
        C(x,y)=255;
    end
end
subplot(5,3,3)
imshow(C)
title('Image C');
C=uint8(C);
% Logical Operations
lr1=bitand(A,C);
subplot(5,3,4)
imshow(lr1)
title('A and C');
lr2=bitand(B,C);
subplot(5,3,5)
imshow(lr2)
title('B and C');
lr3=bitand(A,B);
subplot(5,3,6)
imshow(lr3)
title('A and B');
lr4=bitor(A,C);
subplot(5,3,7)
imshow(lr2)
title('A or C');
lr5=bitor(B,C);
subplot(5,3,8)
imshow(lr5)
title('B or C');
lr6=bitor(A,B);
subplot(5,3,9)
imshow(lr6)
title('A or B');
lr7=bitxor(A,C);
subplot(5,3,10)
imshow(lr7)
title('A exor C');
lr8=bitxor(B,C);
subplot(5,3,11)
imshow(lr8)
title('B exor C');
lr9=bitxor(A,B);
subplot(5,3,12)
imshow(lr9)
title('A exor B');
lr10= bitcmp(A);
subplot(5,3,13)
imshow(lr10)
title('Not A');
lr11= bitcmp(B);
subplot(5,3,14)
imshow(lr11)
title('Not B');
lr12= bitcmp(C);
subplot(5,3,15)
imshow(lr12)
title('Not C');
% Arithmetic Operations
lr13=imadd(A,C);
figure,subplot(4,3,1)
imshow(uint8(lr13))
title('A + C');
lr14=imadd(B,C);
subplot(4,3,2)
imshow(uint8(lr14))
title('B + C');
lr15=imadd(A,B);
subplot(4,3,3)
imshow(uint8(lr15))
title('A + B');
lr16=imsubtract(A,C);
subplot(4,3,4)
imshow(uint8(lr16))
title('A - C');
lr17=imsubtract(B,C);
subplot(4,3,5)
imshow(uint8(lr17))
title('B - C');
lr18=imsubtract(A,B);
subplot(4,3,6)
imshow(uint8(lr18))
title('A - B');
lr19=immultiply(A,C);
subplot(4,3,7)
imshow(uint8(lr19))
title('A * C');
lr20=immultiply(B,C);
subplot(4,3,8)
imshow(uint8(lr20))
title('B * C');
lr21=immultiply(A,B);
subplot(4,3,9)
imshow(uint8(lr21))
title('A * B');
lr22=imdivide(A,C);
subplot(4,3,10)
imshow(uint8(lr22))
title('A/C');
lr23=imdivide(B,C);
subplot(4,3,11)
imshow(uint8(lr23))
title('B/C');
lr24=imdivide(A,B);
subplot(4,3,12)
imshow(uint8(lr24))
title('A/B');