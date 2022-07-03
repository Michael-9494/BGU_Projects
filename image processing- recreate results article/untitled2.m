
% 203833041 michael polonik
clear all
close all
clc
% load image and Luminosity method was used to convert colored image to gray scaled image
%  intensity = 0.21*red +0.71*green + 0.007*blue
iptsetpref('ImshowAxesVisible','on');
Back1_JPG = imread('IMG_0085.JPG');
Back1 = 0.21*Back1_JPG(:,:,1) +0.71*Back1_JPG(:,:,2) + 0.007*Back1_JPG(:,:,3);
Im1_JPG = imread('IMG_0084.JPG');
Im1 =  0.21*Im1_JPG(:,:,1) +0.71*Im1_JPG(:,:,2) + 0.007*Im1_JPG(:,:,3);

figure, imshowpair(Im1,Back1,'montage');
title(['FIG 3: Image extracted from the video and converted to gray scale']);


% LoG = fspecial('log'); %creating LoG filter
% Im11 = imfilter(Im1,LoG); %filteing the image
% figure,imshow(Im11)
% Im11 = (Im1-(Im11)); figure,imshow(Im11)

% step 2: substraction
O1 = Im1-Back1; 
figure, imshow(O1,[]);
title(['FIG 4: Image output after background subtraction'])
impixelinfo;
figure, imhist(O1)
% Filtering the Fourier transform and then inverse
% high-pass circle filter
%read the image file into a matrix:
keaton = O1;
[ly,lx]=size(keaton); %determine image size
dx = (0:(lx-1))-round(lx/2); dy = (0:(ly-1))-round(ly/2);
%a grid with the same size as image, with center (0,0):
[x,y] = meshgrid(dx,dy);
R = sqrt((x.^2)+(y.^2));
circ=(R >20); %circle (high-pass filter)
FKe=fftshift(fft2(keaton)); %Fourier transform
FKefil=FKe.*circ; %filtering in the frequency domain
IKe=ifft2(FKefil); %inverse transform
uIKe=uint8(abs(IKe)); %convert to unsigned 8-bit
figure(1)
subplot(1,2,1)
imshow(keaton); %display the photo
title('Fourier circle high-pass filtering');
ylabel('original');
subplot(1,2,2)
imshow(circ); %the filter
ylabel('the filter');
figure(2)
imshow(uIKe);
title('filtered image');

O1 = uIKe;
% step 3: Image Segmentation
count = 0;
T1 = mean2(O1);
done = false;
while ~done
    count = count + 1;
    g = O1 > T1;
    Tnext = 0.5*(mean(O1 (g)) + mean(O1(~g)));
    done = abs(T1 - Tnext) < 0.5;
    T1 = Tnext;
end

G1 = O1>T1;

figure, imshow(G1)
title(['FIG 5: Result after Multilevel thresholding']);

% step 4: invert:
GG = 1-G1;
figure, imshow(GG)
title(['FIG 6: Image output after thresholding and inversion']);


% step 5: Median filter:
M1 = medfilt2(GG,[5 5]);
figure, imshow(M1);
title(['FIG 7: Image output after applying Median filter']);
impixelinfo;










