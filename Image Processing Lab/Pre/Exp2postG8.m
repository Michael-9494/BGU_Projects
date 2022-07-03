%% Exp2preG8 
%% Q1
% 1.2
clear all; clc;
beautiful_eden = imread('WhatsApp Image 2021-11-17 at 15.51.07.jpeg');
beautiful_eden = rgb2gray(beautiful_eden);
new_beautiful_eden = beautiful_eden;
new_beautiful_eden(beautiful_eden<0.3*256 | beautiful_eden>0.6*256) = 0;

figure
subplot(2,1,1)
imshow(beautiful_eden)
subplot(2,1,2)
imhist(beautiful_eden)

figure
subplot(2,1,1)
imshow(new_beautiful_eden)
subplot(2,1,2)
imhist(new_beautiful_eden);ylim([0 2e+4])

%1.3
figure
imshow(Negative(beautiful_eden))

%1.4
clear all; clc;
skeleton = imread('skeleton.jpg');
[ResultPic]=AddToMandi(skeleton);
figure; imshow(ResultPic);

%% Q2 

% 2.3
% two filters- padd them and show fft2
%Laplacian
h_Lap = [-1,-1,-1 ; -1,8,-1 ; -1,-1,-1];
h_Lap = double(h_Lap);
%LP
h_LP = (1/9)*ones(3,3);
h_LP = double(h_LP)

% padd them
h_Lap_pad = padarray(h_Lap,[14 14]);
h_Lap_pad = h_Lap_pad(2:end,2:end);% get it to be (30X30)

h_LP_pad = padarray(h_LP,[14 14]);
h_LP_pad = h_LP_pad(2:end,2:end);% get it to be (30X30)

% FFT
H_Lap = fftshift(fft2(h_Lap_pad,30,30));% do fft and shift dc to center of picture
figure, imshow(abs(H_Lap),[])
title 'fft2- Laplacian '; axis tight


H_Lp = fftshift(fft2(h_LP_pad,30,30));% do fft and shift dc to center of picture
figure, imshow(abs(H_Lp),[])
title 'fft2- smoothing ';axis tight

figure, mesh(abs(H_Lap))
title 'fft2- Laplacian ';axis tight

figure, mesh(abs(H_Lp))
title 'fft2- smoothing ';axis tight




% 2.5
im = imread("lenna.jpg");
im = im2double(im);

% blur filter
h_bl3 = 1/9* ones(3,3);
h_bl4 = 1/16* ones(4,4);
h_bl5 = 1/25* ones(5,5);

% 3X3
im1=imfilter(im,h_bl3);


%try to improve using 2.4 answer
L = [-1,-1,-1 ; -1,8,-1 ; -1,-1,-1]; % laplacian
HH = (fft2(L,size(im1,1),size(im1,2)));% in frequency domain
Im = (fft2(im1)); % fft of image 
G = HH.*Im; % Filtering
g = ifft2((G)); % convert to time domain
% g = g(2:end-1,2:end-1);% take relevant size of image
gg = im + g;
mse_3_2_4 = immse(im,gg)% mse


% improve with fspecial
w = fspecial('laplacian',0);
% convert image to double
f2 = im2double(im1);
g2 = imfilter(f2,w,'replicate');
% enhance with laplacian
g = f2 - g2;


figure,
subplot(2,2,1), imshow(im,[]); title 'lenna- OG'
subplot(2,2,2), imshow(im1);title 'lenna- 3x3 smoothing'
subplot(2,2,3), imshow(gg); title 'lenna_{3x3}- 2.4'
subplot(2,2,4),imshow(g), title 'lenna_{3X3}- fspecial '
mse_3_fspecial = immse(im,g)


% 4X4
im2=imfilter(im,h_bl4);

%try to improve using 2.4 answer
HH = (fft2(L,size(im2,1),size(im2,2)));% in frequency domain
Im = (fft2(im2)); % fft of image 
G = HH.*Im; % Filtering
g = ifft2((G)); % convert to time domain
% g = g(2:end-2,2:end-2);% take relevant size of image
gg = im + g;
mse_4_2_4 = immse(im,gg)% mse


% improve with fspecial
w = fspecial('laplacian',0);
% convert image to double
f2 = im2double(im2);
g2 = imfilter(f2,w,'replicate');
% enhance with laplacian
g = f2 - g2;



figure,
subplot(2,2,1), imshow(im,[]); title 'lenna- OG'
subplot(2,2,2), imshow(im2);title 'lenna- 4x4 smoothing'
subplot(2,2,3), imshow(gg); title 'lenna_{4x4}- 2.4'
subplot(2,2,4),imshow(g), title 'lenna_{4X4}- fspecial '
mse_4_fspecial = immse(im,g)


% 5X5
im3=imfilter(im,h_bl5);
%try to improve using 2.4 answer
HH = (fft2(L,size(im3,1),size(im3,2)));% in shifted frequency domain
Im = (fft2(im3)); % shifted fft of image 
G = HH.*Im; % Filtering
g = ifft2((G)); % convert to time domain
gg = im + g;
mse_5_2_4 = immse(im,gg)% mse


% improve with fspecial
w = fspecial('laplacian',0);
% convert image to double
f2 = im2double(im3);
g2 = imfilter(f2,w,'replicate');
% enhance with laplacian
g = f2 - g2;

figure,
subplot(2,2,1), imshow(im,[]); title 'lenna- OG'
subplot(2,2,2), imshow(im2);title 'lenna- 5x5 smoothing'
subplot(2,2,3), imshow(gg); title 'lenna_{5x5}- 2.4'
subplot(2,2,4),imshow(g), title 'lenna_{5X5}- fspecial '
mse_5_fspecial = immse(im,g)


% 2.6
lagi = imread("Lagi.jpg");
   lagi = rgb2gray(rot90(lagi,3)) ;
figure, imshow(lagi,[])
title 'Lagi image '

L = fftshift(fft2(lagi,size(lagi,1),size(lagi,2)));% do fft and shift dc to center of picture
Lagi = (fft2(lagi,size(lagi,1),size(lagi,2)));% do fft and shift dc to center of picture
figure ; mesh(fftshift(abs(Lagi))); 
% ylim([2000 2040]); xlim([1490 1550]);
title 'Lagi in spectrum '

% find 

% Ideal high pass
P=size(lagi);
M=P(1);N=P(2);
F=fft2(lagi,M,N);

% Set up range of variables.
u = 0:(M - 1);
v = 0:(N - 1);
% Compute the indices for use in meshgrid.
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
% Compute the meshgrid arrays.
[V,U] = meshgrid(v,u);

D=sqrt(U.^2+V.^2);   

u0=4; %remove freq less than  10%
H_10=1-double(D<=u0);
G=H_10.*F;
g=(ifft2(G));% display
figure, imshow(histeq(uint8(g)),[])% Enhance contrast using histogram equalization
title('10% Low frequancy loss')

u0=20; %remove freq less than  50%
H_50=1-double(D<=u0);
G=H_50.*F;
g=(ifft2(G));
figure, imshow(histeq(uint8(g)),[])% Enhance contrast using histogram equalization
title('50% Low frequancy loss')

u0=90; %remove freq less than  90%
H_90=1-double(D<=u0);
G=H_90.*F;
g=(ifft2(G));
figure, imshow(histeq(uint8(g)),[])% Enhance contrast using histogram equalization
title('90% Low frequancy loss')




% 2.7

im = imread("rect_im.jpg");
im = rgb2gray(im);
im = im2double(im);

% filters:
h_23 = (1/23)*ones(1,23); h_25 = (1/25)*ones(1,25); 
h_45 = (1/45)*ones(1,45); h_50 = (1/50)*ones(1,50); 

% filter image:
im_23 = imfilter(im,h_23); im_25 = imfilter(im,h_25);
im_45 = imfilter(im,h_45); im_50 = imfilter(im,h_50);

figure,
subplot(3,2,1), imshow(im,[]); title 'image'
subplot(3,2,2), imshow(im_23);title '23'
subplot(3,2,3), imshow(im_25); title '25'
subplot(3,2,4),imshow(im_45), title '45 '
subplot(3,2,5),imshow(im_50), title '50'

% 2.8

I = imread('eight.tif');

J = imnoise(I,'salt & pepper',0.02);% add noise
out_I_1=CleanSP(J,'gaussian',10,10);
out_I_2=CleanSP(J,'median',5,5);

figure,subplot(2,2,1), imshow(I);title 'coins'
subplot(2,2,2), imshow(J); title 'coins+ salt & pepper noise'
subplot(2,2,3), imshow(out_I_1); title 'gaussian filtering'
subplot(2,2,4), imshow(out_I_2); title 'median filtering'


%% Q3
clc; clear all;

% 3.2
IMG = imread('circles.png');
[n,Coordinates]=LocateCirc(IMG)

%3.3
clear all; clc;
Iexam = imread('Iexam.tif');
noisy01_Iexam = imnoise(Iexam,'salt & pepper',0.01);
noisy03_Iexam = imnoise(Iexam,'salt & pepper',0.03);
noisy20_Iexam = imnoise(Iexam,'salt & pepper',0.2);
figure; 
subplot(3,2,1); imshow(noisy01_Iexam); title('1% noise')
subplot(3,2,3); imshow(noisy03_Iexam); title('3% noise')
subplot(3,2,5); imshow(noisy20_Iexam); title('20% noise')

se = strel('square',2);
filtered01 = imclose(imopen(noisy01_Iexam,se),se)
subplot(3,2,2); imshow(filtered01); title('filtered 1% noise')

filtered03 = imclose(imopen(noisy03_Iexam,se),se)
subplot(3,2,4); imshow(filtered03); title('filtered 3% noise')

filtered20 = imclose(imopen(noisy20_Iexam,se),se)
subplot(3,2,6); imshow(filtered20); title('filtered 20% noise')

%3.4
clear all; clc;
stick(3,3)=256; stick(4,2)=256; stick(4,4)=256;stick(5,2)=256;stick(5,4)=256;stick(6,3)=256;
stick(6:17,3)=256; stick(8,4)=256; stick(8,5)=256; stick(8,2)=256; stick(8,1)=256; 
stick(18,4)=256; stick(19,5)=256; stick(18,2)=256; stick(19,1)=256; stick(18,3)=256;

N=20;X=zeros(N);
[n,m]=size(stick);
X(N-n+1:N,2:m+1)=stick;

H1=[0 1 1];H2=[1 1 0];
for i=0:N
X=imdilate(X,H1);
X=imerode(X,H2);
figure(2)
imshow(X,'InitialMagnification','fit')
drawnow 
clear ts; ts=tic;
while toc(ts)<0.1;end
end

%3.5
clear all; clc;
[M]=IMove([1,40;2,30;1,-40;2,-30])

L = size(M); N = L(3);
for i=1:N
figure(2)
imshow(M(:,:,i),'InitialMagnification','fit')
drawnow 
clear ts; ts=tic;
while toc(ts)<0.01;end
end


%3.6
a = [1 1 1 0 0]
B = [1 0 0 0]
B_1 = [1 1 0 0 ]
B_2 = [0 0 0 1]
C = imdilate(B,a)
C_1 = imdilate(B_1,a)
C_2 = imdilate(B_2,a)

%% Q4
%4.2


A=imread('nicole&oli.jpg'); %Read in image
A = rgb2gray(A) ;


[bw,thresh]=edge(A,'log',0.001); %Edge detection on original LoG filter
figure 

 imshowpair(A,bw,'montage'); title 'Original Image and Edge detection with LoG filter';





