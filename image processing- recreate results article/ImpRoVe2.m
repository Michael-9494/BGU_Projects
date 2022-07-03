%% 203833041 michael polonik
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

% step 2: substraction
SubtractedIm = Im1-Back1; 
figure, imshow(SubtractedIm,[]);
title(['FIG 4: Image output after background subtraction'])
impixelinfo;
figure, imhist(SubtractedIm)


% results

figure, histogram(SubtractedIm,100);
[counts,centers]=hist(double(SubtractedIm(:)),100);
v =centers;
h = counts;
% normlizied histogram
norm_factor=sum(h);
h=h./norm_factor;
display(round(sum(h)) == 1)
figure;
plot(v,h);
norm_factor=sum(h);
h=h./norm_factor;

% find optimal min by optimization tool
M=2;
Aeq = zeros(1,3*M);
Aeq(2*M+1:3*M)=1;
beq = 1;
A=-eye(3*M,3*M);
b=zeros(3*M,1);
x0 =[1*randn(2*M,1);ones(M,1)/M] ;    %rand mu,sjgma, even probabilities
fun = @(x)(sum((histogram_model(v,M,x(1:M),x(M+1:2*M),x(2*M+1:3*M))-h).^2));
Minfval = 1e10;
for i=1:10
    x0 =[abs(255*rand(2*M,1));ones(M,1)/M] ;    %rand mu,sjgma, even probabilities
    [x,fval] = fmincon(fun,x0,A,b,Aeq,beq);
    if(fval< Minfval )
        Minfval = fval;
        Bestx = x;
    end
end
x = Bestx;
%x(end-1:end)
BestHist = histogram_model(v,M,x(1:M),x(M+1:2*M),x(2*M+1:3*M));
figure;plot(v,BestHist);hold on;plot(v,h,'--');legend('HistEst','H');

% find optimal threshold
T=find_threshold(x,M);
% color cells in green
ThresholdedIm=SubtractedIm>T;

% step 4: invert:
ThresholdedIm = 1-ThresholdedIm;
figure, imshow(ThresholdedIm)
title(['FIG 6: Image output after thresholding and inversion']);

% step 5: Median filter:
MedIm = medfilt2(ThresholdedIm,[7 7]);
figure, imshow(MedIm);
title(['FIG 7: Image output after applying Median filter']);
impixelinfo;

%step 6: Morphological Operations
B1_di = ones(7);
B2_er = ones(5);
% first, start with Dilation: NAND operation ~(A & B);
MM1 =  imdilate(MedIm,B1_di);
figure, imshow(MM1,[])
% then, Erosion: AND operation (A & B);
Im1_after_di_and_er =imerode(MM1,B2_er) ;
figure, imshow(Im1_after_di_and_er);
title(['FIG 8: Output after applying Morphological operations']);
impixelinfo;

redChannel = SubtractedIm; % Initialize
redChannel(ThresholdedIm) = 0;
greenChannel = SubtractedIm; % Initialize
greenChannel(ThresholdedIm) = 255;
blueChannel = SubtractedIm; % Initialize
blueChannel(ThresholdedIm) = 0;
rgbImage = cat(3, redChannel, greenChannel, blueChannel);
rgbImage(rgbImage<T) = 0;
%
figure;
imshow(rgbImage);
title('Recognition of the cells-part B');