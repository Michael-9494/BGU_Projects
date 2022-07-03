%% 203833041 michael polonik
clear all
close all
clc
tic
% load image and Luminosity method was used to convert colored image to
% gray scaled image with intensity = 0.21*red +0.71*green + 0.007*blue
iptsetpref('ImshowAxesVisible','off');

Back1_JPG = imread('IMG_0085.JPG');
Back1 = 0.21*Back1_JPG(:,:,1) +0.71*Back1_JPG(:,:,2) + 0.007*Back1_JPG(:,:,3);
Back2_JPG = imread('IMG_0073.JPG');
Back2 = 0.21*Back2_JPG(:,:,1) +0.71*Back2_JPG(:,:,2) + 0.007*Back2_JPG(:,:,3);
Im1_JPG = imread('IMG_0084.JPG');
Im1 =  0.21*Im1_JPG(:,:,1) +0.71*Im1_JPG(:,:,2) + 0.007*Im1_JPG(:,:,3);
Im2_JPG = imread('IMG_0074.JPG');
Im2 =  0.21*Im2_JPG(:,:,1) +0.71*Im2_JPG(:,:,2) + 0.007*Im2_JPG(:,:,3);

figure, imshowpair(Im1,Back1,'montage');
title(['FIG 3: Image extracted from the video and converted to gray scale']);
figure, imshowpair(Im2,Back2,'montage');
title(['FIG 3: Image extracted from the video and converted to gray scale']);

% step 2: substraction
SubtractedIm1 = Im1-Back1;
SubtractedIm2 = Im2-Back2;
figure(2), imshowpair(SubtractedIm1,SubtractedIm2,'montage');
title(['FIG 4: Image output after background subtraction'])


% froad = imread("Picture1.png");
% figure, imshow(froad)
% figure,imhist(froad)

% step 3: Image Segmentation
[Qsignal,Q] = U_quantizer(SubtractedIm1,2);
% figure;
% imshow(Qsignal);impixelinfo;
% figure, imhist(Qsignal)

ThresholdedIm1 = Qsignal>0.1;


[Qsignal,~] = U_quantizer(SubtractedIm2,2);
% figure;
% imshow(Qsignal);impixelinfo;
% figure, imhist(Qsignal)
ThresholdedIm2 = Qsignal>0.1;

figure,imshowpair(ThresholdedIm1,ThresholdedIm2,'montage');
title(['FIG 5: Result after Multilevel thresholding']);

% step 4: invert:
InvertIm1 = 1-ThresholdedIm1;
InvertIm2 = 1-ThresholdedIm2;
figure, imshowpair(InvertIm1,InvertIm2,'montage')
title(['FIG 6: Image output after thresholding and inversion']);

% step 5: Median filter:
InvertAndMedIm1 = medfilt2(InvertIm1,[7 7]);
InvertAndMedIm2 = medfilt2(InvertIm2,[9 9]);
figure, imshowpair(InvertAndMedIm1,InvertAndMedIm2,'montage') ;
title(['FIG 7: Image output after applying Median filter']);

%step 6: Morphological Operations
B1_di = ones(7);
B2_er = ones(7);
% first, start with Dilation
MM1 =  imdilate(InvertAndMedIm1,B1_di);
MM2 =  imdilate(InvertAndMedIm2,B1_di);
% figure, imshow(MM1,[])
% then, Erosion
Im1_after_di_and_er =imerode(MM1,B2_er) ;
Im2_after_di_and_er =imerode(MM2,B2_er) ;
figure, imshowpair(Im1_after_di_and_er,Im2_after_di_and_er,'montage') ;
title(['FIG 8: Output after applying Morphological operations']);

Run_time = toc;







