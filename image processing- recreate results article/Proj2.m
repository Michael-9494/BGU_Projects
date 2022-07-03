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
figure, imshowpair(SubtractedIm1,SubtractedIm2,'montage');
title(['FIG 4: Image output after background subtraction'])

figure, imhist(SubtractedIm1);title(['histogram-Subtracted Image 1'])
figure,imhist(SubtractedIm2);title(['histogram-Subtracted Image 2'])
%  froad = imread("Picture1.png");
%  figure, imshow(froad)
%  figure,imhist(froad)

% step 3: Image Segmentation
%BASIC GLOBAL THRESHOLDING
count = 0;
% Segment the image using T. This will produce two groups of pixels: Gn
% consisting of all pixels with intensity values greater than T and G2,
% consisting of pixels with values less than or equal to T.
T1 = mean2(SubtractedIm1);
done = false;
while ~done
    count = count + 1;
    g = SubtractedIm1 > T1;
    % Compute the average intensity values m1 and m2 of the pixels in
    % regions G1 and G2, respectively.
    m1 = mean(SubtractedIm1 (g)); m2 = mean(SubtractedIm1(~g));
    % Compute a new threshold value:
    Tnext = 0.5*(m1 + m2);
    % Repeat steps until the difference in T in successive
    % iterations is smaller than a predefined value 0.5.
    done = abs(T1 - Tnext) < 0.5;
    T1 = Tnext;
end

ThresholdedIm1 = SubtractedIm1>T1;

%BASIC GLOBAL THRESHOLDING
count = 0;
% Segment the image using T. This will produce two groups of pixels: Gn
% consisting of all pixels with intensity values greater than T and G2,
% consisting of pixels with values less than or equal to T.
T2 = mean2(SubtractedIm2);
done = false;
while ~done
    count = count + 1;
    g = SubtractedIm2 > T2;
    % Compute the average intensity values m1 and m2 of the pixels in
    % regions G1 and G2, respectively.
    m1 = mean(SubtractedIm2 (g)); m2 = mean(SubtractedIm2(~g));
    % Compute a new threshold value:
    Tnext = 0.5*(m1 + m2);
    % Repeat steps until the difference in T in successive
    % iterations is smaller than a predefined value T1.
    done = abs(T2 - Tnext) < 0.5;
    T2 = Tnext;
end
ThresholdedIm2 = SubtractedIm2>T2;

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

Run_time1 = toc;

%% improve:
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
title(['Image output after background subtraction'])

% step 3: Image Segmentation
[Qsignal,Q] = U_quantizer(SubtractedIm1,2);
ThresholdedIm1 = Qsignal>0.1;

[Qsignal,Q2] = U_quantizer(SubtractedIm2,2);
ThresholdedIm2 = Qsignal>0.1;

figure,imshowpair(ThresholdedIm1,ThresholdedIm2,'montage');
title([' Result after Uniform quantization and thresholding']);

% step 4: invert:
InvertIm1 = 1-ThresholdedIm1;
InvertIm2 = 1-ThresholdedIm2;
figure, imshowpair(InvertIm1,InvertIm2,'montage')
title(['Image output after thresholding and inversion']);

% step 5: Median filter:
InvertAndMedIm1 = medfilt2(InvertIm1,[7 7]);
InvertAndMedIm2 = medfilt2(InvertIm2,[9 9]);
figure, imshowpair(InvertAndMedIm1,InvertAndMedIm2,'montage') ;
title(['Image output after applying Median filter']);

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
title(['Output after applying Morphological operations']);

Im11 = medfilt2(Im1_after_di_and_er,[5 5]);
Im22 = medfilt2(Im2_after_di_and_er,[11 11]);
figure, imshowpair(Im11,Im22,'montage') ;
title(['image after another median filter']);
Run_time2 = toc;

%% Test the error
IM1_from_recreation = 1-Im1_after_di_and_er;
IM1_from_recreation(2781:3247,3214:3760) = 0;
IM1_from_recreation(1700:2200,2200:2700) = 0;
figure, imshow(IM1_from_recreation,[])
Sum_im1_from_recreation = sum(sum(IM1_from_recreation))

IM2_from_recreation = 1-Im2_after_di_and_er ;
IM2_from_recreation(2300:2700,2600:3100) = 0;
figure, imshow(IM2_from_recreation,[])
Sum_im2_from_recreation = sum(sum(IM2_from_recreation))

IMM11 = 1-Im11;
IMM11(2781:3247,3214:3760) = 0;
IMM11(1700:2200,2200:2700) = 0;
figure, imshow(IMM11,[]);impixelinfo;
Sum_im1 = sum(sum(IMM11))


IMM22 = 1-Im22;
IMM22(2300:2700,2600:3100) = 0;
figure, imshow(IMM22,[]);impixelinfo;
Sum_im2 = sum(sum(IMM22))

