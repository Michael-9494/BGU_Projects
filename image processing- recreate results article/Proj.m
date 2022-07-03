%% 203833041 michael polonik
clear all
close all
clc
% load image and Luminosity method was used to convert colored image to gray scaled image
%  intensity = 0.21*red +0.71*green + 0.007*blue
iptsetpref('ImshowAxesVisible','on');
Back1_JPG = imread('N_Back_1.JPG');
Back1 = 0.21*Back1_JPG(:,:,1) +0.71*Back1_JPG(:,:,2) + 0.007*Back1_JPG(:,:,3);
Back2_JPG = imread('N_Back_2.JPG');
Back2 =  0.21*Back2_JPG(:,:,1) +0.71*Back2_JPG(:,:,2) + 0.007*Back2_JPG(:,:,3);
Im1_JPG = imread('N_IM_1.JPG');
Im1 =  0.21*Im1_JPG(:,:,1) +0.71*Im1_JPG(:,:,2) + 0.007*Im1_JPG(:,:,3);
Im2_JPG = imread('N_IM_2.JPG');
Im2 =  0.21*Im2_JPG(:,:,1) +0.71*Im2_JPG(:,:,2) + 0.007*Im2_JPG(:,:,3);

Back3_JPG = imrotate(imread('IMG_0050.JPG'),0);
Back3 = 0.21*Back3_JPG(:,:,1) +0.71*Back3_JPG(:,:,2) + 0.007*Back3_JPG(:,:,3);
Im3_JPG = imrotate(imread('IMG_0051.JPG'),0);
Im3 =  0.21*Im3_JPG(:,:,1) +0.71*Im3_JPG(:,:,2) + 0.007*Im3_JPG(:,:,3);

froad = imread("Picture1.png");
figure, imshow(froad)
figure,imhist(froad)
figure,
subplot(3,2,1),imshow(Im1,[]);
subplot(3,2,2),imshow(Back1,[]);
subplot(3,2,3),imshow(Im2,[]);
subplot(3,2,4),imshow(Back2,[]);
subplot(3,2,5),imshow(Im3,[]);
subplot(3,2,6),imshow(Back3,[]);
title(['FIG 3: Image extracted from the video and converted to gray scale']);

% step 2: substraction
O1 = Im1-Back1;
O2 = Im2-Back2;
O3 = Im3-Back3;
figure, imshow(O1,[]);
title(['FIG 4: Image output after background subtraction'])
impixelinfo;
figure, imshow(O2,[]);
title(['FIG 4: Image output after background subtraction'])
impixelinfo;
figure, imshow(O3,[]);
title(['FIG 4: Image output after background subtraction'])
impixelinfo;
figure, imhist(O1)
figure, imhist(O2)
figure, imhist(O3)

%% Image Segmentation
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


count = 0;
T2 = mean2(O2);
done = false;
while ~done
    count = count + 1;
    g = O2 > T2;
    Tnext2 = 0.5*(mean(O2 (g)) + mean(O2(~g)));
    done = abs(T2 - Tnext2) < 0.5;
    T2 = Tnext2;
end

G2 = O2>T2;
figure, imshow(G2)
title(['FIG 5: Result after Multilevel thresholding']);

% T = multithresh(O1,2);
% T2 = multithresh(O2,2);
%  AA = (O1 < T(1)); AAA = (O1 > T(2));
% Binary_Image_O1 = (AA | AAA);
%  BB = (O2 < T2(1)); BBB = (O2 > T2(2));
%  Binary_Image_O2 = (BB | BBB);
%
%  figure, imshow(Binary_Image_O1,[])
%  title(['FIG 5: Result after Multilevel thresholding']);
%  figure, imshow(Binary_Image_O2,[])
%  title(['FIG 5: Result after Multilevel thresholding']);
count = 0;
T3 = mean2(O3);
done = false;
while ~done
    count = count + 1;
    g3 = O3 > T3;
    Tnext3 = 0.5*(mean(O3 (g3)) + mean(O3(~g3)));
    done = abs(T3 - Tnext3) < 0.5;
    T3 = Tnext3;
end

G3 = O3>T3;
figure, imshow(G3)
title(['FIG 5: Result after Multilevel thresholding']);

% invert:
GG = 1-G1;
figure, imshow(GG)
title(['FIG 6: Image output after thresholding and inversion']);

GG2 = 1-G2;
figure, imshow(GG2)
title(['FIG 6: Image output after thresholding and inversion']);

GG3 = 1-G3;
figure, imshow(GG3)
title(['FIG 6: Image output after thresholding and inversion']);

%% Median filter:
M1 = medfilt2(GG);
M2 = medfilt2(GG2);
M3 = medfilt2(GG3);
figure, imshow(M1);
title(['FIG 7: Image output after applying Median filter']);
impixelinfo;
figure, imshow(M2);
title(['FIG 7: Image output after applying Median filter']);
impixelinfo;
figure, imshow(M3);
title(['FIG 7: Image output after applying Median filter']);
impixelinfo;

% Morphological Operations
B1_di = ones(3);
B2_er = ones(3);
% first, start with Dilation: NAND operation ~(A & B);
MM1 =  imdilate(M1,B1_di);
figure, imshow(MM1,[])
% then, Erosion: AND operation (A & B);
Im1_after_di_and_er =imerode(MM1,B2_er) ;

MM2 = imdilate(M2,B1_di);
figure, imshow(MM2)
Im2_after_di_and_er =imerode(MM2,B2_er) ;

MM3 = imdilate(M3,B1_di);
figure, imshow(MM3)
Im3_after_di_and_er =imerode(MM3,B2_er) ;

figure, imshow(Im1_after_di_and_er);
title(['FIG 8: Output after applying Morphological operations']);
impixelinfo;
figure, imshow(Im2_after_di_and_er);
title(['FIG 8: Output after applying Morphological operations']);
impixelinfo;
figure, imshow(Im3_after_di_and_er);
title(['FIG 8: Output after applying Morphological operations']);
impixelinfo;