%Matlab code What is happening?
length=18; tlevel=0.2; %Define SE and percent

A=im2gray(imread('circuit.jpg'));
figure, subplot(2,3,1), imshow(A) %Read image and display
B=~(im2bw(A,tlevel)); subplot(2,3,2), imshow(B); %Threshold image and

SE=ones(3,length); bw1=imerode(B,SE); %Erode vertical lines
subplot(2,3,3), imshow(bw1); %Display result
title(['Erode vertical lines']);
bw2=imerode(bw1,SE'); subplot(2,3,4), imshow(bw2); %Erode horizontal lines
title(['Erode horizontal lines']);
bw3=imdilate(bw2,SE');bw4=imdilate(bw3,SE); %Dilate back
subplot(2,3,5), imshow(bw4); %Display
title(['Dilate back']);
boundary=bwperim(bw4);
[i,j]=find(boundary); %Superimpose boundaries
 subplot(2,3,6), imshow(A); hold on; plot(j,i,'r.');
 title(['Superimpose boundaries']);