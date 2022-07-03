

%exp 2 - image processing
%% Q-1----------------------------------

%part 1
clear all;clc;

%1.1
Bcr= imread('rice.png');%Bc for Blood Cell r for Raw
Bcr = double(Bcr)/255; %scale as double between 0-1
% estimating the bckground for better masking
se = strel('disk',10);
Bco = imopen(Bcr,se);
Bc = Bcr - Bco;%substracting the background

figure
subplot(121)
imshow(Bcr)
title('Blood cell')
subplot(122)
imshow(Bc)
title('Blood cell with Background substraction')

%1.2
figure
sgtitle('Histogram')
subplot(121)
imhist(Bcr)
title('Original Blood cell ')
xlabel('Gray level')
ylabel('Count')
subplot(122)
imhist(Bc)
title('Blood cell')
xlabel('Gray level')
ylabel('Count')
%1.3
% treshholding
Bc_MASK= Bc;
Bc_MASK(find(Bc<=0.15))=0;
Bc_MASK(find(Bc>0.15))=1;

figure
imshow(Bc_MASK)
title('mask after tresholding')

%1.4
element = strel('disk',3); %element to do imopen
Bc_cleanMask = imopen(Bc_MASK,element); figure, imshow(Bc_cleanMask)
Only_Bc = Bc_cleanMask.*Bcr;

figure
imshow(Only_Bc)
title('Only the Blood Cells with no background')
% %1.5
% %1.52

%1.53
Bc_MASK2= Bc_cleanMask;
Bc_MASK2(find(Bc_cleanMask<0.25))=0;
Bc_MASK2(find(Bc_cleanMask>0.25))=1;

figure
subplot(121)
imhist(Only_Bc)
title('Only Blood cell Histogram')
xlabel('Gray level')
ylabel('Count')
subplot(122)
imshow(Bc_MASK2)
title('Mask by tresholding')

LoG = fspecial('log'); %creating LoG filter
BcAfterLoG = imfilter(Bc_MASK2,LoG); %filteing the image

figure
sgtitle('Blood Cells Border')
subplot(121)
imshow(BcAfterLoG)
title('Laplacian of Gaussian filter')

LaplaceFilter = [-1 -1 -1 ; -1 8 -1 ; -1 -1 -1]; %laplacian matrix
BcLapla = imfilter(Bc_MASK2,LaplaceFilter);
subplot(122)
imshow(BcLapla)
title('Laplacian filter')

%1.6
Label = bwlabel(Bc_MASK2);
Estimate_Bc_Num = max(max(Label)); % finding the highest element number wich will be thr estimation
disp('Estimation of Blood Cell Number is:')
disp([Estimate_Bc_Num])
Orientation=cell2mat(struct2cell(regionprops(Label,'Orientation')));
[~,I] = find(Orientation>0 & Orientation<90); % Marking all the Bc that are tilted between 0 - 90 degree

%1.8
% we will make a loop to flag the element number providing by the bwlabel
% with the a scalar which is equal to 1 + num of element found
Only_Right_Mask = zeros(size(Label));
for i = 1:length(I)

    Only_Right_Mask(find(Label==I(i)))=1;

end
Only_Right = Only_Right_Mask.*Bcr;
figure
imshow(Only_Right)
title('Only right tilted Blood Cells')


%% Q-2--------------------------------

%% 2.1
clear all

SPveg = imread('Nvegetables.tif'); %loading Image
SPveg = im2double(SPveg);
% [mv,nv] = size(SPveg); %size image
veg = imread('vegetables.tif');
veg = im2double(veg);

m = [7 5 4 4 3 3 2 1 1]; %size of filter
n = [7 5 5 4 4 3 2 3 2];

restoredefaultpath
rehash toolboxcache

Imafter77 = CleanSP(SPveg,'Median',m(1),n(1)); %filtering the image using median filter with different sizes
Imafter55 = CleanSP(SPveg,'Median',m(2),n(2));
Imafter45 = CleanSP(SPveg,'Median',m(3),n(3));
Imafter44 = CleanSP(SPveg,'Median',m(4),n(4));
Imafter34 = CleanSP(SPveg,'Median',m(5),n(5));
Imafter33 = CleanSP(SPveg,'Median',m(6),n(6));
Imafter22 = CleanSP(SPveg,'Median',m(7),n(7));
Imafter13 = CleanSP(SPveg,'Median',m(8),n(8));
Imafter12 = CleanSP(SPveg,'Median',m(9),n(9));


MSE_77 = mse(veg,Imafter77); %MSE for each filtering
MSE_55 = mse(veg,Imafter55);
MSE_45 = mse(veg,Imafter45);
MSE_44 = mse(veg,Imafter44);
MSE_34 = mse(veg,Imafter34);
MSE_33 = mse(veg,Imafter33);
MSE_22 = mse(veg,Imafter22);
MSE_13 = mse(veg,Imafter13);
MSE_12 = mse(veg,Imafter12);

MSEtot = [MSE_77 MSE_55 MSE_45 MSE_44 MSE_34 MSE_33 MSE_22 MSE_13 MSE_12]; %total MSE
CatName = categorical({'7X7','5X5','4X5','4X4','3X4','3X3','2X2','1X3','1X2'}); %names of filters sizes

figure %plotting the MSE
bar(CatName,MSEtot)
title('MSE after filtering with different filters sizes')
ylabel('MSE')
grid on

figure %best filter and good image
subplot(1,2,1)
imshow(veg)
title('original clean image')
subplot(1,2,2)
imshow(Imafter33)
title('image after using median filter 3X3')

%% 2.2

clear all

veg = imread('vegetables.tif'); %loading image
veg = im2double(veg);

MSE_77 = zeros(1,100); %MSE intializing
MSE_55 = zeros(1,100);
MSE_45 = zeros(1,100);
MSE_44 = zeros(1,100);
MSE_34 = zeros(1,100);
MSE_33 = zeros(1,100);
MSE_22 = zeros(1,100);
MSE_13 = zeros(1,100);
MSE_12 = zeros(1,100);

m = [7 5 4 4 3 3 2 1 1]; %size of filter
n = [7 5 5 4 4 3 2 3 2];

for i = 1:100 %repeting exp 100 times
NoiseVeg = imnoise(veg,'salt & pepper',0.2); %aadding noise

Imafter77 = CleanSP(NoiseVeg,'Median',m(1),n(1)); %filtering the image using median filter with different sizes
Imafter55 = CleanSP(NoiseVeg,'Median',m(2),n(2));
Imafter45 = CleanSP(NoiseVeg,'Median',m(3),n(3));
Imafter44 = CleanSP(NoiseVeg,'Median',m(4),n(4));
Imafter34 = CleanSP(NoiseVeg,'Median',m(5),n(5));
Imafter33 = CleanSP(NoiseVeg,'Median',m(6),n(6));
Imafter22 = CleanSP(NoiseVeg,'Median',m(7),n(7));
Imafter13 = CleanSP(NoiseVeg,'Median',m(8),n(8));
Imafter12 = CleanSP(NoiseVeg,'Median',m(9),n(9));

MSE_77(i) = mse(veg,Imafter77); %MSE for each filtering
MSE_55(i) = mse(veg,Imafter55);
MSE_45(i) = mse(veg,Imafter45);
MSE_44(i) = mse(veg,Imafter44);
MSE_34(i) = mse(veg,Imafter34);
MSE_33(i) = mse(veg,Imafter33);
MSE_22(i) = mse(veg,Imafter22);
MSE_13(i) = mse(veg,Imafter13);
MSE_12(i) = mse(veg,Imafter12);

end

MEAN_77 = mean(MSE_77); %mean of MSE
MEAN_55 = mean(MSE_55);
MEAN_45 = mean(MSE_45);
MEAN_44 = mean(MSE_44);
MEAN_34 = mean(MSE_34);
MEAN_33 = mean(MSE_33);
MEAN_22 = mean(MSE_22);
MEAN_13 = mean(MSE_13);
MEAN_12 = mean(MSE_12);

STD_77 = std(MSE_77); %std of MSE
STD_55 = std(MSE_55); 
STD_45 = std(MSE_45); 
STD_44 = std(MSE_44); 
STD_34 = std(MSE_34); 
STD_33 = std(MSE_33); 
STD_22 = std(MSE_22); 
STD_13 = std(MSE_13); 
STD_12 = std(MSE_12); 

meanMSEtot = [MEAN_77 MEAN_55 MEAN_45 MEAN_44 MEAN_34 MEAN_33 MEAN_22 MEAN_13 MEAN_12]; %total MEAN MSE
stdMSEtot = [STD_77 STD_55 STD_45 STD_44 STD_34 STD_33 STD_22 STD_13 STD_12]; %total STD MSE

CatName = categorical({'7X7','5X5','4X5','4X4','3X4','3X3','2X2','1X3','1X2'}); %names of filters sizes

figure %plotting the MSE
sgtitle('MSE after filtering with different filters sizes for 100 times')
subplot(1,2,1)
bar(CatName,meanMSEtot)
title('mean')
ylabel('MSE')
grid on
subplot(1,2,2)
bar(CatName,stdMSEtot)
title('std')
ylabel('MSE')
grid on


MSEtable = table(MSE_77', MSE_55', MSE_45' ,MSE_44', MSE_34' ,MSE_33' ,MSE_22' ,MSE_13',MSE_12'); %creating a table

MSEtable.Properties.VariableNames{1} = '7X7';
MSEtable.Properties.VariableNames{2} = '5X5';
MSEtable.Properties.VariableNames{3} = '4X5';
MSEtable.Properties.VariableNames{4} = '4X4';
MSEtable.Properties.VariableNames{5} = '3X4';
MSEtable.Properties.VariableNames{6} = '3X3';
MSEtable.Properties.VariableNames{7} = '2X2';
MSEtable.Properties.VariableNames{8} = '1X3';
MSEtable.Properties.VariableNames{9} = '1X2';

MSEtot = [MSE_77'; MSE_55'; MSE_45' ;MSE_44'; MSE_34' ;MSE_33'; MSE_22' ;MSE_13' ;MSE_12']; %total MSE

MSE_total = readtable('MSEtotal.txt'); %MSE for boxplot
% boxplot(MSE_total{:,1},MSE_total{:,2})


%using anova test
% p = anova(MSE_total{:,1},{MSE_total{:,2}});
[p,anovatab] = anova1(MSE_total{:,1},MSE_total{:,2});
title('boxplot')
xlabel('filter size')
ylabel('MSE')

%% Q3
clear all

%% loading the images
pics = load('exp1_pics.mat'); %loading the calibration images
number1 = pics.pics(1);
number2 = pics.pics(2);
number3 = pics.pics(3);
number4 = pics.pics(4);

number1 = number1{1,1};
number2 = number2{1,1};
number3 = number3{1,1};
number4 = number4{1,1};

%% finding Iris location for each eye
% q 3.1
cali1_3 = number1{3};
cali1_3 = rgb2gray(cali1_3); %color to BW photo
cali1_3 = im2double(cali1_3);
cali1_3 = cali1_3(230:277,155:390);

IMAGE = cali1_3; %current Frame
[Eye_Pos]= EyePosition_FUNC(IMAGE); %finding poistion of Irises for each eye in one frame
N = length(IMAGE(1,:));

figure %plotting the result
sgtitle('Iris found using the function')
subplot(1,2,2)
imshow(IMAGE(:,ceil(N/2+10):end))
viscircles(Eye_Pos(2,:), 4,'Color','b');
title('Right eye')
subplot(1,2,1)
imshow(IMAGE(:,1:ceil(N/2)))
viscircles(Eye_Pos(1,:), 4,'Color','b');
title('Left eye')

%% creating Clibration frames matrix
% q 3.3
CalibFrames = [number1 ; number2 ; number3 ; number4]; %creating calibration frame matrix
CalibFrames = CalibFrames(:,[1:2,4:end]);
CalibFrames = CalibFrames(:,[1:9,11:end]);
CalibFrames = CalibFrames(:,[1:2,4:end]);
CalibFrames = CalibFrames(:,[1:7,9:end]);
CalibFrames = CalibFrames(:,[1:5,7:end]);

eye_side = [1 2]; %1 for left eye, 2 for right eye
EyeCalib_L = EyeCalibration_FUNC(CalibFrames,eye_side(1)); %returns iris position matrix
EyeCalib_R = EyeCalibration_FUNC(CalibFrames,eye_side(2)); %returns iris position ma

figure %plotting histogram for each eye
histogram2(EyeCalib_L(1,:),EyeCalib_L(2,:),10)
title('Iris indx for left eye')
xlabel('x [pixel]')
ylabel('y [pixel]')
zlabel('counts')
view(-57,30)

figure
histogram2(EyeCalib_R(1,:),EyeCalib_R(2,:),10)
title('Iris indx for right eye')
xlabel('x [pixel]')
ylabel('y [pixel]')
zlabel('counts')

%% q 3.4

CalibFrames = CalibFrames(:,1:end-1); %taking onli N-1 from calibration, the last one will be for checking the function functonality
FRAMES = CalibFrames(:,end);

%correcting the frames for each number
Frame1 = FRAMES{1,1};
Frame1 = rgb2gray(Frame1); %color to BW photo
Frame1 = im2double(Frame1);
Frame1 = Frame1(230:277,155:390);

Frame2 = FRAMES{2,1};
Frame2 = rgb2gray(Frame2); %color to BW photo
Frame2 = im2double(Frame2);
Frame2 = Frame2(230:277,155:390);

Frame3 = FRAMES{3,1};
Frame3 = rgb2gray(Frame3); %color to BW photo
Frame3 = im2double(Frame3);
Frame3 = Frame3(230:277,155:390);

Frame4 = FRAMES{4,1};
Frame4 = rgb2gray(Frame4); %color to BW photo
Frame4 = im2double(Frame4);
Frame4 = Frame4(230:277,155:390);

Eye_Look1 = EyeLook_FUNC(Frame1,EyeCalib_R); %returns the number with the highest probability the subject looked at in the frame
Eye_Look2 = EyeLook_FUNC(Frame2,EyeCalib_R); 
Eye_Look3 = EyeLook_FUNC(Frame3,EyeCalib_R); 
Eye_Look4 = EyeLook_FUNC(Frame4,EyeCalib_R); 

%% q 3.5

%segmentation by looking at the video
% 0:1 sec 1
% 1:3 sec 2
% 3:5 sec 3
% 5:7 sec 2
% 7:8 sec 3 
% 8:10 sec 4
% 10:11 sec 3
% 11:13 sec 2
% 13:14 sec 1
% 14:16 sec 2
% 16:17 sec 1
% 17:18 sec 2
% 18:20 sec 3
% 20:22 sec 1
% 22:24 sec 2


numVecSegmentation = [1*ones(1,30) 2*ones(1,2*30) 3*ones(1,2*30) 2*ones(1,2*30) 3*ones(1,30) 4*ones(1,2*30) 3*ones(1,30)...
    2*ones(1,2*30) 1*ones(1,30) 2*ones(1,2*30) 1*ones(1,30) 2*ones(1,30) 3*ones(1,2*30) 1*ones(1,2*30) 2*ones(1,2*30) 1*ones(1,2*30)];
Error = zeros(1,length(numVecSegmentation)); %error vector

videoEye = VideoReader('exp1video5.avi');
% vidEye = read(videoEye);
i = 1;
while hasFrame(videoEye)
    frameVID = readFrame(videoEye);
    frameVID = rgb2gray(frameVID); %color to BW photo
    frameVID = im2double(frameVID);
    frameVID = frameVID(230:277,155:390);

    Eye_Look = EyeLook_FUNC(frameVID,EyeCalib_R);

      if numVecSegmentation(i) == Eye_Look
          Error(i) = 0;
      else
          Error(i) = 1;
      end   
%     Error(i)
i = i + 1;
end

Error = Error(1:720); %without last frames

figure
stem(Error)
% set(gcf, 'Position', [100,500, 500, 400]);
title('Error for each frame','FontSize',18)
xlim([0 720])
ylim([0 1.1])
xlabel('frame number','FontSize',16)
ylabel('error','FontSize',16)

numberErrors = sum(Error); %number of errors

%% Q-4 -------------------

%% 4

clear all
clc
tic

v = VideoReader('exp2_video2.avi');
vid = read(v);

back = load("background.mat").pics{1,1};
background=double(rgb2gray(cell2mat(back(1,1))))/255;
flag = ones(1,v.NumFrames);

SE11 = strel('disk',2);%erosion the left side
SE22 = Negative(SE11.Neighborhood);%erosion the left side

for i = 1 : v.NumFrames %for loop to apply algorithem each time
    a=i
 Video(i).Num = i;
    % saving each frame as BW
  Video(i).frames = rgb2gray(vid(:,:,:,(i)));
  Video(i).frames = double(Video(i).frames)/255;
  Video(i).background = Video(i).frames-background; %substracting the background
  Video(i).background1 = Video(i).background ;
  Video(i).background(Video(i).background>0.1) = 1; %high lighting the walking man
  Video(i).background(Video(i).background<-0.4) = 1; %high lighting the walking man

 % Video(i).mask = Video(i).background;
 % a figure mask without the markers
  Video(i).mask_nomarker=imclose(Video(i).background,strel('disk',3));
  Video(i).mask_nomarker=imerode(Video(i).mask_nomarker,strel('disk',6));
  % mask eith good shape of the walking man 
  %Video(i).mask_figure=imerode( Video(i).background,strel('disk',6));
  
%  Video(i).mask = Video(i).mask_figure+2*Video(i).mask_nomarker; % the mask will be combination of the two earlier masks
%  Video(i).mask(Video(i).mask<=0.5)=0; % binary mask
%  Video(i).mask(Video(i).mask>0.5)=1;

  Video(i).spots=imopen( Video(i).background,strel('disk',4)); %walking man with figure complitly white with 4 black spots
% important  
  Video(i).spots(Video(i).spots<=0.5)=0; 
  Video(i).spots(Video(i).spots>0.5)=1;

  Video(i).lables=Video(i).spots+Negative(Video(i).mask_nomarker); %adding the negative leaving us with only black spots

  Video(i).lables = imclose(Video(i).lables,strel('disk',4)); %removing noise
  
  temp_black_photo = zeros(360,640);
  temp_black_photo(120:330,170:360) = Negative(Video(i).lables(120:330,170:360));
  Video(i).location = temp_black_photo;
  %Video(i).location = imdilate(Video(i).location,strel('disk',4)) %removing noise
  Video(i).location(Video(i).location<=0.5)=0;

  Centroid =cell2mat(struct2cell(regionprops(bwlabel(Video(i).location,4),'Centroid')));
  n=length(Centroid)/2;
  Centroid = reshape(Centroid,2,n)';
  Video(i).Coordinates = zeros(4,2);
    if n>4
        Video(i).Coordinates=Centroid(end-3:end,:);
        n=4;
    elseif n == 4
        Video(i).Coordinates = Centroid;
    else
        flag(1,i) = 0;
    end

 clear Centroid 
end
%erosion----
Locaition = Video(logical(flag));

frNum = 13;
figure
subplot(231)
imshow(Locaition(frNum).frames)
title('Original frame','FontSize',18)
subplot(232)
imshow(Locaition(frNum).background1)
title('Original minus Background','FontSize',18)
subplot(233)
imshow(Locaition(frNum).background)
title('Background Thresholding ','FontSize',18)
subplot(236)
imshow(Locaition(frNum).mask_nomarker)
title('Mask without marker','FontSize',18)
subplot(235)
imshow(Negative(Locaition(frNum).spots))
title('Mask with marker','FontSize',18)
subplot(234)
imshow(Locaition(frNum).location)
title('4 markers','FontSize',18)


t =[Locaition.Num]/30 ;
coordinate_mat = [Locaition.Coordinates];
N = length(t);
%ploting space over time

x = zeros(4,N);
y = zeros(4,N);

for i = 1:N
[~,I] = sort(coordinate_mat(:,2*i));
coordinate_mat(1:4,(2*i-1):2*i) = coordinate_mat(I,(2*i-1):2*i);
x(1:4,i) = coordinate_mat(1:4,2*i-1);
y(1:4,i) = coordinate_mat(1:4,2*i)*(-1); % in the matrix the y axis is positivly down
end

figure
 plot3(x,t,y)
 title('Locatin in space as function of time view (0,0)')
 xlabel('x')
 ylabel('time [s]')
 zlabel('y')
 view(0,0)
 grid on
legend('Hip','Thigh','Knee','Ankle')
figure
 plot3(x,t,y)
 title('Locatin in space as function of time view (-35,30)')
 xlabel('x')
 ylabel('time [s]')
 zlabel('y')
 view(-35,30)
 grid on
legend('Hip','Thigh','Knee','Ankle')


%4.2
distx = abs(diff(x')');
disty = abs(diff(y')');
distance = sqrt(distx.*distx+disty.*disty);
tdif=diff(t);
v = distance./diff(t);

figure
subplot(221)
plot (t(1:end-1),v(1,:))
title('Hip')
xlabel('time [s]')
ylabel('Velocity [pixel/s]')
subplot(222)
plot (t(1:end-1),v(2,:))
title('Thigh')
xlabel('time [s]')
ylabel('Velocity [pixel/s]')
subplot(223)
plot (t(1:end-1),v(3,:))
title('Knee')
xlabel('time [s]')
ylabel('Velocity [pixel/s]')
subplot(224)
plot (t(1:end-1),v(4,:))
title('Ankle')
xlabel('time [s]')
ylabel('Velocity [pixel/s]')

Walking_distance = (sum(v(4,1:end))/14)*10^(-3);

angle = atand(diff(x)./diff(y));

figure
subplot(121)
plot (t,angle(1,:))
title('Hip to Thigh')
xlabel('time [s]')
ylabel('Angle [degree]')
subplot(122)
plot (t,angle(2,:))
title('Thigh to Knee')
xlabel('time [s]')
ylabel('Angle [degree]')
toc



