function [EyeCalib] = EyeCalibration_FUNC(CalibFrames,eye)
% the function gets a matrix of the calibration frames and returns
%a matrix of the location of eyes for each frame
%for one eye. if eye = 1 left eye, if eye = 2 right eye

NUM = length(CalibFrames(1,:)); %number of calibration images
EyeCalib = zeros(NUM,8); %calibration place of eyes
i = 1;
for n = 1:4 %finding the place of Iris for eye
    for j = 1:NUM
        Frame = CalibFrames{n,j};
        Frame = rgb2gray(Frame); %color to BW photo
        Frame = im2double(Frame);
        Frame = Frame(230:277,155:390);

        Eye_Pos = EyePosition_FUNC(Frame); %finding eye position

        EyeCalib(j,i:i+1) = Eye_Pos(eye,:); %filling tha clibration matrix with the correct eye position
    end
i = i + 2;
end


end


