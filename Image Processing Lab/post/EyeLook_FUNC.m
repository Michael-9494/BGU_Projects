function [Eye_Look]= EyeLook_FUNC(Frame,EyeCalib)
% the function gets a frame and the calibration matrix and returns the
% number with the highest probability that the eye is looking at.
% by using the right eye coordinates

Eye_Pos = EyePosition_FUNC(Frame); %eye position for each eye in the frame
Right_Eye = Eye_Pos(2,:);

Mean_Number = mean(EyeCalib(:,1:2:end)); %mean for each number in X coordinate

%min distance from mean cali
NumLOOK_R = find(abs(Mean_Number-Right_Eye(1).*ones(1,4)) == min(abs(Mean_Number-Right_Eye(1).*ones(1,4))));

Eye_Look = NumLOOK_R; %number looking at in frame

end