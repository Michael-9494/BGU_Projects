function [Eye_Pos]= EyePosition_FUNC(Frame)
% the function gets frane and returns the position of the Iris

IMAGE = Frame;
N = length(IMAGE(1,:));
Pattern_R = IMAGE(:,ceil(N/2+10):end); %image for each eye
Pattern_L = IMAGE(:,1:ceil(N/2));
[centers_R, ~] = imfindcircles(Pattern_R,[6 45],'ObjectPolarity','dark','Method','TwoStage'); %finding the Iris using find circle function
[centers_L, ~] = imfindcircles(Pattern_L,[5 35],'ObjectPolarity','dark','Method','TwoStage');
      
if isempty(centers_L) %if didn't find the Iris
   centers_L = [0 0];
else 
    if length(centers_L(:,1)) > 1 %if find more than one
       GOODPosition_L = find(centers_L(:,2)==max(centers_L(:,2))); %row number of nearest circle
       centers_L = centers_L(GOODPosition_L,:);
    end
end

if isempty(centers_R) %if didn't find the Iris
   centers_R = [0 0];
else
    if length(centers_R(:,1)) > 1 %if find more than one
       GOODPosition_R = find(centers_R(:,2)==min(centers_R(:,2))); %row number of nearest circle
       centers_R = centers_R(GOODPosition_R,:);
    end
end

Eye_Pos = [round(centers_L) ; round(centers_R)]; %position of Irises

end
