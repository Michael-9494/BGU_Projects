function [answer] = commaornot(x_t)
%this function reveals if a 1 second signal its a word or a comma by
%cheking its avreage power, if the avreage power is 0 so its a comma
%for a comma answer=0, and for a word answer=1.

 power2 = x_t.^2;
 avpower = 1/8000*(sum(power2));
 if avpower == 0 
     answer = 0
 else
     answer = 1
 end 