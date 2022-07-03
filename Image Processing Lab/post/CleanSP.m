function out_I=CleanSP(in_I,Type,var1,var2)
% the function gets an image and filters the noise using different filter
% type, depends on the fiter type discribed in Type variable.
% var 1 = number of raws
% var 2 = number of columns

    m = var1; %size of filter
    n = var2;
   
if strcmp(Type,'Gaussian')
    %create low pass filter of type Gaussian
    GausFilter = fspecial('gaussian',[m,n]); %creating gaussian filter of size [m ,n]
    %filtering the signal with the Gaussian filter
    out_I = imrotate(filter2(in_I,GausFilter),180);
else 
    if strcmp(Type,'Median') 
         %create filter of type Median and filtering the signal with the Median filter
        out_I = medfilt2(in_I,[m n]); 
       
    end
end


end