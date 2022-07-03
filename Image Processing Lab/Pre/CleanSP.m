function out_I=CleanSP(in_I,Type,var1,var2)
% this function take image (in_I) ad considering the Type it will filter
% the image

    switch lower(Type)
        case 'gaussian'
            h = fspecial('gaussian',[var1 var2]);
            out_I = imfilter(in_I,h,'replicate');
        case 'median'
            out_I = medfilt2(in_I,[var1 var2]);
        otherwise
            error('Unknown filter type.')
    end
end