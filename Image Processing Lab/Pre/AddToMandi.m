function [ResultPic]=AddToMandi(Picture)
mandi = imread('mandi.tif');
Picture = im2gray(Picture);

% create frame for skeleton
mask = Picture;
mask(mask<240)=0;
mask(mask>240)=255;
mask = 255-mask;

% insert zeros to free some space for the skeleton in mandi
mandiblank=mandi;
mandiblank(500:length(mask)+499,700:width(mask)+699) = mandi(500:length(mask)+499,700:width(mask)+699)-mask;


% adding skeleton to mandi
mandiplusPicture = mandiblank;
Picture(Picture>240)=0;
mandiplusPicture(500:length(mask)+499,700:width(mask)+699) = mandiplusPicture(500:length(mask)+499,700:width(mask)+699)+Picture;


ResultPic = mandiplusPicture;
