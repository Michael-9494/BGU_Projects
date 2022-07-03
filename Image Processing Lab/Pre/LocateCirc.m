function [n,Coordinates]=LocateCirc(IMG)
% Inputs:
% IMG - Gray levels image
%
% Outputs:
% n - number of identified patterns.
% Coordinates - a vector with the XY coordinates of the
% required pattern [ n * 2 (X Y)]

circle = imread('circle.jpeg');

IMG = im2gray(IMG);
circle = im2gray(circle);
interval = double(circle);
interval(circle<10)=(-1);interval(circle>200)=1;interval((10<circle)&(circle<200))=0;


findings = bwhitmiss(IMG,interval);

n = sum(sum(findings));
[x,y] = find(findings);
Coordinates = [x,y];