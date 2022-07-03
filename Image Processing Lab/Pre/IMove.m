function [M]=IMove(Track)
% Input:
% Track - a vector [n x 2] contains movement instruction for
% stick-man. the first column contains axis:
% 1=movement in the X-axis
% 2=movement in the Y-axis
% the second column contains number of steps in the same
% direction. %
% Output:
% M - a Matrix [ 50 x 50 x (n*steps) ] contains the movements % step-by-step.

stick(5,3)=256; stick(4,3)=256; stick(3,3)=256; stick(4,2)=256; stick(4,4)=256;stick(5,2)=256;stick(5,4)=256;stick(6,3)=256;
stick(6:17,3)=256; stick(8,4)=256; stick(8,5)=256; stick(8,2)=256; stick(8,1)=256; 
stick(18,4)=256; stick(19,5)=256; stick(18,2)=256; stick(19,1)=256; stick(18,3)=256;

N=50;M=zeros(N,N,sum(abs(Track(:,2))));
[n,m]=size(stick);
M(N-n:N-1,2:m+1,1)=stick;

j=1; k=1;
while j<=sum(abs(Track(:,2))) || k<length(Track)
    if Track(k,1) == 1
        if Track(k,2)>0
            H1=[0 1 1];H2=[1 1 0];
            for i=1:Track(k,2)
                M(:,:,j+i)=imdilate(M(:,:,j+i-1),H1);
                M(:,:,j+i)=imerode(M(:,:,j+i),H2);
            end
            j=j+i-1;
        end
        if Track(k,2)<0
            H1=[1 1 0];H2=[0 1 1];
            for i=1:abs(Track(k,2))
                M(:,:,j+i)=imdilate(M(:,:,j+i-1),H1);
                M(:,:,j+i)=imerode(M(:,:,j+i),H2);
            end
            j=j+i-1;
        end
    end
    if Track(k,1) == 2
        if Track(k,2)>0
            H1=[1 1 0]';H2=[0 1 1]';
            for i=1:Track(k,2)
                M(:,:,j+i)=imdilate(M(:,:,j+i-1),H1);
                M(:,:,j+i)=imerode(M(:,:,j+i),H2);
            end
            j=j+i-1;
        end
        if Track(k,2)<0
            H1=[0 1 1]';H2=[1 1 0]';
            for i=1:abs(Track(k,2))
                M(:,:,j+i)=imdilate(M(:,:,j+i-1),H1);
                M(:,:,j+i)=imerode(M(:,:,j+i),H2);
            end
            j=j+i-1;
        end
    end
    j=j+1; k=k+1;
end
end
