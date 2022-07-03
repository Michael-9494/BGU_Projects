function [ClusterMeans,SegmentedIn] = K_meansRGB(RGB_Im,K)
s=size(RGB_Im);
N = s(1)*s(2);
%making a vector of image
x=[];
for i=1:s(3)
    image = reshape(RGB_Im(:,:,i),N,1);
    x = [image x];
end
x = double(x);
% inital random condition
Idx = randi(size(x,1),2,1);
ClusterMeans = x(Idx,:);
position=zeros(N,1);
distance=zeros(N,K);
while(1)
    %predict matching cluster per pixel
    for k=1:K
        Vec = (x - ClusterMeans(k,:));
        distance(:,k) = sum((Vec.^2),2); % distance of each pixel from cluster
    end
    [~,I] = min(distance,[],2);
    position=I;
    %recalculate clusters based on positions
    Previous_cluster=ClusterMeans;
    for k=1:K
        Idx =find(position==k);
        PixelsAtCluster = x(Idx,:);
        ClusterMeans(k,:) = sum(PixelsAtCluster)/(length(Idx)+1);
    end
    % stop condition
    d = sum(sum(abs(ClusterMeans-Previous_cluster)));
    if(d <1e-3)
        break;
    end
end
SegmentedIn = reshape(position,s(1),s(2));

end




