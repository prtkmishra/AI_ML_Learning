%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script uses the concept of feature mapping to count number of Lego
% blocks in an image based on a template
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T1 = imread("images/blue_2by4brick.jpg");
I = imread("images/training_images/train04.jpg");

T1_Gray = rgb2gray(T1);
I_Gray = rgb2gray(I);

IP_T1 = detectSURFFeatures(T1_Gray);
IP_I  = detectSURFFeatures(I_Gray);

[IP_I_Feat, IP_I_pts] = extractFeatures(I_Gray, IP_I,'FeatureSize',128);
[IP_T1_Feat, IP_T1_pts] = extractFeatures(T1_Gray,IP_T1,'FeatureSize',128);
size(IP_I_pts)
% hist(IP_I_Feat(1,:))
% E = evalclusters(IP_T1_Feat,'kmeans','CalinskiHarabasz','klist',[1:5]);
T1_cluster = kmeans(IP_T1_Feat,3);
I_cluster = kmeans(IP_I_Feat,3);
% ref_pairs = matchFeatures(IP_I_Feat,IP_T1_Feat);
% ref_pairs = matchFeatures(IP_I_Feat,IP_T1_Feat, 'Metric', 'SSD');
ref_pairs = matchFeatures(IP_I_Feat,IP_T1_Feat, 'MatchThreshold', 10);
size(ref_pairs)
OriginalPointsMatched = IP_I_pts(ref_pairs(:,1),:);
refPointsMatched = IP_T1_pts(ref_pairs(:,2),:);
[tform, inlierIdx] = estimateGeometricTransform2D(OriginalPointsMatched,refPointsMatched, 'projective');
origPtsInlier = OriginalPointsMatched(inlierIdx,:);
refPtsInlier  = refPointsMatched(inlierIdx,:);

% boxPolygon = [1, 1;...                           % top-left
%         size(T1, 2), 1;...                 % top-right
%         size(T1, 2), size(T1, 1);... % bottom-right
%         1, size(T1, 1);...                 % bottom-left
%         1, 1];                   % top-left again to close the polygon
% newBoxPolygon = transformPointsForward(tform, boxPolygon);
figure;
showMatchedFeatures(I,T1,OriginalPointsMatched,refPointsMatched,'montage');
% imshow(I);
% hold on;
% plot(IP_I_pts);
% line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
% title('Detected Box');
% figure, showMatchedFeatures(I,T1,origPtsInlier,refPtsInlier,'montage');
% imshowpair(I,IP_I_pts,'montage')
% plot(T1_cluster);
