%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script uses the concept of Bag of Words to count number of Lego
% blocks in an image based on a template
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% imds = imageDatastore('images/images_/','IncludeSubfolders',1,'LabelSource','foldernames','FileExtensions',{'.jpg'});
imds = imageDatastore('images/images_/','IncludeSubfolders',1,'LabelSource','foldernames','FileExtensions',{'.jpg'});
T1 = imread("images/blue_2by4brick.jpg");

bag = bagOfFeatures(imds,'PointSelection','Detector');
classifier = trainImageCategoryClassifier(imds,bag);
[featureVector, words] = encode(bag, T1);
% imds1 = imageDatastore('images/training_images/','LabelSource','foldernames','FileExtensions',{'.jpg'});
confMatrix = evaluate(classifier, imds);

I = imread("images/training_images/train06.jpg");%% Converting image to L a* b* feature space and use a* and b*
test_gray = rgb2gray(I); 
rg = I(:,:,1);                                                              % select red dimensions from test image
gr = I(:,:,2);                                                              % select green dimensions from test image
r = imsubtract(rg,gr);                                                      % Subtract Green from Red to discard any YELLOW pixels
b = I(:,:,3);                                                               % Select blue dimensions from test image

red = imsubtract(r,test_gray);                                              % Subtract identified red segment from main image (Gray)
blue = imsubtract(b,test_gray);                                             % Subtract identified blue segment from main image (Gray)

%% Image segmentation using K-means clustering
iter = 3;
J = imsegkmeans(red,iter,'NumAttempts',iter);

%% Edge detections using gabor masks
orientation = 0:10:359;
smimage = 0;

for i = 1:length(orientation)
    gmask = gabor2(1,0.5,orientation(i),1,90);
    gmask1 = gabor2(1,0.5,orientation(i),1,0);
    ibc = conv2(J,gmask,'same');
    ibc1 = conv2(J,gmask1,'same');
    convimage = sqrt(ibc.^2 + ibc1.^2);   
    smimage = max(smimage,convimage);
end
%% Binarise the output
counts = imhist(smimage,16);
T =  otsuthresh(counts);
H = imbinarize(smimage, T);

%% apply closed morphological operations
nhood = [0 1 0; 1 1 1; 0 1 0];
seg= imclose(H,nhood);

imshow(seg);
hold on;                                    % Generate region properties for red segments

% [labelIdx, scores] = predict(classifier, I);
% classifier.Labels(labelIdx)