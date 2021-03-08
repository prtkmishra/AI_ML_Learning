function [pos2]=find_matches(I1,pos1,I2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introduction:
% Function to identify correspondence between 2 images Img1 and Im2. The 2
% images have been taken from slightly different angles but are of the same
% subject/ scene. The function identifies the correspondence between both
% the images by detection of interest points, feature extraction and
% matching the features. I shall discuss in detail about each step in
% detail now.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function can be used in below format
%    I1=im2double(imread('graffiti/img1.png'));
%    I2=im2double(imread('graffiti/img2.png'));
%    detector = detectHarrisFeatures(rgb2gray(I1)); 
%    pos1=detector.Location;
% 
%    find_matches(I1,pos1,I2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interest Point Detection:
%  Harris corner detecttion technique has been used to identify interest
%  points in both the images. I have used Gaussian Filter Size as 3 as it
%  gave better results overall. The Images had to be converted to 'gray' ti
%  to be used in detectHarrisFeatures() function. I tried using
%  detectSURFFeatures() but it exposed warning when processing few images
%  like graffiti as it reached maxium number of trials. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature Extraction:
% For feature extraction in both the images I have used extractFeatures()
% function where a 'simple square neighbourhood' of size 3 is used as the
% method for feature extraction. Based on several iterations, this method
% has given better results overall.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matching the extracted features:
% I have used matchFeatures() function to match the extracted features from
% both the images with a max ratio of 0.6. By using a lower value, not many
% matching points were identified and hence error observed. Using a higher
% value did not give much better results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transformation Matrix:
% I have used estimateGeometricTransform2D() to calculate the
% transformation matrix by using the matched feature indexes from both the
% images. The transformation corresponds to matching point pairs using MSAC
% algorithm. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculating pos2:
% Finally I have used transformPointsForward() using tform matrix from
% above step and pos1 input from the function to calculate corresponding
% coordinates in Image2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% References:
% Detection, Extraction and matching: https://uk.mathworks.com/help/vision/ug/find-image-rotation-and-scale-using-automated-feature-matching.html
% Transformation Matrix: https://uk.mathworks.com/help/images/ref/affine2d.transformpointsforward.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Interest point detection
Img1 = rgb2gray(I1);
Img2 = rgb2gray(I2);
% 
detector1 = detectHarrisFeatures(Img1,'FilterSize',3);
detector2 = detectHarrisFeatures(Img2,'FilterSize',3);

% detector1 = detectSURFFeatures(Img1);
% detector2 = detectSURFFeatures(Img2);

%% Extract Features

[featureImg1, pointsImg1] = extractFeatures(Img1,detector1,'Block',3);
[featureImg2, pointsImg2] = extractFeatures(Img2,detector2,'Block',3);

%% Match Features
indexPairs = matchFeatures(featureImg1, featureImg2,'MaxRatio',0.6);
% indexPairs = matchFeatures(featureImg1, featureImg2);
%% Segregate points in both the images for matched features
matchedImg1  = pointsImg1(indexPairs(:,1));
matchedImg2 = pointsImg2(indexPairs(:,2));

%% generate a transformation matrix
[tform] = estimateGeometricTransform2D(...
     matchedImg1, matchedImg2, 'similarity');

%% generate pos2 for image2 
[pos2] = transformPointsForward(tform,pos1);


%% Plot

showMatchedFeatures(I1,I2,pos1,pos2, 'montage');

% % 
