
function [seg]=segment_image(I)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input an image as I for segmentation
% 
% Assumptions:
%   The image has been converted to double format using "im2double"
% 
% Process:
%   -   The input image is converted from RGB to L a* b* [1]
%   -   The L a* b* space gives use the information where:
%           L = Luminosity layer
%           a = chromaticity-layer indicating where color falls along the red-green axis
%           b = chromaticity-layer indicating where the color falls along the blue-yellow axis
%   -   The image is converted to "single" to be used in K-means image segmentation algorithm
%   -   MATLAB function imsegkmeans() is used to segment the image based on colours
%   -   Parameters used for k-means clustering are:
%       -   Number of clusters = 3. After much experimentation with different
%           values of clusters ranging between 2 and 30. Number of colours 3 is
%           a preferred value. I can confidently say that for all images number
%           of clusters as 3 is the most optimised value
%       -   Number of attempts = 3. This parameters defines the number of iterations for k-means clustering.
%           Although There is not much visible difference between 1 and 5
%           iterations, value 3 is a preferred value to avoid local minima.
%   -   Once the image has been segmented in 3 clusters using K-means. we apply
%       gabor filters to identify edges of the identified clusters.
%   -   For gabor filters, below parameters were used:
%       -   orientation: multiple orientations with a difference of 10deg
%       -   Sigma: 1. After much experimentation with different sigma
%           values, value 1 is identified as the most fit value for size of
%           pixels
%       -   freq: 0.5
%       -   aspect:1. A circular symmetric envelope
%       -   phase: 0/90. Both the gabor masks are 90deg phase shifted to
%           each other
%       -   The outputs of gabor masks are then convoluted with same shape.
%   -   Once the edges have  been identified, we binarize the image
%   -   To binarize the output of gabor filters we need to identify a threshold. [2] 
%   -   To identify threshold for above step, we use an inbuilt MATLAB
%       function called otsuthresh() to use Otsu's method for global hist
%       thresh
%   -   Once the image is binarised, we use a 3x3 vector as structural
%       element as neighbourhood for Closed morphological operation.
%   -   The output is binary image with same size as of the input image.
%   
% Limitations:
%   -   The applied method fails to identify objects when there is not much
%       variance in a* and b* between foreground and background in the image
%   -   This method may fail to correctly identify edges where variance is
%       very minimal.
%   -   Images with cluttered background creates noise
% 
% Other approaches tried:
%   -   Other methods like thresholding and region based segmentation
%   -   Thresholding combined with LoG did not provide as good results
%   -   canny edge detector with k-means produced image with higher noise
%   -   Other methods like Hough transform did not work well with all the
%       images
% 
% References:
%   -   [1] https://uk.mathworks.com/help/images/color-based-segmentation-using-k-means-clustering.html
%   -   [2] https://uk.mathworks.com/help/images/ref/imhist.html
%   -   [2] https://uk.mathworks.com/help/images/ref/imbinarize.html


%% Converting image to L a* b* feature space and use a* and b*
image = rgb2lab(I);
abspace = image(:,:,2:3);
abspace = im2single(abspace);

%% Image segmentation using K-means clustering
iter = 3;
J = imsegkmeans(abspace,iter,'NumAttempts',iter);

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

%% plot output
figure(1), clf, 
imagesc(seg),colormap('gray');