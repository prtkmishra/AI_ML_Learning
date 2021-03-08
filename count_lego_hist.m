%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script uses the concept of binarizing to identify number of Lego
% blocks in an image based on a template
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Isample = imread("images/red_2by2brick.jpg"); 
I = imread("images/training_images/train06.jpg");

test_graysample = rgb2gray(Isample);                                                    % Convert I into gray scale
test_gray = rgb2gray(I);                                                    % Convert I into gray scale
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Image Segmentation based on Colour
%       Segment Image in red and blue segments
%       RED = (255,0,0); GREEN = (0,255,0); BLUE = (0,0,255); YELLOW =
%       (255,255,0); AQUA = (0,255,255); Magenta = (255,0,255)
rgsample = Isample(:,:,1);                                                              % select red dimensions from test image
grsample = Isample(:,:,2);                                                              % select green dimensions from test image
rsample = imsubtract(rgsample,grsample);                                                      % Subtract Green from Red to discard any YELLOW pixels
bsample = Isample(:,:,3);                                                               % Select blue dimensions from test image

redsample = imsubtract(rsample,test_graysample);                                              % Subtract identified red segment from main image (Gray)
bluesample = imsubtract(bsample,test_graysample);                                             % Subtract identified blue segment from main image (Gray)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rg = I(:,:,1);                                                              % select red dimensions from test image
gr = I(:,:,2);                                                              % select green dimensions from test image
r = imsubtract(rg,gr);                                                      % Subtract Green from Red to discard any YELLOW pixels
b = I(:,:,3);                                                               % Select blue dimensions from test image

red = imsubtract(r,test_gray);                                              % Subtract identified red segment from main image (Gray)
blue = imsubtract(b,test_gray);                                             % Subtract identified blue segment from main image (Gray)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Binarize the Images

red_binsample = imbinarize(redsample,0.15);                                             % Binarize red segmented image                                
blue_binsample = imbinarize(bluesample,0.15);                                           % Binarize blue segmented image


red_bin = imbinarize(red,0.15);                                             % Binarize red segmented image                                
blue_bin = imbinarize(blue,0.15);                                           % Binarize blue segmented image
figure;
subplot(2,1,1), imshow(red_bin)
subplot(2,1,2), imshow(red_binsample)