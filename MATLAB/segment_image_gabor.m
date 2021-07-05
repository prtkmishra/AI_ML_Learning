%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to segment image using gabor function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
I = im2double(imread('im4.jpg'));
X = rgb2gray(I);
orientation = [0,15,30,45,60,75,90,105,120,135,150,165];
figure(3), clf

image = 0;
for i = 1:length(orientation)
gmask = gabor2(1,0.5,orientation(i),1,90);
gmask1 = gabor2(1,0.5,orientation(i),1,0);
ibc = conv2(X,gmask,'valid');
ibc1 = conv2(X,gmask1,'valid');
convimage = sqrt(ibc.^2 + ibc1.^2);   
image = max(image,convimage);
end
X = im2uint8(image);
K = imsegkmeans(X,3,'NumAttempts',3);
counts = imhist(K,256);
T =  otsuthresh(counts);

nhood = [0 1 0; 1 1 1; 0 1 0];

BW = imbinarize(K,T);
J = imclose(BW,nhood);

subplot(2,2,1), imagesc(K), colormap('gray');
subplot(2,2,2), imagesc(BW), colormap('gray');
subplot(2,2,3), imagesc(J), colormap('gray');
% imagesc(J), colormap('gray');