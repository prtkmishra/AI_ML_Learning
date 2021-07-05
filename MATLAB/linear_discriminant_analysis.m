%% Linear Discriminant Analysis
% LDA is used to calculate the weigths that can maximise the cost function
% J(w) = sb/sw
clc;
%% Input Parameters
fprintf('#################################')
fprintf(' Input variables ');
fprintf('#################################')
% Configure input samples for each class
% Input data is given as below:for 2 samples (x1,x2) and (x3,x4) the input
% is
% X = [ x1 x3
%     x2 x4]
C1 = [1 2 3
    2 1 3]
C2 = [6 7
    5 8]
% Weights are given in a column vector
W = [2
    -3];

%% Calculate Mean
fprintf('#################################')
fprintf(' Mean of each class ');
fprintf('#################################')
meanC1 = mean(C1,2)
meanC2 = mean(C2,2)

%% Calculate sb (Between class scatter)
fprintf('#################################')
fprintf(' Between class scatter ');
fprintf('#################################')
sb = norm(W'*(meanC1-meanC2)).^2

%% Calculate sw (within class scatter)
fprintf('#################################')
fprintf(' within class scatter ');
fprintf('#################################')
sw =[];
% Loop for Class 1
for i = 1:size(C1,2)
   sw = [sw (W'*(C1(:,i)-meanC1)).^2];
end
% Loop for Class 2
for i = 1:size(C2,2)
   sw = [sw (W'*(C2(:,i)-meanC2)).^2];
end
% Sum of sw
sw = sum(sw)

%% Calculate the final cost of the weights
fprintf('#################################')
fprintf(' Total Cost ');
fprintf('#################################')
Cost = sb/sw

%% Projection of data
fprintf('#################################')
fprintf(' Projection of data ');
fprintf('#################################')
Y1 = W'*C1;
Y2 = W'*C2;
X = [C1 C2]
Y = [Y1 Y2]