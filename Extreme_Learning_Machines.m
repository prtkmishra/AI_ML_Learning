%% Extreme Learning Machines

clc;
%% Input Parameters
fprintf('#################################')
fprintf(' Input Parameters ');
fprintf('#################################')
% Input matrix given as below. where rows correspond to samples and
% columns correspond to dimensions. for e.g. (x1,x2) and (x3,x4) can be
% written as 
% X = [x1 x3
%     x2 x4]
X = [1
    0.90
    -0.90
    -1.0]

% Configure the random matrix used between input and hidden layers
V = [0.90 0.80 0.50 -1.0
    -0.90 -0.90 0.70 -1.0
    0.10 -0.80 -0.30 -0.50
    -0.90 1.0 0.10 0.60
    0.10 -0.40 -0.90 0.20]

% Define the weights between hidden and output layers
W = [0.65 0.65 0.00 -0.60 0.00 -0.60
    0.05 0.05 0.00 0.40 0 0.40];


%% Run the algorithm for classification
V*X
H = heaviside(V*X)
% Y = [repmat(1,1,size(X,2)); heaviside(V*X)]
Y = [repmat(1,1,size(X,2)); H]
Z = W*Y