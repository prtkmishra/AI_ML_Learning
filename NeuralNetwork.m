%% FeedForward Neural Network
% This script calculates output for given inputs and weights for a FF neural network
% This is a 4-3-2 NN. i.e. 4 input nodes, 1 hidden layer with 3 nodes and 2
% output nodes
%% Input Variables
clc;
fprintf('#################################')
fprintf(' Input variables ');
fprintf('#################################')
% The input X is give as column vector
X = [0.5
    0.4]
% The weight matrix is given in matrix form. Where row belongs to weights
% from input nodes to a single hidden node. Here we have 4 input nodes and
% 3 hidden nodes
Wji = [-2 3
    -5 -4
    -5 -3]
% Weights for bias from input to hidden node
Wjzero = [-5
    -4
    4]

% Weights from hidden nodes to output nodes
Wkj = [4 4 5
    -1 -3 -1]

% Weights from hidden bias node to output nodes
Wkzero = [4
    3]

%% FeedForward NN
fprintf('#################################')
fprintf(' Hidden Layer output ');
fprintf('#################################')
Y = tansig(Wji*X + Wjzero) % Hidden Layer 1 Activation Function
fprintf('#################################')
fprintf(' Final output ');
fprintf('#################################')
Z = logsig(Wkj*Y+Wkzero) % Activation 