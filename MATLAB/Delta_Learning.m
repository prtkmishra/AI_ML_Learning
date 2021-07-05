%% Delta learning Rule
% This script calculates weight for Delta learning rule
% w ← w + η(t − y)x'
%% System Variables
MODE = 1; % 0=Sequential, 1 = Batch

%% Input Parameters
% Weights are given in a row vector; Weights = [1 1]
% Theta is a scalar
% Learning rate is a scalar
% Input sample is the column in X. For e.g., (0,0) (0,1) (1,0) (1,1) will
% be written as below in matrix form
% Input = [0 0 1 1
%     0 1 0 1]
% target label y is given in a row vector form; y = [0 0 0 1]
% Define number of epochs for the training; epoch = 10
clc;
fprintf('#################################')
fprintf(' Input Variables ');
fprintf('#################################')

Input = [0 1] % Input samples
y = [1 0] % Target output
Weights = [2] % Weights
theta = 1.5 % Threshold
learning_rate = 1; % Learning rate for training

epoch = 10; % Number of epochs for training

%% Augmented Notation
fprintf('#################################')
fprintf(' Augmented Notation ');
fprintf('#################################')

W = [-theta Weights]

X = [repmat(1,1,size(Input,2));Input]
%% Train the algorithm
UW = 0
for e = 1:epoch
    fprintf('#################################')
    fprintf(' Weights at the end of epoch %d ',e);
    fprintf('#################################')
    if MODE == 0
        for i = 1:size(X,2)
            W = W+learning_rate*(y(:,i)-heaviside(W*X(:,i)))*X(:,i)';
        end
    else
        UW =0;
        for i = 1:size(X,2)
            UW = UW+(learning_rate*(y(:,i)-heaviside(W*X(:,i)))*X(:,i)');
        end
        W = W+UW
    end
end

%% Final Weights
fprintf('#################################')
fprintf(' Final Weights at the end of epoch %d ',e);
fprintf('#################################')
W
Theta = -W(1,1)
Weigths = W(:,2:size(W,2))
