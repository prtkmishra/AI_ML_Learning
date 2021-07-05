%% Negative Feedback Network
% e = x − W'y
% y ← y + αWe

%% Input variables
% Weights of the network are given as a matrix. 
% Row corresponds to weights between Input node x(i)and output Y1 and Y2 correspondingly
clc;
fprintf('#################################')
fprintf(' Input variables ');
fprintf('#################################')
W = [1 1 0
    1 1 1]
X = [1
    1
    0]
alpha = 0.5
% Initialize output as 0
y = [0
    0]
epoch = 5; % Number of iterations for training

%% train the network for y
for i = 1:epoch
    fprintf('#################################')
    fprintf(' Updated Y at end of epoch %d ',i);
    fprintf('#################################')
    e = X-(W'*y)
    y = y+alpha*W*e
end

    